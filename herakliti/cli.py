"""The command line — Herakliti's face.

Windows console note, first thing in the file for a reason: this console is cp1252
by default, which silently mangles every character this system exists to handle
(ë, ç, and the em dash) into '?'. Reconfiguring the streams to UTF-8 must happen
before anything imports rich or writes a byte, because rich snapshots the stream
encoding when it builds a Console. stdin is included: the user types Albanian too,
and a mangled question retrieves nothing.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":  # pragma: no cover - platform specific
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        # pythonw / some test harnesses hand us a stream without reconfigure().
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from herakliti import __version__, config
from herakliti.engine import loader
from herakliti.engine.registry import DEFAULT_ALIAS, MODELS, ModelSpec, resolve

if TYPE_CHECKING:
    from herakliti.brain.agent import Herakliti
    from herakliti.knowledge.types import Answer, Chunk

console = Console()
err = Console(stderr=True)

app = typer.Typer(
    name="herakliti",
    help="A local, citing question-answering system. Grounded or silent.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Timing:
    """What the user actually waited for. On this hardware they deserve to know."""

    total: float = 0.0
    ttft: float | None = None
    tokens: int = 0
    exact: bool = True


def _prog() -> str:
    """However this was invoked, so 'run X' hints are copy-pasteable."""
    name = Path(sys.argv[0]).name
    if name.startswith("herakliti"):
        return "herakliti"
    if name == "main.py":
        return "python main.py"
    return "python -m herakliti.cli"


def _apply_globals(model: str | None, offline: bool, verbose: bool) -> None:
    """Flags win over env defaults, but absent flags must not clobber the env."""
    if model:
        config.SETTINGS.model = model
    if offline:
        config.SETTINGS.offline = True
    if verbose:
        config.SETTINGS.verbose = True


def _spec_or_exit(model: str | None) -> ModelSpec:
    try:
        return resolve(model or config.SETTINGS.model)
    except KeyError as e:
        err.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(2)


def _require_downloaded(spec: ModelSpec) -> None:
    """A first run must say what is missing, not block silently for five minutes."""
    if loader.is_downloaded(spec):
        return
    err.print(
        Panel(
            f"Model [bold]{spec.alias}[/bold] ({spec.params}, {spec.size_gb:.1f} GB) "
            f"is not on this machine yet.\n\n"
            f"Fetch it once:\n\n"
            f"    [bold cyan]{_prog()} pull {spec.alias}[/bold cyan]\n\n"
            f"[dim]It lands in {escape(str(loader.local_path(spec)))} and is reused forever.\n"
            f"{escape(spec.note)}[/dim]",
            title="[yellow]No model yet[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    raise typer.Exit(1)


def _agent(model: str | None, offline: bool) -> "Herakliti":
    from herakliti.brain.agent import Herakliti  # llama_cpp/torch live under here

    return Herakliti(model=model or config.SETTINGS.model, offline=offline or config.SETTINGS.offline)


def _store_stats() -> dict[str, Any]:
    """Read the store's own numbers.

    Deliberately not via Herakliti.stats(): that reports the backend by touching
    .llm, which loads the weights — measured at 18s and 2.7 GB to print a table.
    KnowledgeStore is sqlite + faiss only, and the engine facts below are already
    knowable without a model.
    """
    try:
        from herakliti.knowledge.store import KnowledgeStore

        # No close(): close() saves, and saving this read-only snapshot could
        # overwrite the index of whatever process is actually writing.
        return KnowledgeStore().stats()
    except Exception as e:
        err.print(f"[dim]store unreadable: {escape(str(e))}[/dim]")
        return {}


def _backend() -> str:
    """Mirror LLM.backend without loading a model — the probe only needs the build.

    Muted at fd level because merely importing llama_cpp makes ggml announce the
    Vulkan device on stderr, which lands in the middle of whatever we're drawing.
    """
    try:
        from herakliti.engine.llm import _quiet, gpu_offload_default, has_gpu_backend

        with _quiet():
            return "gpu" if gpu_offload_default() != 0 and has_gpu_backend() else "cpu"
    except Exception:
        return "unknown"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _disk_bytes() -> int:
    return sum(p.stat().st_size for p in (config.DB_PATH, config.FAISS_PATH) if p.exists())


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _kv(data: dict[str, Any]) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    t.add_column(style="cyan")
    for k, v in data.items():
        t.add_row(str(k), escape(str(v)))
    return t


def _render_stats(spec: ModelSpec) -> None:
    store = _kv(_store_stats())
    store.add_row("index size", _human(_disk_bytes()))
    store.add_row("index dir", str(config.INDEX_DIR))
    console.print(Panel(store, title="Store", border_style="cyan", padding=(0, 1)))

    engine = _kv(
        {
            "model": spec.label,
            "file": spec.filename,
            "downloaded": "yes" if loader.is_downloaded(spec) else f"no — `{_prog()} pull {spec.alias}`",
            "backend": _backend(),
            "threads": config.SETTINGS.n_threads,
            "context": f"{config.SETTINGS.n_ctx} tokens",
            "embeddings": config.SETTINGS.embed_model,
            "offline": "yes" if config.SETTINGS.offline else "no",
        }
    )
    console.print(Panel(engine, title="Engine", border_style="cyan", padding=(0, 1)))


def _print_sources(ans: "Answer") -> None:
    if not ans.citations:
        if ans.used_retrieval:
            console.print("[dim]No sources survived filtering — answered without them.[/dim]")
        return
    # Plain positional numbering, matching the [1], [2]... blocks the model was shown:
    # Answer.citations is guaranteed one-chunk-per-url by dedupe_by_url upstream (see
    # herakliti.knowledge.types), so index-in-list *is* the citation number here.
    lines = [
        f"[bold]{i}.[/bold] {escape(c.title)}\n   [blue]{escape(c.url)}[/blue]"
        for i, c in enumerate(ans.citations, 1)
    ]
    console.print(Panel("\n".join(lines), title="Sources", border_style="cyan", padding=(0, 1)))


def _print_trace(ans: "Answer") -> None:
    head = f"used_retrieval={ans.used_retrieval} · agent elapsed {ans.elapsed:.2f}s"
    body = [f"[dim]{head}[/dim]"]
    body += [f"[dim]{i:>2}[/dim] {escape(step)}" for i, step in enumerate(ans.trace, 1)]
    console.print(Panel("\n".join(body), title="Trace", border_style="magenta", padding=(0, 1)))


def _print_timing(t: _Timing, ans: "Answer") -> None:
    approx = "" if t.exact else "~"
    bits = [f"{t.total:.1f}s"]
    if t.ttft is not None:
        bits.append(f"{t.ttft:.1f}s to first token")  # prefill: the real bottleneck here
    bits.append(f"{approx}{t.tokens} tok")
    decode = t.total - (t.ttft or 0.0)
    if t.tokens and decode > 0.05:
        bits.append(f"{approx}{t.tokens / decode:.1f} tok/s")
    # Confidence is only meaningful for a grounded answer. For a reasoning answer the number
    # would be noise, and for the fallback the honest label is "no source", not a percentage.
    if ans.used_retrieval:
        if ans.confidence:
            bits.append(f"confidence {ans.confidence:.0%}")
    elif any("unverified" in s for s in ans.trace):
        bits.append("[yellow]unverified — no source found[/yellow]")
    console.print("[dim]" + " · ".join(bits) + "[/dim]")


def _answer(h: "Herakliti", question: str, *, stream: bool) -> tuple["Answer", _Timing] | None:
    """Run one question, printing the answer as it arrives. None means it failed."""
    from herakliti.knowledge.types import Answer

    t = _Timing()
    t0 = time.perf_counter()
    status = console.status("[dim]Thinking…[/dim]", spinner="dots")

    if not stream:
        try:
            status.start()
            ans = h.ask(question)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
            return None
        except Exception as e:
            err.print(f"[red]Query failed:[/red] {escape(str(e))}")
            return None
        finally:
            status.stop()
        console.print(ans.text, markup=False, highlight=False)
        t.total = time.perf_counter() - t0
        # No per-token signal without streaming; ~4 chars/token, and say so with '~'.
        t.tokens = max(1, len(ans.text) // 4)
        t.exact = False
        return ans, t

    buf: list[str] = []
    try:
        status.start()
        for piece in h.stream_ask(question):
            if t.ttft is None:
                t.ttft = time.perf_counter() - t0
                status.stop()
            t.tokens += 1  # llama.cpp emits one delta per token
            buf.append(piece)
            console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return None
    except Exception as e:
        err.print(f"\n[red]Query failed:[/red] {escape(str(e))}")
        return None
    finally:
        status.stop()

    console.print()
    t.total = time.perf_counter() - t0
    ans = h.last_answer or Answer(text="".join(buf), elapsed=t.total)
    return ans, t


def _respond(h: "Herakliti", question: str, *, stream: bool, verbose: bool) -> "Answer | None":
    got = _answer(h, question, stream=stream)
    if got is None:
        return None
    ans, timing = got
    console.print()
    _print_sources(ans)
    if verbose:
        _print_trace(ans)
    _print_timing(timing, ans)
    return ans


# --------------------------------------------------------------------------
# Taught memory — teach / forget / recall a fact
# --------------------------------------------------------------------------


def _teach(h: "Herakliti", fact: str, note: str = "") -> bool:
    """Store one fact, printing the agent's own confirmation. False on failure."""
    try:
        confirmation = h.teach(fact, note=note)
    except Exception as e:
        err.print(f"[red]Could not store that:[/red] {escape(str(e))}")
        return False
    console.print(f"[green]✓[/green] {escape(str(confirmation))}")
    return True


def _teach_auto(h: "Herakliti", line: str) -> None:
    """The auto-detected path: keep the acknowledgement light, matching the chat flow."""
    try:
        h.teach(line)
    except Exception as e:
        err.print(f"[red]Could not store that:[/red] {escape(str(e))}")
        return
    console.print("[green]✓ Got it — I will remember that.[/green]")


def _print_memories(mems: "list[Chunk]") -> None:
    """Render taught facts, or a friendly nudge when there are none."""
    if not mems:
        console.print(
            Panel(
                "You have not taught me anything yet.\n"
                f'[dim]Try: [cyan]{_prog()} teach "My dog is named Rex"[/cyan][/dim]',
                title="Memories",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        return

    lines: list[str] = []
    for i, c in enumerate(mems, 1):
        text = escape(str(getattr(c, "text", "") or "").strip())
        # The teach layer parks a human label (often a date) in the title; show it
        # only when it adds something the fact text does not already say.
        title = escape(str(getattr(c, "title", "") or "").strip())
        line = f"[bold]{i}.[/bold] {text}"
        if title and title.lower() not in text.lower():
            line += f"\n   [dim]{title}[/dim]"
        lines.append(line)
    console.print(Panel("\n".join(lines), title=f"Memories ({len(mems)})", border_style="cyan", padding=(0, 1)))


def _show_memories(h: "Herakliti") -> None:
    try:
        mems = h.memories()
    except Exception as e:
        err.print(f"[red]Could not read memories:[/red] {escape(str(e))}")
        return
    _print_memories(mems)


def _is_teaching(line: str) -> bool:
    """Is this line the user telling us a fact rather than asking a question?

    Fail soft: if the (concurrently written) learn module is unavailable, treat the
    line as an ordinary question so the REPL never breaks over a missing import.
    """
    try:
        from herakliti.brain import learn

        return learn.is_teaching(line)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="fast | default | quality", metavar="ALIAS")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="No network: answer from the local store only.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show the retrieval trace and timings.")]


def _version(value: bool) -> None:
    if value:
        console.print(f"herakliti {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """Herakliti — retrieval is the point. Every answer carries its sources."""


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in any language.")],
    model: ModelOpt = None,
    offline: OfflineOpt = False,
    verbose: VerboseOpt = False,
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Print tokens as they are produced.")] = True,
) -> None:
    """Ask one question. Prints the answer, then its sources."""
    _apply_globals(model, offline, verbose)
    _require_downloaded(_spec_or_exit(model))
    h = _agent(model, offline)
    if _respond(h, question, stream=stream, verbose=verbose) is None:
        raise typer.Exit(1)


@app.command()
def chat(model: ModelOpt = None, offline: OfflineOpt = False, verbose: VerboseOpt = False) -> None:
    """Interactive session. /help lists the commands."""
    _apply_globals(model, offline, verbose)
    spec = _spec_or_exit(model)
    _require_downloaded(spec)
    h = _agent(model, offline)

    console.print(
        Panel(
            f"[bold]Herakliti {__version__}[/bold] · {spec.label} · {_backend()} · "
            f"{config.SETTINGS.n_threads} threads"
            + (" · [yellow]offline[/yellow]" if config.SETTINGS.offline else "")
            + "\n[dim]/help for commands, /exit to leave.[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # Load the model now, behind a spinner, rather than stalling silently on the first
    # message. The weights take a few seconds to map; doing it here keeps the first reply
    # snappy and the prompt free of a mysterious pause.
    try:
        with console.status("[dim]Waking the model…[/dim]", spinner="dots"):
            _ = h.llm
    except Exception as e:
        err.print(f"[red]Could not load the model:[/red] {escape(str(e))}")
        raise typer.Exit(1)

    history: list[tuple[str, "Answer"]] = []
    while True:
        try:
            line = console.input("\n[bold cyan]›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not _slash(h, line, history, spec):
                break
            continue
        # "Just tell it things": a statement of fact is stored, not retrieved against.
        if _is_teaching(line):
            _teach_auto(h, line)
            continue
        ans = _respond(h, line, stream=True, verbose=verbose)
        if ans is not None:
            history.append((line, ans))

    console.print("[dim]Everything flows.[/dim]")


def _slash(h: "Herakliti", line: str, history: list[tuple[str, "Answer"]], spec: ModelSpec) -> bool:
    """Handle a /command. False means: leave the REPL."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""  # original case preserved for facts

    if cmd in ("/exit", "/quit"):
        return False

    if cmd == "/clear":
        history.clear()
        console.clear()
        console.print("[dim]Transcript cleared.[/dim]")

    elif cmd == "/sources":
        if not history:
            console.print("[dim]Nothing asked yet.[/dim]")
        else:
            _print_sources(history[-1][1])

    elif cmd == "/stats":
        _render_stats(spec)

    elif cmd == "/remember":
        if not arg:
            console.print("[yellow]Usage: /remember <fact to remember>[/yellow]")
        else:
            _teach(h, arg)

    elif cmd == "/forget":
        # A bare /forget wipes everything, so confirm first; a targeted one does not.
        if arg:
            try:
                n = h.forget(arg)
                console.print(f"[green]✓[/green] Forgot {n} fact(s).")
            except Exception as e:
                err.print(f"[red]Could not forget:[/red] {escape(str(e))}")
        elif console.input("[yellow]Forget ALL taught facts? [y/N] [/yellow]").strip().lower() in ("y", "yes"):
            try:
                n = h.forget()
                console.print(f"[green]✓[/green] Forgot {n} fact(s).")
            except Exception as e:
                err.print(f"[red]Could not forget:[/red] {escape(str(e))}")
        else:
            console.print("[dim]Kept everything.[/dim]")

    elif cmd == "/memories":
        _show_memories(h)

    elif cmd == "/help":
        t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        t.add_column(style="bold cyan")
        for name, what in (
            ("/exit", "leave (Ctrl-D works too)"),
            ("/clear", "forget this session's transcript and clear the screen"),
            ("/sources", "show the sources behind the last answer"),
            ("/remember <fact>", "teach me a fact to recall and cite later"),
            ("/memories", "list everything you have taught me"),
            ("/forget", "forget all taught facts (asks first)"),
            ("/stats", "store and engine stats"),
            ("/help", "this list"),
        ):
            t.add_row(name, what)
        console.print(t)
        console.print(
            f"[dim]{len(history)} question(s) this session. "
            f"State a fact and I will remember it; anything else is a question.[/dim]"
        )

    else:
        console.print(f"[yellow]Unknown command {escape(cmd)}. /help for the list.[/yellow]")

    return True


@app.command()
def teach(
    fact: Annotated[str, typer.Argument(help='A fact to remember, e.g. "My dog is named Rex".')],
    note: Annotated[str, typer.Option("--note", help="An optional aside stored alongside the fact.")] = "",
    model: ModelOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Teach a fact. It joins the knowledge store and is recalled and cited like any source."""
    _apply_globals(model, offline, False)
    # No _require_downloaded: teaching only embeds and stores, it never runs the model.
    h = _agent(model, offline)
    if not _teach(h, fact, note):
        raise typer.Exit(1)


@app.command()
def forget(
    query: Annotated[str | None, typer.Argument(help="Which fact to forget. Omit to wipe them all.")] = None,
    model: ModelOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Forget one taught fact, or all of them."""
    _apply_globals(model, offline, False)
    if query is None and not typer.confirm("Forget ALL taught facts? This cannot be undone.", default=False):
        console.print("[dim]Kept everything.[/dim]")
        return
    h = _agent(model, offline)
    try:
        n = h.forget(query)
    except Exception as e:
        err.print(f"[red]Could not forget:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Forgot {n} fact(s).")


@app.command()
def memories(model: ModelOpt = None, offline: OfflineOpt = False) -> None:
    """List everything you have taught Herakliti."""
    _apply_globals(model, offline, False)
    h = _agent(model, offline)
    _show_memories(h)


@app.command()
def models() -> None:
    """List the model registry and what is already on disk."""
    t = Table(title="Herakliti models", title_style="bold", header_style="bold")
    t.add_column("alias", style="bold cyan")
    t.add_column("params", justify="right")
    t.add_column("size", justify="right")
    t.add_column("downloaded", justify="center")
    t.add_column("path", overflow="fold", style="dim")

    for spec in MODELS.values():
        path = loader.local_path(spec)
        here = loader.is_downloaded(spec)
        size = f"{path.stat().st_size / 1e9:.2f} GB" if here else f"{spec.size_gb:.2f} GB"
        t.add_row(
            spec.alias + (" *" if spec.alias == DEFAULT_ALIAS else ""),
            spec.params,
            size,
            "[green]yes[/green]" if here else "[dim]no[/dim]",
            str(path),
        )

    console.print(t)
    console.print(f"[dim]* default · fetch one with `{_prog()} pull <alias>`[/dim]")


@app.command()
def pull(alias: Annotated[str, typer.Argument(help="fast | default | quality")]) -> None:
    """Download a model's weights once. They are reused forever after."""
    spec = _spec_or_exit(alias)
    if loader.is_downloaded(spec):
        console.print(f"[green]✓[/green] {spec.alias} is already here: [dim]{escape(str(loader.local_path(spec)))}[/dim]")
        return

    config.ensure_dirs()
    console.print(f"Pulling [bold]{spec.filename}[/bold] ([bold]{spec.size_gb:.1f} GB[/bold]) from {spec.repo_id}")
    console.print(f"[dim]{escape(spec.note)}[/dim]")
    t0 = time.perf_counter()
    try:
        # huggingface_hub prints its own byte-accurate bar; a rich spinner on top
        # would only fight it for the cursor.
        path = loader.ensure_model(spec.alias, on_progress=lambda m: None)
    except KeyboardInterrupt:
        err.print("\n[yellow]Cancelled. Re-running resumes where it stopped.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        err.print(f"[red]Download failed:[/red] {escape(str(e))}")
        raise typer.Exit(1)

    dt = time.perf_counter() - t0
    console.print(
        f"[green]✓[/green] {spec.alias} ready — {path.stat().st_size / 1e9:.2f} GB in {dt:.0f}s\n"
        f"[dim]{escape(str(path))}[/dim]"
    )


def _fetch(target: str) -> list[tuple[Any, str]]:
    """A URL means that page; anything else is a topic to look up on Wikipedia."""
    from herakliti.knowledge.sources import web, wikipedia

    if target.startswith(("http://", "https://")):
        got = web.fetch_page(target)
        return [got] if got else []
    return wikipedia.search_and_fetch(target)


@app.command()
def ingest(
    target: Annotated[str, typer.Argument(help="A URL, or a topic to look up and index.")],
    verbose: VerboseOpt = False,
) -> None:
    """Fetch something now and index it, so later questions can answer offline."""
    _apply_globals(None, False, verbose)
    if config.SETTINGS.offline:
        # Every fetcher silently no-ops when offline; say so instead of "found nothing".
        err.print("[red]ingest needs the network, but offline mode is on.[/red]")
        raise typer.Exit(2)

    t0 = time.perf_counter()
    try:
        with console.status(f"[dim]Fetching {escape(target)}…[/dim]", spinner="dots"):
            found = _fetch(target)

        if not found:
            err.print(f"[yellow]Nothing usable found for {escape(target)}.[/yellow]")
            raise typer.Exit(1)

        from herakliti.knowledge.retriever import Retriever

        r = Retriever()
        rows: list[tuple[str, str]] = []
        with console.status("[dim]Chunking and embedding…[/dim]", spinner="dots"):
            for source, text in found:
                n = r.ingest(source, text)
                rows.append((source.title, f"{n} chunk(s) · {source.kind}"))
            # Retriever.ingest persists rows but leaves FAISS in memory; without this
            # the vectors die with the process and dense search goes blind.
            r.store.save()
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        err.print(f"[red]Ingest failed:[/red] {escape(str(e))}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] indexed [dim]in {time.perf_counter() - t0:.1f}s[/dim]")
    console.print(_kv(dict(rows)))


@app.command()
def stats(model: ModelOpt = None, offline: OfflineOpt = False) -> None:
    """What the store holds and which engine will run."""
    _apply_globals(model, offline, False)
    _render_stats(_spec_or_exit(model))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address. 0.0.0.0 exposes it to your LAN.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    model: ModelOpt = None,
    offline: OfflineOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Serve the OpenAI-compatible HTTP API."""
    _apply_globals(model, offline, verbose)
    spec = _spec_or_exit(model)
    _require_downloaded(spec)

    import os

    import uvicorn

    # uvicorn imports the factory in this process, but exporting the choice keeps it
    # true for any child a reloader might spawn.
    os.environ["HERAKLITI_MODEL"] = spec.alias
    if config.SETTINGS.offline:
        os.environ["HERAKLITI_OFFLINE"] = "1"

    console.print(
        Panel(
            f"[bold]http://{host}:{port}[/bold]  ·  {spec.label}  ·  {_backend()}\n"
            f"[dim]POST /v1/chat/completions · POST /ask · GET /v1/models · GET /health\n"
            f"docs at http://{host}:{port}/docs · Ctrl-C to stop[/dim]",
            title="Herakliti serving",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # One worker, always: see the lock in server/api.py. A second worker would mean a
    # second copy of the weights in RAM and two llama.cpp contexts racing.
    uvicorn.run(
        "herakliti.server.api:create_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
        log_level="info" if verbose else "warning",
    )


def run() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":
    run()

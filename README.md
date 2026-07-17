# Herakliti-AI

A fully local, pure-Python question-answering system. No cloud API, no external
daemon, no API keys. It runs a small language model on your own machine and makes it
*accurate* by retrieving facts from live Wikipedia, Wikidata and the web — then it
**cites its sources** so you can check every answer.

> Named after Heraclitus of Ephesus, who held that everything flows. Herakliti's
> knowledge is never fixed: it grows with every question you ask.

## The idea

Training a model that has memorised every fact is not something a laptop can do — and it
is not how modern systems stay accurate anyway. They **retrieve**. A 4-billion-parameter
model with good retrieval answers factual questions *better* than a 70B model without it,
because it reads a source before answering instead of reciting from memory — and it hands
you the link.

So Herakliti is built in three layers:

```
        ┌─────────────────────────── brain ───────────────────────────┐
Question │ route → plan → retrieve → ground → check → cite             │ Answer + sources
        └───────────────┬──────────────────────────────┬──────────────┘
                        │                               │
        ┌───────── knowledge ──────────┐        ┌──── engine ────┐
        │ Wikipedia · Wikidata · Web    │        │ llama.cpp      │
        │ chunk → embed → FAISS+SQLite  │        │ (Qwen3.5 GGUF) │
        │ BM25 + dense → RRF → rerank   │        └────────────────┘
        └───────────────────────────────┘
```

llama.cpp is the muscle; the brain and knowledge layers are what make it smart. Every
document it fetches is embedded and stored, so the second time you ask about a topic it
answers from local memory — faster, and offline.

## Install

Requires **Python 3.12** (not 3.13/3.14 — `llama-cpp-python` has no wheel there yet).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then install the GPU-accelerated engine. On an Intel Iris Xe iGPU this made prompt
processing **5.5× faster** (measured: 77 s → 14 s for a 1730-token prompt) — and it needs
no NVIDIA card. If you have no usable Vulkan device, skip this and the CPU wheel is used
automatically; Herakliti detects the backend at runtime.

```powershell
.venv\Scripts\python -m pip install --force-reinstall --no-deps `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan `
  llama-cpp-python==0.3.34
```

## Use

```powershell
# Download the default model once (2.7 GB, Apache-2.0, yours to keep offline)
python -m herakliti.cli pull default

# Ask a question — streams the answer, then prints numbered sources
python -m herakliti.cli ask "What is the capital of Albania?"

# Albanian works too (Qwen3.5 speaks Tosk Albanian)
python -m herakliti.cli ask "Kryeqyteti i Shqipërisë është?"

# Interactive chat with memory (/help, /sources, /stats, /clear, /exit)
python -m herakliti.cli chat

# Quick reasoning — maths, writing, translation need no lookup, so they don't do one
python -m herakliti.cli ask "Find x: x = 5 + 3 - 3 * 5"
python -m herakliti.cli ask "Translate 'good morning' into Albanian"
python -m herakliti.cli ask "Summarize this: <paste any text>"

# Teach it something — it remembers, recalls and cites you, and survives restarts
python -m herakliti.cli teach "My project codename is Blue Falcon"
python -m herakliti.cli memories                # everything you've taught it
python -m herakliti.cli forget "Blue Falcon"    # make it forget

# Pre-load a topic so it can be answered later with no internet
python -m herakliti.cli ingest "History of Tirana"

# What's in the local knowledge store, and which engine will run
python -m herakliti.cli stats

# OpenAI-compatible HTTP server (POST /v1/chat/completions, /ask, /health)
python -m herakliti.cli serve
```

In `chat`, you can also just *say* "Remember that …" and it stores the fact automatically.

`main.py` is a thin shim, so `python main.py ask "..."` works identically.

## Models

| alias     | model            | size    | when to use                                  |
|-----------|------------------|---------|----------------------------------------------|
| `fast`    | Qwen3.5-2B Q4_K_M | 1.28 GB | quickest replies; roughly 2× the prefill speed |
| `default` | Qwen3.5-4B Q4_K_M | 2.74 GB | best quality that still feels interactive     |
| `quality` | Qwen3.5-9B Q4_K_M | 5.68 GB | hardest questions; slow — submit and wait     |

Pick one with `--model`, e.g. `ask --model fast "..."`.

## How it decides what to do

Prompt processing is the bottleneck on a laptop CPU: every ~1000 tokens of context costs
several seconds before the first word of the answer. So context is spent carefully, and
questions take the cheapest route that can actually answer them:

- **Fact** ("capital of X", "population of Y") → a structured Wikidata lookup. Tiny
  context, ~3 seconds, exact.
- **Reason** (maths, "write…", "translate…", "summarize this: …") → answered by the model's
  own skill, no retrieval. `5 + 3 - 3 * 5` needs working-out, not a web search.
- **Chat** (greetings, "who are you") → answered directly, no retrieval.
- **Look-ups** (who/what/when about the world) → the full pipeline: search what's cached,
  fetch live if coverage is thin, fuse lexical + semantic results, rerank, and answer from
  the best few passages — **with citations**.

When a look-up finds a genuine source, Herakliti answers only from it and cites it. When it
finds nothing, it falls back to the model's own knowledge and **labels the answer
"unverified"** rather than either inventing a cited fact or dead-ending — so you always know
whether an answer is sourced or a best effort.

## Configuration

Every tunable is an environment variable prefixed `HERAKLITI_` (see `herakliti/config.py`):
`HERAKLITI_MODEL`, `HERAKLITI_N_CTX`, `HERAKLITI_N_THREADS`, `HERAKLITI_OFFLINE=1`,
`HERAKLITI_K_CONTEXT`, and more. Runtime data (models, index, caches) lives in
`~/.herakliti/` and is safe to delete — it regenerates.

## Layout

```
herakliti/
  engine/     model registry, GGUF download, llama.cpp runtime
  knowledge/  sources (wikipedia/wikidata/web), chunker, embedder, reranker, store, retriever
  brain/      prompts, router, planner, grounding, memory, agent (the orchestrator)
  server/     FastAPI, OpenAI-compatible
  cli.py      the command-line interface
```

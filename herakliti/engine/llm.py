"""The inference runtime — Herakliti's muscle.

Wraps llama.cpp (via llama-cpp-python) behind a small, honest interface.
Loading is lazy and instances are cached per (model, ctx, offload) so that
importing this module stays free and repeated calls reuse the loaded weights.
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from herakliti import config
from herakliti.engine.loader import ensure_model
from herakliti.engine.registry import ModelSpec, resolve

log = logging.getLogger(__name__)

_INSTANCES: dict[tuple, "LLM"] = {}

_LOGGING_QUIETED = False


def _quiet_llama_logging() -> None:
    """Raise llama-cpp-python's own log gate to CRITICAL instead of installing our own
    callback.

    llama_cpp/_logger.py registers a persistent, module-level callback at import time —
    it never gets garbage-collected, so it is not the source of noise. It gates on Python's
    "llama-cpp-python" logger level, which verbose=False already sets to ERROR; this just
    raises the bar further. Swapping in our own callback instead (an earlier attempt) risked
    exactly the ctypes-callback-GC hazard this avoids: replacing a live C-side function
    pointer mid-flight, with a message already in transit, is how "Exception ignored on
    calling ctypes callback function" happens. Cooperating with their gate is the safe path.
    """
    global _LOGGING_QUIETED
    if _LOGGING_QUIETED or config.SETTINGS.verbose:
        return
    logging.getLogger("llama-cpp-python").setLevel(logging.CRITICAL)
    _LOGGING_QUIETED = True


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    _kernel32.GetStdHandle.restype = wintypes.HANDLE
    _kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    _kernel32.SetStdHandle.restype = wintypes.BOOL
    _STD_ERROR_HANDLE = wintypes.DWORD(-12).value  # WinBase.h


@contextmanager
def _quiet():
    """Mute stderr around anything that can print without going through llama.cpp's own
    log gate — confirmed by reading llama-cpp-python's source: model construction with GPU
    offload (`internals.LlamaModel(...)`, where the Vulkan backend picks a device) is not
    wrapped by their own suppress_stdout_stderr, unlike llama_backend_init()/llama_numa_init().

    On Windows this redirects two separate things, because one alone was not enough:
      - `os.dup2` rewires the CRT's fd table, which a normal fprintf(stderr, ...) respects.
      - Code that calls Win32's GetStdHandle(STD_ERROR_HANDLE) directly — which some
        Vulkan/ggml paths do, specifically when attached to a real console rather than a
        redirected pipe — reads the process's *standard handle*, separate state os.dup2
        never touches. That is why testing this via a piped capture looked clean while a
        real interactive terminal still showed the device-enumeration dump: the two
        environments exercise different code paths inside the driver/loader. Redirecting
        both is what closes the gap in an actual console.
    """
    if config.SETTINGS.verbose:
        yield
        return
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_handle = None
    try:
        os.dup2(devnull, fd)
        if sys.platform == "win32":
            try:
                import msvcrt

                saved_handle = _kernel32.GetStdHandle(_STD_ERROR_HANDLE)
                _kernel32.SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(devnull))
            except Exception:  # pragma: no cover - never let this break model loading
                saved_handle = None
        yield
    finally:
        if sys.platform == "win32" and saved_handle is not None:
            _kernel32.SetStdHandle(_STD_ERROR_HANDLE, saved_handle)
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def gpu_offload_default() -> int:
    """-1 (offload every layer) when the build can talk to a GPU, else 0.

    The Vulkan build on an integrated GPU is a large prefill win, and because the
    iGPU is UMA there is no VRAM ceiling to worry about. The plain CPU wheel
    ignores n_gpu_layers, so -1 is harmless there — but we detect anyway so the
    CLI can *report* which path is live instead of leaving the user guessing.
    """
    override = os.getenv("HERAKLITI_N_GPU_LAYERS")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass
    return -1 if has_gpu_backend() else 0


def has_gpu_backend() -> bool:
    """True if this llama_cpp build exposes a non-CPU backend (Vulkan/CUDA/Metal)."""
    try:
        import llama_cpp.llama_cpp as C

        # supports_gpu_offload() is the documented probe; older builds lack it.
        fn = getattr(C, "llama_supports_gpu_offload", None)
        if fn is not None:
            return bool(fn())
    except Exception as e:  # pragma: no cover - defensive
        log.debug("gpu probe failed: %s", e)
    return False


class LLM:
    """A loaded model. Construct via `LLM.get()` to reuse instances."""

    def __init__(
        self,
        model: str | None = None,
        *,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int | None = None,
    ) -> None:
        s = config.SETTINGS
        self.spec: ModelSpec = resolve(model or s.model)
        self.path: Path = ensure_model(self.spec.alias)
        self.n_ctx = n_ctx or s.n_ctx
        self.n_threads = n_threads or s.n_threads

        # Everything that touches llama_cpp goes inside one muted block. First contact loads
        # the shared library, and GPU-offload model construction is where the Vulkan backend
        # enumerates devices — a "ggml_vulkan: Found 1 Vulkan devices" dump that bypasses
        # llama-cpp-python's own log gate (see _quiet's docstring). Redirecting stderr for
        # the whole first-contact window is what reliably swallows it.
        with _quiet():
            from llama_cpp import Llama  # deferred: importing costs ~1s

            _quiet_llama_logging()  # raise their log gate too, belt-and-suspenders
            self.n_gpu_layers = (
                gpu_offload_default() if n_gpu_layers is None else n_gpu_layers
            )
            self._llm = Llama(
                model_path=str(self.path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=s.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                verbose=config.SETTINGS.verbose,
            )
        log.info(
            "loaded %s (ctx=%d threads=%d gpu_layers=%d)",
            self.spec.filename, self.n_ctx, self.n_threads, self.n_gpu_layers,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def get(cls, model: str | None = None, **kw: Any) -> "LLM":
        spec = resolve(model or config.SETTINGS.model)
        key = (spec.alias, kw.get("n_ctx") or config.SETTINGS.n_ctx, kw.get("n_gpu_layers"))
        inst = _INSTANCES.get(key)
        if inst is None:
            inst = cls(model, **kw)
            _INSTANCES[key] = inst
        return inst

    # -- generation --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        s = config.SETTINGS
        with _quiet():
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=s.temperature if temperature is None else temperature,
                max_tokens=max_tokens or s.max_tokens,
                stop=stop or [],
            )
        return (out["choices"][0]["message"].get("content") or "").strip()

    def stream(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        s = config.SETTINGS
        with _quiet():
            for chunk in self._llm.create_chat_completion(
                messages=messages,
                temperature=s.temperature if temperature is None else temperature,
                max_tokens=max_tokens or s.max_tokens,
                stop=stop or [],
                stream=True,
            ):
                piece = chunk["choices"][0].get("delta", {}).get("content")
                if piece:
                    yield piece

    def json(self, messages: list[dict], schema: dict, *, max_tokens: int = 256) -> dict:
        """Schema-constrained generation.

        llama.cpp compiles the JSON schema into a GBNF grammar, so the output is
        *structurally* guaranteed — no regex-scraping a model's prose for JSON.
        """
        with _quiet():
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object", "schema": schema},
            )
        raw = out["choices"][0]["message"].get("content") or "{}"
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            log.warning("constrained decode still produced non-JSON: %.120s", raw)
            return {}

    # -- utilities ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8", "replace")))

    def reset(self) -> None:
        self._llm.reset()

    @property
    def backend(self) -> str:
        return "gpu" if self.n_gpu_layers != 0 and has_gpu_backend() else "cpu"

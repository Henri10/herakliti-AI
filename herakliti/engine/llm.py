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

# Held at module scope on purpose. A ctypes callback that gets garbage-collected and is
# then invoked by the C library raises "Exception ignored on calling ctypes callback
# function" — the exact noise the Vulkan backend was spraying into the chat prompt. Keeping
# a permanent reference is what stops that.
_LOG_CALLBACK: Any = None
_LOGS_SILENCED = False


def _silence_llama_logs() -> None:
    """Route llama.cpp's C-level logging (ggml/Vulkan device dumps included) into a no-op.

    Passing verbose=False is not enough: the Vulkan backend logs through a callback during
    init, on its own thread, and the default handler was both printing device chatter and
    raising mid-callback. A single silent, never-collected callback replaces it cleanly.
    """
    global _LOG_CALLBACK, _LOGS_SILENCED
    if _LOGS_SILENCED or config.SETTINGS.verbose:
        return
    try:
        import ctypes

        import llama_cpp

        @llama_cpp.llama_log_callback
        def _sink(level, text, user_data):  # noqa: ARG001 - fixed C ABI signature
            pass

        _LOG_CALLBACK = _sink  # keep the reference alive for the process lifetime
        llama_cpp.llama_log_set(_sink, ctypes.c_void_p(0))
        _LOGS_SILENCED = True
    except Exception:  # pragma: no cover - never let log-silencing break loading
        log.debug("could not install silent llama log callback", exc_info=True)


@contextmanager
def _quiet():
    """Belt-and-suspenders for any raw writes to fd 2 that bypass the log callback."""
    if config.SETTINGS.verbose:
        yield
        return
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
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
        # the shared library and the GPU probe (llama_supports_gpu_offload) makes the Vulkan
        # backend enumerate devices — a "ggml_vulkan: Found 1 Vulkan devices" dump printed
        # straight to fd 2, before any callback can intercept it. Redirecting the fd for the
        # whole first-contact is the only thing that reliably swallows it.
        with _quiet():
            from llama_cpp import Llama  # deferred: importing costs ~1s

            _silence_llama_logs()  # no-op callback for any later logging
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

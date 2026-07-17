"""HTTP surface: OpenAI-compatible, plus Herakliti's own endpoints.

The OpenAI shape exists so that clients people already own (openai-python, Open
WebUI, curl snippets) work against this box unmodified. `/ask` is the honest one:
it returns citations as data instead of smuggling them into the message text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from herakliti import __version__, config
from herakliti.engine.registry import MODELS

if TYPE_CHECKING:
    from herakliti.brain.agent import Herakliti
    from herakliti.knowledge.types import Answer

log = logging.getLogger(__name__)

_DONE = object()  # sentinel: next() returning this means the generator is spent


# --------------------------------------------------------------------------
# Wire types
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    # Accepted so OpenAI clients don't trip a 422, then ignored: this is a RAG
    # pipeline, and the agent owns its decoding budget. Context is the scarce
    # resource on this hardware, not the caller's preference. See SETTINGS.
    temperature: float | None = None
    max_tokens: int | None = None

    def question(self) -> str:
        """The last real user turn.

        Herakliti.ask() answers one question against retrieval; there is no
        multi-turn state to hand it, so earlier turns are deliberately dropped
        rather than concatenated into a prompt nobody budgeted for.
        """
        for m in reversed(self.messages):
            if m.role == "user" and m.content.strip():
                return m.content.strip()
        return ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    elapsed: float
    confidence: float


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _citations(ans: "Answer | None") -> list[dict[str, str]]:
    """List index (0-based here) + 1 matches the [1], [2]... the model was shown: Answer.
    citations is guaranteed one-chunk-per-url by dedupe_by_url upstream (see
    herakliti.knowledge.types), so no dedup is needed at this display layer."""
    if ans is None:
        return []
    return [{"title": c.title, "url": c.url} for c in ans.citations]


def _store_stats() -> dict[str, Any]:
    """The store's own numbers, read straight from sqlite + faiss.

    Deliberately not Herakliti.stats(): that reports the backend by touching .llm,
    which loads the weights. A readiness probe that costs 18s and 2.7 GB of RAM on
    the first poll is not a readiness probe.
    """
    try:
        from herakliti.knowledge.store import KnowledgeStore

        # No close(): close() saves, and saving this read-only snapshot could
        # overwrite the index of the agent that is actually writing.
        return KnowledgeStore().stats()
    except Exception as e:  # /health must answer even when the store is unhappy
        log.warning("store unreadable: %s", e)
        return {"error": type(e).__name__}


def _backend() -> str:
    """Mirror LLM.backend without loading a model — /health must stay cheap.

    Muted at fd level: importing llama_cpp makes ggml greet stderr, and /health
    should not spray the server log on every poll.
    """
    try:
        from herakliti.engine.llm import _quiet, gpu_offload_default, has_gpu_backend

        with _quiet():
            return "gpu" if gpu_offload_default() != 0 and has_gpu_backend() else "cpu"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app(model: str | None = None, offline: bool | None = None) -> FastAPI:
    """Build the app.

    A factory so uvicorn can construct it itself (`--factory`) and tests can get a
    fresh instance. Arguments default to SETTINGS, which the CLI has already
    adjusted in-process before uvicorn imports this module.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from herakliti.brain.agent import Herakliti  # heavy: keep it off module scope

        app.state.agent = Herakliti(
            model=model or config.SETTINGS.model,
            offline=config.SETTINGS.offline if offline is None else offline,
        )
        # llama.cpp holds one mutable KV cache per context. Two requests generating
        # at once would interleave into each other's state and corrupt both answers,
        # so inference is serialized here. This is also why the server is one worker
        # and one process: a second worker means a second copy of the weights in RAM.
        app.state.lock = asyncio.Lock()
        log.info("herakliti ready (model=%s)", model or config.SETTINGS.model)
        yield
        app.state.agent = None

    app = FastAPI(
        title="Herakliti",
        version=__version__,
        summary="Local retrieval-grounded answers, with sources.",
        lifespan=lifespan,
    )

    def _live() -> "Herakliti":
        agent = getattr(app.state, "agent", None)
        if agent is None:
            raise HTTPException(503, "agent not ready")
        return agent

    # -- OpenAI-compatible ------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest) -> Any:
        agent = _live()
        question = req.question()
        if not question:
            raise HTTPException(400, "no user message with content")

        model_id = req.model or config.SETTINGS.model
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _sse(app, agent, cid, created, model_id, question),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with app.state.lock:
            ans = await asyncio.to_thread(agent.ask, question)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": ans.text}, "finish_reason": "stop"}
            ],
            # Non-standard, and the point of the whole system. Clients that don't
            # know the key ignore it; ours reads it.
            "citations": _citations(ans),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        from herakliti.engine import loader

        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.alias,
                    "object": "model",
                    "created": created,
                    "owned_by": "herakliti",
                    "params": spec.params,
                    "size_gb": spec.size_gb,
                    "context_length": config.SETTINGS.n_ctx,
                    "downloaded": loader.is_downloaded(spec),
                }
                for spec in MODELS.values()
            ],
        }

    # -- native -----------------------------------------------------------

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        agent = _live()
        question = req.question.strip()
        if not question:
            raise HTTPException(400, "question must not be empty")
        async with app.state.lock:
            ans = await asyncio.to_thread(agent.ask, question)
        return AskResponse(
            answer=ans.text,
            citations=[Citation(**c) for c in _citations(ans)],
            elapsed=ans.elapsed,
            confidence=ans.confidence,
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        agent = getattr(app.state, "agent", None)
        store = await asyncio.to_thread(_store_stats)  # sqlite + a faiss read: keep it off the loop
        return {
            "status": "ok" if agent is not None else "starting",
            "model": model or config.SETTINGS.model,
            "backend": _backend(),
            "store": store,
        }

    return app


async def _sse(
    app: FastAPI,
    agent: "Herakliti",
    cid: str,
    created: int,
    model_id: str,
    question: str,
) -> AsyncIterator[str]:
    """OpenAI's chunked stream: 'data: {...}\\n\\n' frames, then 'data: [DONE]'."""

    def frame(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        # ensure_ascii=False: 'ë' should travel as UTF-8, not as an escape.
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # Held across the whole generation, not per chunk: the KV cache is single-writer.
    async with app.state.lock:
        yield frame({"role": "assistant", "content": ""})
        pieces: Iterator[str] = agent.stream_ask(question)
        try:
            while True:
                # next() in a worker thread: llama.cpp blocks for ~140ms per token
                # on this CPU and would otherwise stall the whole event loop.
                piece = await asyncio.to_thread(next, pieces, _DONE)
                if piece is _DONE:
                    break
                yield frame({"content": piece})
        except Exception as e:
            log.exception("stream failed")
            yield frame({"content": f"\n[error: {e}]"}, "stop")
            yield "data: [DONE]\n\n"
            return
        yield frame({}, "stop", citations=_citations(agent.last_answer))
    yield "data: [DONE]\n\n"

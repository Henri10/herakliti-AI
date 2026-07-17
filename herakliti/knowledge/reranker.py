"""Cross-encoder reranking — a strictly optional precision pass.

Bi-encoder retrieval scores query and passage independently; a cross-encoder reads
both together and is far better at rejecting topical-but-wrong passages. On this
machine that precision is worth real money: prefill costs 10-45s per 1000 tokens,
so one wrong chunk in the prompt costs more seconds than the entire rerank pass.

This module must never be load-bearing. A weak CPU, a missing model, a broken
remote-code shim — any of these degrade the answer, none of them may end a query.
"""

from __future__ import annotations

import threading

from ..config import SETTINGS
from .types import Chunk

_MAX_CHARS = 512
"""Cross-encoder attention is quadratic in sequence length. The lead of a chunk
carries the topic; paying for its tail is not worth the seconds on this CPU."""


class Reranker:
    """Reorders chunks by relevance. Use :meth:`get`, not the constructor."""

    _instance: "Reranker | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._broken = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self):
        """Deferred: importing torch costs seconds, so importing herakliti must not."""
        if self._model is None and not self._broken:
            with self._lock:
                if self._model is None and not self._broken:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(SETTINGS.rerank_model, device="cpu")
                    except Exception:
                        self._broken = True
        return self._model

    @property
    def available(self) -> bool:
        """True only if the model is genuinely usable. Forces the lazy load, so this
        answers honestly rather than optimistically."""
        if not SETTINGS.use_reranker:
            return False
        return self._load() is not None

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Best `top_k` chunks, scored desc. Falls back to `chunks[:top_k]` on any failure."""
        if not chunks or top_k <= 0:
            return []
        if not SETTINGS.use_reranker or self._broken:
            return chunks[:top_k]

        # Retrieval already ranked these; the tail past k_rerank is not worth CPU.
        head = chunks[: SETTINGS.k_rerank]
        tail = chunks[SETTINGS.k_rerank :]

        try:
            model = self._load()
            if model is None:
                return chunks[:top_k]
            scores = model.predict(
                [(query, c.text[:_MAX_CHARS]) for c in head],
                show_progress_bar=False,
            )
            if len(scores) != len(head):
                return chunks[:top_k]
            for c, s in zip(head, scores):
                c.score = float(s)
        except Exception:
            self._broken = True
            return chunks[:top_k]

        ranked = sorted(head, key=lambda c: c.score, reverse=True)
        # Only reached when top_k exceeds k_rerank; keeps the contract that we
        # return top_k chunks whenever that many exist.
        return (ranked + tail)[:top_k]

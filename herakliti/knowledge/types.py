"""Shared data types.

This module is the contract every other layer is written against. It imports
nothing heavy, so it is safe to import from anywhere.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["wikipedia", "wikidata", "web", "user"]


def content_id(*parts: str) -> str:
    """Stable short id derived from content, so re-ingesting the same text is a no-op."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def dedupe_by_url(chunks: "list[Chunk]") -> "list[Chunk]":
    """One chunk per source URL, keeping the first (highest-ranked) occurrence.

    Any list of chunks handed to `build_context` becomes both the [1], [2]... numbers the
    model is told to cite (positional — every chunk counts, duplicates included) and,
    separately, the citation list shown to the user. If two chunks share a URL, those two
    numbering schemes drift apart: the model cites a block by its true position, a display
    that skips repeats never shows that number, and the citation looks broken or invented.
    Call this on the final chunk list before it is ever numbered, so every consumer —
    prompt, CLI, server — agrees on what "citation 4" means.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.url in seen:
            continue
        seen.add(c.url)
        out.append(c)
    return out


@dataclass(slots=True)
class Source:
    """A document fetched from the outside world."""

    url: str
    title: str
    kind: SourceKind
    fetched_at: float = field(default_factory=time.time)
    lang: str = "en"

    @property
    def doc_id(self) -> str:
        return content_id(self.url)


@dataclass(slots=True)
class Chunk:
    """A retrievable passage. `score` is populated by whichever stage ranked it."""

    text: str
    doc_id: str
    title: str
    url: str
    kind: SourceKind
    position: int
    id: str = ""
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = content_id(self.url, str(self.position), self.text[:256])

    def cite(self) -> str:
        return f"{self.title} — {self.url}"


@dataclass(slots=True)
class Answer:
    """The result of a question. `trace` explains how it was reached."""

    text: str
    citations: list[Chunk] = field(default_factory=list)
    used_retrieval: bool = False
    confidence: float = 0.0
    elapsed: float = 0.0
    trace: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        """Unique source URLs, in citation order."""
        seen: dict[str, None] = {}
        for c in self.citations:
            seen.setdefault(c.url, None)
        return list(seen)

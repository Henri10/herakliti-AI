"""Turning documents into retrievable passages.

Retrieval quality is decided here, before any model runs. Two choices matter:

1. *Never cut mid-sentence.* A chunk truncated mid-clause loses the very fact it
   was supposed to carry, and the embedding of a fragment is noise. So we split
   recursively — paragraphs, then sentences, then (only if a single sentence is
   monstrous) a hard wrap at a word boundary.
2. *Prepend the title.* Prose is full of pronouns: "It borders Montenegro to the
   north." That chunk is unretrievable for "Albania borders" — the subject lives
   three paragraphs up. Gluing "Albania — " to the front is a cheap stand-in for
   contextual retrieval and lifts both the dense and the lexical arm.
"""

from __future__ import annotations

import re
from typing import Sequence

from herakliti import config
from herakliti.knowledge.types import Chunk, Source

MIN_CHARS = 80
"""Below this a chunk is navigation cruft, a stub heading or a stray caption."""

TITLE_SEP = " — "

_PARA_RE = re.compile(r"\n\s*\n+")
_WS_RE = re.compile(r"[ \t ]+")

# Split after ., !, ?, … (plus any closing quote/bracket) when followed by space.
_SENT_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")

# Fragments that only *look* like sentence ends. Splitting on these shreds prose.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e",
    "cf", "al", "fig", "no", "op", "ca", "approx", "inc", "ltd", "co", "dept",
    "est", "gen", "col", "capt", "sgt", "lt", "rev", "hon", "pres", "univ",
}
_ABBREV_END_RE = re.compile(r"(?:^|\s)([\w.]{1,6})\.$", re.UNICODE)


def _normalise(text: str) -> str:
    """Collapse the whitespace noise that HTML-to-text extraction leaves behind."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ends_with_abbrev(fragment: str) -> bool:
    m = _ABBREV_END_RE.search(fragment)
    if not m:
        return False
    token = m.group(1).rstrip(".").lower()
    # A lone initial ("J.") or a known abbreviation — not a sentence boundary.
    return token in _ABBREV or len(token) == 1


def _sentences(text: str) -> list[str]:
    """Split into sentences, re-joining fragments cut at an abbreviation."""
    parts = _SENT_RE.split(text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and _ends_with_abbrev(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Last resort for a single sentence longer than a whole chunk.

    Breaks at the last space before the limit; only mid-word if there is no space
    at all (long URLs, CJK, concatenated table dumps).
    """
    out: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def _units(text: str, max_chars: int) -> list[str]:
    """Atomic pieces, each guaranteed <= max_chars, split at the coarsest level that fits."""
    units: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _sentences(para):
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                units.extend(_hard_wrap(sent, max_chars))
    return units


def _joined_len(parts: Sequence[str]) -> int:
    return sum(len(p) for p in parts) + max(0, len(parts) - 1)


def _tail_overlap(text: str, overlap: int) -> list[str]:
    """The trailing whole sentences of `text` that fit in `overlap` chars.

    Deliberately re-splits the emitted chunk instead of reusing the packing units:
    a unit is often a whole paragraph, and no paragraph ever fits an overlap
    budget — carrying by unit would silently produce no overlap at all on exactly
    the well-formed prose we care most about.
    """
    if overlap <= 0:
        return []
    carry: list[str] = []
    for sent in reversed(_sentences(text)):
        if _joined_len([sent, *carry]) > overlap:
            break
        carry.insert(0, sent)
    return carry


def _pack(units: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily fill chunks, repeating trailing sentences as the overlap.

    Overlapping by sentence rather than by character keeps the seam readable: the
    repeated text is always a complete thought, so a fact straddling a boundary
    survives whole in the second chunk instead of being halved into noise.
    """
    chunks: list[str] = []
    cur: list[str] = []
    size = 0

    for unit in units:
        # +1 for the space joining it to what is already in the chunk.
        cost = len(unit) + (1 if cur else 0)
        if cur and size + cost > max_chars:
            chunk = " ".join(cur)
            chunks.append(chunk)
            cur = _tail_overlap(chunk, overlap)
            # Carried context must never push the new chunk past the limit; units
            # are all <= max_chars, so dropping from the front always converges.
            while cur and _joined_len(cur) + 1 + len(unit) > max_chars:
                cur.pop(0)
            size = _joined_len(cur)
            cost = len(unit) + (1 if cur else 0)
        cur.append(unit)
        size += cost

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_text(
    text: str,
    source: Source,
    *,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping, title-prefixed Chunks ready for embedding.

    `max_chars` and `overlap` bound the *body*; the title prefix is added after
    packing, so a chunk's final text runs len(title)+3 characters longer.
    """
    s = config.SETTINGS
    max_chars = max_chars or s.chunk_chars
    overlap = s.chunk_overlap if overlap is None else overlap
    if max_chars < MIN_CHARS:
        raise ValueError(f"max_chars={max_chars} is below MIN_CHARS={MIN_CHARS}")
    # An overlap at or above the chunk size would carry every chunk forward forever.
    overlap = max(0, min(overlap, max_chars // 2))

    body = _normalise(text)
    if not body:
        return []

    prefix = f"{source.title.strip()}{TITLE_SEP}" if source.title.strip() else ""

    chunks: list[Chunk] = []
    for raw in _pack(_units(body, max_chars), max_chars, overlap):
        raw = raw.strip()
        if len(raw) < MIN_CHARS:
            continue
        chunks.append(
            Chunk(
                text=f"{prefix}{raw}",
                doc_id=source.doc_id,
                title=source.title,
                url=source.url,
                kind=source.kind,
                position=len(chunks),
            )
        )
    return chunks

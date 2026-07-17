"""Retrieval orchestration — where Herakliti decides what it actually knows.

The flow, and why it is this shape:

  1. Search what we already have (dense + lexical, fused with RRF).
  2. If local coverage looks thin, go fetch from the live world, ingest it, search again.
     Ingesting is what makes the system *learn*: ask the same thing tomorrow and step 1
     answers it with no network at all.
  3. Rerank the survivors with a cross-encoder and hand back only the best few.

Step 3 is deliberately stingy. On this machine prefill costs ~10-45s per 1000 tokens, so
context is the scarcest resource in the system — six good chunks beat twenty mediocre ones,
and cost a quarter as much time.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from typing import Callable

import numpy as np

from herakliti import config
from herakliti.knowledge.chunker import chunk_text
from herakliti.knowledge.types import Chunk, Source, dedupe_by_url

log = logging.getLogger(__name__)

Trace = Callable[[str], None] | None


def _noop(_: str) -> None:
    pass


def rrf_fuse(rankings: list[list[Chunk]], k: int | None = None) -> list[Chunk]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1/(k + rank(d)).

    Fuses rankings by *position*, not by score, which is the whole point: BM25 scores and
    cosine similarities live on incomparable scales and normalising them is guesswork.
    Rank is the common currency. k=60 damps the top-1 bias of any single ranker.
    """
    k = k or config.SETTINGS.rrf_k
    scores: dict[str, float] = {}
    best: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, ch in enumerate(ranking, start=1):
            scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (k + rank)
            best.setdefault(ch.id, ch)
    out = []
    for cid, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        ch = best[cid]
        ch.score = sc
        out.append(ch)
    return out


class Retriever:
    def __init__(self, store=None, embedder=None, reranker=None) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._healed = False

    # -- lazily-built collaborators (keep import + construction cheap) ------

    @property
    def store(self):
        if self._store is None:
            from herakliti.knowledge.store import KnowledgeStore

            self._store = KnowledgeStore(dim=self.embedder.dim)
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            from herakliti.knowledge.embedder import Embedder

            self._embedder = Embedder.get()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from herakliti.knowledge.reranker import Reranker

            self._reranker = Reranker.get()
        return self._reranker

    # -- ingestion ---------------------------------------------------------

    def ingest(self, source: Source, text: str, trace: Trace = None) -> int:
        """Chunk, embed and persist one fetched document. Returns chunks added."""
        t = trace or _noop
        chunks = chunk_text(text, source)
        # A user-taught fact is usually one short sentence ("my dog is Rex"), below the
        # chunker's prose-cruft floor — chunk_text drops it. Such a fact is exactly what we
        # must keep, so fall back to a single verbatim chunk rather than lose it.
        if not chunks and source.kind == "user" and text.strip():
            chunks = [self._fact_chunk(source, text)]
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        n = self.store.add_document(source, chunks, vectors)
        t(f"indexed {n} chunks from {source.title}")
        return n

    @staticmethod
    def _fact_chunk(source: Source, text: str) -> Chunk:
        """One verbatim chunk for a taught fact, bypassing the chunker's length floor."""
        return Chunk(
            text=" ".join(text.split()),
            doc_id=source.doc_id,
            title=source.title,
            url=source.url,
            kind=source.kind,
            position=0,
        )

    def _fetch_live(self, query: str, entity: str | None, trace: Trace = None) -> int:
        """Fetch from every source in parallel. One source failing must not sink the query."""
        t = trace or _noop
        if config.SETTINGS.offline:
            t("offline: skipping live fetch")
            return 0

        from herakliti.knowledge.sources import web, wikidata, wikipedia

        jobs: dict[str, Callable[[], list[tuple[Source, str]]]] = {
            "wikipedia": lambda: wikipedia.search_and_fetch(query, limit=2),
            "web": lambda: web.search_and_fetch(query),
        }
        if entity:
            def _wd() -> list[tuple[Source, str]]:
                r = wikidata.lookup(entity)
                return [r] if r else []

            jobs["wikidata"] = _wd

        fetched: list[tuple[Source, str]] = []
        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {ex.submit(fn): name for name, fn in jobs.items()}
            for fut in cf.as_completed(futures, timeout=60):
                name = futures[fut]
                try:
                    got = fut.result() or []
                    t(f"{name}: {len(got)} document(s)")
                    fetched.extend(got)
                except Exception as e:
                    t(f"{name}: failed ({type(e).__name__})")
                    log.debug("source %s failed", name, exc_info=True)

        added = 0
        for source, text in fetched:
            if not text or self.store.has_url(source.url):
                continue
            try:
                added += self.ingest(source, text, trace)
            except Exception as e:
                t(f"ingest failed for {source.url}: {type(e).__name__}")
        return added

    # -- retrieval ---------------------------------------------------------

    def _heal_dense_index(self, trace: Trace = None) -> None:
        """Re-embed any chunks that exist in sqlite but not in FAISS.

        Runs once per process, on the first retrieval. It repairs the case the
        store cannot fix itself — a faiss.bin lost or corrupted while the database
        survived — which would otherwise silently reduce every query to lexical-only
        with no error to explain the worse answers.
        """
        if self._healed:
            return
        self._healed = True
        try:
            missing = self.store.unvectored_chunks()
        except Exception:
            return
        if not missing:
            return
        t = trace or _noop
        t(f"dense index out of sync: re-embedding {len(missing)} chunk(s)")
        log.warning("faiss/sqlite drift: re-embedding %d chunks", len(missing))
        try:
            vecs = self.embedder.embed_documents([c.text for c in missing])
            rebuilt = {c.id: v for c, v in zip(missing, vecs)}
            n = self.store.reindex_vectors(rebuilt)
            if n:
                self.store.save()
                t(f"repaired {n} vector(s)")
        except Exception as e:
            t(f"reindex failed ({type(e).__name__}) — degrading to lexical")
            log.debug("reindex failed", exc_info=True)

    def _hybrid(self, query: str, k: int) -> list[Chunk]:
        qvec = self.embedder.embed_query(query)
        dense = [c for c, _ in self.store.search_dense(qvec, k)]
        lexical = [c for c, _ in self.store.search_lexical(query, k)]
        return rrf_fuse([dense, lexical])

    def _rank(self, query: str, fused: list[Chunk], k: int) -> list[Chunk]:
        """Cross-encoder rerank of the fused candidates, or the fused head if unavailable.

        When the reranker runs, each returned chunk carries an absolute relevance logit in
        `.score` — that is what `_covers` reads to decide whether the local store actually
        answers the question.
        """
        if not fused:
            return []
        head = fused[: config.SETTINGS.k_rerank]
        if self.reranker.available:
            # Rank the WHOLE head rather than asking for just k — same cross-encoder cost
            # either way, since it scores all of `head` regardless — so dedup below can
            # backfill from the next-best chunk instead of quietly handing back fewer than
            # k sources whenever the top candidates cluster on one page.
            full = self.reranker.rerank(query, head, len(head))
        else:
            full = head
        # Dedup over the FULL ranked head, not just the first k — a page that dominates
        # the top ranks (several chunks of one long article) would otherwise collapse to a
        # single citation and hand back fewer than k sources. Deduping before truncating lets
        # the next-best distinct page backfill instead. See types.dedupe_by_url for why the
        # dedup itself has to happen (it is a citation-numbering invariant, not just tidiness).
        deduped = dedupe_by_url(full)[:k]
        return self._prefer_user_facts(deduped)

    @staticmethod
    def _prefer_user_facts(ranked: list[Chunk]) -> list[Chunk]:
        """Nudge genuinely-relevant taught facts above competing web/wiki chunks.

        The reranker leaves an absolute logit in `.score`; adding user_fact_boost to
        kind="user" chunks lets a relevant taught fact (~+8 -> ~+10) win and satisfy
        `_covers`, so we do not fetch the web for something the user already told us. The
        boost is small enough that an off-topic taught fact (~-6 -> ~-4) stays below the
        gate. No user chunk present -> the list is returned untouched.
        """
        boost = config.SETTINGS.user_fact_boost
        if not boost or not any(c.kind == "user" for c in ranked):
            return ranked
        for c in ranked:
            if c.kind == "user":
                c.score += boost
        return sorted(ranked, key=lambda c: c.score, reverse=True)

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        allow_fetch: bool = True,
        entity: str | None = None,
        trace: Trace = None,
    ) -> list[Chunk]:
        t = trace or _noop
        s = config.SETTINGS
        k = k or s.k_context

        self._heal_dense_index(trace)
        fused = self._hybrid(query, s.k_retrieve)
        ranked = self._rank(query, fused, k)
        t(f"local search: {len(fused)} candidates, top score "
          f"{ranked[0].score:.2f}" if ranked else "local search: nothing cached")

        # The decision to go online is about relevance, not count. A store full of
        # Albania articles returns plenty of chunks for "who was Ada Lovelace" — all
        # useless. The reranker's absolute score is what tells us the difference, so we
        # fetch when the *best* local hit is still off-topic, not when there are few hits.
        if allow_fetch and not config.SETTINGS.offline and not self._covers(ranked):
            t("local store does not cover this — fetching from the live world")
            added = self._fetch_live(query, entity, trace)
            if added:
                fused = self._hybrid(query, s.k_retrieve)
                ranked = self._rank(query, fused, k)
                t(f"after fetch: {len(fused)} candidates, top score "
                  f"{ranked[0].score:.2f}" if ranked else "after fetch: still nothing")

        return ranked

    def _covers(self, ranked: list[Chunk]) -> bool:
        """Does the local store genuinely answer this, or just return neighbours?

        With the cross-encoder available we trust its absolute score: a top hit above
        `rerank_gate` is real coverage, anything below means "go look it up". Without the
        reranker we have no calibrated signal, so we fall back to a count heuristic and
        accept that it will sometimes fetch when it needn't.
        """
        if not ranked:
            return False
        if self.reranker.available:
            return ranked[0].score >= config.SETTINGS.rerank_gate
        return len(ranked) >= 3

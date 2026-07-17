"""Herakliti — the orchestrator.

The tiering here is a direct consequence of measured hardware limits. Prefill runs at
~120 tok/s with iGPU offload, so every 1000 tokens of context costs ~8 seconds before a
single token comes back. That makes context the scarcest resource in the system, and it
means the right answer to "what is the capital of Albania" is *not* to stuff six Wikipedia
chunks into the prompt.

So questions are routed by how much context they actually need:

  FACT     -> Wikidata gives a structured answer in ~3s. Context: ~40 tokens. Cite and done.
  REASON   -> the model's own skill: maths, writing, translation, summarising pasted text.
              No retrieval — forcing a source here is what made "solve x = 5+3-3*5" refuse.
  CHAT     -> greeting / small talk, no retrieval.
  RETRIEVE -> the full hybrid pipeline, capped hard at k_context chunks, cited.

Cheap paths are tried first and fall through to expensive ones only on failure. And when
retrieval finds no real source, the answer falls back to general knowledge flagged as
unverified rather than refusing outright — a labelled best effort beats a dead end.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterator

from herakliti import config
from herakliti.brain import grounding

# A bare "hello" carries no language signal, and left to itself the model will cheerfully
# answer it in Polish. The user writes English and Albanian, so we detect which and tell the
# model outright. Albanian shows itself through its diacritics or a few unmistakable words;
# anything else defaults to English, which is what stops the stray-language drift.
_ALBANIAN_RE = re.compile(
    r"[ëçËÇ]"
    r"|\b(?:pershendetje|tung|tungjatjeta|faleminderit|flm|mirupafshim|miredita"
    r"|miremengjes|mirembrema|naten|ckemi|si\s+je|si\s+jeni|kryeqyteti|shqip"
    r"|cfare|kush|sepse|eshte|nuk|dua|jam)\b",
    re.IGNORECASE,
)


def _reply_language(text: str) -> str:
    return "Albanian" if _ALBANIAN_RE.search(text) else "English"
from herakliti.brain.memory import Memory
from herakliti.knowledge.types import Answer, Chunk, dedupe_by_url

log = logging.getLogger(__name__)


class Herakliti:
    """The public API. Construction is cheap; nothing loads until the first ask()."""

    def __init__(self, model: str | None = None, *, offline: bool | None = None) -> None:
        self.model = model
        if offline is not None:
            config.SETTINGS.offline = offline
        config.ensure_dirs()
        self.memory = Memory()
        self.last_answer: Answer | None = None
        self._llm = None
        self._retriever = None

    # -- lazy collaborators ------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from herakliti.engine.llm import LLM

            self._llm = LLM.get(self.model)
        return self._llm

    @property
    def retriever(self):
        if self._retriever is None:
            from herakliti.knowledge.retriever import Retriever

            self._retriever = Retriever()
        return self._retriever

    # -- the cheap, high-precision path ------------------------------------

    def _try_fact(self, question: str, trace: list[str]) -> tuple[list[Chunk], bool]:
        """Wikidata lookup: ~3s, ~40 tokens of context, and structurally correct.

        Returns (chunks, handled). Falls through to retrieval when the entity does not
        resolve — an unknown entity is exactly when we need real documents.
        """
        from herakliti.brain.planner import extract_entity
        from herakliti.knowledge.sources import wikidata
        from herakliti.knowledge.chunker import chunk_text

        entity = extract_entity(question)
        if not entity:
            trace.append("fact: no entity extracted, falling back to retrieval")
            return [], False
        trace.append(f"fact: entity = {entity!r}")
        if config.SETTINGS.offline:
            return [], False
        try:
            got = wikidata.lookup(entity)
        except Exception as e:
            trace.append(f"fact: wikidata failed ({type(e).__name__})")
            return [], False
        if not got:
            trace.append("fact: wikidata had nothing, falling back to retrieval")
            return [], False
        source, text = got
        trace.append(f"fact: wikidata -> {text[:80]}")
        chunks = chunk_text(text, source)
        try:
            self.retriever.ingest(source, text)  # remember it for next time
        except Exception:
            pass
        return chunks, bool(chunks)

    # -- main entry --------------------------------------------------------

    def ask(self, question: str) -> Answer:
        started = time.time()
        from herakliti.brain import learn

        if learn.is_teaching(question):
            confirmation = self.teach(question)
            answer = Answer(
                text=confirmation,
                citations=[],
                used_retrieval=False,
                confidence=1.0,
                elapsed=time.time() - started,
                trace=["learned a new fact from you"],
            )
            self.last_answer = answer
            return answer

        trace: list[str] = []
        chunks, mode = self._gather(question, trace)

        if mode == "grounded":
            text = self._answer_grounded(question, chunks, trace, stream=False)  # type: ignore[assignment]
        elif mode in ("reason", "fallback"):
            text = self._answer_reason(question, trace, stream=False, hedge=mode == "fallback")  # type: ignore[assignment]
        else:
            text = self._answer_chat(question, trace, stream=False)  # type: ignore[assignment]

        answer = self._finish(question, text, chunks, mode == "grounded", trace, started)
        return answer

    def stream_ask(self, question: str) -> Iterator[str]:
        started = time.time()
        from herakliti.brain import learn

        if learn.is_teaching(question):
            confirmation = self.teach(question)
            self.last_answer = Answer(
                text=confirmation,
                citations=[],
                used_retrieval=False,
                confidence=1.0,
                elapsed=time.time() - started,
                trace=["learned a new fact from you"],
            )
            yield confirmation
            return

        trace: list[str] = []
        chunks, mode = self._gather(question, trace)

        if mode == "grounded":
            gen = self._answer_grounded(question, chunks, trace, stream=True)
        elif mode in ("reason", "fallback"):
            gen = self._answer_reason(question, trace, stream=True, hedge=mode == "fallback")
        else:
            gen = self._answer_chat(question, trace, stream=True)

        pieces: list[str] = []
        for piece in gen:  # type: ignore[union-attr]
            pieces.append(piece)
            yield piece

        self._finish(question, "".join(pieces), chunks, mode == "grounded", trace, started)

    # -- internals ---------------------------------------------------------

    def _gather(self, question: str, trace: list[str]) -> tuple[list[Chunk], str]:
        """Route, then collect context by the cheapest path that can answer.

        Returns (chunks, mode) where mode is one of:
          "grounded"  answer only from `chunks`, with citations
          "reason"    the model answers from its own skill (maths, writing, translation)
          "chat"      greeting / small talk, no retrieval
          "fallback"  retrieval found no real source, so answer from general knowledge,
                      clearly flagged as unverified rather than refusing outright
        """
        from herakliti.brain.planner import extract_entity, rewrite
        from herakliti.brain.router import Route, route

        r = route(question, llm=None)
        trace.append(f"route = {r.value}")

        if r is Route.CHAT:
            return [], "chat"
        if r is Route.REASON:
            return [], "reason"

        q = rewrite(question, self.memory.recent(2) or None, llm=self.llm)
        if q != question:
            trace.append(f"rewritten -> {q!r}")

        if r is Route.FACT:
            chunks, handled = self._try_fact(q, trace)
            if handled:
                # Retriever.retrieve() already guarantees one chunk per url; _try_fact does
                # not go through it, so the same guarantee is enforced here — the single
                # point where a chunk list becomes both prompt context and Answer.citations.
                return dedupe_by_url(chunks), "grounded"

        t0 = time.time()
        chunks = self.retriever.retrieve(
            q, entity=extract_entity(q), trace=trace.append
        )
        trace.append(f"retrieved {len(chunks)} chunks in {time.time()-t0:.1f}s")
        if self._has_coverage(chunks):
            return dedupe_by_url(chunks), "grounded"
        trace.append("no supporting source found — answering from general knowledge (unverified)")
        return [], "fallback"

    def _has_coverage(self, chunks: list[Chunk]) -> bool:
        """Do the retrieved chunks actually answer the question, or just sit nearby?

        When the cross-encoder ran, its top score is the honest signal: below the coverage
        gate means we found neighbours, not an answer, and should say so instead of forcing
        a grounded reply that will refuse. Without a reranker we have no calibrated score,
        so any hit counts and the grounded prompt itself handles a weak match.
        """
        if not chunks:
            return False
        try:
            if self.retriever.reranker.available:
                return chunks[0].score >= config.SETTINGS.rerank_gate
        except Exception:
            pass
        return True

    def _answer_grounded(self, question: str, chunks: list[Chunk], trace: list[str], *, stream: bool):
        from herakliti.brain.prompts import SYSTEM_GROUNDED, build_context

        ctx = build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_GROUNDED},
            {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"},
        ]
        trace.append(f"context = {self.llm.count_tokens(ctx)} tokens across {len(chunks)} chunks")
        return self.llm.stream(messages) if stream else self.llm.chat(messages)

    def _answer_reason(self, question: str, trace: list[str], *, stream: bool, hedge: bool = False):
        """Answer from the model's own ability — maths, writing, translation — or, when
        `hedge` is set, from general knowledge after retrieval found no source. The hedge
        note keeps the fallback honest: better a flagged best-effort answer than either a
        flat refusal or a confident guess passed off as sourced fact."""
        from herakliti.brain.prompts import SYSTEM_REASON

        user = question
        if hedge:
            user = (
                f"{question}\n\n(No source was found for this. Answer from your own "
                "knowledge, and if you are unsure or it may be out of date, say so briefly.)"
            )
        messages = [{"role": "system", "content": SYSTEM_REASON}]
        messages.extend(self.memory.recent(2))
        messages.append({"role": "user", "content": user})
        trace.append("answered from reasoning" + (" (unverified fallback)" if hedge else ""))
        return self.llm.stream(messages) if stream else self.llm.chat(messages)

    def _answer_chat(self, question: str, trace: list[str], *, stream: bool):
        from herakliti.brain.prompts import SYSTEM_CHAT

        lang = _reply_language(question)
        system = f"{SYSTEM_CHAT}\nReply in {lang}."
        messages = [{"role": "system", "content": system}]
        messages.extend(self.memory.recent(3))
        messages.append({"role": "user", "content": question})
        trace.append(f"answering without retrieval (in {lang})")
        return self.llm.stream(messages) if stream else self.llm.chat(messages)

    def _finish(
        self,
        question: str,
        text: str,
        chunks: list[Chunk],
        used_retrieval: bool,
        trace: list[str],
        started: float,
    ) -> Answer:
        conf = grounding.confidence(text, chunks, used_retrieval)
        if used_retrieval and conf < 0.35 and not grounding.is_refusal(text):
            trace.append(f"low groundedness ({conf:.2f}) — answer may not be supported")
        answer = Answer(
            text=text,
            citations=chunks if used_retrieval else [],
            used_retrieval=used_retrieval,
            confidence=conf,
            elapsed=time.time() - started,
            trace=trace,
        )
        self.memory.add(question, text)
        self.last_answer = answer
        return answer

    # -- teachable memory --------------------------------------------------

    def teach(self, fact: str, *, note: str = "") -> str:
        """Store a fact the user taught, as a kind="user" document in the knowledge store.

        Persisted immediately (save()) so a fresh process recalls it, and idempotent by
        the fact's content-addressed url so re-teaching the same thing is a no-op.
        """
        from herakliti.brain import learn

        cleaned = learn.clean_fact(fact)
        if not cleaned:
            return "I didn't catch a fact to remember there — tell me what to keep in mind."
        source = learn.memory_source(cleaned, note)
        try:
            self.retriever.ingest(source, cleaned)
            self.retriever.store.save()
        except Exception as e:
            log.debug("could not store taught fact", exc_info=True)
            return f"I ran into a problem storing that ({type(e).__name__})."
        return f"Got it — I'll remember that: {cleaned}"

    def forget(self, query: str | None = None) -> int:
        """Delete taught facts: all of them when query is None, else those it matches.

        Returns the number of chunks removed. Matching is a folded whole-word containment
        check (no model): a fact is dropped when every content word of `query` appears in it.
        """
        store = self.retriever.store
        prefix = "herakliti://memory/"
        if query is None:
            removed = store.delete_by_url_prefix(prefix)
        else:
            from herakliti.brain.router import fold

            wanted = {w for w in fold(query).split() if len(w) > 1}
            if not wanted:
                return 0
            urls = {
                c.url for c in store.chunks_by_kind("user")
                if wanted <= set(fold(c.text).split())
            }
            removed = sum(store.delete_by_url_prefix(u) for u in urls)
        if removed:
            store.save()
        return removed

    def memories(self) -> list[Chunk]:
        """Every fact the user has taught, oldest first."""
        return self.retriever.store.chunks_by_kind("user")

    # -- misc --------------------------------------------------------------

    def stats(self) -> dict:
        from herakliti.engine.registry import resolve

        spec = resolve(self.model or config.SETTINGS.model)
        out: dict = {
            "model": spec.filename,
            "params": spec.params,
            "threads": config.SETTINGS.n_threads,
            "offline": config.SETTINGS.offline,
        }
        try:
            out["backend"] = self.llm.backend
        except Exception:
            out["backend"] = "unloaded"
        try:
            out["store"] = self.retriever.store.stats()
        except Exception as e:
            out["store"] = {"error": type(e).__name__}
        return out

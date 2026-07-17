"""Groundedness — the check that keeps Herakliti honest.

A small model asked a question it cannot answer from context will, left alone, produce a
confident and entirely invented answer. That failure is worse than useless in a system
whose whole promise is "you can verify this". So we score how much of the answer is
actually supported by the retrieved text.

The cheap lexical check runs always. The LLM check is opt-in, because a second model call
costs seconds we usually cannot justify.
"""

from __future__ import annotations

import re

from herakliti.knowledge.types import Chunk

_WORD = re.compile(r"\w+", re.UNICODE)

# Hedges a well-behaved model uses when the context is insufficient. If the answer is
# mostly one of these, it is *correctly* refusing and must not be scored as ungrounded.
_REFUSALS = (
    "i don't know",
    "i do not know",
    "don't have enough",
    "do not have enough",
    "sources don't say",
    "sources do not say",
    "not in the provided",
    "no information",
    "cannot answer",
    "can't answer",
    "nuk e di",
    "nuk ka informacion",
)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 3}


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(r in low for r in _REFUSALS)


def lexical_support(answer: str, chunks: list[Chunk]) -> float:
    """Fraction of the answer's content words that appear in the retrieved context.

    Crude on purpose: it is free, and it reliably catches the failure we care about —
    an answer full of specifics that appear nowhere in the sources. It cannot catch a
    fluent paraphrase that subtly misstates the source; `llm_support` is for that.
    """
    if not chunks:
        return 0.0
    ans = _tokens(answer)
    if not ans:
        return 0.0
    ctx = set()
    for c in chunks:
        ctx |= _tokens(c.text)
        ctx |= _tokens(c.title)
    return len(ans & ctx) / len(ans)


def confidence(answer: str, chunks: list[Chunk], used_retrieval: bool) -> float:
    """A number we are willing to show the user, on [0, 1]."""
    if is_refusal(answer):
        # Refusing when the context is thin is the correct behaviour, and we report it
        # honestly rather than dressing it up as a confident answer.
        return 0.0
    if not used_retrieval:
        return 0.5  # answered from the model's own weights: plausible, unverifiable
    support = lexical_support(answer, chunks)
    # Map support onto something calibrated-ish. Perfect lexical overlap is neither
    # achievable nor desirable (the model should paraphrase), so we saturate early.
    return max(0.0, min(1.0, 0.25 + 0.95 * support))


def llm_support(answer: str, chunks: list[Chunk], llm) -> bool:
    """Ask the model whether the context actually entails the answer.

    Only worth spending on when the lexical signal is ambiguous.
    """
    from herakliti.brain.prompts import GROUNDING_CHECK, build_context

    schema = {
        "type": "object",
        "properties": {"grounded": {"type": "boolean"}},
        "required": ["grounded"],
    }
    try:
        out = llm.json(
            [{"role": "user", "content": GROUNDING_CHECK.format(
                context=build_context(chunks), answer=answer)}],
            schema=schema,
            max_tokens=16,
        )
        return bool(out.get("grounded", True))
    except Exception:
        return True  # never let the checker itself break a query

"""Query planning — rewrite, decompose, and pull out the subject.

Same economics as the router: an llm.json() call is 5-30 seconds, a regex is free. So
every function here has a rule-based gate in front of the model, and each gate is there
to answer one question: would the model actually change anything? A self-contained
question does not need rewriting, and a single-hop question does not need decomposing —
running the model anyway just makes the user wait to be told what we already knew.

Imports nothing heavy — safe to import from anywhere.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from herakliti.brain.prompts import (
    DECOMPOSE_SCHEMA,
    DECOMPOSE_SYSTEM,
    REWRITE_SCHEMA,
    REWRITE_SYSTEM,
)
from herakliti.brain.router import fold, looks_multi_hop

if TYPE_CHECKING:
    from herakliti.engine.llm import LLM

log = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3

# A question only needs history if it actually points at something outside itself:
# a pronoun, a demonstrative, or an elliptical opener. Length is not a signal —
# "who is Edi Rama?" is four words and perfectly self-contained.
_DEPENDENT_RE = re.compile(
    r"\b(?:it|its|it's|he|his|him|she|her|hers|they|them|their|theirs|that|those|these"
    r"|this|there|then|same)\b"
    r"|^(?:and|but|so|ok|okay)\b"
    r"|^(?:what|how)\s+about\b"
    r"|\b(?:ai|ajo|ata|ato|tij|saj|tyre|atje|aty|ky|kjo|keta|keto|atij|asaj|kete|ate"
    r"|njejten|njejtin)\b"
    r"|^(?:po|edhe|dhe)\b"
)

# Ordered most-specific first: the anchored "when was X born" shapes must win over the
# generic "<attribute> of X", or "when was the capital of Albania founded" would hand
# back "Albania founded".
_ENTITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^(?:when|what\s+year|what\s+date)\s+(?:was|were|did|is)\s+(?P<ent>.+?)\s+"
        r"(?:born|founded|established|created|built|die|died|invented|written|released"
        r"|published)\b",
        r"\bhow\s+many\s+(?:people|inhabitants|residents)\s+"
        r"(?:live\s+in|are\s+there\s+in)\s+(?P<ent>.+)$",
        r"\bsa\s+(?:banore|njerez)\s+(?:ka|jetojne\s+ne)\s+(?P<ent>.+)$",
        r"^kur\s+(?:lindi|u\s+lind|vdiq|u\s+themelua|u\s+krijua|u\s+ndertua)\s+(?P<ent>.+)$",
        r"\b(?:capital|population|area|currency|language|flag|anthem|president"
        r"|prime\s+minister|mayor|founder|author|director|inventor|birthplace|gdp"
        r"|religion|climate)\s+(?:city\s+)?of\s+(?:the\s+)?(?P<ent>.+)$",
        r"\b(?:kryeqyteti|kryeqytet|popullsia|popullsi|siperfaqja|siperfaqe|presidenti"
        r"|kryeministri|kryetari|monedha|gjuha|flamuri|autori|themeluesi)\s+"
        r"(?:i|e|te|se)\s+(?P<ent>.+)$",
    )
)

# Interrogative scaffolding to shave off the front when no pattern above matches.
# Every optional article demands a trailing space: without it "(?:a)?" happily eats the
# "A" of "Albania" and leaves "lbania".
_LEAD_RE = re.compile(
    r"^(?:"
    r"(?:who|what|whats|what's|when|where|which|why|how|whose|whom)\s+"
    r"(?:(?:is|are|was|were|does|do|did|can|could|will|many|much|long|old|far)\s+)?"
    r"(?:(?:the|a|an)\s+)?"
    r"|(?:tell\s+me\s+about|tell\s+me|explain|describe|define|summari[sz]e|give\s+me)\s+"
    r"(?:(?:the|a|an)\s+)?"
    r"|(?:kush|cfare|c'eshte|cili|cila|cilat|cilet|kur|ku|si|sa|pse)\s+"
    r"(?:(?:eshte|jane|ishte|ishin|ka|ben|kane)\s+)?"
    r"|(?:me\s+trego\s+per|trego\s+per|shpjego|pershkruaj)\s+"
    r")"
)

_LEADING_ARTICLE_RE = re.compile(r"^(?:the|një|nje)\s+", re.IGNORECASE)
_TRIM = " \t\"'“”‘’?!.,;:—–-…"

# A trailing copula is the tell of a "X of Y is?" phrasing ("kryeqyteti i Shqipërisë
# është?" -> the pattern captures "Shqipërisë është"). Left on, it garbles the Wikidata
# search — "Shqipërisë është" fuzzy-matched to a county in Iran, while bare "Shqipërisë"
# resolves cleanly to Albania. Both diacritic and folded Albanian forms are listed.
_TRAILING_COPULA_RE = re.compile(
    r"[\s,]+(?:is|are|was|were|be|been|does|do"
    r"|është|eshte|janë|jane|ishte|ishin|qe|është\?)$",
    re.IGNORECASE,
)


def _clean_entity(text: str) -> str | None:
    cand = _LEADING_ARTICLE_RE.sub("", text.strip(_TRIM)).strip(_TRIM)
    prev = None
    while cand and cand != prev:
        prev = cand
        cand = _TRAILING_COPULA_RE.sub("", cand).strip(_TRIM)
    return cand or None


def extract_entity(question: str) -> str | None:
    """Pull the subject out of a question, for the Wikidata lookup. No LLM, no network.

    Heuristic and cheap by design — it is the front half of a path whose whole reason to
    exist is being ~3s end to end. It is meaningful for FACT-routed questions; on a
    multi-hop question it will happily return "country whose capital is Tirana", which is
    why route() sends those to RETRIEVE instead. A miss is safe: agent falls back to
    retrieval when the entity does not resolve.
    """
    raw = re.sub(r"\s+", " ", question).strip().rstrip("?!.…").strip()
    if not raw:
        return None

    folded = fold(raw)  # same length as raw, so spans transfer (see router.fold)
    for rx in _ENTITY_PATTERNS:
        m = rx.search(folded)
        if m:
            start, end = m.span("ent")
            cand = _clean_entity(raw[start:end])
            if cand:
                return cand

    lead = _LEAD_RE.match(folded)
    return _clean_entity(raw[lead.end() :] if lead else raw)


def _history_snippet(history: list[dict], turns: int = 2, max_chars: int = 300) -> str:
    """Just enough conversation to resolve a pronoun. Truncated because this text is
    prefill, and prefill is the bill we are trying not to run up."""
    lines = []
    for m in history[-turns:]:
        text = " ".join(str(m.get("content", "")).split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        lines.append(f"{m.get('role', 'user')}: {text}")
    return "\n".join(lines)


def rewrite(question: str, history: list[dict] | None = None, llm: "LLM | None" = None) -> str:
    """Resolve pronouns/ellipsis against history: "what about its population?" ->
    "what is the population of Tirana?".

    Returns the question untouched when there is no history, when nothing in it refers
    outward, or when no model was supplied — three cases where the model could only
    hand back what it was given, ten seconds later. Only genuine reference resolution
    needs meaning, and that is the one thing regex cannot do.
    """
    if not history or not _DEPENDENT_RE.search(fold(question)) or llm is None:
        return question
    try:
        out = llm.json(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": f"{_history_snippet(history)}\nfollow-up: {question}",
                },
            ],
            REWRITE_SCHEMA,
            max_tokens=64,
        )
        new = " ".join(str(out.get("question", "")).split())
    except Exception as e:
        log.debug("rewrite failed (%s), keeping original", e)
        return question
    # A rewrite that ballooned is a model that started answering instead of rewriting.
    if not new or len(new) > 3 * len(question) + 60:
        return question
    return new


def decompose(question: str, llm: "LLM | None" = None) -> list[str]:
    """Split a multi-hop question into an ordered chain of standalone sub-questions.

    Gated on `looks_multi_hop`, because the overwhelming majority of questions are
    single-hop and would come straight back unchanged — the model is only asked when a
    nested clause is actually present. Capped at 3: each sub-question is a full retrieval
    round, and four of them is no longer a question, it is a coffee break.
    """
    if llm is None or not looks_multi_hop(question):
        return [question]
    try:
        out = llm.json(
            [
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user", "content": question},
            ],
            DECOMPOSE_SCHEMA,
            max_tokens=192,
        )
        subs = [
            " ".join(str(s).split()) for s in out.get("questions", []) if str(s).strip()
        ]
    except Exception as e:
        log.debug("decompose failed (%s), treating as single-hop", e)
        return [question]
    return subs[:MAX_SUB_QUESTIONS] or [question]

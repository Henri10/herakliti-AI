"""Teach-intent detection and fact cleaning — rules only, never the model.

A user teaching Herakliti a fact ("remember that my dog is Rex") is a per-turn decision
that must be effectively free: paying a multi-second llm.json() call to notice an
imperative would defeat the point. So intent is decided by a small, high-precision
regex over the diacritic-folded text (see router.fold), in English and Albanian.

Precision is chosen over recall on purpose. A false positive stores junk as if it were a
fact the user asked us to keep; a false negative merely means the user reaches for the
explicit /remember command instead. So the detector refuses anything shaped like a
question and only fires on an unambiguous teaching lead-in.

Imports nothing heavy — safe to import from anywhere.
"""

from __future__ import annotations

import re

from herakliti.brain.router import fold
from herakliti.config import SETTINGS
from herakliti.knowledge.types import Source, content_id

# A teaching lead-in, English and Albanian, anchored to the start. The trailing
# connective (that / qe / a colon or comma) is folded into the match so clean_fact strips
# it along with the phrase. Written against fold(text): lowercase, no diacritics, so
# "shëno" is matched as "sheno" and "dua të dish" as "dua te dish".
_LEADIN = (
    r"(?:please\s+|pls\s+|just\s+|hey\s+|ok(?:ay)?\s+)?"
    r"(?:"
    r"remember(?:\s+this|\s+that)?"
    r"|note(?:\s+that)?"
    r"|make\s+a\s+note(?:\s+(?:that|of))?"
    r"|(?:do\s+not|don'?t)\s+forget(?:\s+that)?"
    r"|keep\s+in\s+mind(?:\s+that)?"
    r"|for\s+the\s+record"
    r"|fyi"
    r"|i\s+want\s+you\s+to\s+know(?:\s+that)?"
    r"|i'?d?\s+want\s+you\s+to\s+know(?:\s+that)?"
    r"|i\s+want\s+to\s+tell\s+you(?:\s+that)?"
    r"|mban\s+mend(?:\s+qe)?"
    r"|mos\s+harro(?:\s+qe)?"
    r"|sheno(?:\s+qe)?"
    r"|kujto(?:\s+qe)?"
    r"|dua\s+te\s+dish(?:\s+qe)?"
    r"|dua\s+te\s+dini(?:\s+qe)?"
    r")"
    r"\b[\s:,\.\-—–]*"
)
_LEADIN_RE = re.compile(r"^\s*" + _LEADIN)

# A question, by surface form: a question mark anywhere, or an interrogative opener in
# either language. This is the guard that keeps "remember what my dog's name is?" (a
# question) from being stored as if it were a fact. `ku` needs a word boundary so it does
# not fire on the Albanian teaching verb "kujto".
_QUESTION_RE = re.compile(
    r"^\s*(?:who|what|whats|what's|when|where|why|how|which)\b"
    r"|^\s*(?:kush|cfare|ku|kur|si|sa|pse|cili|cila|cilat|cilet)\b"
)


def is_teaching(text: str) -> bool:
    """True when the user is asking Herakliti to remember a fact, not asking a question.

    High precision by design: a question ("remember what he said?") or anything without a
    recognised teaching lead-in returns False, so at worst the user falls back to the
    explicit command rather than having junk stored.
    """
    if not text or not text.strip():
        return False
    folded = fold(text)
    if "?" in folded or _QUESTION_RE.search(folded):
        return False
    return bool(_LEADIN_RE.match(folded))


def clean_fact(text: str) -> str:
    """Strip a leading teaching phrase and surrounding quotes/punctuation, keep the fact.

    "remember that my dog is Rex" -> "my dog is Rex". No lead-in -> returned unchanged.
    Matches the lead-in on the folded copy (diacritic-insensitive) but slices the original,
    because fold preserves length and the fact must keep its real casing and diacritics.
    """
    raw = text.strip()
    if not raw:
        return ""
    m = _LEADIN_RE.match(fold(raw))
    body = raw[m.end():] if m else raw
    return body.strip(" \t\n\r\"'“”‘’«»`:;,.!?()[]—–-…")


def memory_source(fact: str, note: str = "") -> Source:
    """A kind="user" Source for a taught fact, with a stable url so re-teaching is a no-op.

    The url is content-addressed on the fact itself, so the store (which dedupes by
    url/chunk_id) treats teaching the same thing twice as a single document.
    """
    return Source(
        url="herakliti://memory/" + content_id(fact),
        title=note or "Something you taught me",
        kind="user",
        lang=SETTINGS.lang,
    )

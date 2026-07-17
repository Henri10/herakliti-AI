"""Query routing — pick the cheapest path that can still be right.

Routing runs before every single turn, so it has to be effectively free. An llm.json()
call costs 5-30 seconds on this hardware; a regex costs microseconds. So rules decide,
and the model is consulted only for the thin band of inputs no rule claims — and even
then only if the caller already holds a loaded model. route() never loads one itself:
paying a multi-second model load to *classify* a question you then have to answer anyway
is the worst trade in the system.

Every rule is written in ASCII and matched against fold(question), so Albanian works
whether or not the user typed the diacritics ("përshëndetje" == "pershendetje").

Imports nothing heavy — safe to import from anywhere.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

from herakliti.brain.prompts import ROUTER_SCHEMA, ROUTER_SYSTEM

if TYPE_CHECKING:
    from herakliti.engine.llm import LLM

log = logging.getLogger(__name__)


class Route(str, Enum):
    CHAT = "chat"
    RETRIEVE = "retrieve"
    FACT = "fact"
    REASON = "reason"


def fold(text: str) -> str:
    """Lowercase and strip diacritics *without changing the string's length*.

    Length preservation is the whole point, not an accident: it lets `extract_entity`
    locate a span in the folded text and slice that span out of the *original*, so
    "kryeqyteti i Shqipërisë" yields "Shqipërisë" with its diacritics intact for the
    Wikidata lookup. NFD alone would shift every index to the right of an ë.

    Exactly one character is appended per input character, so len(fold(s)) == len(s).
    """
    out: list[str] = []
    for ch in text:
        base = "".join(
            c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c)
        )
        if len(base) != 1:  # e.g. 'ß' -> 'ss'; keep the original rather than shift indices
            base = ch
        low = base.lower()
        out.append(low if len(low) == 1 else base)
    return "".join(out)


# --------------------------------------------------------------------------
# CHAT rules. Anchored end-to-end: "hello" is chitchat, but "hello, what is the
# capital of Albania?" is a question with a greeting bolted on.
# --------------------------------------------------------------------------

_GREET = (
    r"(?:hi+|hey+|hello+|helo|yo|greetings|good\s+(?:morning|afternoon|evening|day|night)"
    r"|thanks?|thank\s+you|thx|ty|cheers|bye|goodbye|see\s+you|ok(?:ay)?|kk|cool|nice|great"
    r"|awesome|lol|sure|yes|no|yep|nope|sorry"
    r"|pershendetje|tung(?:jatjeta)?|c'?kemi|miredita|miremengjes|mirembrema|naten\s+e\s+mire"
    r"|faleminderit|flm|mirupafshim|dakord|mire|po|jo|si\s+je(?:ni)?)"
)
_FILLER = r"(?:there|herakliti|again|friend|mate|all|everyone|guys|shoku|miku|ti|ju)"

_GREETING_RE = re.compile(
    rf"^{_GREET}\b(?:[\s,.!?-]*(?:{_GREET}|{_FILLER})\b)*[\s,.!?]*$"
)

# A greeting bolted onto the front of a real request ("hello, who are you?",
# "hi, what is the capital of Albania?"). Stripped before routing so the request
# underneath is classified on its own merits instead of being dragged to RETRIEVE by
# its question word. Only a leading greeting clause followed by a separator is removed.
_LEADING_GREET_RE = re.compile(rf"^{_GREET}\b[\s,.!?-]+(?=\S)")

_META_RE = re.compile(
    r"^(?:so\s+|hey\s+|ok\s+)?(?:"
    r"who\s+(?:are|r)\s+(?:you|u)"
    r"|what\s+(?:are|r)\s+you"
    r"|what(?:'s|\s+is)\s+your\s+name"
    r"|what\s+can\s+you\s+do"
    r"|what\s+do\s+you\s+do"
    r"|how\s+do\s+you\s+work"
    r"|are\s+you\s+(?:an?\s+)?(?:ai|bot|human|real|robot|chatgpt|llm)"
    r"|introduce\s+yourself"
    r"|help"
    r"|kush\s+je(?:\s+ti)?"
    r"|kush\s+jeni"
    r"|si\s+quhesh"
    r"|si\s+e\s+ke\s+emrin"
    r"|cfare\s+(?:je|mund\s+te\s+besh)"
    r"|si\s+funksionon"
    r")\b[\s\W]*$"
)

# Arithmetic only. Requires at least one operator between numbers so that "what is 42"
# is not mistaken for a calculation. Retrieval cannot help with 2+2 — the model computes it.
_MATH_RE = re.compile(
    r"^(?:what\s+(?:is|are)\s+|calculate\s+|compute\s+|solve\s+|how\s+much\s+is\s+"
    r"|sa\s+(?:bejne|ben|eshte)\s+)?"
    r"\(*\d[\d\s.,)]*(?:[-+*/^%x]\s*\(*\d[\d\s.,)]*)+\s*=?\s*\??$"
)

# --------------------------------------------------------------------------
# REASON — work the model does from its OWN skill, not from sources: arithmetic
# and equations, generation (write/compose/code), and transforming text the user
# supplied (translate/summarise/rewrite). Routing these to grounded retrieval was the
# bug that made "find x in x = 5 + 3 - 3 * 5" answer "the sources do not contain that".
# --------------------------------------------------------------------------

# Computation beyond the bare expression _MATH_RE catches: an explicit equation, a
# "find x ... =" phrasing, or a strong arithmetic operator. Deliberately NOT a lone
# hyphen between numbers — "2020-2021" is a date range, not a subtraction to evaluate.
_COMPUTE_RE = re.compile(
    r"\b(?:solve|calculat|comput|evaluat|simplif|factor|zgjidh|llogarit)"
    r"|=\s*[-+(]?\s*\d"                          # an equation:  x = 5 ...
    r"|\bfind\s+[a-z]\b[^?]*="                   # find x ... =
    r"|\d\s*[*/^%×÷]\s*\(*\s*\d"                 # strong operator: 12 * (3 ...
    r"|\d\s*[-+]\s*\d\s*[-+*/^%×÷]\s*\d"         # a chain of >= 2 operators: 5 + 3 - ...
)

# Generative asks. Anchored, so "who created Facebook?" (a lookup) is untouched.
_MAKE_RE = re.compile(
    r"^(?:write|draft|compose|create|generate|make\s+me|give\s+me\s+(?:a|an|some)|"
    r"come\s+up\s+with|invent|brainstorm|code|program|"
    r"shkruaj|krijo|harto|gjenero)\b"
)

# Transforms of text the user provides. These need CONTENT in the message, else they are
# a topic request ("summarise the history of Albania") that genuinely needs retrieval.
_TRANSFORM_RE = re.compile(
    r"^(?:summari[sz]e|tl;?dr|rewrite|re-write|rephrase|reword|paraphrase|proofread"
    r"|correct|polish|shorten|condense|expand|permbledh|permblidh|thjeshto)\b"
)
_TRANSLATE_RE = re.compile(r"^(?:translate|perkthe(?:j|ni)?)\b")

_PROVIDED_COLON_RE = re.compile(r":\s*\S")

# --------------------------------------------------------------------------
# Multi-hop. A nested relative clause means the subject of the question is itself the
# answer to another question, so there is no single entity to look up in Wikidata.
# Over-firing here is nearly free — it only costs the FACT shortcut, and RETRIEVE is
# the safe default anyway.
# --------------------------------------------------------------------------

_MULTIHOP_RE = re.compile(
    r"\bwhose\b"
    r"|\bthe\s+(?:country|city|state|company|band|team|person|man|woman|author|director"
    r"|film|movie|book|album)\s+(?:whose|that|which|where)\b"
    r"|\bof\s+(?:the\s+)?\w+\s+(?:that|which|who)\s+"
    r"(?:is|was|has|had|won|wrote|founded|created|discovered|invented|directed|born)\b"
    r"|\b(?:i|e|te|se)\s+cil(?:it|es|ave|eve|in)\b"
    r"|\bvendit\s+qe\b"
    r"|\?.+\?"
    r"|\band\s+(?:who|what|when|where|which|how|why)\b"
)

# --------------------------------------------------------------------------
# FACT. Deliberately narrow: only shapes that map onto a Wikidata property verified to
# work on this project, and only when the pattern structurally guarantees a trailing
# entity to look up ("capital of X", never a bare "kryeqyteti"). "president of X" is
# absent on purpose — P35 head-of-state is right for Albania and wrong for the UK, and
# a wrong high-precision answer is worse than a slow grounded one.
# --------------------------------------------------------------------------

_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), prop)
    for p, prop in (
        (r"\bcapital\s+(?:city\s+)?of\s+\S", "P36"),
        (r"\bkryeqyteti\s+(?:i|e|te|se)\s+\S", "P36"),
        (r"\bpopulation\s+of\s+\S", "P1082"),
        (r"\bhow\s+many\s+(?:people|inhabitants|residents)\s+\S", "P1082"),
        (r"\bpopullsia\s+(?:i|e|te|se)\s+\S", "P1082"),
        (r"\bsa\s+(?:banore|njerez)\s+(?:ka|jetojne)\s+\S", "P1082"),
        (r"\bwhen\s+(?:was|were)\b.+\bborn\b", "P569"),
        (r"\bkur\s+(?:lindi|u\s+lind)\s+\S", "P569"),
        (r"\bwhen\s+did\b.+\bdie\b", "P570"),
        (r"\bkur\s+vdiq\s+\S", "P570"),
        (r"\bwhen\s+(?:was|were)\b.+\b(?:founded|established|created)\b", "P571"),
        (r"\bkur\s+u\s+(?:themelua|krijua)\s+\S", "P571"),
    )
)

# Anything interrogative or imperative-informational. Reaching here means no cheap path
# claimed it, so it needs real documents.
_INFO_RE = re.compile(
    r"\?"
    r"|^(?:who|what|whats|when|where|which|why|how|whose|whom|is|are|was|were|do|does|did"
    r"|can|could|list|name)\b"
    r"|^(?:tell|explain|describe|summari[sz]e|compare|define|give|show|find)\b"
    r"|^(?:kush|cfare|c'eshte|cili|cila|cilat|cilet|kur|ku|si|sa|pse|nga|a)\b"
    r"|^(?:me\s+trego|trego|shpjego|pershkruaj|permblidh|krahaso|gjej)\b"
)


def looks_multi_hop(question: str) -> bool:
    """True when the question chains one lookup onto another ("...whose capital is X")."""
    return bool(_MULTIHOP_RE.search(fold(question)))


def _has_provided_content(question: str) -> bool:
    """True when the message carries text to operate on, not just a topic to look up.

    "summarise this: <paragraph>" has content; "summarise the history of Albania" does not
    and should retrieve instead. Signals: a newline, a quoted span, a colon followed by real
    text, or simply a long block that is clearly pasted rather than typed as a question.
    """
    q = question.strip()
    if "\n" in q or '"' in q or "'" in q or "“" in q or "«" in q:
        return True
    m = _PROVIDED_COLON_RE.search(q)
    if m and len(q) - m.start() > 40:
        return True
    return len(q) > 160


def looks_reasoning(question: str) -> bool:
    """True when the model should answer from its own skill, not from retrieved sources:
    arithmetic/equations, generation, translation, or transforming supplied text."""
    core = _LEADING_GREET_RE.sub("", fold(question).strip()).strip()
    if _MATH_RE.match(core) or _COMPUTE_RE.search(core):
        return True
    if _MAKE_RE.match(core) or _TRANSLATE_RE.match(core):
        return True
    if _TRANSFORM_RE.match(core) and _has_provided_content(question):
        return True
    return False


def fact_property(question: str) -> str | None:
    """The Wikidata property id a FACT question is asking for, e.g. "P36" for a capital.

    Exposed because route() alone would force the Wikidata layer to re-derive the very
    thing routing just worked out. Returns None for anything that is not a single
    entity-attribute lookup.
    """
    if looks_multi_hop(question):
        return None
    q = fold(question)
    for rx, prop in _FACT_PATTERNS:
        if rx.search(q):
            return prop
    return None


def route(question: str, llm: "LLM | None" = None) -> Route:
    """Classify a question. Rules only, unless `llm` is supplied *and* rules abstain.

    Defaults to RETRIEVE whenever it is unsure: a needless retrieval costs seconds, an
    ungrounded answer costs the user's trust.
    """
    q = fold(question).strip()
    if not q:
        return Route.CHAT
    if _GREETING_RE.match(q):
        return Route.CHAT

    # Peel a leading greeting ("hello, who are you?") so the real request under it is
    # what gets routed. If nothing survives the peel it was pure chitchat.
    core = _LEADING_GREET_RE.sub("", q).strip()
    if not core:
        return Route.CHAT
    if _META_RE.match(core):
        return Route.CHAT
    # REASON before FACT/RETRIEVE: maths and generation must never be sent looking for a
    # source. A FACT shape still wins if it is literally a "capital of X" lookup, so this
    # only claims computation/generation/translation and transforms of supplied text.
    if looks_reasoning(question):
        return Route.REASON
    if fact_property(core):
        return Route.FACT
    if _INFO_RE.search(core):
        return Route.RETRIEVE
    if llm is None:
        return Route.RETRIEVE
    return _route_llm(question, llm)


def _route_llm(question: str, llm: "LLM") -> Route:
    """Last resort: a bare fragment with no question mark, no question word, no greeting.

    Rules genuinely cannot separate "Albanian history" (retrieve) from "cool, thanks"
    (chat) — there is no surface feature to key on, only meaning. Worth one constrained
    call, but only because the caller already has the model in memory.

    Chooses chat-or-retrieve only; FACT is unreachable here by construction (see
    ROUTER_SYSTEM). The grammar cannot emit anything outside the enum, so the Route()
    conversion below is total.
    """
    try:
        out = llm.json(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ],
            ROUTER_SCHEMA,
            max_tokens=16,
        )
        return Route(str(out.get("route", "")).strip().lower())
    except Exception as e:
        # Broad on purpose: an unroutable question must still get answered. A bad enum
        # value, a llama.cpp error, a context overflow — all mean "fall back to grounded".
        log.debug("router llm fallback failed (%s), defaulting to retrieve", e)
        return Route.RETRIEVE

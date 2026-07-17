"""Wikidata facts over the REST API.

The cheap, high-precision path for factual questions: a handful of structured
claims instead of an article to prefill. SPARQL answers the same questions in
13s versus 3s here, which this hardware cannot afford.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from herakliti.config import SETTINGS
from herakliti.knowledge.types import Source

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)

_API = "https://www.wikidata.org/w/api.php"
_client: "httpx.Client | None" = None

# Insertion order is output order in lookup(); identifying facts first, since
# the character budget truncates the tail.
_PROPS: dict[str, str] = {
    "P36": "capital",
    "P17": "country",
    "P30": "continent",
    "P37": "official language",
    "P38": "currency",
    "P1082": "population",
    "P571": "inception",
    "P569": "date of birth",
    "P570": "date of death",
    "P106": "occupation",
    "P27": "citizenship",
    "P31": "instance of",
    "P625": "coordinates",
}

_MAX_FACT_CHARS = 400
_LABEL_BATCH = 50  # wbgetentities hard-caps ids at 50

_STOP = {
    "what", "whats", "what's", "who", "whos", "who's", "where", "when", "which",
    "why", "how", "is", "are", "was", "were", "the", "a", "an", "tell", "me",
    "about", "of", "in", "for", "on", "at", "by", "from", "do", "does", "did",
    "give", "list", "name", "many", "much",
}


def _http() -> "httpx.Client":
    """Pooled client; Wikimedia rejects a UA without contact info outright."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(
            headers={"User-Agent": SETTINGS.user_agent},
            timeout=SETTINGS.http_timeout,
            follow_redirects=True,
        )
    return _client


def _get_json(params: dict[str, str]) -> dict | None:
    try:
        r = _http().get(_API, params=params)
    except Exception as exc:
        log.debug("wikidata %s failed: %s", params.get("action"), exc)
        return None
    if r.status_code != 200:
        log.debug("wikidata %s -> HTTP %s", params.get("action"), r.status_code)
        return None
    try:
        return r.json()
    except Exception as exc:
        log.debug("wikidata %s -> bad json: %s", params.get("action"), exc)
        return None


def _candidates(name: str) -> list[str]:
    """Search terms to try, in order.

    wbsearchentities matches labels, not meaning: "capital of Albania" returns
    zero hits. The full phrase goes first so "Bank of England" resolves as
    itself instead of being shortened to "England"; only a miss walks the tail.
    """
    cleaned = re.sub(r"[?!.]+$", "", name.strip())
    out: list[str] = []

    def add(cand: str) -> None:
        cand = cand.strip(" ,;:'\"")
        if cand and cand.casefold() not in {o.casefold() for o in out}:
            out.append(cand)

    add(cleaned)
    words = cleaned.split()
    for i in range(1, len(words)):
        if words[i].casefold() in _STOP:
            continue
        add(" ".join(words[i:]))
    return out[:4]  # bound latency: each miss is a ~0.9s round trip


def _best_snak(statements: list[dict]) -> dict | None:
    """Rank-aware pick.

    Wikidata keeps every historical value, so statement[0] for US population is
    the 1790 census (3.9M). The preferred rank is the current value; deprecated
    is wrong on purpose.
    """
    fallback: dict | None = None
    for st in statements:
        rank = st.get("rank")
        if rank == "deprecated":
            continue
        snak = st.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        if rank == "preferred":
            return snak
        if fallback is None:
            fallback = snak
    return fallback


def _fmt_time(value: dict) -> str:
    """Precision-aware: "+2023-00-00T00:00:00Z" carries month/day 00 and would
    blow up any real date parser."""
    m = re.match(r"([+-])(\d{4,})-(\d{2})-(\d{2})", str(value.get("time", "")))
    if not m:
        return ""
    sign, year, month, day = m.groups()
    y = str(int(year))
    if sign == "-":
        return f"{y} BC"
    prec = value.get("precision", 11)
    if prec >= 11 and month != "00" and day != "00":
        return f"{y}-{month}-{day}"
    if prec == 10 and month != "00":
        return f"{y}-{month}"
    return y


def _fmt_value(dtype: str, value: object) -> tuple[str, bool]:
    """-> (text, is_qid). is_qid means it still needs a label lookup."""
    if dtype == "wikibase-entityid" and isinstance(value, dict):
        return str(value.get("id") or ""), True
    if dtype == "time" and isinstance(value, dict):
        return _fmt_time(value), False
    if dtype == "quantity" and isinstance(value, dict):
        return str(value.get("amount", "")).lstrip("+"), False
    if dtype == "globecoordinate" and isinstance(value, dict):
        lat, lon = value.get("latitude"), value.get("longitude")
        if lat is None or lon is None:
            return "", False
        return f"{float(lat):g}, {float(lon):g}", False
    if dtype == "monolingualtext" and isinstance(value, dict):
        return str(value.get("text") or ""), False
    if dtype == "string":
        return str(value or ""), False
    return "", False


def _labels(qids: list[str], lang: str) -> dict[str, str]:
    """Batched label resolution — one call per 50 ids, not one call per claim."""
    out: dict[str, str] = {}
    uniq = list(dict.fromkeys(q for q in qids if q))
    langs = "|".join(dict.fromkeys([lang, "en"]))
    for i in range(0, len(uniq), _LABEL_BATCH):
        data = _get_json(
            {
                "action": "wbgetentities",
                "ids": "|".join(uniq[i : i + _LABEL_BATCH]),
                "props": "labels",
                "languages": langs,
                "format": "json",
            }
        )
        if not data:
            continue
        for qid, ent in (data.get("entities") or {}).items():
            labels = ent.get("labels") or {}
            for code in (lang, "en"):
                if code in labels:
                    out[qid] = labels[code]["value"]
                    break
    return out


def find_entity(name: str, lang: str = "en") -> tuple[str, str] | None:
    """Resolve a name (or a whole question) to (qid, label)."""
    if SETTINGS.offline:
        return None
    for cand in _candidates(name):
        data = _get_json(
            {
                "action": "wbsearchentities",
                "search": cand,
                "language": lang,
                "uselang": lang,
                "format": "json",
                "limit": "5",
            }
        )
        if not data:
            continue
        for hit in data.get("search", []) or []:
            # "Wikimedia disambiguation page" / "Wikimedia list article" are
            # navigation stubs with no claims worth citing.
            if "wikimedia" in (hit.get("description") or "").casefold():
                continue
            qid = hit.get("id")
            if qid:
                return qid, hit.get("label") or cand
    return None


def get_facts(qid: str, lang: str = "en") -> dict[str, str]:
    """Readable {property name: value} for the mapped properties this entity has."""
    if SETTINGS.offline:
        return {}
    data = _get_json({"action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"})
    if not data:
        return {}
    claims = ((data.get("entities") or {}).get(qid) or {}).get("claims") or {}
    if not claims:
        return {}

    staged: dict[str, tuple[str, bool]] = {}
    pending: list[str] = []
    for pid, label in _PROPS.items():
        snak = _best_snak(claims.get(pid, []) or [])
        if snak is None:
            continue
        datavalue = snak.get("datavalue") or {}
        text, is_qid = _fmt_value(str(datavalue.get("type") or ""), datavalue.get("value"))
        if not text:
            continue
        staged[label] = (text, is_qid)
        if is_qid:
            pending.append(text)

    resolved = _labels(pending, lang) if pending else {}
    out: dict[str, str] = {}
    for label, (text, is_qid) in staged.items():
        final = resolved.get(text, "") if is_qid else text
        if final:
            out[label] = final
    return out


def lookup(query: str, lang: str = "en") -> tuple[Source, str] | None:
    """Compact fact sentences for `query`, capped so it stays cheap to prefill."""
    if SETTINGS.offline:
        return None
    found = find_entity(query, lang=lang)
    if found is None:
        return None
    qid, label = found
    facts = get_facts(qid, lang=lang)
    if not facts:
        return None

    head = f"{label} —"
    parts: list[str] = []
    used = len(head)
    for key, value in facts.items():
        piece = f" {key}: {value}."
        if used + len(piece) > _MAX_FACT_CHARS:
            break
        parts.append(piece)
        used += len(piece)
    if not parts:
        return None

    src = Source(
        url=f"https://www.wikidata.org/wiki/{qid}", title=label, kind="wikidata", lang=lang
    )
    return src, head + "".join(parts)

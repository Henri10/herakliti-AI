"""Wikipedia access.

Two paths, deliberately separate. ``summary()`` hits the REST endpoint for a
~200-token extract; ``fetch()`` pulls the whole article. Prefill is the
bottleneck on this box, so callers should reach for ``summary()`` unless they
genuinely need the body text.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from herakliti.config import SETTINGS
from herakliti.knowledge.types import Source

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)

_client: "httpx.Client | None" = None

_CRUFT = (
    "References",
    "External links",
    "See also",
    "Notes",
    "Bibliography",
    "Further reading",
    "Sources",
    "Citations",
    "Footnotes",
)

# Level 2 only: a "=== Sources ===" subsection buried mid-article must not
# truncate the body, and the tail sections we want gone are always level 2.
_CRUFT_RE = re.compile(
    r"^==[ \t]*(?:" + "|".join(_CRUFT) + r")[ \t]*==[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_BLANKS_RE = re.compile(r"\n{3,}")
_DISAMBIG_RE = re.compile(r"\b(?:may|can)\s+refer\s+to\b|\brefers?\s+to:", re.IGNORECASE)
_LEADIN_RE = re.compile(r"^.*?\brefers?\s+to\b:?", re.IGNORECASE)


def _http() -> "httpx.Client":
    """One pooled client: Wikimedia 403s a UA without contact info, and TLS
    handshakes are not free at 15W."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(
            headers={"User-Agent": SETTINGS.user_agent},
            timeout=SETTINGS.http_timeout,
            follow_redirects=True,
        )
    return _client


def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _quote(title: str) -> str:
    return quote(title.replace(" ", "_"), safe="/:()',!-")


def _page_url(title: str, lang: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{_quote(title)}"


def _get_json(url: str, params: dict[str, str] | None = None) -> dict | None:
    """Single network chokepoint: one dead source must never kill a query."""
    try:
        r = _http().get(url, params=params)
    except Exception as exc:
        log.debug("wikipedia GET %s failed: %s", url, exc)
        return None
    if r.status_code != 200:
        log.debug("wikipedia GET %s -> HTTP %s", url, r.status_code)
        return None
    try:
        return r.json()
    except Exception as exc:
        log.debug("wikipedia GET %s -> bad json: %s", url, exc)
        return None


def _clean(text: str) -> str:
    cut = _CRUFT_RE.search(text)
    if cut is not None:
        text = text[: cut.start()]
    return _BLANKS_RE.sub("\n\n", text).strip()


def _disambig_target(extract: str, title: str) -> str | None:
    """Pick the primary sense off a disambiguation page.

    The extract lists senses in page order, so entry #1 is the primary meaning.
    ``prop=links`` is not an option: it returns links alphabetically, which
    makes "Anna Kavan" the first link on the Mercury page.
    """
    for raw in extract.splitlines():
        line = _LEADIN_RE.sub("", raw, count=1).strip()
        cand = line.split(",")[0].strip(" .;:–—")
        if cand and len(cand) < 120 and cand.casefold() != title.casefold():
            return cand
    return None


def search(query: str, limit: int = 5, lang: str = "en") -> list[str]:
    """Page titles best matching `query`, best first."""
    if SETTINGS.offline:
        return []
    data = _get_json(
        _api(lang),
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": str(limit),
        },
    )
    if not data:
        return []
    hits = data.get("query", {}).get("search", []) or []
    return [h["title"] for h in hits if h.get("title")]


def summary(title: str, lang: str = "en", _depth: int = 0) -> tuple[Source, str] | None:
    """Short REST extract — the cheap path that matters when prefill costs seconds."""
    if SETTINGS.offline:
        return None
    data = _get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{_quote(title)}")
    if not data:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None

    real_title = data.get("title") or title
    if data.get("type") == "disambiguation" or _DISAMBIG_RE.search(extract[:400]):
        target = _disambig_target(extract, real_title)
        if target is None or _depth >= 1:
            return None
        return summary(target, lang, _depth + 1)

    canonical = (data.get("content_urls", {}).get("desktop", {}) or {}).get("page") or _page_url(
        real_title, lang
    )
    src = Source(url=canonical, title=real_title, kind="wikipedia", lang=lang)
    return src, _clean(extract)


def fetch(title: str, lang: str = "en", _depth: int = 0) -> tuple[Source, str] | None:
    """Full article plaintext, cruft stripped. Redirects are resolved server-side."""
    if SETTINGS.offline:
        return None
    data = _get_json(
        _api(lang),
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "format": "json",
            "redirects": "1",
        },
    )
    if not data:
        return None

    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        if "missing" in page:
            continue
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        real_title = page.get("title") or title

        if _DISAMBIG_RE.search(extract[:400]):
            if _depth >= 1:
                return None
            target = _disambig_target(extract, real_title)
            return fetch(target, lang, _depth + 1) if target else None

        body = _clean(extract)
        if not body:
            continue
        src = Source(url=_page_url(real_title, lang), title=real_title, kind="wikipedia", lang=lang)
        return src, body
    return None


def search_and_fetch(query: str, limit: int = 2, lang: str = "en") -> list[tuple[Source, str]]:
    """Search, then fetch until `limit` articles actually resolve.

    Over-searches because some hits are disambiguation pages or dead titles that
    yield nothing.
    """
    if SETTINGS.offline:
        return []
    out: list[tuple[Source, str]] = []
    seen: set[str] = set()
    for title in search(query, limit=max(limit * 2, limit + 2), lang=lang):
        if len(out) >= limit:
            break
        got = fetch(title, lang=lang)
        if got is None:
            continue
        src, text = got
        if src.url in seen:  # redirects and disambiguation can converge
            continue
        seen.add(src.url)
        out.append((src, text))
    return out

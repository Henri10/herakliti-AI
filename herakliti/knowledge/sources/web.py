"""Open-web retrieval: DuckDuckGo to discover, trafilatura to extract.

The fallback for when Wikipedia and Wikidata have nothing. DDG rate-limits
hard, so every failure in here is swallowed and logged: a throttled search
degrades a query, it does not end it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from herakliti.config import SETTINGS
from herakliti.knowledge.types import Source, content_id

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)

_client: "httpx.Client | None" = None

_MAX_CHARS = 20_000
_WORKERS = 4  # serial fetching is the difference between 5s and 20s

_BINARY_EXT = (
    ".pdf", ".ps", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".rar", ".7z", ".exe", ".dmg", ".iso",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff",
    ".epub", ".mobi", ".apk", ".rpm", ".deb", ".csv", ".xml", ".rss",
)


def _http() -> "httpx.Client":
    """Pooled and thread-safe; shared across the fetch workers."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(
            headers={"User-Agent": SETTINGS.user_agent},
            timeout=SETTINGS.http_timeout,
            follow_redirects=True,
        )
    return _client


def _usable(url: str) -> bool:
    """Reject non-http(s) schemes and anything that is plainly not an article."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    return not parsed.path.lower().rstrip("/").endswith(_BINARY_EXT)


def _title(html: str, url: str) -> str:
    try:
        from trafilatura.metadata import extract_metadata

        meta = extract_metadata(html)
        if meta is not None and meta.title:
            return str(meta.title).strip()
    except Exception as exc:
        log.debug("metadata extraction failed for %s: %s", url, exc)
    return urlparse(url).netloc or url


def search(query: str, max_results: int = 5) -> list[dict]:
    """DDG results as [{"title","href","body"}]. Returns [] on throttling."""
    if SETTINGS.offline:
        return []
    try:
        from ddgs import DDGS

        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # rate limits, DNS, upstream HTML changes
        log.debug("ddgs search %r failed: %s", query, exc)
        return []

    out: list[dict] = []
    for hit in hits or []:
        href = (hit.get("href") or "").strip()
        if not _usable(href):
            continue
        out.append(
            {
                "title": (hit.get("title") or "").strip(),
                "href": href,
                "body": (hit.get("body") or "").strip(),
            }
        )
    return out


def fetch_page(url: str) -> tuple[Source, str] | None:
    """Fetch and extract readable text, or None if there is nothing worth having."""
    if SETTINGS.offline or not _usable(url):
        return None
    try:
        r = _http().get(url)
    except Exception as exc:
        log.debug("web fetch %s failed: %s", url, exc)
        return None
    if r.status_code != 200:
        log.debug("web fetch %s -> HTTP %s", url, r.status_code)
        return None

    ctype = r.headers.get("content-type", "").lower()
    if ctype and "html" not in ctype and "xml" not in ctype and not ctype.startswith("text/"):
        log.debug("web fetch %s -> skipping content-type %s", url, ctype)
        return None

    try:
        html = r.text
    except Exception as exc:
        log.debug("web fetch %s -> undecodable: %s", url, exc)
        return None
    if not html:
        return None

    try:
        import trafilatura

        text = trafilatura.extract(
            html, include_comments=False, include_tables=False, favor_precision=True
        )
    except Exception as exc:
        log.debug("trafilatura failed on %s: %s", url, exc)
        return None
    if not text or not text.strip():
        return None

    src = Source(url=str(r.url), title=_title(html, url), kind="web", lang=SETTINGS.lang)
    return src, text.strip()[:_MAX_CHARS]


def _fetch_hit(hit: dict) -> tuple[Source, str] | None:
    got = fetch_page(hit["href"])
    if got is None:
        return None
    src, text = got
    if not src.title or src.title == urlparse(src.url).netloc:
        if hit.get("title"):  # DDG's title beats a bare hostname
            src.title = hit["title"]
    return src, text


def search_and_fetch(query: str, max_pages: int | None = None) -> list[tuple[Source, str]]:
    """Search, then fetch the hits concurrently, keeping search-rank order."""
    if SETTINGS.offline:
        return []
    limit = SETTINGS.max_web_pages if max_pages is None else max_pages
    if limit <= 0:
        return []

    # Over-fetch: paywalls and JS-only pages extract to nothing, and they cost
    # us nothing extra in wall-clock while the pool is already open.
    hits = search(query, max_results=max(limit * 2, limit + 2))
    if not hits:
        return []

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        fetched = list(pool.map(_fetch_hit, hits))  # map preserves input order

    out: list[tuple[Source, str]] = []
    seen_urls: set[str] = set()
    seen_text: set[str] = set()
    for got in fetched:
        if got is None:
            continue
        src, text = got
        # URL dedupe alone is not enough: MediaWiki serves redirects like
        # /wiki/Capital_of_Albania as 200 OK at the original URL, so two hits
        # come back with different URLs and byte-identical bodies. Paying
        # prefill twice for one article is the worst trade in this system.
        fingerprint = content_id(text[:1000])
        if src.url in seen_urls or fingerprint in seen_text:
            continue
        seen_urls.add(src.url)
        seen_text.add(fingerprint)
        out.append((src, text))
        if len(out) >= limit:
            break
    return out

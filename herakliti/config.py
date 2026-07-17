"""Central configuration.

Every tunable lives here. Values are overridable via environment variables
prefixed with ``HERAKLITI_`` (e.g. ``HERAKLITI_N_THREADS=8``).

Imports nothing heavy — safe to import from anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths — all runtime data lives under ~/.herakliti and is regenerable.
# --------------------------------------------------------------------------

HOME: Path = Path(os.getenv("HERAKLITI_HOME", str(Path.home() / ".herakliti")))
MODELS_DIR: Path = HOME / "models"
INDEX_DIR: Path = HOME / "index"
CACHE_DIR: Path = HOME / "cache"

DB_PATH: Path = INDEX_DIR / "herakliti.db"
FAISS_PATH: Path = INDEX_DIR / "faiss.bin"


def ensure_dirs() -> None:
    for d in (HOME, MODELS_DIR, INDEX_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str) -> str:
    return os.getenv(f"HERAKLITI_{key}", default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _default_threads() -> int:
    """llama.cpp is fastest at roughly the physical core count.

    This is a hybrid Intel part (P-cores + E-cores); oversubscribing logical
    threads makes it slower, not faster. Leave one core for the OS.
    """
    return max(1, (os.cpu_count() or 4) - 2)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Settings:
    # --- engine ---
    model: str = field(default_factory=lambda: _env("MODEL", "default"))
    n_ctx: int = field(default_factory=lambda: _env_int("N_CTX", 8192))
    n_threads: int = field(default_factory=lambda: _env_int("N_THREADS", _default_threads()))
    n_batch: int = field(default_factory=lambda: _env_int("N_BATCH", 512))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 640))

    # --- embeddings / rerank (CPU-friendly, multilingual: Albanian works) ---
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "
    embed_batch: int = field(default_factory=lambda: _env_int("EMBED_BATCH", 32))

    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", "1") == "1")

    # --- chunking (Wikipedia prose) ---
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # --- retrieval ---
    k_retrieve: int = field(default_factory=lambda: _env_int("K_RETRIEVE", 30))   # per lexical/dense arm
    k_rerank: int = field(default_factory=lambda: _env_int("K_RERANK", 20))       # fed to cross-encoder
    k_context: int = field(default_factory=lambda: _env_int("K_CONTEXT", 6))      # placed in the prompt
    rrf_k: int = 60                                                              # standard RRF constant
    min_relevance: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE", 0.25))
    # Cross-encoder logit below which the best local hit counts as "not real coverage",
    # so we go fetch from the live world. Calibrated on this reranker: genuinely relevant
    # passages score +7..+11, irrelevant ones -6..-10, so +1.0 separates them with margin.
    rerank_gate: float = field(default_factory=lambda: _env_float("RERANK_GATE", 1.0))
    # Added to a user-taught chunk's cross-encoder score so a genuinely relevant fact the
    # user taught outranks a competing web/wiki chunk and counts as coverage (no live fetch
    # for something already taught). Small on purpose: an off-topic user fact still scores
    # ~-6, so +2 lifts it to ~-4 — well under rerank_gate — while a relevant ~+8 becomes ~+10.
    user_fact_boost: float = field(default_factory=lambda: _env_float("USER_FACT_BOOST", 2.0))

    # --- network ---
    user_agent: str = field(
        default_factory=lambda: _env(
            "USER_AGENT",
            "Herakliti-AI/1.0 (local research assistant; +https://github.com/herakliti-ai)",
        )
    )
    http_timeout: float = field(default_factory=lambda: _env_float("HTTP_TIMEOUT", 20.0))
    max_web_pages: int = field(default_factory=lambda: _env_int("MAX_WEB_PAGES", 3))
    lang: str = field(default_factory=lambda: _env("LANG", "en"))

    # --- behaviour ---
    offline: bool = field(default_factory=lambda: _env("OFFLINE", "0") == "1")
    verbose: bool = field(default_factory=lambda: _env("VERBOSE", "0") == "1")


SETTINGS = Settings()

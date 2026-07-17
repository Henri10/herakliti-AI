"""Dense embeddings with a persistent content-addressed cache.

Two invariants live here so callers cannot break them:

1. e5 models are asymmetric — they require "query: " / "passage: " prefixes.
   Getting these wrong does not raise, it silently degrades retrieval quality.
   Callers never pass prefixes; they pick a method and the prefix follows.
2. Vectors are always L2-normalized, so FAISS ``IndexFlatIP`` is exactly cosine.

The cache is SQLite rather than a .npz because embedding runs on a 15W CPU where
a re-ingest is measured in minutes, and because SQLite gives us atomic commits
and multi-process access for free — an interrupted write cannot leave a torn file.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from typing import TYPE_CHECKING

from ..config import CACHE_DIR, SETTINGS

if TYPE_CHECKING:
    import numpy as np

_CACHE_PATH = CACHE_DIR / "embeddings.db"

_DEFAULT_DIM = 384
"""multilingual-e5-small's width. Superseded by the real value once loaded."""


class Embedder:
    """Sentence embeddings, cached by content hash. Use :meth:`get`, not the constructor."""

    _instance: "Embedder | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._dim = _DEFAULT_DIM
        self._db: sqlite3.Connection | None = None
        self._load_lock = threading.Lock()
        # Separate from _load_lock: one shared connection must be serialized, but
        # cache reads must not block behind a multi-second model load.
        self._db_lock = threading.Lock()

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def dim(self) -> int:
        """Real width once the model is loaded; the known default beforehand."""
        return self._dim

    # -- model ---------------------------------------------------------------

    def _load(self):
        """Deferred: importing torch costs seconds, so importing herakliti must not."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(SETTINGS.embed_model, device="cpu")
                    self._dim = model.get_sentence_embedding_dimension() or _DEFAULT_DIM
                    self._model = model
        return self._model

    # -- cache ---------------------------------------------------------------

    def _key(self, prefixed: str) -> str:
        """Keyed on the model too: embed_model is env-overridable, and vectors from a
        different model are not merely stale but wrong (and may differ in width)."""
        h = hashlib.sha256()
        h.update(SETTINGS.embed_model.encode("utf-8"))
        h.update(b"\x1f")
        h.update(prefixed.encode("utf-8", "replace"))
        return h.hexdigest()

    def _conn(self) -> sqlite3.Connection | None:
        """WAL + a busy timeout is what makes the cache safe for a second Herakliti
        process: concurrent readers never block, and a kill mid-write rolls back."""
        if self._db is None:
            with self._db_lock:
                if self._db is None:
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        db = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False, timeout=5.0)
                        db.execute("PRAGMA journal_mode=WAL")
                        db.execute("PRAGMA synchronous=NORMAL")
                        db.execute("PRAGMA busy_timeout=5000")
                        db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
                        db.commit()
                        self._db = db
                    except sqlite3.Error:
                        return None
        return self._db

    def _cache_get(self, keys: list[str]) -> dict[str, bytes]:
        db = self._conn()
        if db is None or not keys:
            return {}
        out: dict[str, bytes] = {}
        try:
            with self._db_lock:
                for i in range(0, len(keys), 500):  # stay under SQLITE_MAX_VARIABLE_NUMBER
                    batch = keys[i : i + 500]
                    q = f"SELECT k, v FROM vec WHERE k IN ({','.join('?' * len(batch))})"
                    out.update({k: v for k, v in db.execute(q, batch)})
        except sqlite3.Error:
            return out
        return out

    def _cache_put(self, rows: list[tuple[str, bytes]]) -> None:
        """A failed cache write costs a recompute, never a query — swallow it."""
        db = self._conn()
        if db is None or not rows:
            return
        try:
            with self._db_lock:
                db.executemany("INSERT OR REPLACE INTO vec (k, v) VALUES (?, ?)", rows)
                db.commit()
        except sqlite3.Error:
            pass

    # -- api -----------------------------------------------------------------

    def _encode(self, prefixed: list[str]) -> "np.ndarray":
        import numpy as np

        model = self._load()
        vecs = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=SETTINGS.embed_batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32).reshape(len(prefixed), -1)

    def embed_documents(self, texts: list[str]) -> "np.ndarray":
        """(n, dim) float32, L2-normalized. Applies the passage prefix for you."""
        import numpy as np

        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        prefixed = [SETTINGS.embed_passage_prefix + t for t in texts]
        keys = [self._key(p) for p in prefixed]
        cached = self._cache_get(keys)

        # Dedupe misses: a batch often repeats text, and encoding is the expensive part.
        missing: list[str] = []
        missing_keys: list[str] = []
        seen: set[str] = set()
        for k, p in zip(keys, prefixed):
            if k not in cached and k not in seen:
                seen.add(k)
                missing_keys.append(k)
                missing.append(p)

        if missing:
            fresh = self._encode(missing)
            self._dim = fresh.shape[1]
            self._cache_put([(k, v.tobytes()) for k, v in zip(missing_keys, fresh)])
            cached.update({k: v.tobytes() for k, v in zip(missing_keys, fresh)})

        out = np.stack([np.frombuffer(cached[k], dtype=np.float32) for k in keys])
        return np.ascontiguousarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> "np.ndarray":
        """(dim,) float32, L2-normalized. Applies the query prefix for you."""
        import numpy as np

        prefixed = SETTINGS.embed_query_prefix + text
        key = self._key(prefixed)
        hit = self._cache_get([key]).get(key)
        if hit is not None:
            return np.frombuffer(hit, dtype=np.float32).copy()

        vec = self._encode([prefixed])[0]
        self._dim = vec.shape[0]
        self._cache_put([(key, vec.tobytes())])
        return np.ascontiguousarray(vec, dtype=np.float32)

"""Persistent hybrid memory: SQLite (facts + BM25) beside FAISS (meaning).

This is what makes Herakliti *learn*. Every document fetched from the network is
written here once; the second time a topic comes up it is served from disk in
milliseconds with no HTTP and no re-embedding.

Two indexes over one corpus, joined by `chunks.rowid`:
  * FTS5/BM25 catches names, numbers and rare tokens that embeddings blur away.
  * FAISS/cosine catches paraphrase, translation and synonymy.
The int64 id handed to FAISS *is* the sqlite rowid, so a dense hit resolves back
to its Chunk with one indexed lookup. Fusing the two arms is the caller's job.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from herakliti import config
from herakliti.knowledge.types import Chunk, Source

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

MAX_FTS_TERMS = 32
"""A pasted paragraph as a query is pathological for BM25 and slow. Truncate."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    doc_id     TEXT PRIMARY KEY,
    url        TEXT UNIQUE,
    title      TEXT,
    kind       TEXT,
    lang       TEXT,
    fetched_at REAL
);
CREATE TABLE IF NOT EXISTS chunks(
    rowid    INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE,
    doc_id   TEXT,
    title    TEXT,
    url      TEXT,
    kind     TEXT,
    position INT,
    text     TEXT
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);

-- remove_diacritics 2 folds ë->e and ç->c, so a query for "Tirane" still finds
-- "Tiranë". The corpus is multilingual; the default tokenizer is not enough.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    title,
    content='chunks',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2"
);

-- Triggers, not explicit inserts: the FTS shadow table cannot drift out of sync
-- no matter which code path writes. On INSERT OR IGNORE that ignores, the
-- trigger never fires, so re-ingesting stays a genuine no-op.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.rowid, new.text, new.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title)
    VALUES ('delete', old.rowid, old.text, old.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title)
    VALUES ('delete', old.rowid, old.text, old.title);
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.rowid, new.text, new.title);
END;
"""

# Word characters only: anything FTS5 could read as an operator (" * : ^ - ( )
# AND OR NOT NEAR) cannot survive tokenisation into the MATCH string.
_TERM_RE = re.compile(r"\w[\w']*", re.UNICODE)


def _fts_query(text: str) -> str:
    """Compile arbitrary user text into a MATCH expression that cannot raise.

    Unsanitised input is not a theoretical problem: a question as ordinary as
    `Who wrote "Hamlet" AND when?` is a syntax error inside MATCH and would take
    the whole query down. Every term is extracted, double-quoted into a literal
    and OR-ed — OR because recall is the lexical arm's job; precision comes from
    BM25 ranking and the reranker downstream.
    """
    seen: dict[str, None] = {}
    for term in _TERM_RE.findall(text):
        seen.setdefault(term.lower(), None)
        if len(seen) >= MAX_FTS_TERMS:
            break
    # The escape is unreachable through _TERM_RE today; it stays because this is
    # the one place untrusted text meets SQL, and the regex may loosen later.
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in seen)


class KnowledgeStore:
    """Hybrid store over sqlite + FAISS. Not a context manager by accident: call
    `save()` when you are done ingesting, or `close()` which saves for you."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        faiss_path: str | Path | None = None,
        dim: int = 384,
    ) -> None:
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        self.faiss_path = Path(faiss_path) if faiss_path else config.FAISS_PATH
        self.dim = dim
        self._lock = threading.Lock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

        self._index, self._ids = self._load_index()

    # -- index lifecycle ---------------------------------------------------

    def _load_index(self) -> tuple[Any, set[int]]:
        """Read the FAISS index, falling back to an empty one on cold start or rot.

        A truncated or dimension-mismatched index must not be fatal: the vectors
        are derivable from text we still hold, and `add_document` re-adds any
        chunk whose rowid is missing from the index, so a wiped index heals on
        the next ingest instead of bricking the store.
        """
        import faiss

        if self.faiss_path.exists():
            try:
                idx = faiss.read_index(str(self.faiss_path))
                if idx.d != self.dim:
                    raise ValueError(f"index dim {idx.d} != expected {self.dim}")
                # AttributeError here means it is not an IndexIDMap — also unusable.
                return idx, set(faiss.vector_to_array(idx.id_map).tolist())
            except Exception as e:
                log.warning("faiss index at %s unusable (%s); starting empty", self.faiss_path, e)

        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim)), set()

    # -- writing -----------------------------------------------------------

    def add_document(self, source: Source, chunks: list[Chunk], vectors: "np.ndarray") -> int:
        """Persist a document and its chunks. Returns the number of vectors newly indexed.

        Idempotent: re-ingesting the same url duplicates nothing, and a chunk is
        only embedded into FAISS if its rowid is not already there.
        """
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if len(chunks) and (vectors.ndim != 2 or vectors.shape[1] != self.dim):
            raise ValueError(f"vectors must be (n, {self.dim}), got {vectors.shape}")

        with self._lock:
            with self._db:  # one transaction: either the document lands or it does not
                self._db.execute(
                    "INSERT OR IGNORE INTO documents(doc_id, url, title, kind, lang, fetched_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (source.doc_id, source.url, source.title, source.kind,
                     source.lang, source.fetched_at),
                )
                rowids: list[int] = []
                for c in chunks:
                    self._db.execute(
                        "INSERT OR IGNORE INTO chunks(chunk_id, doc_id, title, url, kind, position, text)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (c.id, c.doc_id, c.title, c.url, c.kind, c.position, c.text),
                    )
                    # Re-read rather than trust lastrowid: on IGNORE it is stale.
                    row = self._db.execute(
                        "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.id,)
                    ).fetchone()
                    rowids.append(int(row["rowid"]))

            # `picked` guards the in-batch case: a duplicate id inside one
            # add_with_ids call leaves FAISS with two vectors under one rowid,
            # which no later dedup can undo.
            fresh: list[tuple[int, Any]] = []
            picked: set[int] = set()
            for rid, v in zip(rowids, vectors):
                if rid not in self._ids and rid not in picked:
                    picked.add(rid)
                    fresh.append((rid, v))
            if not fresh:
                return 0

            ids = np.asarray([r for r, _ in fresh], dtype=np.int64)
            vecs = np.ascontiguousarray(np.stack([v for _, v in fresh]), dtype=np.float32)
            self._index.add_with_ids(vecs, ids)
            self._ids.update(int(i) for i in ids)
            return len(fresh)

    # -- reading -----------------------------------------------------------

    def search_dense(self, qvec: "np.ndarray", k: int) -> list[tuple[Chunk, float]]:
        """Cosine nearest neighbours. `qvec` must be L2-normalised (score = inner product)."""
        import numpy as np

        if k <= 0 or self._index.ntotal == 0:
            return []

        q = np.ascontiguousarray(np.asarray(qvec, dtype=np.float32).reshape(1, self.dim))
        with self._lock:
            scores, ids = self._index.search(q, min(k, self._index.ntotal))

        hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        by_rowid = self._chunks_by_rowid(r for r, _ in hits)
        return [(by_rowid[r], s) for r, s in hits if r in by_rowid]

    def search_lexical(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """BM25 over FTS5, best first, score normalised to (0, 1].

        bm25() returns *negative* numbers and more-negative means better, so the
        raw value is negated. Magnitudes are not comparable across queries (a rare
        term scores ~1.0, a term in half the corpus ~1e-6), so the score is scaled
        against the best hit in this result set — meaningful as a ranking, not as
        an absolute.
        """
        match = _fts_query(query)
        if not match or k <= 0:
            return []

        try:
            # Title is weighted below text: the chunker already prepends the title
            # to the body, and double-counting it drowns the actual content.
            rows = self._db.execute(
                "SELECT rowid, bm25(chunks_fts, 1.0, 0.5) AS score FROM chunks_fts"
                " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 0.5) ASC LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.Error as e:
            log.warning("lexical search failed for %.60r: %s", query, e)
            return []

        if not rows:
            return []

        raw = [(int(r["rowid"]), -float(r["score"])) for r in rows]
        best = max(s for _, s in raw) or 1.0
        by_rowid = self._chunks_by_rowid(r for r, _ in raw)
        return [(by_rowid[r], s / best) for r, s in raw if r in by_rowid]

    def _chunks_by_rowid(self, rowids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(rowids)
        if not ids:
            return {}
        q = f"SELECT rowid, * FROM chunks WHERE rowid IN ({','.join('?' * len(ids))})"
        return {int(r["rowid"]): _to_chunk(r) for r in self._db.execute(q, ids)}

    def has_url(self, url: str) -> bool:
        """True if this url is already learned — the check that skips a network round trip."""
        return self._db.execute(
            "SELECT 1 FROM documents WHERE url = ? LIMIT 1", (url,)
        ).fetchone() is not None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._db.execute(
            "SELECT rowid, * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _to_chunk(row) if row else None

    def chunks_by_kind(self, kind: str) -> list[Chunk]:
        """Every stored chunk of one SourceKind, oldest first — how the memory layer
        enumerates what the user has taught (kind="user")."""
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid, chunk_id, doc_id, title, url, kind, position, text"
                " FROM chunks WHERE kind = ? ORDER BY rowid", (kind,)
            ).fetchall()
        return [_to_chunk(r) for r in rows]

    def delete_by_url_prefix(self, prefix: str) -> int:
        """Delete every chunk (and its document + dense vector) whose url starts with prefix.

        Returns the number of chunk rows removed. Keeps all three stores consistent:
        the AFTER DELETE trigger drops each FTS shadow row, and remove_ids drops the
        vector by rowid so a stale vector can never resurface in a dense search. Persisting
        the FAISS change is the caller's job — call save() afterwards, as with add_document.
        """
        import faiss
        import numpy as np

        like = _like_prefix(prefix)
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid FROM chunks WHERE url LIKE ? ESCAPE '\\'", (like,)
            ).fetchall()
            rowids = [int(r["rowid"]) for r in rows]
            with self._db:  # one transaction so chunks and their document fall together
                self._db.execute(
                    "DELETE FROM chunks WHERE url LIKE ? ESCAPE '\\'", (like,)
                )
                self._db.execute(
                    "DELETE FROM documents WHERE url LIKE ? ESCAPE '\\'", (like,)
                )
            if rowids:
                try:
                    self._index.remove_ids(faiss.IDSelectorBatch(
                        np.asarray(rowids, dtype=np.int64)))
                except Exception:
                    # A dangling vector is harmless (search resolves rowid->row and drops
                    # the miss), so a faiss hiccup must not fail the delete.
                    log.warning("faiss remove_ids failed for %d id(s)", len(rowids),
                                exc_info=True)
                self._ids.difference_update(rowids)
        return len(rowids)

    def stats(self) -> dict:
        docs = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size = sum(
            p.stat().st_size
            for p in (self.db_path, self.faiss_path,
                      self.db_path.with_suffix(self.db_path.suffix + "-wal"))
            if p.exists()
        )
        return {
            "documents": int(docs),
            "chunks": int(chunks),
            "vectors": int(self._index.ntotal),
            "size_mb": round(size / 1e6, 2),
        }

    # -- self-healing ------------------------------------------------------

    def unvectored_chunks(self) -> list[Chunk]:
        """Chunks present in sqlite but absent from the FAISS index.

        Non-empty means the two stores drifted — typically a lost or corrupt
        faiss.bin outliving the database. Left alone, every such chunk is dead
        weight the dense arm can never return, and the normal fetch path won't
        re-embed it (the document's url is already known). This is how the
        retriever finds what to repair.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid, chunk_id, doc_id, title, url, kind, position, text FROM chunks"
            ).fetchall()
        out: list[Chunk] = []
        for r in rows:
            if int(r["rowid"]) in self._ids:
                continue
            out.append(Chunk(
                text=r["text"], doc_id=r["doc_id"], title=r["title"], url=r["url"],
                kind=r["kind"], position=int(r["position"]), id=r["chunk_id"],
            ))
        return out

    def reindex_vectors(self, rebuilt: dict[str, "np.ndarray"]) -> int:
        """Add vectors for chunks identified by chunk_id. Returns how many landed.

        `rebuilt` maps chunk_id -> embedding. The store cannot embed on its own
        (that would drag torch into this module), so the caller supplies vectors
        for the chunk_ids from `unvectored_chunks()`.
        """
        import numpy as np

        if not rebuilt:
            return 0
        with self._lock:
            added = 0
            for chunk_id, vec in rebuilt.items():
                row = self._db.execute(
                    "SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()
                if row is None:
                    continue
                rid = int(row["rowid"])
                if rid in self._ids:
                    continue
                v = np.ascontiguousarray(
                    np.asarray(vec, dtype=np.float32).reshape(1, self.dim)
                )
                self._index.add_with_ids(v, np.asarray([rid], dtype=np.int64))
                self._ids.add(rid)
                added += 1
        return added

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Flush sqlite and write the FAISS index via a temp file + atomic replace.

        Writing in place would leave a half-written index if the process dies
        mid-write — and that file is the only copy of work that cost minutes of
        CPU to embed.
        """
        import faiss

        with self._lock:
            self._db.commit()
            tmp = self.faiss_path.with_suffix(self.faiss_path.suffix + ".tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, self.faiss_path)

    def close(self) -> None:
        try:
            self.save()
        finally:
            self._db.close()


def _like_prefix(prefix: str) -> str:
    r"""A LIKE pattern matching `prefix` followed by anything, with % _ \ neutralised.

    The url scheme carries no wildcards today, but escaping is the safe contract for the
    one place a caller-supplied string reaches a LIKE clause."""
    esc = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return esc + "%"


def _to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        text=row["text"],
        doc_id=row["doc_id"],
        title=row["title"],
        url=row["url"],
        kind=row["kind"],
        position=int(row["position"]),
        id=row["chunk_id"],
    )

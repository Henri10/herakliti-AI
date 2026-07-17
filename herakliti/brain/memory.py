"""Conversation memory.

Deliberately small. Every remembered token is re-prefilled on the next turn, and prefill
is the dominant cost on this hardware — a generous history would quietly make each turn
slower than the last. We keep a short window and drop the oldest pairs.

Persisted to disk (one small JSON file) so context survives restarting the CLI or the
server, not just an active session — knowledge already worked this way (every fetched
document lands in the store); this closes the same gap for the last few things you said.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Memory:
    max_turns: int = 6
    path: Path | None = None
    _turns: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None, max_turns: int = 6) -> "Memory":
        """Restore turns from disk if a path is given and something is there.

        Never raises: a missing file, a first run, a corrupt write interrupted mid-flight —
        all just mean starting with empty history rather than losing the conversation entirely.
        """
        mem = cls(max_turns=max_turns, path=path)
        if path is None:
            return mem
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            turns = [(str(q), str(a)) for q, a in raw]
            mem._turns = turns[-max_turns:]
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            log.debug("could not load conversation memory from %s", path, exc_info=True)
        return mem

    def add(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))
        if len(self._turns) > self.max_turns:
            del self._turns[: len(self._turns) - self.max_turns]
        self._save()

    def as_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for q, a in self._turns:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    def recent(self, n: int = 2) -> list[dict]:
        """Just the last n exchanges — enough for pronoun resolution, cheap to prefill."""
        msgs: list[dict] = []
        for q, a in self._turns[-n:]:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    def clear(self) -> None:
        self._turns.clear()
        self._save()

    def _save(self) -> None:
        """Best-effort and atomic. A failed write must never break the conversation — it
        only means this turn will not survive a restart, which is not worth surfacing."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._turns, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self.path)
        except OSError:
            log.debug("could not persist conversation memory to %s", self.path, exc_info=True)

    def __len__(self) -> int:
        return len(self._turns)

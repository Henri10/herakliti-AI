"""Conversation memory.

Deliberately small. Every remembered token is re-prefilled on the next turn, and prefill
is the dominant cost on this hardware — a generous history would quietly make each turn
slower than the last. We keep a short window and drop the oldest pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Memory:
    max_turns: int = 6
    _turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))
        if len(self._turns) > self.max_turns:
            del self._turns[: len(self._turns) - self.max_turns]

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

    def __len__(self) -> int:
        return len(self._turns)

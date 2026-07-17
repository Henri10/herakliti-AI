"""Herakliti-AI — a fully local, pure-Python question-answering system.

llama.cpp is the muscle; Herakliti is the brain. Factual accuracy comes from
retrieval against live Wikipedia/Wikidata/web rather than from memorised weights,
so a small model on a laptop CPU can answer accurately — and cite its sources.

Named after Heraclitus of Ephesus (Herakliti), who held that everything flows:
this system's knowledge is never fixed. It grows with every question asked.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["Herakliti", "Answer", "__version__"]


def __getattr__(name: str):
    # Lazy re-export: importing `herakliti` must stay instant and must not drag in
    # torch/llama_cpp. Only touch the heavy modules when the symbol is actually used.
    if name == "Herakliti":
        from herakliti.brain.agent import Herakliti

        return Herakliti
    if name == "Answer":
        from herakliti.knowledge.types import Answer

        return Answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

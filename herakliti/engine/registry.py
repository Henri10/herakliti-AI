"""Model catalog — the "ollama pull" equivalent, but pure Python.

Every entry here was verified live against the Hugging Face API: the repo exists,
is ungated, and the exact filename resolves. Do not add an entry you have not checked.

Sizes are the real GGUF byte sizes (decimal GB), measured, not advertised.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    repo_id: str
    filename: str
    size_gb: float
    params: str
    ctx_native: int
    note: str

    @property
    def label(self) -> str:
        return f"{self.alias} ({self.params}, {self.size_gb:.1f} GB)"


# Qwen3.5 (Feb 2026) — 201 languages *including Tosk Albanian*, native tool calling,
# and non-thinking by default on the Small series (verified from the GGUF's Jinja
# template: `enable_thinking` undefined => no <think> block). That default matters:
# a thinking model burning 500-2000 reasoning tokens at ~7 tok/s is unusable here.
MODELS: dict[str, ModelSpec] = {
    "fast": ModelSpec(
        alias="fast",
        repo_id="unsloth/Qwen3.5-2B-GGUF",
        filename="Qwen3.5-2B-Q4_K_M.gguf",
        size_gb=1.28,
        params="2B",
        ctx_native=262144,
        note="Roughly 2x the prefill of the 4B — the tier that actually feels interactive on a laptop CPU.",
    ),
    "default": ModelSpec(
        alias="default",
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        filename="Qwen3.5-4B-Q4_K_M.gguf",
        size_gb=2.74,
        params="4B",
        ctx_native=262144,
        note="Best quality that stays usable. Strong at grounded RAG and Albanian.",
    ),
    "quality": ModelSpec(
        alias="quality",
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        filename="Qwen3.5-9B-Q4_K_M.gguf",
        size_gb=5.68,
        params="9B",
        ctx_native=262144,
        note="Highest quality that fits in 16GB. Slow: submit-and-walk-away, not interactive.",
    ),
}

DEFAULT_ALIAS = "default"


def resolve(name: str | None) -> ModelSpec:
    """Resolve an alias ('fast'), or a bare/gguf filename, to a ModelSpec."""
    if not name:
        return MODELS[DEFAULT_ALIAS]
    if name in MODELS:
        return MODELS[name]
    for spec in MODELS.values():
        if name == spec.filename or name == spec.filename.removesuffix(".gguf"):
            return spec
    known = ", ".join(MODELS)
    raise KeyError(f"Unknown model {name!r}. Known aliases: {known}")

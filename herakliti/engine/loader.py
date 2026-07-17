"""Model acquisition: resolve an alias to a local GGUF path, downloading if needed.

This is the "ollama pull" step. Nothing here imports llama_cpp — keeping download
separate from load means `herakliti pull` stays fast and doesn't touch the runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from herakliti import config
from herakliti.engine.registry import ModelSpec, resolve

log = logging.getLogger(__name__)


def local_path(spec: ModelSpec) -> Path:
    return config.MODELS_DIR / spec.filename


def is_downloaded(spec: ModelSpec) -> bool:
    p = local_path(spec)
    # A partially-downloaded file is worse than none: require a plausible size.
    return p.exists() and p.stat().st_size > 0.5 * spec.size_gb * 1e9


def ensure_model(
    name: str | None = None,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Return the local path to the model's GGUF, downloading it if absent.

    Raises RuntimeError if the model is missing and we are in offline mode.
    """
    spec = resolve(name)
    dest = local_path(spec)
    if is_downloaded(spec):
        return dest

    if config.SETTINGS.offline:
        raise RuntimeError(
            f"Model {spec.alias!r} is not downloaded and offline mode is on.\n"
            f"Run without --offline once to fetch it ({spec.size_gb:.1f} GB)."
        )

    config.ensure_dirs()
    if on_progress:
        on_progress(f"Downloading {spec.filename} ({spec.size_gb:.1f} GB) from {spec.repo_id}")
    log.info("downloading %s from %s", spec.filename, spec.repo_id)

    from huggingface_hub import hf_hub_download  # heavy-ish; import on use

    got = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(config.MODELS_DIR),
    )
    return Path(got)

"""Configuration discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def load_config(explicit: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Load configuration from the most relevant location."""

    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    env_value = os.environ.get("CONAN_DUB_CONFIG")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path("configs/default.yaml"))

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}, candidate.resolve()

    return {}, None

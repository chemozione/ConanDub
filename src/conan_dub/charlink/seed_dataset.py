"""Utilities to assemble the initial character seed dataset."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SeedConfig:
    """Configuration for seed dataset creation."""

    frames_dir: Path
    output_dir: Path
    limit_per_character: int = 20


def collect_seed_crops(
    character: str,
    config: SeedConfig,
    crop_paths: Iterable[Path],
    dry_run: bool = False,
) -> List[Path]:
    """Copy provided crops into the character dataset."""

    dest_dir = config.output_dir / character
    dest_dir.mkdir(parents=True, exist_ok=True)

    stored: List[Path] = []
    for idx, crop_path in enumerate(crop_paths):
        if idx >= config.limit_per_character:
            break
        target = dest_dir / f"{character}_{idx:03d}{crop_path.suffix}"
        if dry_run:
            logger.info("Dry-run seed copy %s -> %s", crop_path.name, target.name)
            target.write_text("dry-run crop placeholder\n", encoding="utf-8")
        else:
            shutil.copyfile(crop_path, target)
        stored.append(target)
    return stored


"""Utilities for saving and loading dubbing manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import DubbingManifest, Utterance, as_records, to_parquet


def load_manifest(source: Path) -> DubbingManifest:
    """Load a manifest from JSON."""

    source = source.resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = [Utterance(**item) for item in payload["items"]]
    return DubbingManifest(items=items)


def save_manifest_json(manifest: DubbingManifest, destination: Path) -> Path:
    """Persist a manifest to JSON."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {"items": list(as_records(manifest))}
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return destination


def save_manifest_parquet(
    manifest: DubbingManifest, destination: Path, overwrite: bool = False
) -> Path:
    """Persist a manifest to Parquet."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_parquet(manifest, destination, overwrite=overwrite)
    return destination


def ensure_manifest(destination: Path) -> None:
    """Create an empty manifest file if it does not exist."""

    if destination.exists():
        return
    empty = DubbingManifest(items=[])
    save_manifest_json(empty, destination)


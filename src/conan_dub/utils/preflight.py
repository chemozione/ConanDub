"""Environment preflight helpers."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


def check_env(outputs: Iterable[Path], dry_run: bool = False) -> List[str]:
    """Return human-readable preflight messages for the current environment."""

    messages: List[str] = []
    if sys.version_info < (3, 10):
        messages.append(
            f"Python 3.10+ is recommended; detected {sys.version.split()[0]}. Update your interpreter."
        )

    if shutil.which("ffmpeg") is None:
        note = (
            "ffmpeg not found on PATH. Install it (e.g., `choco install ffmpeg` on Windows) "
            "before running real separation/muxing."
        )
        if dry_run:
            note += " Continuing because dry-run mode is enabled."
        messages.append(note)

    for target in outputs:
        parent = _resolve_parent(target)
        if parent is None:
            continue
        try:
            parent.mkdir(parents=True, exist_ok=True)
            test_file = parent / ".conan_dub_write_test"
            with test_file.open("w", encoding="utf-8") as handle:
                handle.write("ok")
            if test_file.exists():
                test_file.unlink()
        except OSError as exc:
            messages.append(f"Cannot write to {parent}: {exc}")

    return messages


def ffmpeg_available() -> bool:
    """Return True when ffmpeg is discoverable on PATH."""

    return shutil.which("ffmpeg") is not None


def python_version() -> str:
    """Return the short Python version string."""

    return sys.version.split()[0]


def platform_summary() -> str:
    """Return a platform descriptor suitable for diagnostics."""

    return platform.platform()


def _resolve_parent(path: Path) -> Path | None:
    """Return the directory to probe for write permissions."""

    if path.exists():
        return path if path.is_dir() else path.parent
    if path.suffix:
        return path.parent
    return path

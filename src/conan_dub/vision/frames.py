"""Utilities for extracting frames from video assets."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    ffmpeg_bin: str = "ffmpeg",
    dry_run: bool = False,
) -> Path:
    """Extract frames using ffmpeg."""

    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run frame extraction for %s", video_path.name)
        placeholder = output_dir / "frame_0001.txt"
        placeholder.write_text("dry-run frame placeholder\n", encoding="utf-8")
        return output_dir

    command = [
        ffmpeg_bin,
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(output_dir / "frame_%04d.png"),
    ]
    logger.info("Running ffmpeg frame extraction: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg binary not found; install ffmpeg or use dry-run mode.") from exc
    return output_dir


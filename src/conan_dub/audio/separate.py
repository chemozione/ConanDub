"""Audio separation interfaces for voice/background stems."""

from __future__ import annotations

import logging
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

SeparatorName = Literal["vocal", "demucs", "uvr5"]


@dataclass(slots=True)
class SeparationRequest:
    """Collect arguments for a separation run."""

    input_path: Path
    output_dir: Path
    separator: SeparatorName = "vocal"
    dry_run: bool = False
    extras: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class SeparationResult:
    """Paths to separated stems."""

    vocal_path: Path
    accompaniment_path: Path


def run_separation(request: SeparationRequest) -> SeparationResult:
    """Execute the requested separation pipeline.

    A dry-run writes short silent placeholders so the downstream pipeline
    can be tested without invoking heavy ML models.
    """

    input_path = request.input_path.expanduser().resolve()
    if not input_path.exists() and not request.dry_run:
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    request.output_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = request.output_dir / "voice.wav"
    accompaniment_path = request.output_dir / "background.wav"

    if request.dry_run:
        logger.info("Dry-run separation: creating stub stems in %s", request.output_dir)
        _write_stub_wave(vocal_path)
        _write_stub_wave(accompaniment_path)
        return SeparationResult(vocal_path=vocal_path, accompaniment_path=accompaniment_path)

    logger.info("Running %s separator for %s", request.separator, input_path.name)

    if request.separator == "vocal":
        _fallback_passthrough(input_path, vocal_path, accompaniment_path)
    elif request.separator in {"demucs", "uvr5"}:
        logger.warning(
            "%s adapter not implemented yet; falling back to pass-through output.",
            request.separator,
        )
        _fallback_passthrough(input_path, vocal_path, accompaniment_path)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported separator: {request.separator}")

    return SeparationResult(vocal_path=vocal_path, accompaniment_path=accompaniment_path)


def _fallback_passthrough(source: Path, vocal_path: Path, accompaniment_path: Path) -> None:
    """Copy the original file as the vocal stem and emit silence for background."""

    shutil.copyfile(source, vocal_path)
    _write_stub_wave(accompaniment_path)


def _write_stub_wave(destination: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a silent PCM wave file for dry runs."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "w") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)  # 16-bit samples
        wav_handle.setframerate(sample_rate)
        frame_count = int(duration_seconds * sample_rate)
        wav_handle.writeframes(b"\x00\x00" * frame_count)

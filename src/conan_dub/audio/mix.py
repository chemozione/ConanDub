"""Audio mixing and muxing helpers."""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def mix_tracks(
    tts_directory: Path,
    output_path: Path,
    background_path: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Combine synthesized speech with an optional backing track."""

    tts_directory = tts_directory.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run mix: creating stub wave at %s", output_path)
        _write_stub_wave(output_path)
        return output_path

    candidates = sorted(tts_directory.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No TTS wav files found in {tts_directory}")

    primary = candidates[0]
    logger.warning(
        "Mixing logic not yet implemented; copying %s to %s as placeholder.",
        primary.name,
        output_path,
    )
    output_path.write_bytes(primary.read_bytes())
    return output_path


def mux_with_video(
    video_path: Path, audio_path: Path, output_path: Path, ffmpeg_bin: str = "ffmpeg", dry_run: bool = False
) -> Path:
    """Combine the rendered audio with the original video using ffmpeg."""

    video_path = video_path.expanduser().resolve()
    audio_path = audio_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run mux: not invoking ffmpeg, touching %s", output_path)
        output_path.write_text("dry-run mux placeholder\n", encoding="utf-8")
        return output_path

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c",
        "copy",
        str(output_path),
    ]
    logger.info("Running ffmpeg mux: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg binary not found; install ffmpeg to enable muxing.") from exc
    return output_path


def _write_stub_wave(path: Path, duration_seconds: float = 1.0, sample_rate: int = 22050) -> None:
    """Emit a silent wave file used for dry-run tests."""

    with wave.open(str(path), "w") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        frame_count = int(duration_seconds * sample_rate)
        wav_handle.writeframes(b"\x00\x00" * frame_count)


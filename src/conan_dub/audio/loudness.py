"""Loudness measurement and normalization helpers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def measure_integrated_loudness(path: Path) -> float:
    """Return an approximate LUFS value for the provided WAV file.

    Falls back to a heuristic when pyloudnorm and soundfile are unavailable.
    """

    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
        import pyloudnorm as pyln  # type: ignore
    except ModuleNotFoundError:
        logger.warning(
            "pyloudnorm + soundfile not found; returning placeholder loudness value."
        )
        return -24.0

    data, rate = sf.read(path)
    meter = pyln.Meter(rate)
    return float(meter.integrated_loudness(data))


def normalize_loudness(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -16.0,
    dry_run: bool = False,
) -> Path:
    """Normalize loudness towards the requested LUFS target."""

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run loudness normalization for %s", input_path.name)
        output_path.write_text("dry-run loudness placeholder\n", encoding="utf-8")
        return output_path

    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError:
        logger.warning(
            "librosa/soundfile not available; copying audio without loudness changes."
        )
        if input_path != output_path:
            shutil.copyfile(input_path, output_path)
        return output_path

    data, rate = librosa.load(input_path, sr=None, mono=True)
    current_lufs = measure_integrated_loudness(input_path)
    delta = target_lufs - current_lufs
    gain = 10 ** (delta / 20.0)
    destination = output_path
    if input_path.resolve() == output_path.resolve():
        destination = output_path.with_suffix(".tmp_norm.wav")
    sf.write(destination, data * gain, rate)
    if destination != output_path:
        Path(destination).replace(output_path)
    logger.info(
        "Adjusted loudness for %s from %.2f LUFS to target %.2f LUFS",
        input_path.name,
        current_lufs,
        target_lufs,
    )
    return output_path

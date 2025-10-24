"""Deterministic placeholder file generators for dry-run operation."""

from __future__ import annotations

import json
import random
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Optional

PLACEHOLDER_NAMESPACE = "conan-dub"


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def make_silence_wav(
    path: Path,
    seconds: float = 1.0,
    sample_rate: int = 16000,
    rng: Optional[random.Random] = None,
) -> Path:
    """Emit a silent mono WAV file using only stdlib dependencies."""

    path = path.expanduser().resolve()
    ensure_dir(path.parent)
    frame_count = max(1, int(seconds * sample_rate))
    amplitude = 0
    if rng is not None:
        amplitude = rng.randint(-1024, 1024)
    frame_bytes = struct.pack("<h", amplitude)
    with wave.open(str(path), "w") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)  # 16-bit PCM
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(frame_bytes * frame_count)
    return path


def make_blank_json(
    path: Path,
    payload: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
) -> Path:
    """Persist a minimal JSON payload."""

    path = path.expanduser().resolve()
    ensure_dir(path.parent)
    data: Dict[str, Any] = dict(payload) if payload is not None else {}
    if rng is not None:
        data.setdefault(
            "_placeholder",
            {
                "source": PLACEHOLDER_NAMESPACE,
                "seed": f"{rng.getrandbits(32):08x}",
            },
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def make_text_placeholder(
    path: Path,
    message: str,
    rng: Optional[random.Random] = None,
) -> Path:
    """Create a plain-text sentinel file describing dry-run output."""

    path = path.expanduser().resolve()
    ensure_dir(path.parent)
    suffix = ""
    if rng is not None:
        suffix = f"\nseed={rng.getrandbits(32):08x}"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{message}\nsource={PLACEHOLDER_NAMESPACE}{suffix}\n")
    return path

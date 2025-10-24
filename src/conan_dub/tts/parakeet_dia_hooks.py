"""Parakeet/Dia multispeaker TTS integration hooks."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from .interfaces import SynthesisRequest, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)


class ParakeetDiaEngine(TTSEngine):
    """Stubbed Parakeet/Dia interface pending model integration."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        output_dir = request.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if request.dry_run:
            logger.info("Dry-run Parakeet/Dia synthesis for %d items", len(request.manifest.items))
            for item in request.manifest.items:
                self._write_stub_wave(output_dir / f"{item.id}.wav", item.duration)
            return SynthesisResult(output_dir=output_dir)

        logger.warning("Parakeet/Dia synthesis not implemented; generating dry-run outputs.")
        return self.synthesize(
            SynthesisRequest(
                manifest=request.manifest,
                output_dir=output_dir,
                checkpoint_dir=request.checkpoint_dir,
                dry_run=True,
                target_duration_tolerance=request.target_duration_tolerance,
            )
        )

    def _write_stub_wave(self, path: Path, duration: float) -> None:
        """Create a silent wave file with a duration hint."""

        with wave.open(str(path), "w") as wav_handle:
            wav_handle.setnchannels(1)
            wav_handle.setsampwidth(2)
            wav_handle.setframerate(self.sample_rate)
            frame_count = max(1, int(duration * self.sample_rate))
            wav_handle.writeframes(b"\x00\x00" * frame_count)


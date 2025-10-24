"""WhisperX alignment wrapper with CTM/VAD metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WhisperXConfig:
    """Runtime configuration for WhisperX."""

    model_name: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16


@dataclass(slots=True)
class AlignmentSegment:
    """Aligned transcript with CTM detail."""

    id: str
    start: float
    end: float
    text: str
    speaker: Optional[str]
    ctm: list[tuple[float, float, str]]


class WhisperXAligner:
    """Interface to WhisperX transcription and alignment."""

    def __init__(self, config: WhisperXConfig):
        self.config = config

    def transcribe(
        self,
        audio_path: Path,
        language: str = "ja",
        dry_run: bool = False,
    ) -> List[AlignmentSegment]:
        """Run transcription + alignment returning CTM-enhanced segments."""

        audio_path = audio_path.expanduser().resolve()
        if dry_run:
            logger.info("Dry-run WhisperX alignment for %s", audio_path.name)
            return [
                AlignmentSegment(
                    id="seg-000",
                    start=0.0,
                    end=1.5,
                    text="テスト",
                    speaker=None,
                    ctm=[(0.0, 0.5, "テ"), (0.5, 1.0, "スト")],
                )
            ]

        try:
            import whisperx  # type: ignore
        except ModuleNotFoundError:
            logger.warning(
                "whisperx is unavailable; returning placeholder alignment instead."
            )
            return self.transcribe(audio_path, language=language, dry_run=True)

        logger.info("Loading WhisperX model %s", self.config.model_name)
        model = whisperx.load_model(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        transcription = model.transcribe(str(audio_path), batch_size=self.config.batch_size, language=language)
        align_model, metadata = whisperx.load_align_model(language=language, device=self.config.device)
        result = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            str(audio_path),
            device=self.config.device,
        )
        segments: List[AlignmentSegment] = []
        for idx, segment in enumerate(result["segments"]):
            ctm_entries = [
                (float(token["start"]), float(token["end"]), token["text"])
                for token in segment.get("tokens", [])
                if token.get("start") is not None and token.get("end") is not None
            ]
            segments.append(
                AlignmentSegment(
                    id=f"seg-{idx:03d}",
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=str(segment["text"]).strip(),
                    speaker=segment.get("speaker"),
                    ctm=ctm_entries,
                )
            )
        return segments


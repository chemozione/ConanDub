"""pyannote.audio diarization adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiarizationConfig:
    """Configuration for pyannote diarization and speaker embeddings."""

    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/embedding"
    use_auth_token: bool = False
    device: Optional[str] = None


@dataclass(slots=True)
class DiarizationSegment:
    """Container describing a diarized speech segment."""

    start: float
    end: float
    speaker_id: str
    embedding: list[float]


class PyannoteDiarizer:
    """Wrapper around pyannote pipelines."""

    def __init__(self, config: DiarizationConfig):
        self.config = config

    def diarize(self, audio_path: Path, dry_run: bool = False) -> List[DiarizationSegment]:
        """Return diarization segments with embeddings."""

        audio_path = audio_path.expanduser().resolve()
        if dry_run:
            logger.info("Dry-run diarization for %s", audio_path.name)
            return [
                DiarizationSegment(
                    start=0.0,
                    end=1.5,
                    speaker_id="spk-test",
                    embedding=[0.0, 0.1, -0.1],
                )
            ]

        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ModuleNotFoundError:
            logger.warning(
                "pyannote.audio not installed; returning placeholder diarization result."
            )
            return self.diarize(audio_path, dry_run=True)

        diar_pipeline = Pipeline.from_pretrained(
            self.config.diarization_model,
        )
        if self.config.device:
            try:
                diar_pipeline.to(self.config.device)
            except AttributeError:
                logger.debug("pyannote Pipeline missing 'to' method for device assignment")
        diarization = diar_pipeline(str(audio_path))

        try:
            from pyannote.audio import Inference  # type: ignore
        except ModuleNotFoundError:
            logger.warning("pyannote.audio Inference unavailable; embeddings will be empty.")
            embeddings = None
        else:
            embeddings = Inference(self.config.embedding_model, device=self.config.device)

        segments: List[DiarizationSegment] = []
        for idx, turn in enumerate(diarization.itertracks()):
            segment, track, speaker = turn
            vector: list[float]
            if embeddings is not None:
                import numpy as np  # type: ignore

                clip = embeddings.crop(str(audio_path), segment, duration=segment.duration)
                vector = clip.tolist() if isinstance(clip, np.ndarray) else list(clip)
            else:
                vector = []

            segments.append(
                DiarizationSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_id=speaker if speaker else f"spk-{idx:03d}",
                    embedding=vector,
                )
            )
        return segments

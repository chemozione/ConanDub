"""Speech emotion recognition adapter built on Hugging Face pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..align.whisperx_runner import AlignmentSegment
from ..data.schema import EmotionTag

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmotionConfig:
    """Configuration for the SER pipeline."""

    model_id: str = "superb/hubert-large-superb-er"
    device: int | str = "cpu"


class EmotionRecognizer:
    """Run SER on aligned segments."""

    def __init__(self, config: EmotionConfig):
        self.config = config
        self._pipeline = None

    def _lazy_load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except ModuleNotFoundError:
            logger.warning("transformers not installed; falling back to neutral emotions.")
            self._pipeline = None
            return

        self._pipeline = pipeline(
            task="audio-classification",
            model=self.config.model_id,
            device=self.config.device,
        )

    def analyze(
        self,
        audio_path: Path,
        segments: Iterable[AlignmentSegment],
        dry_run: bool = False,
    ) -> List[EmotionTag]:
        """Return an emotion label per alignment segment."""

        audio_path = audio_path.expanduser().resolve()
        items = list(segments)
        if dry_run:
            logger.info("Dry-run SER for %s", audio_path.name)
            return [
                EmotionTag(segment_id=item.id, emotion="neutral", confidence=0.5)
                for item in items
            ]

        self._lazy_load()
        if self._pipeline is None:
            return self.analyze(audio_path, items, dry_run=True)

        tags: List[EmotionTag] = []
        for item in items:
            result = self._pipeline(
                str(audio_path),
                chunk_length_s=item.end - item.start,
                stride=(item.start, item.end),
            )
            top = max(result, key=lambda entry: entry["score"])
            tags.append(
                EmotionTag(
                    segment_id=item.id,
                    emotion=str(top["label"]),
                    confidence=float(top["score"]),
                )
            )
        return tags


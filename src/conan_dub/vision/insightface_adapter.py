"""Optional InsightFace detector adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .animeface import FaceDetection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InsightFaceConfig:
    """Configuration for InsightFace detection."""

    model: str = "insightface/antelopev2"
    det_thresh: float = 0.4


class InsightFaceAdapter:
    """Thin wrapper around the InsightFace detection pipeline."""

    def __init__(self, config: InsightFaceConfig):
        self.config = config
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ModuleNotFoundError:
            logger.warning(
                "insightface not installed; returning placeholder detections."
            )
            self._model = None
            return

        self._model = FaceAnalysis(name=self.config.model)
        self._model.prepare(ctx_id=0)

    def detect(self, frame_path: Path, dry_run: bool = False) -> List[FaceDetection]:
        """Detect faces leveraging InsightFace when available."""

        frame_path = frame_path.expanduser().resolve()
        if dry_run:
            logger.info("Dry-run InsightFace detection for %s", frame_path.name)
            return [
                FaceDetection(
                    frame_path=frame_path,
                    bbox=(8, 8, 248, 248),
                    confidence=0.6,
                )
            ]

        self._lazy_load()
        if self._model is None:
            return self.detect(frame_path, dry_run=True)

        faces = self._model.get(frame_path)
        detections: List[FaceDetection] = []
        for face in faces:
            bbox = tuple(int(x) for x in face.bbox)
            detections.append(
                FaceDetection(frame_path=frame_path, bbox=bbox, confidence=float(face.det_score))
            )
        return detections


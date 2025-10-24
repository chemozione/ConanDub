"""AnimeFace detector adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceDetection:
    """Simple face detection result."""

    frame_path: Path
    bbox: tuple[int, int, int, int]
    confidence: float


class AnimeFaceDetector:
    """Wrapper around the AnimeFace ONNX runtime."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self._runtime = None

    def _lazy_load(self) -> None:
        if self._runtime is not None:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError:
            logger.warning("onnxruntime missing; AnimeFace detections will be placeholders.")
            self._runtime = None
            return

        if self.model_path is None:
            raise RuntimeError("AnimeFace model path must be provided when onnxruntime is available.")
        self._runtime = ort.InferenceSession(str(self.model_path))

    def detect(self, frame_path: Path, dry_run: bool = False) -> List[FaceDetection]:
        """Detect faces for a single frame."""

        frame_path = frame_path.expanduser().resolve()
        if dry_run or self._runtime is None:
            logger.info("Dry-run AnimeFace detection for %s", frame_path.name)
            return [
                FaceDetection(
                    frame_path=frame_path,
                    bbox=(0, 0, 256, 256),
                    confidence=0.5,
                )
            ]

        self._lazy_load()
        # Detailed model invocation is left to future implementation.
        logger.warning("AnimeFace runtime present but invocation not implemented yet.")
        return self.detect(frame_path, dry_run=True)


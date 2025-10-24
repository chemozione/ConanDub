"""Lip activity estimation via SyncNet or optical flow heuristics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MethodName = Literal["syncnet", "optical_flow"]


@dataclass(slots=True)
class LipActivity:
    """Represents the activity score for a given frame or segment."""

    frame_path: Path
    score: float


class LipActivityEstimator:
    """Modular estimator supporting SyncNet and optical-flow heuristics."""

    def __init__(self, method: MethodName = "syncnet"):
        if method not in {"syncnet", "optical_flow"}:
            raise ValueError(f"Unsupported lip activity method: {method}")
        self.method = method

    def estimate(self, frame_path: Path, dry_run: bool = False) -> LipActivity:
        """Return a lip-activity score between 0 and 1."""

        frame_path = frame_path.expanduser().resolve()
        if dry_run:
            logger.info("Dry-run lip activity inference for %s", frame_path.name)
            return LipActivity(frame_path=frame_path, score=0.5)

        if self.method == "syncnet":
            logger.warning("SyncNet inference not yet implemented; returning placeholder score.")
            return LipActivity(frame_path=frame_path, score=0.6)
        if self.method == "optical_flow":
            logger.warning(
                "Optical-flow heuristic not yet implemented; returning placeholder score."
            )
            return LipActivity(frame_path=frame_path, score=0.4)
        raise AssertionError("Unreachable branch.")


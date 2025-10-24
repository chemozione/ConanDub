"""Interfaces for multispeaker TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..data.schema import DubbingManifest


@dataclass(slots=True)
class SynthesisRequest:
    """Information required to run synthesis."""

    manifest: DubbingManifest
    output_dir: Path
    checkpoint_dir: Path | None = None
    dry_run: bool = False
    target_duration_tolerance: float = 0.15


@dataclass(slots=True)
class SynthesisResult:
    """Response from the TTS engine."""

    output_dir: Path


class TTSEngine(ABC):
    """Abstract base class for TTS implementations."""

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Run synthesis for the given manifest."""


def load_engine(name: str, **kwargs) -> TTSEngine:
    """Instantiate a configured TTS engine."""

    if name == "parakeet_dia":
        from .parakeet_dia_hooks import ParakeetDiaEngine

        return ParakeetDiaEngine(**kwargs)

    raise ValueError(f"Unsupported TTS engine: {name}")


"""Data models describing the Detective Conan dubbing manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

MANIFEST_FIELDS: Sequence[str] = (
    "id",
    "start",
    "end",
    "speaker_id",
    "character",
    "emotion",
    "text_ja",
    "text_it",
    "audio_path",
    "speaker_embedding",
)


class Utterance(BaseModel):
    """Single aligned utterance ready for dubbing."""

    id: str
    start: float
    end: float
    speaker_id: str
    character: str
    emotion: str
    text_ja: str
    text_it: str | None = None
    audio_path: Path
    speaker_embedding: List[float] = Field(default_factory=list)

    @field_validator("audio_path", mode="before")
    @classmethod
    def _coerce_audio_path(cls, value: str | Path) -> Path:
        return Path(value)

    @model_validator(mode="after")
    def _check_timing(self) -> "Utterance":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self

    @property
    def duration(self) -> float:
        """Return utterance duration in seconds."""
        return self.end - self.start


class DubbingManifest(BaseModel):
    """Collection of utterances representing a dubbing manifest."""

    items: List[Utterance]

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.items)

    def characters(self) -> List[str]:
        """Return the unique set of characters referenced in the manifest."""
        return sorted({item.character for item in self.items})

    def speakers(self) -> List[str]:
        """Return the unique set of speaker identifiers."""
        return sorted({item.speaker_id for item in self.items})

    def total_duration(self) -> float:
        """Compute the sum of segment durations."""
        return sum(item.duration for item in self.items)


def as_records(manifest: DubbingManifest) -> Iterable[dict[str, object]]:
    """Convert a manifest into a flat iterable of dictionaries."""

    for item in manifest:
        yield {
            "id": item.id,
            "start": item.start,
            "end": item.end,
            "speaker_id": item.speaker_id,
            "character": item.character,
            "emotion": item.emotion,
            "text_ja": item.text_ja,
            "text_it": item.text_it,
            "audio_path": str(item.audio_path),
            "speaker_embedding": item.speaker_embedding,
        }


def to_arrow_table(manifest: DubbingManifest):
    """Return an Arrow table when pyarrow is available."""

    try:
        import pyarrow as pa  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pyarrow is required to produce Arrow tables; install pyarrow first."
        ) from exc

    return pa.Table.from_pylist(list(as_records(manifest)))


def to_parquet(manifest: DubbingManifest, destination: Path, overwrite: bool = False):
    """Serialize the manifest to a Parquet file."""

    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")

    table = to_arrow_table(manifest)
    table.write_parquet(destination)


@dataclass(slots=True)
class EmotionTag:
    """Simple container for SER output prior to manifest enrichment."""

    segment_id: str
    emotion: str
    confidence: float

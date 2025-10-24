"""Fusion logic for combining diarization, vision, and lip activity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..align.diarize import DiarizationSegment
from ..align.whisperx_runner import AlignmentSegment
from ..data.schema import DubbingManifest, Utterance

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FusionConfig:
    """Configure modality weights for character assignment."""

    lip_weight: float = 0.5
    vision_weight: float = 0.3
    diar_weight: float = 0.2
    min_confidence: float = 0.4


def fuse_segments(
    alignment: Iterable[AlignmentSegment],
    diar: Iterable[DiarizationSegment],
    face_votes: Dict[str, str] | None = None,
    lip_scores: Dict[str, float] | None = None,
    translations: Dict[str, str] | None = None,
    config: FusionConfig | None = None,
) -> DubbingManifest:
    """Produce a dubbing manifest by fusing multimodal cues."""

    config = config or FusionConfig()
    diar_segments = list(diar)
    face_votes = face_votes or {}
    lip_scores = lip_scores or {}
    translations = translations or {}
    entries: List[Utterance] = []

    for segment in alignment:
        speaker_id, diar_conf = _match_speaker(segment, diar_segments)
        lip_conf = lip_scores.get(segment.id, 0.0)
        vision_character = face_votes.get(segment.id)
        confidence = diar_conf * config.diar_weight + lip_conf * config.lip_weight
        character = vision_character or speaker_id
        if confidence < config.min_confidence:
            logger.info(
                "Segment %s below confidence threshold %.2f < %.2f",
                segment.id,
                confidence,
                config.min_confidence,
            )
        entries.append(
            Utterance(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker_id=speaker_id,
                character=character,
                emotion="neutral",
                text_ja=segment.text,
                text_it=translations.get(segment.id),
                audio_path=f"tts/{segment.id}.wav",
                speaker_embedding=[],
            )
        )
    return DubbingManifest(items=entries)


def _match_speaker(
    segment: AlignmentSegment,
    diar_segments: List[DiarizationSegment],
) -> tuple[str, float]:
    """Return the best matching speaker using intersection-over-union."""

    best_score = 0.0
    best_speaker = segment.speaker or "unknown"
    for diag in diar_segments:
        overlap = _temporal_overlap(segment.start, segment.end, diag.start, diag.end)
        if overlap > best_score:
            best_score = overlap
            best_speaker = diag.speaker_id
    return best_speaker, best_score


def _temporal_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap ratio using IoU style metric."""

    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0.0:
        return 0.0
    return intersection / union

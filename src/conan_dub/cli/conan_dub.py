"""Typer-based CLI for the Detective Conan dubbing pipeline."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

from .. import __version__
from ..align.diarize import DiarizationConfig, DiarizationSegment, PyannoteDiarizer
from ..align.whisperx_runner import AlignmentSegment, WhisperXAligner, WhisperXConfig
from ..audio.loudness import normalize_loudness
from ..audio.mix import mix_tracks, mux_with_video
from ..audio.separate import SeparationRequest, run_separation
from ..charlink.embed_cluster import EmbeddingConfig, cluster_embeddings, embed_faces
from ..charlink.fuse import FusionConfig, fuse_segments
from ..charlink.seed_dataset import SeedConfig, collect_seed_crops
from ..data.export import load_manifest, save_manifest_json
from ..data.schema import DubbingManifest, EmotionTag, Utterance
from ..data.validate import validate_manifest_file, validate_manifest_payload
from ..config import load_config
from ..ser.emotion import EmotionConfig, EmotionRecognizer
from ..translate.nllb import NLLBTranslator, TranslationConfig
from ..tts.interfaces import SynthesisRequest, load_engine
from ..vision.animeface import AnimeFaceDetector
from ..vision.frames import extract_frames
from ..vision.insightface_adapter import InsightFaceAdapter, InsightFaceConfig
from ..utils import (
    check_env,
    configure_logging,
    ensure_dir,
    ffmpeg_available,
    make_blank_json,
    make_silence_wav,
    make_text_placeholder,
    platform_summary,
    python_version,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, rich_markup_mode="markdown")

DEFAULT_SEED = 1337
VALID_DEVICE_CHOICES = {"auto", "cpu", "cuda"}


def _version_callback(value: bool) -> bool:
    """Print version information when requested."""

    if value:
        typer.echo(__version__)
        raise typer.Exit()
    return value


def _resolve_device_option(choice: str) -> Tuple[str, Optional[str]]:
    normalized = choice.lower()
    if normalized not in VALID_DEVICE_CHOICES:
        raise typer.BadParameter(
            f"Invalid device '{choice}'. Choose from {sorted(VALID_DEVICE_CHOICES)}."
        )
    if normalized == "cpu":
        return "cpu", None

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        torch_available = False
    else:
        torch_available = bool(torch.cuda.is_available())

    if normalized == "auto":
        return ("cuda" if torch_available else "cpu"), None

    # normalized == "cuda"
    if torch_available:
        return "cuda", None
    return "cpu", "CUDA requested but not available; falling back to CPU."


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a YAML configuration file (defaults to configs/default.yaml).",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase log verbosity, repeatable."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce log output to warnings."),
    version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        callback=_version_callback,
        help="Show package version and exit.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        min=0,
        help="Seed controlling deterministic placeholder generation.",
        show_default=True,
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Preferred compute device for GPU-enabled stages (auto/cpu/cuda).",
    ),
) -> None:
    """Configure logging and shared context."""

    del version  # handled via _version_callback
    verbosity = -1 if quiet else min(verbose, 2)
    configure_logging(verbosity)
    cfg, cfg_path = load_config(config)
    resolved_device, device_warning = _resolve_device_option(device)
    if device_warning:
        typer.secho(device_warning, fg="yellow")
    ctx.obj = {
        "config": cfg,
        "config_path": cfg_path,
        "verbosity": verbosity,
        "seed": seed,
        "rng": random.Random(seed),
        "device_requested": device.lower(),
        "device": resolved_device,
    }


def _preflight(paths: Iterable[Path], dry_run: bool) -> None:
    for message in check_env(paths, dry_run=dry_run):
        typer.secho(f"[preflight] {message}", fg="yellow")


def _resolve_dry_run(flag: Optional[bool], config: Dict[str, Any]) -> bool:
    if flag is not None:
        return flag
    dry_cfg = config.get("dry_run", {})
    return bool(dry_cfg.get("enabled", False))


def _ctx_config(ctx: typer.Context) -> Dict[str, Any]:
    return ctx.obj.get("config", {}) if ctx.obj else {}


def _ctx_config_path(ctx: typer.Context) -> Optional[Path]:
    return ctx.obj.get("config_path") if ctx.obj else None


def _ctx_seed(ctx: typer.Context) -> int:
    return ctx.obj.get("seed", DEFAULT_SEED) if ctx.obj else DEFAULT_SEED


def _ctx_rng(ctx: typer.Context) -> random.Random:
    if ctx.obj is None:
        ctx.obj = {}
    rng = ctx.obj.get("rng")
    if rng is None:
        rng = random.Random(_ctx_seed(ctx))
        ctx.obj["rng"] = rng
    return rng


def _ctx_device(ctx: typer.Context) -> str:
    return ctx.obj.get("device", "cpu") if ctx.obj else "cpu"


def _ctx_requested_device(ctx: typer.Context) -> str:
    return ctx.obj.get("device_requested", "auto") if ctx.obj else "auto"


def _hf_device_arg(device: str) -> int | str:
    return 0 if device == "cuda" else "cpu"


@app.command()
def diagnostics(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Assess environment using dry-run assumptions.",
    ),
) -> None:
    """Print environment and configuration diagnostics."""

    cfg = _ctx_config(ctx)
    cfg_path = _ctx_config_path(ctx)
    seed = _ctx_seed(ctx)
    device_requested = _ctx_requested_device(ctx)
    device_effective = _ctx_device(ctx)
    sample_manifest = {
        "items": [
            {
                "id": "diagnostics-seg",
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "spk-diagnostics",
                "character": "placeholder",
                "emotion": "neutral",
                "text_ja": "テスト",
                "text_it": None,
                "audio_path": "tts/diagnostics.wav",
                "speaker_embedding": [],
            }
        ]
    }
    schema_ok = validate_manifest_payload(sample_manifest).ok
    dry_run_default = _resolve_dry_run(None, cfg)
    info = {
        "version": __version__,
        "python": python_version(),
        "platform": platform_summary(),
        "config": str(cfg_path) if cfg_path else "auto",
        "ffmpeg": "found" if ffmpeg_available() else "missing",
        "schema": "ok" if schema_ok else "missing",
        "dry_run_supported": "yes",
        "dry_run_requested": "yes" if dry_run else "no",
        "dry_run_default": "yes" if dry_run_default else "no",
        "device_requested": device_requested,
        "device_effective": device_effective,
        "seed": seed,
    }
    for key, value in info.items():
        typer.echo(f"{key}={value}")


def _cfg_get(cfg: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aligned_from_json(path: Path) -> List[AlignmentSegment]:
    payload = _read_json(path)
    return [
        AlignmentSegment(
            id=item["id"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            speaker=item.get("speaker"),
            ctm=item.get("ctm", []),
        )
        for item in payload.get("segments", [])
    ]


def _diar_from_json(path: Path) -> List[DiarizationSegment]:
    payload = _read_json(path)
    return [
        DiarizationSegment(
            start=item["start"],
            end=item["end"],
            speaker_id=item["speaker_id"],
            embedding=item.get("embedding", []),
        )
        for item in payload.get("segments", [])
    ]


def _ser_map(path: Path) -> Dict[str, EmotionTag]:
    payload = _read_json(path)
    result: Dict[str, EmotionTag] = {}
    for entry in payload.get("tags", []):
        tag = EmotionTag(
            segment_id=entry["segment_id"],
            emotion=entry["emotion"],
            confidence=entry["confidence"],
        )
        result[tag.segment_id] = tag
    return result


def _face_votes(path: Path) -> Dict[str, str]:
    payload = _read_json(path)
    votes: Dict[str, str] = {}
    for idx, entry in enumerate(payload.get("detections", [])):
        votes.setdefault(entry.get("segment_id", f"seg-{idx:03d}"), entry.get("character", "unknown"))
    return votes


def _assign_speaker(segment: AlignmentSegment, diar_segments: Iterable[DiarizationSegment]) -> Tuple[str, float]:
    best_score = 0.0
    best_speaker = segment.speaker or "unknown"
    for diag in diar_segments:
        score = _temporal_overlap(segment.start, segment.end, diag.start, diag.end)
        if score > best_score:
            best_score = score
            best_speaker = diag.speaker_id
    return best_speaker, best_score


def _temporal_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _speaker_embedding(speaker_id: str, diar_segments: Iterable[DiarizationSegment]) -> List[float]:
    for diag in diar_segments:
        if diag.speaker_id == speaker_id and diag.embedding:
            return diag.embedding
    return []


@app.command()
def split(
    ctx: typer.Context,
    input: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    separator: str = typer.Option("vocal", help="Separator backend: vocal|demucs|uvr5"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Separate voice/background stems."""

    cfg = _ctx_config(ctx)
    device = _ctx_device(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not input.exists():
        if use_dry:
            make_text_placeholder(
                input,
                "conan-dub dry-run placeholder video container",
                rng=rng,
            )
        else:
            raise typer.BadParameter(f"Input file not found: {input}", param_hint="input")
    _preflight([out], use_dry)
    request = SeparationRequest(
        input_path=input,
        output_dir=out,
        separator=separator,  # type: ignore[arg-type]
        dry_run=use_dry,
    )
    result = run_separation(request)
    typer.echo(f"Voice stem: {result.vocal_path}")
    typer.echo(f"Background stem: {result.accompaniment_path}")


@app.command()
def diarize(
    ctx: typer.Context,
    voice: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Run pyannote diarization and store segments.json."""

    cfg = _ctx_config(ctx)
    device = _ctx_device(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not voice.exists():
        if use_dry:
            make_silence_wav(voice, rng=rng)
        else:
            raise typer.BadParameter(f"Voice file not found: {voice}", param_hint="voice")
    _preflight([out], use_dry)
    diarizer = PyannoteDiarizer(
        DiarizationConfig(
            diarization_model=_cfg_get(cfg, ("paths", "pyannote_diarization"), "pyannote/speaker-diarization-3.1"),
            embedding_model=_cfg_get(cfg, ("paths", "pyannote_embedding"), "pyannote/embedding"),
            device=None if device == "cpu" else device,
        )
    )
    result = diarizer.diarize(voice, dry_run=use_dry)
    payload = {
        "segments": [
            {
                "start": item.start,
                "end": item.end,
                "speaker_id": item.speaker_id,
                "embedding": item.embedding,
            }
            for item in result
        ]
    }
    output_path = out / "segments.json"
    _write_json(output_path, payload)
    typer.echo(f"Wrote diarization segments to {output_path}")


@app.command()
def align(
    ctx: typer.Context,
    voice: Path = typer.Argument(..., resolve_path=True),
    segments: Path = typer.Option(..., resolve_path=True, help="Diarization segments.json path"),
    out: Path = typer.Option(..., writable=True, resolve_path=True, help="Aligned transcript output path"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Run WhisperX alignment with CTM output."""

    cfg = _ctx_config(ctx)
    device = _ctx_device(ctx)
    hf_device = _hf_device_arg(device)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not voice.exists():
        if use_dry:
            make_silence_wav(voice, rng=rng)
        else:
            raise typer.BadParameter(f"Voice file not found: {voice}", param_hint="voice")
    if not segments.exists():
        if use_dry:
            make_blank_json(segments, {"segments": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Diarization segments missing: {segments}", param_hint="segments")
    _preflight([out], use_dry)
    aligner = WhisperXAligner(
        WhisperXConfig(
            model_name=_cfg_get(cfg, ("paths", "whisperx_model"), "medium"),
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
        )
    )
    diar_segments = _diar_from_json(segments)
    aligned_segments = aligner.transcribe(
        voice,
        language="ja",
        dry_run=use_dry,
    )
    for segment in aligned_segments:
        speaker_id, score = _assign_speaker(segment, diar_segments)
        if segment.speaker is None:
            segment.speaker = speaker_id
        logger.debug("Assigned speaker %s to %s with score %.2f", segment.speaker, segment.id, score)
    payload = {
        "segments": [
            {
                "id": item.id,
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "speaker": item.speaker,
                "ctm": item.ctm,
            }
            for item in aligned_segments
        ]
    }
    _write_json(out, payload)
    typer.echo(f"Wrote aligned segments to {out}")


@app.command()
def frames(
    ctx: typer.Context,
    input: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    fps: float = typer.Option(2.0, min=0.25, help="Frames per second for extraction"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Extract frames for face detection and lip-sync analysis."""

    cfg = _ctx_config(ctx)
    device = _ctx_device(ctx)
    hf_device = _hf_device_arg(device)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not input.exists():
        if use_dry:
            make_text_placeholder(
                input,
                "conan-dub dry-run placeholder video container",
                rng=rng,
            )
        else:
            raise typer.BadParameter(f"Input file not found: {input}", param_hint="input")
    _preflight([out], use_dry)
    extract_frames(
        video_path=input,
        output_dir=out,
        fps=fps,
        dry_run=use_dry,
    )
    typer.echo(f"Frames written under {out}")


@app.command("detect-faces")
def detect_faces(
    ctx: typer.Context,
    frames_dir: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    detector: str = typer.Option("animeface", help="animeface|insightface"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Detect faces for extracted frames."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not frames_dir.exists():
        if use_dry:
            ensure_dir(frames_dir)
            make_text_placeholder(
                frames_dir / "frame_0001.txt",
                "conan-dub dry-run frame placeholder",
                rng=rng,
            )
        else:
            raise typer.BadParameter(f"Frames directory not found: {frames_dir}", param_hint="frames-dir")
    _preflight([out], use_dry)
    frames = sorted(frames_dir.glob("*"))
    detections: List[Dict[str, Any]] = []
    if detector == "animeface":
        detector_impl = AnimeFaceDetector()
        for frame in frames:
            for item in detector_impl.detect(frame, dry_run=use_dry):
                detections.append(
                    {"frame": str(item.frame_path), "bbox": list(item.bbox), "confidence": item.confidence}
                )
    elif detector == "insightface":
        detector_impl = InsightFaceAdapter(
            InsightFaceConfig(model=_cfg_get(cfg, ("paths", "insightface_repo"), "insightface/antelopev2"))
        )
        for frame in frames:
            for item in detector_impl.detect(frame, dry_run=use_dry):
                detections.append(
                    {"frame": str(item.frame_path), "bbox": list(item.bbox), "confidence": item.confidence}
                )
    else:
        raise typer.BadParameter(f"Unsupported detector {detector}")
    _write_json(out, {"detections": detections})
    typer.echo(f"Detected faces written to {out}")


@app.command("char-seed")
def char_seed(
    ctx: typer.Context,
    frames_dir: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    character: str = typer.Option("unknown", help="Character label for the seed crops"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Create a seed dataset of character crops."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not frames_dir.exists():
        if use_dry:
            ensure_dir(frames_dir)
            make_text_placeholder(
                frames_dir / "frame_0001.txt",
                "conan-dub dry-run frame placeholder",
                rng=rng,
            )
        else:
            raise typer.BadParameter(f"Frames directory not found: {frames_dir}", param_hint="frames-dir")
    _preflight([out], use_dry)
    frame_paths = sorted(frames_dir.glob("*"))
    seed_config = SeedConfig(frames_dir=frames_dir, output_dir=out, limit_per_character=20)
    stored = collect_seed_crops(
        character=character,
        config=seed_config,
        crop_paths=frame_paths,
        dry_run=use_dry,
    )
    typer.echo(f"Stored {len(stored)} seed crops under {out}")


@app.command("char-augment")
def char_augment(
    ctx: typer.Context,
    chars: Path = typer.Option(..., resolve_path=True),
    faces: Path = typer.Option(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Augment character dataset using embeddings and clustering."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    _preflight([out], use_dry)
    if not chars.exists():
        if use_dry:
            ensure_dir(chars)
        else:
            raise typer.BadParameter(f"Character directory not found: {chars}", param_hint="chars")
    if not faces.exists():
        if use_dry:
            make_blank_json(faces, {"detections": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Faces JSON not found: {faces}", param_hint="faces")
    crops = [path for path in chars.glob("**/*") if path.is_file()]
    embeddings = embed_faces(crops, EmbeddingConfig(), dry_run=use_dry)
    clusters = cluster_embeddings(embeddings)
    report_path = out / "clusters.json"
    payload = {str(label): [str(path) for path in members] for label, members in clusters.items()}
    _write_json(report_path, payload)
    faces_payload = _read_json(faces)
    _write_json(out / "faces.json", faces_payload)
    typer.echo(f"Cluster report written to {report_path}")


@app.command()
def ser(
    ctx: typer.Context,
    voice: Path = typer.Argument(..., resolve_path=True),
    aligned: Path = typer.Option(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Run speech emotion recognition per aligned segment."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not voice.exists():
        if use_dry:
            make_silence_wav(voice, rng=rng)
        else:
            raise typer.BadParameter(f"Voice file not found: {voice}", param_hint="voice")
    if not aligned.exists():
        if use_dry:
            make_blank_json(aligned, {"segments": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Aligned JSON not found: {aligned}", param_hint="aligned")
    _preflight([out], use_dry)
    segments = _aligned_from_json(aligned)
    recognizer = EmotionRecognizer(
        EmotionConfig(
            model_id=_cfg_get(cfg, ("paths", "ser_model"), "superb/hubert-large-superb-er"),
            device=hf_device,
        )
    )
    tags: List[EmotionTag] = recognizer.analyze(
        audio_path=voice,
        segments=segments,
        dry_run=use_dry,
    )
    payload = {
        "tags": [
            {"segment_id": tag.segment_id, "emotion": tag.emotion, "confidence": tag.confidence}
            for tag in tags
        ]
    }
    _write_json(out, payload)
    typer.echo(f"Emotion tags written to {out}")


@app.command()
def fuse(
    ctx: typer.Context,
    aligned: Path = typer.Option(..., resolve_path=True),
    diar: Path = typer.Option(..., resolve_path=True),
    ser_path: Path = typer.Option(..., resolve_path=True),
    faces: Path = typer.Option(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Fuse multimodal cues into a dubbing manifest."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not aligned.exists():
        if use_dry:
            make_blank_json(aligned, {"segments": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Required file not found: {aligned}", param_hint="aligned")
    if not diar.exists():
        if use_dry:
            make_blank_json(diar, {"segments": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Required file not found: {diar}", param_hint="diar")
    if not ser_path.exists():
        if use_dry:
            make_blank_json(ser_path, {"tags": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Required file not found: {ser_path}", param_hint="ser-path")
    if not faces.exists():
        if use_dry:
            make_blank_json(faces, {"detections": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Required file not found: {faces}", param_hint="faces")
    _preflight([out], use_dry)
    aligned_segments = _aligned_from_json(aligned)
    diar_segments = _diar_from_json(diar)
    ser_tags = _ser_map(ser_path)
    face_votes = _face_votes(faces)
    fusion_cfg = FusionConfig(
        lip_weight=_cfg_get(cfg, ("fusion", "lip_weight"), 0.5),
        vision_weight=_cfg_get(cfg, ("fusion", "vision_weight"), 0.3),
        diar_weight=_cfg_get(cfg, ("fusion", "diar_weight"), 0.2),
        min_confidence=_cfg_get(cfg, ("fusion", "min_confidence"), 0.4),
    )
    manifest = fuse_segments(
        alignment=aligned_segments,
        diar=diar_segments,
        face_votes=face_votes,
        lip_scores=None,
        translations=None,
        config=fusion_cfg,
    )

    enriched_items: List[Utterance] = []
    for item in manifest.items:
        updates: Dict[str, Any] = {}
        if item.id in ser_tags:
            updates["emotion"] = ser_tags[item.id].emotion
        embedding = _speaker_embedding(item.speaker_id, diar_segments)
        if embedding:
            updates["speaker_embedding"] = embedding
        if updates:
            enriched_items.append(item.model_copy(update=updates))
        else:
            enriched_items.append(item)
    manifest = DubbingManifest(items=enriched_items)
    save_manifest_json(manifest, out)
    validation = validate_manifest_file(out)
    if not validation.ok and not use_dry:
        raise typer.Exit(code=1)
    typer.echo(f"Manifest written to {out}")


@app.command()
def translate(
    ctx: typer.Context,
    manifest: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Translate Japanese text to Italian."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not manifest.exists():
        if use_dry:
            make_blank_json(manifest, {"items": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Manifest not found: {manifest}", param_hint="manifest")
    _preflight([out], use_dry)
    data = load_manifest(manifest)
    translator = NLLBTranslator(
        TranslationConfig(
            model_id=_cfg_get(cfg, ("paths", "nllb_model"), "facebook/nllb-200-distilled-600M"),
            device=hf_device,
            max_length=_cfg_get(cfg, ("translation", "max_length"), 512),
        )
    )
    outputs = translator.translate_texts(
        [item.text_ja for item in data.items],
        dry_run=use_dry,
    )
    updated_items = [
        item.model_copy(update={"text_it": translation}) for item, translation in zip(data.items, outputs, strict=False)
    ]
    result = DubbingManifest(items=updated_items)
    save_manifest_json(result, out)
    validation = validate_manifest_file(out)
    if not validation.ok and not use_dry:
        raise typer.Exit(code=1)
    typer.echo(f"Translated manifest written to {out}")


@app.command()
def synth(
    ctx: typer.Context,
    manifest: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., dir_okay=True, writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Synthesize speech using the configured TTS engine."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not manifest.exists():
        if use_dry:
            make_blank_json(manifest, {"items": []}, rng=rng)
        else:
            raise typer.BadParameter(f"Manifest not found: {manifest}", param_hint="manifest")
    _preflight([out], use_dry)
    manifest_data = load_manifest(manifest)
    engine_name = _cfg_get(cfg, ("tts", "engine"), "parakeet_dia")
    engine = load_engine(engine_name, sample_rate=_cfg_get(cfg, ("tts", "sample_rate"), 22050))
    request = SynthesisRequest(
        manifest=manifest_data,
        output_dir=out,
        checkpoint_dir=Path(_cfg_get(cfg, ("paths", "parakeet_dia_checkpoint"), "checkpoints/parakeet_dia")),
        dry_run=use_dry,
        target_duration_tolerance=_cfg_get(cfg, ("tts", "target_duration_tolerance"), 0.15),
    )
    result = engine.synthesize(request)
    typer.echo(f"TTS outputs available under {result.output_dir}")


@app.command()
def mix(
    ctx: typer.Context,
    tts: Path = typer.Argument(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    bg: Optional[Path] = typer.Option(None, resolve_path=True, help="Optional background"),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Mix synthesized speech with optional background audio."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not tts.exists():
        if use_dry:
            ensure_dir(tts)
            make_silence_wav(tts / "seg-000.wav", rng=rng)
        else:
            raise typer.BadParameter(f"TTS directory not found: {tts}", param_hint="tts")
    if bg is not None and not bg.exists():
        if use_dry:
            make_silence_wav(bg, rng=rng)
        else:
            raise typer.BadParameter(f"Background audio not found: {bg}", param_hint="bg")
    _preflight([out], use_dry)
    mixed = mix_tracks(
        tts_directory=tts,
        output_path=out,
        background_path=bg,
        dry_run=use_dry,
    )
    if not use_dry:
        normalize_loudness(
            input_path=mixed,
            output_path=mixed,
            target_lufs=_cfg_get(cfg, ("mixing", "loudness_lufs"), -16.0),
            dry_run=False,
        )
    typer.echo(f"Mixed audio written to {mixed}")


@app.command()
def mux(
    ctx: typer.Context,
    video: Path = typer.Argument(..., resolve_path=True),
    audio: Path = typer.Option(..., resolve_path=True),
    out: Path = typer.Option(..., writable=True, resolve_path=True),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run/--no-dry-run", help="Force dry-run behaviour"),
):
    """Mux the final audio with the source video using ffmpeg."""

    cfg = _ctx_config(ctx)
    rng = _ctx_rng(ctx)
    use_dry = _resolve_dry_run(dry_run, cfg)
    if not video.exists():
        if use_dry:
            make_text_placeholder(
                video,
                "conan-dub dry-run placeholder video container",
                rng=rng,
            )
        else:
            raise typer.BadParameter(f"Video not found: {video}", param_hint="video")
    if not audio.exists():
        if use_dry:
            make_silence_wav(audio, rng=rng)
        else:
            raise typer.BadParameter(f"Audio not found: {audio}", param_hint="audio")
    _preflight([out], use_dry)
    mux_with_video(
        video_path=video,
        audio_path=audio,
        output_path=out,
        dry_run=use_dry,
        ffmpeg_bin=_cfg_get(cfg, ("mixing", "mux_tool"), "ffmpeg"),
    )
    typer.echo(f"Muxed video written to {out}")


def main() -> None:
    """Entry-point for console script."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

---
jupyter:
  jupytext:
    formats: md:myst
    text_representation:
      extension: .md
      format_name: myst
      format_version: 0.13
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Conan Dub – Colab GPU Test Harness

This notebook drives an end-to-end test run on Google Colab with optional GPU acceleration. It assumes you have already executed the shell setup commands below in a Colab cell:

```bash
bash colab/setup_colab.sh
bash scripts/colab_pull_inputs.sh || true
```

The helper scripts install GPU requirements, ensure `ffmpeg` is present, and pull any previously uploaded media from `MyDrive/conan_inputs/`.

## 1. Mount Google Drive

```{code-cell} python
from google.colab import drive

drive.mount("/content/drive")
```

## 2. Run Configuration

Define a run identifier, base directories, and utility helpers for timing, logging, and reporting.

```{code-cell} python
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from conan_dub.utils.report import append_warning, record_stage, start_run, write_report

SEED = 1337
DEVICE = "cuda"  # change to "cpu" to disable GPU usage
DRY_RUN = False   # toggle to True for lightweight verification

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_seed{SEED}"
RUN_ROOT = Path("/content/drive/MyDrive/conan_dub_runs") / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MEDIA_CANDIDATES = [Path("/content/test.mp4"), Path("/content/test.mkv")]
MEDIA_PATH = next((p for p in MEDIA_CANDIDATES if p.exists()), None)
if MEDIA_PATH is None:
    raise FileNotFoundError("Upload test.mp4 or test.mkv to /content or Drive (scripts/colab_pull_inputs.sh)")

report = start_run(RUN_ROOT, SEED, RUN_ID)
print(f"Run ID: {RUN_ID}")
print(f"Output root: {RUN_ROOT}")
print(f"Media file: {MEDIA_PATH}")
print(f"Device: {DEVICE}")
print(f"Dry run: {DRY_RUN}")

# Common directories
AUDIO_DIR = RUN_ROOT / "audio"
DIAR_DIR = RUN_ROOT / "diar"
FRAMES_DIR = RUN_ROOT / "frames"
CHAR_DIR = RUN_ROOT / "characters"
CHAR_AUG_DIR = RUN_ROOT / "char_aug"
TTS_DIR = RUN_ROOT / "tts_wavs"

ALIGNED_PATH = RUN_ROOT / "aligned.json"
FACES_PATH = RUN_ROOT / "faces.json"
SER_PATH = RUN_ROOT / "emotion.json"
MANIFEST_PATH = RUN_ROOT / "manifest.json"
MANIFEST_IT_PATH = RUN_ROOT / "manifest_it.json"
FINAL_AUDIO = RUN_ROOT / "final_audio.wav"
FINAL_VIDEO = RUN_ROOT / "output_private.mkv"
```

### Helper Functions

```{code-cell} python
def _size_of(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if path.exists():
        return path.stat().st_size
    return 0


def run_stage(name: str, args: list[str], output_paths: list[Path]) -> None:
    cmd = ["python", "-m", "conan_dub.cli.conan_dub", *args]
    print(f"[stage] {name}:", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True)
    end = time.perf_counter()

    log_path = RUN_ROOT / f"{name.replace(' ', '_')}.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    outputs = {path: _size_of(path) for path in output_paths}
    outputs[log_path] = log_path.stat().st_size
    record_stage(report, name, start, end, outputs)

    if result.returncode != 0:
        append_warning(report, f"{name} exited with code {result.returncode}")
        raise RuntimeError(f"Stage '{name}' failed. See {log_path}")


def stage_args(base: list[str]) -> list[str]:
    args = base + ["--device", DEVICE, "--seed", str(SEED)]
    if DRY_RUN:
        args.append("--dry-run")
    return [str(a) for a in args]
```

## 3. Pipeline Execution

Each cell below runs one pipeline stage, capturing timings and output sizes.

### Split (Voice/Background)

```{code-cell} python
run_stage(
    "split",
    stage_args([
        "split",
        str(MEDIA_PATH),
        "--out",
        str(AUDIO_DIR),
    ]),
    [AUDIO_DIR / "voice.wav", AUDIO_DIR / "background.wav"],
)
```

### Diarize

```{code-cell} python
run_stage(
    "diarize",
    stage_args([
        "diarize",
        str(AUDIO_DIR / "voice.wav"),
        "--out",
        str(DIAR_DIR),
    ]),
    [DIAR_DIR / "segments.json"],
)
```

### Align (WhisperX)

```{code-cell} python
run_stage(
    "align",
    stage_args([
        "align",
        str(AUDIO_DIR / "voice.wav"),
        "--segments",
        str(DIAR_DIR / "segments.json"),
        "--out",
        str(ALIGNED_PATH),
    ]),
    [ALIGNED_PATH],
)
```

### Frames

```{code-cell} python
run_stage(
    "frames",
    stage_args([
        "frames",
        str(MEDIA_PATH),
        "--out",
        str(FRAMES_DIR),
    ]),
    [FRAMES_DIR],
)
```

### Detect Faces

```{code-cell} python
run_stage(
    "detect-faces",
    stage_args([
        "detect-faces",
        str(FRAMES_DIR),
        "--out",
        str(FACES_PATH),
    ]),
    [FACES_PATH],
)
```

### Character Seed and Augment

```{code-cell} python
run_stage(
    "char-seed",
    stage_args([
        "char-seed",
        str(FRAMES_DIR),
        "--out",
        str(CHAR_DIR),
    ]),
    [CHAR_DIR],
)

run_stage(
    "char-augment",
    stage_args([
        "char-augment",
        "--chars",
        str(CHAR_DIR),
        "--faces",
        str(FACES_PATH),
        "--out",
        str(CHAR_AUG_DIR),
    ]),
    [CHAR_AUG_DIR / "clusters.json", CHAR_AUG_DIR / "faces.json"],
)
```

### Speech Emotion Recognition

```{code-cell} python
run_stage(
    "ser",
    stage_args([
        "ser",
        str(AUDIO_DIR / "voice.wav"),
        "--aligned",
        str(ALIGNED_PATH),
        "--out",
        str(SER_PATH),
    ]),
    [SER_PATH],
)
```

### Fuse Modalities

```{code-cell} python
run_stage(
    "fuse",
    stage_args([
        "fuse",
        "--aligned",
        str(ALIGNED_PATH),
        "--diar",
        str(DIAR_DIR / "segments.json"),
        "--ser-path",
        str(SER_PATH),
        "--faces",
        str(FACES_PATH),
        "--out",
        str(MANIFEST_PATH),
    ]),
    [MANIFEST_PATH],
)
```

### Translate

```{code-cell} python
run_stage(
    "translate",
    stage_args([
        "translate",
        str(MANIFEST_PATH),
        "--out",
        str(MANIFEST_IT_PATH),
    ]),
    [MANIFEST_IT_PATH],
)
```

### Synthesis

```{code-cell} python
run_stage(
    "synth",
    stage_args([
        "synth",
        str(MANIFEST_IT_PATH),
        "--out",
        str(TTS_DIR),
    ]),
    [TTS_DIR],
)
```

### Mixing

```{code-cell} python
run_stage(
    "mix",
    stage_args([
        "mix",
        str(TTS_DIR),
        "--out",
        str(FINAL_AUDIO),
        "--bg",
        str(AUDIO_DIR / "background.wav"),
    ]),
    [FINAL_AUDIO],
)
```

### Muxing

```{code-cell} python
run_stage(
    "mux",
    stage_args([
        "mux",
        str(MEDIA_PATH),
        "--audio",
        str(FINAL_AUDIO),
        "--out",
        str(FINAL_VIDEO),
    ]),
    [FINAL_VIDEO],
)
```

## 4. Generate Report

```{code-cell} python
json_path, md_path = write_report(report, RUN_ROOT)
print("JSON report:", json_path)
print("Markdown report:", md_path)
with md_path.open("r", encoding="utf-8") as handle:
    print(handle.read())
```

## 5. Artifacts Overview

```{code-cell} python
for child in RUN_ROOT.iterdir():
    if child.is_file():
        size = _size_of(child)
        print(f"FILE {child.name:25s} {size/1024:.1f} KB")
    else:
        size = _size_of(child)
        print(f"DIR  {child.name:25s} {size/1024:.1f} KB")
```

Notebook complete – check the Drive folder for audio, manifests, logs, and the generated reports.

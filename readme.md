# Detective Conan AI Dubbing System - Production Plan

[![Windows CI](https://github.com/your-org/conan-dub/actions/workflows/windows.yml/badge.svg)](https://github.com/your-org/conan-dub/actions/workflows/windows.yml)
[![Build CI](https://github.com/your-org/conan-dub/actions/workflows/build.yml/badge.svg)](https://github.com/your-org/conan-dub/actions/workflows/build.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_colab_starter.md)

**Scope:** Build an AI dubbing reference system that translates Japanese -> Italian and synthesizes multi-speaker, emotion-aware voices. Architecture based on Parakeet/Dia; preprocessing on GPU/Colab; training on TPUs (JAX/Flax NNX); synthesis & mixing locally or in cloud.

**Usage note & rights:** Only process **content you own or are licensed to use**. Outputs are for research/education and private use. Do **not** redistribute copyrighted video/audio without permission.

---

## Quickstart (Dry-Run)

1. `pip install -e .[cpu,dev]`
2. `conan-dub split input.mp4 --out work/audio --dry-run`
3. `conan-dub diarize work/audio/voice.wav --out work/diar --dry-run`
4. `conan-dub align work/audio/voice.wav --segments work/diar/segments.json --out work/aligned.json --dry-run`
5. `conan-dub fuse --aligned work/aligned.json --diar work/diar/segments.json --ser-path work/emotions.json --faces work/faces.json --out work/manifest.json --dry-run`
6. `conan-dub translate work/manifest.json --out work/manifest_it.json --dry-run`
7. `conan-dub synth work/manifest_it.json --out work/tts --dry-run`
8. `conan-dub mix work/tts --out work/final_mix.wav --dry-run`
9. `conan-dub mux input.mp4 --audio work/final_mix.wav --out output_private.mkv --dry-run`

The CLI honours `--dry-run/--no-dry-run` flags so every stage can be validated without invoking GPU-dependent models. Drop the flag when the required adapters and checkpoints are installed.

---

## Windows CPU Quickstart

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\win\setup.ps1
.\scripts\win\demo_dry_run.ps1
.\scripts\win\test.ps1
```

Preflight warnings will highlight optional tools (for example `ffmpeg`) that become mandatory once you leave dry-run mode.

---

## Deterministic Dry-Run (No ffmpeg)

- Placeholder assets (silent WAVs, empty JSON manifests, text video stubs) are generated in pure Python, guaranteeing deterministic outputs even when `ffmpeg` or GPU runtimes are absent.
- Schema validation runs on every manifest write; violations downgrade to warnings while `--dry-run` is active.
- Verbosity can be adjusted globally with `-v/--verbose` and `-q/--quiet` to surface troubleshooting details when needed.
- Use `conan-dub diagnostics --dry-run` for an environment snapshot (version, config source, schema, ffmpeg).
- `conan-dub --version` reports the installed package build (default seed is `1337`).

---

## Colab Handoff

- Generate the bundle locally with `scripts\win\make_colab_bundle.ps1` (Windows) or `scripts/make_colab_bundle.sh` (macOS/Linux). The archive is written to `dist/colab_bundle.zip` along with `COLAB_README.md`.
- In Colab, run:
  ```bash
  bash colab/setup_colab.sh
  bash scripts/colab_pull_inputs.sh || true
  ```
  Then open `notebooks/02_colab_tests.md` (or `01_colab_starter.md` for manual setup) and execute the cells to produce a GPU-backed report under `MyDrive/conan_dub_runs/<run_id>/`.
- The CLI remains deterministic; pass `--seed` for reproducibility and toggle `--device cuda|cpu` per stage.

---

## Pipeline Overview

```
input.mp4
   |
   +--> Audio Separation (vocal | demucs | uvr5)
   |       |
   |       +--> Diarization (WhisperX + pyannote)
   |       +--> Alignment + Emotion Tags + Translation
   |
   +--> Frame Extraction --> Face Detection (AnimeFace / InsightFace)
           \--> Character Linking (seed + clustering + fusion)
   |
   +--> TTS (Parakeet/Dia) --> Mixing & Loudness --> Private mux
```

---

## Documentation Index

- [01 - Preprocessing & Dataset Creation](docs/01_preprocessing_dataset.md)
- [02 - Model Training](docs/02_model_training.md)
- [03 - Automatic Dubbing & Synthesis](docs/03_dubbing_synthesis.md)
- [04 - Engineering Notes](docs/04_engineering_notes.md)
- [05 - Codegen Prompt](docs/05_codegen_prompt.md)
- [Colab Notebook Scaffold](notebooks/01_colab_starter.md)

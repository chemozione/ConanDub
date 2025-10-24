# 🧾 One-Shot Prompt to Generate the Open-Source Utility (feed to your codegen AI)

> Paste this verbatim into your codegen tool to scaffold the repo, respecting the fixed constraints.


```
You are generating a public, research/education-only open-source utility named
"conan-dub" that builds an anime dubbing pipeline per the following requirements.

NON-NEGOTIABLES:
- Integrate WhisperX alignment/VAD/CTM with Whisper transcription (Japanese), and pyannote.audio
  diarization with speaker embeddings.
- Default voice/background separator is "vocal"; include optional Demucs and UVR5 CLI adapters,
  behind flags.
- Face detection defaults to AnimeFace; include an optional InsightFace path.
- Provide a minimal character dataset workflow: seed 10–20 crops per character, optional
  CLIP/DeepFace/InsightFace embedding + clustering to semi-auto expand, with a small
  CLI for human verification.
- Fuse visual presence + lip activity (SyncNet or optical-flow heuristic) with diarized
  speech to assign character names to speaker turns. Make the fusion logic modular.
- Emotion tags per utterance using a Hugging Face audio-classification pipeline; model is
  configurable in a YAML file.
- Assemble a Parquet/Arrow (or HF Dataset) with fields:
  id, start, end, speaker_id, character, emotion, text_ja, text_it, audio_path, speaker_embedding.
- Provide a training interface for Parakeet/Dia-style multispeaker Italian TTS with
  speaker + emotion conditioning. Implement loaders for DAC-compressed audio (optional),
  but DO NOT fabricate metrics or timelines—log whatever is empirically measured.
- Synthesis: chunked generation using the trained model; optional target-duration nudging
  (time-stretch within a safe range).
- Mixing/mastering: voice timeline placement; light room match; loudness normalization;
  final mux with original video (for personal use). Do not publish the output in examples.

REPO LAYOUT:
- README.md with quickstart, Colab badge, and high-level diagram.
- very simple LICENSE (MIT or Apache-2.0). Will generate a complete one only when everything will work.
- /src/conan_dub/ with subpackages:
  - /audio/{separate.py,mix.py,loudness.py}
  - /align/{whisperx_runner.py, diarize.py}
  - /vision/{frames.py, animeface.py, insightface_adapter.py, lipsync.py}
  - /charlink/{seed_dataset.py, embed_cluster.py, fuse.py}
  - /ser/{emotion.py}
  - /data/{schema.py, export.py}
  - /translate/{nllb.py}
  - /tts/{interfaces.py, parakeet_dia_hooks.py}
  - /cli/{conan_dub.py}  (argparse / Typer)
- /configs/default.yaml with model names, thresholds, paths.
- /notebooks/01_colab_starter.ipynb mirroring the scaffold blocks.
- /tests/ with smoke tests for each stage (use tiny dummy fixtures).
- /scripts/ convenience bash scripts (prepare_dataset.sh, separate.sh, mux.sh).
- /examples/ with a synthetic tiny sample (non-copyrighted) demonstrating the JSON schema ONLY.

CODING STANDARDS:
- Python 3.10+, type hints, docstrings, black + ruff, small pure functions.
- All external models behind adapters; no hard-coded credentials.
- No placeholder “assert SDR > …” or fabricated numbers; log measured stats only.

CLI UX (Typer recommended):
- `conan-dub split --input input.mp4 --out outdir --separator vocal`
- `conan-dub diarize --voice outdir/voice.wav --out outdir/diar`
- `conan-dub align --voice outdir/voice.wav --segments outdir/diar/segments.json --out outdir/aligned.json`
- `conan-dub frames --input input.mp4 --fps 2 --out outdir/frames`
- `conan-dub detect-faces --frames outdir/frames --detector animeface --out outdir/faces.json`
- `conan-dub char-seed --frames outdir/frames --out data/characters`
- `conan-dub char-augment --chars data/characters --faces outdir/faces.json --out data/char_augmented`
- `conan-dub ser --voice outdir/voice.wav --segments outdir/aligned.json --out outdir/emotion.json`
- `conan-dub fuse --aligned outdir/aligned.json --faces outdir/faces.json --ser outdir/emotion.json --out outdir/manifest.json`
- `conan-dub translate --in outdir/manifest.json --out outdir/manifest_it.json --model nllb-200`
- `conan-dub synth --manifest outdir/manifest_it.json --tts-config configs/default.yaml --out outdir/tts_wavs/`
- `conan-dub mix --tts outdir/tts_wavs --bg outdir/background.wav --out outdir/final_audio.wav`
- `conan-dub mux --video input.mp4 --audio outdir/final_audio.wav --out output_private.mkv`

TESTING:
- Include minimal unit tests that validate CLI flows on tiny fixtures.
- Add a “dry-run” mode that runs the full pipeline but skips heavy models.

DOCUMENTATION:
- README: quickstart, Colab link, pipeline diagram, rights notice, limitations.
- Docs stress that metrics/timelines/costs are NOT guaranteed and must be measured.

DELIVERABLE:
- A working scaffold (installable with `pip install -e .`) that can run the
  pipeline stages independently with stubs where GPU models are unavailable.
```

---
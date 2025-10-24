# Phase 1 — Preprocessing & Dataset Creation

## 1) Voice / Background Separation

* **Primary (per original brief):** `vocal` (fast TFC stem separation).
* **Alternatives (optional):** Demucs v4, UVR5 CLI for experiments.
* **Outputs:** `voice.wav` (dialogue) and `background.wav` (music/SFX) derived via polarity inversion or multi-stem recombination.
* **Store:** WAV + `metadata.json` (duration, RMS, silence ratio).
  *Illustrative code only; choose one separator per environment.*

## 2) Diarization + Forced Alignment (WhisperX)

* **Diarization:** `pyannote.audio` for speaker turns and embeddings.
* **Transcription:** `whisper-large-v3` (or equivalent).
* **Alignment/VAD/CTM (FIXED OMISSION):** **Integrate WhisperX** to snap word/phoneme timings to audio and to harmonize with diarization. Use WhisperX’s VAD + alignment to tighten segment boundaries.
* **Outputs:** speaker-labeled segments with word-level timestamps + speaker embeddings (for later character linking).

## 3) Face Detection & Frame Sampling

* Extract frames (e.g., `ffmpeg -r 2 …`).
* **Anime-oriented detector:** AnimeFace; **optional modern alt:** InsightFace.
* Output per frame: `{frame_id, ts, faces:[bbox, confidence]}`.

## 4) Character Image Dataset (Manual Seed)

* Label ~10–20 clear crops per main character (folder per character).
* Keep a YAML manifest of counts and notes.

## 5) Visual Augmentation (Semi-auto)

* Extract embeddings (CLIP/DeepFace/InsightFace).
* Cluster unlabeled crops; assign labels top-k nearest to seeds; human verify.

## 6) Character Recognition & Who-Is-Speaking

* Train a lightweight CNN/ViT for character ID.
* **Lip activity:** optical flow or SyncNet for speaking frames.
* **Fusion:** overlap diarized speech with on-screen speaker; resolve conflicts by majority vote + confidence thresholds.
* **Output:** `speaker_id, character_name, t_start, t_end`.

## 7) Speech Emotion Recognition

* Baseline: Whisper-emotion model or WavLM-based classifier.
* Output per utterance: `emotion`, `confidence`, timestamps.

## 8) Transcription + Metadata Fusion

Merge:

* text (aligned by WhisperX),
* timestamps,
* `speaker_id` + `character_name`,
* `emotion`.
  Result schema (per original brief):

```json
{
  "id": "scene_000123",
  "start": 12.34,
  "end": 15.78,
  "speaker": "Sakura",
  "emotion": "surprise",
  "text": "Eh?! Davvero hai detto questo?",
  "audio_path": "voices/sakura_000123.wav"
}

## 9) TTS Training Dataset Assembly

* Keep sample rate ≥16 kHz (24 kHz OK if DAC used later).

* Split train/val/test (e.g., 80/10/10).

* Storage: Arrow/Parquet or HF Datasets.

* No fabricated stats: dataset size/duration are TBD empirically based on your clips.
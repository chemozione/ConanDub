# Phase 2 — Model Training (Italian, Multi-speaker, Emotion)

## 10) Multispeaker Italian TTS (Parakeet/Dia)

* Base options: Dia 1.6B (Parakeet-style) / VibeVoice fine-tuning / smaller Bark variants for prototyping.
* Conditioning: speaker embeddings + emotion tokens (e.g., `[emotion:angry]`, `[speaker:Conan]`).
* Infra: JAX/Flax NNX on TPUs (or A100s).
* **Data handling:** optional DAC compression for faster I/O; keep bucket region **co-located** with TPU.
* **Metrics/curves:** treat as **TBD, measured during training** (no preset losses or hours).

## 11) Evaluation (No invented numbers)

* Human MOS (naturalness), emotion match, voice consistency, Italian quality, inference speed. Target **qualitative goals** (“high naturalness,” “consistent emotion”), but **report empirical results** once measured.

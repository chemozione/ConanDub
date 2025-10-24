# Phase 3 — Automatic Dubbing & Synthesis

## 12) Transcript & Translation (JP→IT)

* Transcribe + align with WhisperX.
* Translate using NLLB-200 (distilled 600M acceptable) or larger if resources allow.
* Output JSON with timings + `jp_text` + `it_translation`.

## 13) Chunked Dub Generation

* For each aligned segment:

[speaker:Conan] [emotion:surprise]
"La verità è sempre una!"

* Keep segments short (<~10s) to maintain sync; allow optional duration-constrained prosody.

## 14) Audio Reconstruction & Mix

* Place dubbed clips on timeline using segment timestamps.
* Reverb/room match lightly to original; loudness normalize for consistent levels.
* Merge with `background.wav`.


#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-.conan_out}
DEMO_VIDEO=${2:-examples/tiny_input.mp4}

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

touch "${DEMO_VIDEO}"

commands=(
  "python -m conan_dub.cli.conan_dub split \"${DEMO_VIDEO}\" --out \"${OUT_DIR}\" --dry-run"
  "python -m conan_dub.cli.conan_dub diarize \"${OUT_DIR}/voice.wav\" --out \"${OUT_DIR}/diar\" --dry-run"
  "python -m conan_dub.cli.conan_dub align \"${OUT_DIR}/voice.wav\" --segments \"${OUT_DIR}/diar/segments.json\" --out \"${OUT_DIR}/aligned.json\" --dry-run"
  "python -m conan_dub.cli.conan_dub frames \"${DEMO_VIDEO}\" --out \"${OUT_DIR}/frames\" --dry-run"
  "python -m conan_dub.cli.conan_dub detect-faces \"${OUT_DIR}/frames\" --out \"${OUT_DIR}/faces.json\" --dry-run"
  "python -m conan_dub.cli.conan_dub char-seed \"${OUT_DIR}/frames\" --out \"${OUT_DIR}/characters\" --dry-run"
  "python -m conan_dub.cli.conan_dub char-augment --chars \"${OUT_DIR}/characters\" --faces \"${OUT_DIR}/faces.json\" --out \"${OUT_DIR}/char_aug\" --dry-run"
  "python -m conan_dub.cli.conan_dub ser \"${OUT_DIR}/voice.wav\" --aligned \"${OUT_DIR}/aligned.json\" --out \"${OUT_DIR}/emotion.json\" --dry-run"
  "python -m conan_dub.cli.conan_dub fuse --aligned \"${OUT_DIR}/aligned.json\" --diar \"${OUT_DIR}/diar/segments.json\" --ser-path \"${OUT_DIR}/emotion.json\" --faces \"${OUT_DIR}/faces.json\" --out \"${OUT_DIR}/manifest.json\" --dry-run"
  "python -m conan_dub.cli.conan_dub translate \"${OUT_DIR}/manifest.json\" --out \"${OUT_DIR}/manifest_it.json\" --dry-run"
  "python -m conan_dub.cli.conan_dub synth \"${OUT_DIR}/manifest_it.json\" --out \"${OUT_DIR}/tts_wavs\" --dry-run"
  "python -m conan_dub.cli.conan_dub mix \"${OUT_DIR}/tts_wavs\" --bg \"${OUT_DIR}/background.wav\" --out \"${OUT_DIR}/final_audio.wav\" --dry-run"
  "python -m conan_dub.cli.conan_dub mux \"${DEMO_VIDEO}\" --audio \"${OUT_DIR}/final_audio.wav\" --out \"${OUT_DIR}/output_private.mkv\" --dry-run"
)

for cmd in "${commands[@]}"; do
  echo ">> ${cmd}"
  eval "${cmd}"
done

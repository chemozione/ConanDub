#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:-$(date -u +"%Y%m%d_%H%M%S")}
SEED=${2:-1337}
DEVICE=${3:-cuda}

BASE_DIR="/content/drive/MyDrive/conan_dub_runs/${RUN_ID}"
mkdir -p "${BASE_DIR}"

MEDIA="/content/test.mp4"
if [ ! -f "$MEDIA" ]; then
  echo "[conan-dub] $MEDIA not found. Please run scripts/colab_pull_inputs.sh or upload manually." >&2
  exit 1
fi

echo "[conan-dub] Running quick pipeline into ${BASE_DIR}".

python -m conan_dub.cli.conan_dub split "$MEDIA" --out "${BASE_DIR}/audio" --device "$DEVICE" --seed "$SEED"
python -m conan_dub.cli.conan_dub diarize "${BASE_DIR}/audio/voice.wav" --out "${BASE_DIR}/diar" --device "$DEVICE" --seed "$SEED"
python -m conan_dub.cli.conan_dub align "${BASE_DIR}/audio/voice.wav" --segments "${BASE_DIR}/diar/segments.json" --out "${BASE_DIR}/aligned.json" --device "$DEVICE" --seed "$SEED"

echo "[conan-dub] Quick run complete. Outputs stored in ${BASE_DIR}."

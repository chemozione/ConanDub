#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:-input.mp4}
AUDIO=${2:-final_audio.wav}
OUT=${3:-output_private.mkv}

conan-dub mux --video "$VIDEO" --audio "$AUDIO" --out "$OUT" --dry-run

#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH=${1:-input.mp4}
WORKDIR=${2:-workdir}

conan-dub split "$VIDEO_PATH" --out "$WORKDIR/audio" --separator vocal --dry-run
conan-dub diarize "$WORKDIR/audio/voice.wav" --out "$WORKDIR/diar" --dry-run
conan-dub frames "$VIDEO_PATH" --fps 2 --out "$WORKDIR/frames" --dry-run
conan-dub detect-faces "$WORKDIR/frames" --out "$WORKDIR/faces.json" --dry-run
conan-dub align "$WORKDIR/audio/voice.wav" --segments "$WORKDIR/diar/segments.json" --out "$WORKDIR/aligned.json" --dry-run
conan-dub ser "$WORKDIR/audio/voice.wav" --aligned "$WORKDIR/aligned.json" --out "$WORKDIR/emotions.json" --dry-run
conan-dub fuse --aligned "$WORKDIR/aligned.json" --diar "$WORKDIR/diar/segments.json" --ser-path "$WORKDIR/emotions.json" --faces "$WORKDIR/faces.json" --out "$WORKDIR/manifest.json" --dry-run

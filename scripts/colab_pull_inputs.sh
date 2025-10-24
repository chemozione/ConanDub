#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/content/drive/MyDrive/conan_inputs"
DEST_DIR="/content"

if [ ! -d "$SRC_DIR" ]; then
  echo "[conan-dub] Source directory $SRC_DIR not found. Skipping copy."
  exit 0
fi

copied=false
for name in test.mp4 test.mkv; do
  if [ -f "$SRC_DIR/$name" ]; then
    echo "[conan-dub] Copying $name from Drive..."
    cp "$SRC_DIR/$name" "$DEST_DIR/$name"
    copied=true
  fi
done

if [ "$copied" = false ]; then
  echo "[conan-dub] No test media found in $SRC_DIR."
else
  ls -lh $DEST_DIR/test.* || true
fi

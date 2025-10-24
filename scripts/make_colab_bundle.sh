#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(pwd)
DIST_DIR="$ROOT_DIR/dist"
STAGE_DIR="$DIST_DIR/colab_bundle"
ZIP_PATH="$DIST_DIR/colab_bundle.zip"

rm -f "$ZIP_PATH"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

copy_item() {
  local source="$1"
  local destination="$STAGE_DIR/$source"
  if [ -d "$source" ]; then
    cp -R "$source" "$STAGE_DIR/"
  else
    mkdir -p "$(dirname "$destination")"
    cp "$source" "$destination"
  fi
}

copy_item "README.md"
copy_item "COLAB_README.md"
copy_item "configs"
copy_item "docs"
copy_item "examples"
copy_item "colab"
copy_item "notebooks/01_colab_starter.md"
copy_item "notebooks/02_colab_tests.md"

(cd "$STAGE_DIR" && zip -rq "$ZIP_PATH" .)
rm -rf "$STAGE_DIR"
echo "Created $ZIP_PATH"

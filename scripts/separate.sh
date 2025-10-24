#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-input.mp4}
OUT=${2:-stems}
SEPARATOR=${3:-vocal}

conan-dub split "$INPUT" --out "$OUT" --separator "$SEPARATOR" --dry-run

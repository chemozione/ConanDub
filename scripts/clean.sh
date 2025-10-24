#!/usr/bin/env bash
set -euo pipefail

rm -rf .conan_out build dist
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
echo "Workspace cleaned."

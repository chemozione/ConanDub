#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${1:-.venv}

if [ -d "${VENV_PATH}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

pytest -q

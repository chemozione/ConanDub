#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${1:-.venv}

if [ ! -d "${VENV_PATH}" ]; then
  python -m venv "${VENV_PATH}"
fi

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip
pip install -e ".[cpu,dev]"

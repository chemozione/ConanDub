#!/usr/bin/env bash
set -euo pipefail

echo "[conan-dub] Updating apt cache and installing ffmpeg..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg

echo "[conan-dub] Upgrading pip..."
python -m pip install --upgrade pip

echo "[conan-dub] Installing GPU requirements..."
python -m pip install -r colab/requirements-gpu.txt

echo "[conan-dub] Installing conan-dub in editable mode with GPU extras..."
python -m pip install -e .[gpu]

echo "[conan-dub] Checking CUDA availability..."
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    print("torch.version.cuda=", torch.version.cuda)
except ModuleNotFoundError:
    print("torch is not installed; GPU inference will not be available.")
PY

echo "[conan-dub] Environment setup complete."

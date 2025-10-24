# Colab Handoff Guide

Welcome to the **conan-dub** Colab bundle. This archive is designed to run entirely on Google Colab (CPU or GPU) without relying on private assets.

## Contents

- `README.md` – project overview and quickstart
- `docs/` – authoritative specifications for every pipeline phase
- `configs/default.yaml` – base configuration template
- `examples/` – tiny schema-compliant samples (no copyrighted media)
- `notebooks/01_colab_starter.md` – step-by-step setup for Colab

## Quick Start

1. Upload the unzipped bundle to your Colab workspace (or mount Google Drive).
2. Open `notebooks/01_colab_starter.md` side-by-side while running Colab cells.
3. Install the package in editable mode:
   ```bash
   pip install -e .[cpu]
   ```
4. Run CLI stages with `--dry-run` first. Drop the flag only when GPU runtimes and model checkpoints are configured.

## Notes

- All placeholders are deterministic (seeded via `--seed`, default `1337`).
- `conan-dub diagnostics --dry-run` reports environment readiness (ffmpeg, schema, config).
- Keep outputs private; demo assets are strictly synthetic.

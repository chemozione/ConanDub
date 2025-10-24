# 🤖 Agent Configuration for Cursor / Codex / GPT-based Dev Environments

## Agent roles
- **Architect Agent** → reads `docs/05_codegen_prompt.md` and scaffolds `/src/conan_dub`.
- **Data Agent** → executes Phase 1 scripts and notebook cells.
- **Trainer Agent** → fine-tunes TTS (Phase 2).
- **Synth Agent** → runs translation, synthesis, and mixing (Phase 3).
- **Doc Agent** → validates docs & schema in `/docs` and `/examples`.

## Task orchestration
1. Read `.cursor/rules` for directory mappings.
2. Use `docs/` as authoritative specification.
3. Write outputs under `/src/conan_dub/` according to `05_codegen_prompt.md`.
4. Never overwrite original documentation; commit generated code separately.

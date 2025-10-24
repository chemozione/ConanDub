
# 🔧 Colab Notebook Scaffold (copy/paste skeleton)

> This gives you a runnable, organized starter. Cells reference **vocal** as default, with optional Demucs/UVR5; **WhisperX** is integrated; diarization, faces, fusion, SER, dataset export, and synthesis hooks are stubbed. Replace placeholders (`<…>`) and mount storage as needed.

```python
# %% [markdown]
# # Detective Conan AI Dubbing — Colab Starter
# Research & educational use only. Process only content you own or are licensed to use.

# %% [markdown]
# ## 0) Env setup

# %%
!pip -q install --upgrade pip
!pip -q install vocal pyannote.audio==3.1 torch torchaudio librosa soundfile \
  whisperx==3.1.1 transformers datasets ffmpeg-python opencv-python \
  pillow pyyaml pyarrow pandas tqdm

# Optional extras
# !pip -q install demucs onnxruntime-gpu insightface deepface

# %% [markdown]
# ## 1) I/O helpers

# %%
from pathlib import Path
import json, os, shutil
BASE = Path("/content/conan_dub")
AUDIO = BASE/"audio"
FRAMES = BASE/"frames"
DATA = BASE/"data"
for p in (AUDIO, FRAMES, DATA): p.mkdir(parents=True, exist_ok=True)

def write_json(obj, path): 
    with open(path, "w") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

# %% [markdown]
# ## 2) Voice/Background separation (primary: vocal)

# %%
# Input: /content/input.mp4
INPUT = "/content/input.mp4"  # <-- replace
VOICE_WAV = str(AUDIO/"voice.wav")
BG_WAV = str(AUDIO/"background.wav")
META = DATA/"separation_meta.json"

# vocal produces stems; use polarity inversion to reconstruct background
!vocal --input "{INPUT}" --output "{AUDIO}" --format wav --sample-rate 24000

# TODO: Create background via polarity inversion if only vocals provided, or recombine stems.
# Save metadata (duration, RMS, silence ratio) after analysis.
write_json({"note":"fill with measured stats"}, META)

# %% [markdown]
# ## 3) Diarization + WhisperX alignment (FIXED)

# %%
import torch, torchaudio, subprocess, json
# Diarization (pyannote)
# NOTE: You need a Hugging Face token if using gated models.
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Placeholder: implement diarization pipeline and store segments + embeddings.

# WhisperX for ASR + alignment
# Model download (auto). You can pass device="cuda" if available.
import whisperx
model = whisperx.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
asr_result = model.transcribe(VOICE_WAV, language="ja")

# Voice activity / alignment
align_model, metadata = whisperx.load_align_model(language_code="ja", device=model.device)
asr_aligned = whisperx.align(asr_result["segments"], align_model, metadata, VOICE_WAV, model.device, return_char_alignments=False)

write_json(asr_aligned, DATA/"whisperx_aligned.json")

# %% [markdown]
# ## 4) Face detection (AnimeFace by default; InsightFace optional)

# %%
# Extract frames at 2 FPS
!mkdir -p "{FRAMES}"
!ffmpeg -y -i "{INPUT}" -r 2 "{FRAMES}/frame_%06d.jpg" -hide_banner -loglevel error

# AnimeFace is a binary; on Colab, simplest is to call via subprocess if available,
# otherwise switch to InsightFace (GPU). Here we stub detection outputs.
write_json({"frames":"stub - run AnimeFace or InsightFace and dump bbox+conf per frame"}, DATA/"faces.json")

# %% [markdown]
# ## 5) Character dataset seeding & augmentation

# %%
# Place 10–20 labeled crops per character under DATA/characters/<name>/*.jpg
# Then compute embeddings (CLIP/DeepFace/InsightFace) and cluster to expand labels.
write_json({"todo":"embed, cluster, semi-auto label, verify"}, DATA/"char_augment.json")

# %% [markdown]
# ## 6) Who-is-speaking fusion (visual + diarization + lip activity)

# %%
# Load diarization segments, face tracks, and optionally lip activity (SyncNet/optical flow)
# Compute mapping from SPEAKER_XX -> character name.
write_json({"mapping":"stub - SPEAKER_00 -> Conan, etc."}, DATA/"speaker_character_map.json")

# %% [markdown]
# ## 7) Speech Emotion Recognition (SER)

# %%
# Example with a HF pipeline; chunk VOICE_WAV by aligned segments and run SER.
write_json({"ser":"stub - emotion tags per segment"}, DATA/"emotion.json")

# %% [markdown]
# ## 8) Merge transcript + speaker + emotion into training samples

# %%
merged = {
  "samples": [
    # fill with merged segments including text, speaker_id, character, emotion, timestamps, audio path
  ]
}
write_json(merged, DATA/"training_manifest.json")

# %% [markdown]
# ## 9) Translation JP→IT (NLLB) with optional character-aware prompts

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda" if torch.cuda.is_available() else "cpu")

def translate_jp_to_it(text):
    x = tok(text, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(**x, forced_bos_token_id=tok.lang_code_to_id["ita_Latn"], max_length=256)
    return tok.decode(gen[0], skip_special_tokens=True)

# Iterate merged samples, attach it_translation
write_json({"translation":"stub - apply translate_jp_to_it per segment"}, DATA/"translated.json")

# %% [markdown]
# ## 10) TTS hooks (Parakeet/Dia fine-tuned model)

# %%
# This cell should load your fine-tuned model when ready.
# For now, stub a synthesizer interface that returns silence or a beep per segment.
write_json({"tts":"stub - integrate Parakeet/Dia inference"}, DATA/"tts_outputs.json")

# %% [markdown]
# ## 11) Mix & Master: place dubbed segments over background

# %%
# 1) Build a zeroed voice timeline; 2) drop each TTS clip at start; 3) light reverb; 4) sum with background; 5) normalize.
# Save WAV and then mux with original video via ffmpeg.
write_json({"mix":"stub - implement timeline, loudness, mux"}, DATA/"mix.json")

# %% [markdown]
# ✅ End-to-end skeleton complete.
# Replace stubs with your implementations and measure metrics empirically (no hard-coded targets).
```

---

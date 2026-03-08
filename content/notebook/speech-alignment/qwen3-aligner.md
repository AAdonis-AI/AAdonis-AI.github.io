---
title: "Qwen3 Forced Aligner"
date: 2025-10-03
draft: false
tags: ["alignment", "qwen3", "timestamps"]
summary: "Precise word and phoneme-level timestamps using Qwen3-ASR and its paired forced aligner — multilingual, robust to noise."
---

The Qwen3 ecosystem includes a forced aligner that provides high-resolution word and phoneme timestamps by aligning a known transcript to audio. Unlike MFA, it doesn't require a pronunciation dictionary and handles accented and multilingual speech better.

- **ASR Model:** `Qwen/Qwen3-ASR-1.7B`
- **Forced Aligner:** `Qwen/Qwen3-ForcedAligner-0.6B`

## Word Alignment Script

```python
import torch
from qwen_asr import Qwen3ASRModel

def align_audio(audio_path, text=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=device,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B"
    )

    # If text is provided → forced alignment.
    # If text is None → transcribes first, then aligns.
    results = model.transcribe(
        audio=audio_path,
        language="English",
        return_time_stamps=True
    )

    for ts in results.time_stamps:
        print(f"Word: {ts.text:15s} | Start: {ts.start_time:.3f}s | End: {ts.end_time:.3f}s")

# align_audio("path/to/audio.wav")
```

## Key Features

- **Sub-word precision** — high-resolution timestamps for linguistic analysis
- **Robustness** — handles background noise and varied accents
- **Multilingual** — supports all languages in Qwen3's training set
- **No dictionary required** — unlike MFA, works on any language/accent without a G2P dictionary

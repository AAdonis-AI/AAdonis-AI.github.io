---
title: "EZ-VC"
date: 2025-10-02
draft: false
tags: ["voice-conversion", "ez-vc", "flow-matching"]
summary: "Easy Zero-shot Any-to-Any Voice Conversion — single encoder architecture, excellent cross-lingual performance, and a clean inference script."
---

EZ-VC is a simple zero-shot voice conversion model that combines discrete speech representations from XEUS (a self-supervised model trained on 4000 languages) with a flow-matching diffusion decoder based on F5-TTS. The key differentiator is a **single encoder** architecture — no separate speaker/content encoders needed.

**Paper: [EZ-VC (EMNLP 2025 Findings)](https://arxiv.org/abs/2505.16691)**
**Codebase: [Github](https://github.com/EZ-VC/EZ-VC)**
**HuggingFace: [Model](https://huggingface.co/SPRINGLab/EZ-VC)**

## Performance vs Seed-VC

| Model | SSIM ↑ | NMOS ↑ | SMOS ↑ | UTMOS ↑ |
|-------|--------|--------|--------|---------|
| Seed-VC | 0.69 | 3.55 | 3.78 | 3.02 |
| kNN-VC | 0.59 | 1.94 | 2.05 | 2.42 |
| Vec2Wav2.0 | 0.61 | 3.67 | 3.55 | 3.55 |
| **EZ-VC** | **0.71** | **3.91** | **3.90** | **3.56** |

**When to use EZ-VC:** cross-lingual conversion (especially unseen languages), highest naturalness scores.
**When to use Seed-VC:** real-time conversion, singing, accent/emotion conversion (V2).

## Installation

```bash
git clone https://github.com/EZ-VC/EZ-VC
cd EZ-VC
git submodule update --init --recursive

conda create -n ez-vc python=3.10
conda activate ez-vc

# NVIDIA GPU
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Apple Silicon
# pip install torch torchaudio

pip install -e .

# Install espnet for XEUS (EXACTLY this version)
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
```

## Inference Script

```python
#!/usr/bin/env python3
"""
EZ-VC Voice Conversion Inference
Usage: python inference.py --ref_audio ref.wav --src_audio source.wav --output output.wav
"""

import os, sys, argparse
import soundfile as sf
import torch
from pathlib import Path
from cached_path import cached_path
from omegaconf import OmegaConf
from hydra.utils import get_class

src_path = os.path.join(os.getcwd(), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, target_rms
from f5_tts.infer.utils_xeus import ApplyKmeans, load_xeus_model, extract_units

DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_SWAY_COEF = -1.0

def load_all_models(device, vocoder_name, config_path):
    print(f"Loading models on {device}...")
    xeus_model = load_xeus_model(device).eval()
    apply_kmeans = ApplyKmeans(device)
    vocoder = load_vocoder(vocoder_name=vocoder_name, device=device)

    ckpt_file = str(cached_path("hf://SPRINGLab/EZ-VC/model_2700000.safetensors"))
    vocab_file = str(cached_path("hf://SPRINGLab/EZ-VC/vocab.txt"))

    model_cfg = OmegaConf.load(config_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch

    ema_model = load_model(
        model_cls, model_arch, ckpt_file,
        mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device,
    )
    return xeus_model, apply_kmeans, vocoder, ema_model


def run_inference(ref_audio, src_audio, output_path, device="cuda",
                  nfe=32, speed=1.0,
                  config_path="src/f5_tts/configs/F5TTS_Base_EZ-VC.yaml"):
    xeus_model, apply_kmeans, vocoder, ema_model = load_all_models(device, "bigvgan", config_path)

    print(f"Extracting units from Reference: {ref_audio}")
    ref_text = extract_units(ref_audio, xeus_model, apply_kmeans, device)

    print(f"Extracting units from Source: {src_audio}")
    src_text = extract_units(src_audio, xeus_model, apply_kmeans, device)

    print(f"Running Inference (NFE={nfe})...")
    audio_segment, final_sample_rate, _ = infer_process(
        ref_audio, ref_text, src_text, ema_model, vocoder,
        mel_spec_type="bigvgan", target_rms=target_rms,
        cross_fade_duration=0.15, nfe_step=nfe,
        cfg_strength=DEFAULT_CFG_STRENGTH, sway_sampling_coef=DEFAULT_SWAY_COEF,
        speed=speed, fix_duration=None, device=device,
    )

    sf.write(output_path, audio_segment, final_sample_rate)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EZ-VC Inference")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio (target voice)")
    parser.add_argument("--src_audio", type=str, required=True, help="Source audio (content to convert)")
    parser.add_argument("--output", type=str, default="output_ezvc.wav")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nfe", type=int, default=32, help="Inference steps (higher = better quality)")
    parser.add_argument("--config", type=str, default="src/f5_tts/configs/F5TTS_Base_EZ-VC.yaml")
    args = parser.parse_args()

    run_inference(
        ref_audio=args.ref_audio, src_audio=args.src_audio,
        output_path=args.output, device=args.device,
        nfe=args.nfe, config_path=args.config
    )
```

## Architecture Overview

EZ-VC uses a two-stage pipeline:

1. **Speech-to-Units (XEUS + K-means)**
   - XEUS encoder processes speech at 50 embeddings/second (25ms window, 20ms stride)
   - K-means (500 clusters) quantizes features from the 14th layer
   - Results in discrete speech units capturing linguistic content without speaker identity

2. **Units-to-Speech (F5-TTS based)**
   - Conditional flow matching diffusion decoder
   - Reconstructs speech from discrete units + speaker reference via in-context learning

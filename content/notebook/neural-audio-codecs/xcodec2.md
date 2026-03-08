---
title: "X-Codec2"
date: 2025-10-05
draft: false
tags: ["codecs", "xcodec2", "audio"]
summary: "X-Codec2 setup and usage — single-codebook codec used by Llasa for TTS. Stable version is 1.3.0."
---

X-Codec2 is a single-codebook neural audio codec with a large vocabulary (65536 tokens). It pairs directly with the Llasa TTS model and is the recommended codec for autoregressive speech generation tasks.

**Stable version: `xcodec2==1.3.0`** — newer versions may be unstable.

**HuggingFace: [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)**

## Installation

```sh
conda create -n xcodec2_env python=3.10
conda activate xcodec2_env

pip install torch soundfile transformers xcodec2==1.3.0
```

## Usage

```python
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model

model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
model.eval().cuda()

wav, sr = sf.read("test.wav")
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)

with torch.no_grad():
    vq_code = model.encode_code(input_waveform=wav_tensor)
    print("Code shape:", vq_code.shape)

    recon_wav = model.decode_code(vq_code).cpu()

sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
```

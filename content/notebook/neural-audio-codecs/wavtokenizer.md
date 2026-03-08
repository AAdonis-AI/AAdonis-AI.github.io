---
title: "WavTokenizer"
date: 2025-10-04
draft: false
tags: ["codecs", "wavtokenizer", "audio"]
summary: "WavTokenizer setup — extreme compression (40–75 tok/sec, single codebook). Best for clean TTS; avoid for noisy/degraded speech."
---

WavTokenizer (ICLR 2025) compresses 24kHz audio to just 40 or 75 tokens/sec using a single quantizer. Excellent for clean speech tasks like TTS. **Avoid for speech enhancement** — it was not trained on degraded audio and can introduce additional artifacts when encoding noisy input.

**Paper: [WavTokenizer](https://arxiv.org/pdf/2408.16532)**
**GitHub: [jishengpeng/WavTokenizer](https://github.com/jishengpeng/WavTokenizer)**

## Setup

```bash
git clone https://github.com/jishengpeng/WavTokenizer.git
cd WavTokenizer

conda create -n wavtokenizer python=3.9
conda activate wavtokenizer
pip install -r requirements.txt  # skip fairseq if it fails — inference works without it
```

Select the config matching your model's token rate:
- 40 tok/sec: `./configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml`
- 75 tok/sec: `./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`

Download a checkpoint:
```bash
wget -O wavtokenizer_large_speech_320_v2.ckpt \
  https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_v2.ckpt
```

## Usage

```python
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device = torch.device('cpu')
config_path = "./configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./wavtokenizer_large_unify_600_24k.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path).to(device)

wav, sr = torchaudio.load("audio.wav")
wav = convert_audio(wav, sr, 24000, 1).to(device)
bandwidth_id = torch.tensor([0])

_, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

features = wavtokenizer.codes_to_features(discrete_code)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).cpu()

torchaudio.save("reconstructed.wav", audio_out, 24000)
```

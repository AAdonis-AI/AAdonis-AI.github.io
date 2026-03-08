---
title: "SNAC — Multi-Scale Neural Audio Codec"
date: 2025-10-03
draft: false
tags: ["codecs", "snac", "audio"]
summary: "SNAC setup and encode/decode wrapper — matches DAC quality at significantly lower token rate via multi-scale residual vector quantization."
---

SNAC matches DAC in perceptual reconstruction quality for speech and music while having significantly lower token rate. It uses multi-scale residual vector quantization with downsampled residuals, depthwise convolutions, local attention, and noise blocks. Best general-purpose choice for speech enhancement and editing pipelines.

**Paper: [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/pdf/2410.14411)**
**GitHub: [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)**

## Installation

```bash
pip install snac librosa numpy torch
```

## Wrapper Class

```python
import numpy as np
import torch
import librosa
from snac import SNAC

class SNAC_tools:
    def __init__(self):
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
        self.num_codebooks = 3
        self.device = next(self.model.parameters()).device

    def audio_to_codes(self, audio, sr):
        if sr is None or sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).cuda()
        with torch.inference_mode():
            codes = self.model.encode(audio)
        return codes

    def codes_to_audio(self, codes):
        with torch.inference_mode():
            audio = self.model.decode(codes)
        return audio
```

---
title: "DAC — Descript Audio Codec"
date: 2025-10-02
draft: false
tags: ["codecs", "dac", "audio"]
summary: "DAC setup and a wrapper class for encoding/decoding audio as flattened token sequences — time-major and codebook-major layouts."
---

DAC encodes audio into 9 codebooks at 89 frames/sec, giving 801 tokens/sec when flattened. The high token rate makes it the highest-quality general-purpose codec, but sequences are long — plan accordingly for Transformer context windows.

**Paper: [Descript Audio Codec](https://arxiv.org/pdf/2306.06546)**
**HuggingFace: [descript/dac_44khz](https://huggingface.co/descript/dac_44khz)**
**GitHub: [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)**

## Installation

```sh
pip install transformers soundfile librosa numpy torch
```

## Wrapper Class

The class below handles encode/decode and both flattening layouts (time-major and codebook-major) for use in autoregressive Transformer pipelines.

```python
from transformers import DacModel, AutoProcessor
import torch
import librosa
import soundfile as sf
import numpy as np

class DAC:
    def __init__(self):
        self.model = DacModel.from_pretrained("descript/dac_44khz")
        self.processor = AutoProcessor.from_pretrained("descript/dac_44khz")
        self.num_codebooks = 9
        self.device = next(self.model.parameters()).device

    def audio_to_codebook_matrix(self, audio_wav_path):
        audio, sr = sf.read(audio_wav_path)
        if sr != self.processor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr,
                        target_sr=self.processor.sampling_rate)
        inputs = self.processor(raw_audio=audio,
                        sampling_rate=self.processor.sampling_rate,
                        return_tensors="pt")
        encoder_outputs = self.model.encode(inputs["input_values"].to(self.device))
        return encoder_outputs.audio_codes

    def flatten_matrix_to_vector_time_major(self, codes):
        # Interleaves codebooks: t0_cb0, t0_cb1, ..., t0_cb8, t1_cb0, ...
        return codes[0].T.flatten().tolist()

    def flatten_matrix_to_vector_codebook_major(self, codes):
        # All of codebook 0, then all of codebook 1, ...
        return codes[0].flatten().tolist()

    def vector_time_major_to_matrix(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        num_steps = len(tokens) // self.num_codebooks
        return tokens.view(num_steps, self.num_codebooks).T.unsqueeze(0)

    def vector_codebook_major_to_matrix(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        num_steps = len(tokens) // self.num_codebooks
        return tokens.view(self.num_codebooks, num_steps).unsqueeze(0)

    def codebook_matrix_to_audio(self, audio_codes):
        audio_values = self.model.decode(audio_codes=audio_codes.to(
            self.device)).audio_values
        return audio_values[0].cpu().detach().numpy()

    def audio_array_to_audio_wav(self, audio_array, output_path):
        sf.write(output_path, audio_array, self.processor.sampling_rate)
        return output_path
```

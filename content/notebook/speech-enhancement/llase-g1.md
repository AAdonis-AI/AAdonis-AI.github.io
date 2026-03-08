---
title: "LlaSE-G1"
date: 2025-10-01
draft: false
tags: ["speech-enhancement", "llase", "inference"]
summary: "Setup and inference for LlaSE — a language model-based speech enhancement system that converts degraded audio to high-quality speech."
---

LlaSE is a state-of-the-art speech enhancement model based on a speech language model architecture. It treats enhancement as a conditional generation problem: given discrete tokens of degraded speech, the model generates discrete tokens of clean speech, which are then decoded to a waveform.

**Paper: [LlaSE](https://github.com/Kevin-naticl/LLaSE)**
**Codebase: [Github Repo](https://github.com/Kevin-naticl/LLaSE)**

## Setup

```bash
git clone https://github.com/Kevin-naticl/LLaSE.git
cd LLaSE

conda create -n LLaSE python=3.10
conda activate LLaSE
pip install -r requirements.txt

cd ckpt
bash download.sh
```

## Running Inference

The main script for inference is `inference.py`, which is configured via `./config/test.yml`.

**Step 1** — Edit `./config/test.yml`:
- Adjust `chunk` and `overlap` durations for your audio length
- Set `wav_dir` to the directory where output files will be saved:
  ```yaml
  wav_dir: /path/to/LLASE_outputs
  ```

**Step 2** — Create `filelist.txt` in the LLaSE directory. Each line is an absolute path to a degraded audio file:
```
/absolute/path/to/noisy_speech_1.wav
/absolute/path/to/noisy_speech_2.wav
/absolute/path/to/noisy_speech_3.wav
```
Set its location in the config:
```yaml
filename: filelist.txt
```

**Step 3** — Run inference:
```bash
bash inference.sh
```

The enhanced `.wav` files will be written to `wav_dir`.

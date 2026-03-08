---
title: "Noise & RIR Datasets"
date: 2025-10-02
draft: false
tags: ["datasets", "noise", "rir", "speech-enhancement"]
summary: "Downloading ESC-50, the OpenSLR RIR database, DEMAND, and VGG Sound for noise augmentation and speech enhancement training."
---

These datasets are used to synthesize degraded speech by convolving clean recordings with room impulse responses (RIRs) and mixing with environmental noise at varying SNRs.

## ESC-50

2,000 environmental audio recordings across 50 classes. Lightweight and commonly used.

**HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("ashraq/esc50")
```

**Kaggle:**
```python
import kagglehub
path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
print("Path:", path)
```

---

## Room Impulse Response and Noise Database (OpenSLR #28)

Large collection of real and simulated RIRs and noise recordings.

**Page:** [OpenSLR RIRs](https://www.openslr.org/28/)

```sh
wget https://openslr.trmal.net/resources/28/rirs_noises.zip
unzip rirs_noises.zip -d rirs_noises
```

---

## DEMAND

Diverse Environments Multichannel Acoustic Noise Database — 18 real noise environments.

```sh
TARGET_DIR="/path/to/your/datasets/demand"
mkdir -p "$TARGET_DIR"

curl -L -o "$TARGET_DIR/demand.zip" \
  https://www.kaggle.com/api/v1/datasets/download/chrisfilo/demand

unzip "$TARGET_DIR/demand.zip" -d "$TARGET_DIR"
rm "$TARGET_DIR/demand.zip"
```

---

## VGG Sound

200+ categories of audio-visual events extracted from YouTube. More than 200k 10-second clips.

**Kaggle:** [VGG Sound](https://www.kaggle.com/datasets/codebreaker619/vggsound/data)

```python
import kagglehub
path = kagglehub.dataset_download("codebreaker619/vggsound")
print("Path:", path)
```

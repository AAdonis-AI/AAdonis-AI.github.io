---
title: "Speech Datasets"
description: "Clean speech corpora, noise and RIR databases, and techniques for synthesizing degraded training data."
---

Data quality and diversity are the primary bottlenecks in audio ML. This section covers the datasets I've worked with directly, how to download and prepare them, and how to synthesize training data for degraded-speech tasks.

For speech enhancement in particular, clean data is plentiful (HiFiTTS, LibriSpeech, CommonVoice), but paired clean/noisy data is scarce. The standard solution is on-the-fly augmentation: convolve clean speech with room impulse responses (RIRs) and mix with noise recordings at random SNRs. The quality of this synthetic pipeline largely determines model performance.

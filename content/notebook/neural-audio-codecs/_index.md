---
title: "Neural Audio Codecs"
description: "Discrete neural audio codecs for speech compression and autoregressive modeling — DAC, SNAC, WavTokenizer, and X-Codec2."
---

Neural Audio Codecs (NACs) compress a waveform into a sequence of discrete tokens using a neural encoder and residual vector quantization (RVQ). These tokens are the bridge between raw audio and language models: once audio is tokenized, standard autoregressive Transformers can generate, enhance, or edit speech.

The key design tradeoffs are token rate (tokens/sec), number of codebooks, and reconstruction quality. High token rate (DAC: 801 tok/sec) means high quality but long sequences — expensive for autoregressive modeling. Low token rate (WavTokenizer: 40–75 tok/sec) means shorter sequences but potential quality loss, especially on degraded audio. SNAC and X-Codec2 hit a practical middle ground for most speech tasks.

Choosing the right codec significantly impacts model memory footprint, training speed, and output quality.

---
title: "Codec Comparison & Overview"
date: 2025-10-01
draft: false
tags: ["codecs", "audio", "tokenization"]
summary: "Side-by-side comparison of DAC, SNAC, WavTokenizer, and X-Codec2 — token rates, codebook counts, and when to use each."
---

## Comparison Table

| Codec | Flat Token Rate (tok/sec) | Codebooks | Framerate (Hz) | Vocab Size |
|-------|--------------------------|-----------|----------------|------------|
| **DAC** | 801 | 9 | 89 | 1024 |
| **SNAC 24kHz** | ~150 | 3 | varies | 4096 |
| **WavTokenizer** | 40 or 75 | 1 | — | 4096 |
| **X-Codec2** | ~50 | 1 | — | 65536 |

**Flat Token Rate** — codecs produce a (codebooks × timesteps) matrix. Flattened into a 1D sequence for autoregressive modeling, the total tokens/sec is `codebooks × framerate`.

**DAC** produces 9 codebooks at 89 frames/sec → 801 tok/sec. High quality, but sequences are ~10× longer than WavTokenizer for the same audio duration. Memory-intensive for LLM training.

**SNAC** matches DAC in perceptual quality with ~3–6× lower token rate. Best general-purpose choice for speech tasks.

**WavTokenizer** achieves extreme compression (40–75 tok/sec, single codebook). Excellent for clean TTS tasks. Performs poorly on degraded/noisy speech — it was not trained on such data and can introduce artifacts.

**X-Codec2** — stable at version 1.3.0. Single codebook, low token rate, large vocabulary. Pairs well with Llasa for TTS.

## When to Use Each

| Task | Recommended Codec |
|------|------------------|
| Speech enhancement (noisy input) | SNAC or DAC |
| TTS / zero-shot voice cloning | X-Codec2 or WavTokenizer |
| Speech editing | DAC (fine-grained control) |
| Autoregressive LM on long audio | WavTokenizer or X-Codec2 |
| Codec quality research | DAC (richest representation) |

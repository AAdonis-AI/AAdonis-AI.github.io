---
title: "Voice Conversion"
description: "Transforming a speaker's voice to match a target speaker while preserving linguistic content."
---

Voice conversion aims to transform the voice characteristics of one speaker to match those of another, while preserving the original linguistic content and prosody. The central challenge is cleanly disentangling speaker identity from speech content — a difficult problem because both are encoded together in the waveform.

Applications include speaker anonymization, dubbing, and data augmentation for training ASR and TTS systems. Key evaluation metrics are speaker similarity (SSIM), naturalness (NMOS/UTMOS), and intelligibility (WER). Zero-shot methods — which generalize to unseen speakers at inference time — are now the standard benchmark.

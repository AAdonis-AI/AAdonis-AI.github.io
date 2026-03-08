---
title: "Speech Alignment"
description: "Extracting precise word and phoneme timestamps from audio — tools, models, and common failure modes."
---

Forced alignment maps a known transcript to an audio recording, producing precise timestamps for each word and phoneme. It is a prerequisite for many audio tasks: building paired speech datasets, training TTS systems, dataset quality filtering, and speech editing (to locate the region to mask).

The main challenge is accuracy at phoneme boundaries — especially for fast or non-native speakers, and for stop consonants where the boundary is ambiguous. Another practical challenge on shared HPC clusters is that tools like MFA make frequent network calls, which can hit GitHub rate limits during large dataset processing runs.

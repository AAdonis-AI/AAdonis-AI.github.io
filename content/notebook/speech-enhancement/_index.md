---
title: "Speech Enhancement"
description: "Reconstructing clean speech from degraded audio — models, metrics, and inference setups."
---

Speech enhancement is the task of recovering a high-quality speech signal from a degraded input. Degradations include background noise, reverberation, packet loss, codec artifacts, and reduced sampling rate. The core challenge is learning a mapping from the degraded distribution to clean speech without overfitting to a specific noise type or level.

Recent state-of-the-art approaches use large language models over discrete audio tokens (language model-based speech enhancement), treating enhancement as a conditional generation problem rather than a regression task. This reframing allows leveraging perceptual reward signals and RL fine-tuning, which tends to outperform purely supervised training on standard metrics like DNSMOS, PESQ, and STOI.

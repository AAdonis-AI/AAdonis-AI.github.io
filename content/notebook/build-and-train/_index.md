---
title: "Build & Train Models"
description: "Training pipelines, tokenizers, and infrastructure for large-scale audio and language models."
---

Building and training models at scale requires careful engineering beyond the core ML ideas. This section covers the full stack: designing custom tokenizers that play well with HuggingFace, writing training scripts with the Trainer API, managing distributed runs on HPC clusters, and understanding the infrastructure choices that affect throughput and reproducibility.

Key challenges include tokenizer-model vocabulary alignment, memory management across nodes, handling variable-length audio token sequences, and setting up clean checkpoint/resume workflows. The notes here are drawn from real training runs on multi-GPU ETH cluster setups.

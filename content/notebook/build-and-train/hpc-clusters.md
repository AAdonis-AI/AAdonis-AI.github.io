---
title: "HPC Clusters"
date: 2025-10-01
draft: false
tags: ["hpc", "infrastructure"]
summary: "Redirecting pip, HuggingFace, and PyTorch caches to scratch storage to avoid filling your home directory on shared clusters."
---

When running large-scale training or inference on shared HPC clusters, your home directory quota fills up quickly from pip wheels, HuggingFace model checkpoints, and PyTorch caches. The fix is to redirect all caches to your scratch (net_scratch) partition before installing anything.

## Redirecting Caches to Scratch

```bash
# Base path for your scratch project
export SCRATCH_DIR=/path/to/net_scratch/my_project

# 1. Temporary directory for builds and unpacking wheels
export TMPDIR=$SCRATCH_DIR/tmp_build
mkdir -p $TMPDIR

# 2. PIP cache and build directories
export PIP_CACHE_DIR=$SCRATCH_DIR/pip_cache
mkdir -p $PIP_CACHE_DIR

# 3. Hugging Face cache (for transformers, datasets, etc.)
export HF_HOME=$SCRATCH_DIR/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# 4. PyTorch and other caches (optional)
export TORCH_HOME=$SCRATCH_DIR/torch_cache
mkdir -p $TORCH_HOME

# 5. Now install packages without caching to $HOME
pip install --no-cache-dir -r requirements.txt
```

Add these exports to your job submission script (`.sh` / `.sbatch`) so they apply consistently across runs. Alternatively, put them in `~/.bashrc` if you want them active in all interactive sessions.

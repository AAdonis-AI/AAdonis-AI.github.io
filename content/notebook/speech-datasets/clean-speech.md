---
title: "Clean Speech Datasets"
date: 2025-10-01
draft: false
tags: ["datasets", "speech"]
summary: "Downloading HiFiTTS-2 (high-quality 44kHz) and SparkAudio VoxBox (60k+ hours, multi-language merged corpus)."
---

## HiFiTTS-2

High-quality English speech dataset at 44kHz. Ideal for TTS and codec training.

More information: [HiFiTTS-2 on Hugging Face](https://huggingface.co/datasets/nvidia/hifitts-2)

```sh
mkdir -p ~/datasets/hifitts2
cd ~/datasets/hifitts2

# Download manifest and chapters (replace 44khz with 22khz for lower SR)
wget https://huggingface.co/datasets/nvidia/hifitts-2/resolve/main/44khz/manifest_44khz.json
wget https://huggingface.co/datasets/nvidia/hifitts-2/resolve/main/44khz/chapters_44khz.json
```

Then install [NeMo Speech Data Processor](https://github.com/NVIDIA/NeMo-speech-data-processor) and download the audio:

```sh
python /home/NeMo-speech-data-processor/main.py \
    --config-path="/home/NeMo-speech-data-processor/dataset_configs/english/hifitts2" \
    --config-name="config_44khz.yaml" \
    workspace_dir="/home/hifitts2" \
    max_workers=8
```

---

## SparkAudio VoxBox

A merged corpus of 60k+ hours of English and Chinese speech from CommonVoice, GigaSpeech, LibriSpeech, and others.

**HuggingFace:** [SparkAudio/voxbox](https://huggingface.co/datasets/SparkAudio/voxbox)
**GitHub:** [VoxBox](https://github.com/SparkAudio/VoxBox)

The script below downloads a specific subset (e.g. `casia`, `cremad`, `emns`) by name:

```python
"""
Download a voxbox dataset subset.

Usage:
    python download_voxbox_subset.py --subset casia
    python download_voxbox_subset.py --subset cremad --download_dir ./downloads
"""

import os
import argparse
from huggingface_hub import login, HfApi, hf_hub_download
from tqdm import tqdm


def download_voxbox_subset(subset_name, repo_id="SparkAudio/voxbox",
                            download_dir=None, hf_api_key=None):
    if download_dir is None:
        download_dir = os.path.join(os.environ.get('TMPDIR', './downloads'), 'voxbox_downloads')

    os.makedirs(download_dir, exist_ok=True)
    if hf_api_key:
        login(token=hf_api_key)

    api = HfApi()
    dataset_info = api.dataset_info(repo_id=repo_id)
    all_paths = [s.rfilename for s in dataset_info.siblings]

    downloaded_files = []

    # Download metadata
    metadata_path = f"metadata/{subset_name}.jsonl"
    if metadata_path in all_paths:
        hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=metadata_path, local_dir=download_dir,
            local_dir_use_symlinks=False, token=hf_api_key
        )
        downloaded_files.append(os.path.join(download_dir, metadata_path))
    else:
        print(f"Metadata not found: {metadata_path}")
        available = [p for p in all_paths if p.startswith("metadata/")]
        for m in available[:10]:
            print(f"  - {m}")

    # Download audio tar.gz files
    audio_tars = [f for f in all_paths
                  if f.startswith(f"audios/{subset_name}/") and f.endswith(".tar.gz")]
    if not audio_tars:
        print(f"No audio files found for subset '{subset_name}'")
        return downloaded_files

    for tar_file in tqdm(audio_tars, desc="Downloading audio"):
        hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=tar_file, local_dir=download_dir,
            local_dir_use_symlinks=False, token=hf_api_key
        )
        downloaded_files.append(os.path.join(download_dir, tar_file))

    print(f"Download complete: {len(downloaded_files)} files in {download_dir}")
    return downloaded_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--repo_id', type=str, default='SparkAudio/voxbox')
    parser.add_argument('--download_dir', type=str, default=None)
    parser.add_argument('--hf_api_key', type=str, default=None)
    args = parser.parse_args()

    if args.hf_api_key is None:
        args.hf_api_key = os.environ.get('HF_TOKEN')

    download_voxbox_subset(
        subset_name=args.subset,
        repo_id=args.repo_id,
        download_dir=args.download_dir,
        hf_api_key=args.hf_api_key
    )

if __name__ == "__main__":
    main()
```

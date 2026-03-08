---
title: "Seed-VC"
date: 2025-10-01
draft: false
tags: ["voice-conversion", "seed-vc", "diffusion"]
summary: "Zero-shot voice conversion with Seed-VC — diffusion transformer architecture, real-time inference, batch processing, and a comparison with EZ-VC."
---

Seed-VC is a zero-shot voice conversion framework using a diffusion transformer architecture. Key features:
- **External timbre shifter** during training perturbs source speech timbre, preventing leakage
- **Diffusion transformer** uses full reference speech context for fine-grained timbre capture
- Supports **real-time voice conversion** (~300ms algorithm delay)
- Supports **singing voice conversion** (V1) and **accent/emotion conversion** (V2)

**Paper: [Zero-shot Voice Conversion with Diffusion Transformers](https://arxiv.org/pdf/2411.09943)**
**Codebase: [Github](https://github.com/Plachtaa/seed-vc)**
**HuggingFace: [Model Checkpoints](https://huggingface.co/Plachta/Seed-VC)**

## Available Models

| Version | Name | Purpose | SR | Params |
|---------|------|---------|----|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | Voice Conversion | 22050 | 25M |
| v1.0 | seed-uvit-whisper-small-wavenet | Voice Conversion | 22050 | 98M |
| v1.0 | seed-uvit-whisper-base | Singing Voice Conversion | 44100 | 200M |
| v2.0 | hubert-bsqvae-small | Voice & Accent Conversion | 22050 | 67M+90M |

## Installation

```bash
git clone https://github.com/Plachtaa/seed-vc.git
cd seed-vc

conda create -n seedvc python=3.10
conda activate seedvc

pip install -r requirements.txt

# Optional: ~6x speedup on V2 models (Windows)
pip install triton-windows==3.2.0.post13
```

## Command Line Inference (V1)

```bash
python inference.py \
    --source <source-wav> \
    --target <reference-wav> \
    --output <output-dir> \
    --diffusion-steps 25 \
    --length-adjust 1.0 \
    --inference-cfg-rate 0.7 \
    --f0-condition False \
    --auto-f0-adjust False \
    --semi-tone-shift 0 \
    --fp16 True
```

**Parameter guide:**
- `diffusion-steps`: 25 default; 30–50 for best quality; 4–10 for fastest
- `length-adjust`: <1.0 speeds up, >1.0 slows down
- `f0-condition`: set `True` for singing voice conversion
- `semi-tone-shift`: pitch shift in semitones (SVC only)

## Command Line Inference (V2 — Accent/Emotion)

```bash
python inference_v2.py \
    --source <source-wav> \
    --target <reference-wav> \
    --output <output-dir> \
    --diffusion-steps 25 \
    --intelligibility-cfg-rate 0.7 \
    --similarity-cfg-rate 0.7 \
    --convert-style true \
    --anonymization-only false \
    --top-p 0.9 \
    --temperature 1.0
```

## Web UI Options

```bash
python app_vc.py --fp16 True      # V1 Voice Conversion
python app_svc.py --fp16 True     # Singing Voice Conversion
python app_vc_v2.py --compile     # V2 Model
python app.py --enable-v1 --enable-v2  # Integrated
```

## Real-Time Voice Conversion

```bash
python real-time-gui.py
```

Recommended settings (RTX 3060 Laptop GPU):

| Parameter | Value |
|-----------|-------|
| Diffusion Steps | 10 |
| Inference CFG Rate | 0.7 |
| Max Prompt Length | 3.0s |
| Block Time | 0.18s |
| Crossfade Length | 0.04s |
| Extra Context (left) | 2.5s |
| Extra Context (right) | 0.02s |
| **Latency** | ~430ms |

Use [VB-CABLE](https://vb-audio.com/Cable/) to route GUI output to a virtual microphone.

## Batch Voice Conversion Script

Converts utterances from an input directory using random reference voices — useful for corpus augmentation and speaker anonymization.

```python
"""
Batch Seed-VC augmentation script.
Usage:
    python batch_seedvc_augment.py \
        --seedvc-root ../seed-vc \
        --ref-dir /path/to/reference/voices \
        --input-dir /path/to/input/utterances \
        --output-dir /path/to/output \
        --diffusion-steps 30 \
        --recursive --skip-existing
"""

from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
from typing import Iterable, Sequence

def _iter_audio_files(root: Path, recursive: bool, exts: Sequence[str]) -> Iterable[Path]:
    exts_lc = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    if root.is_file():
        if root.suffix.lower() in exts_lc:
            yield root
        return
    if not root.exists():
        return
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_lc:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts_lc:
                yield p

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Seed-VC corpus augmentation")
    parser.add_argument("--seedvc-root", type=str, default="../seed-vc")
    parser.add_argument("--ref-dir", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--ext", type=str, default="wav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", action="store_true")
    parser.add_argument("--suffix", type=str, default="_seedvc")
    parser.add_argument("--manifest", type=str, default="manifest_seedvc.jsonl")
    args = parser.parse_args()

    seedvc_root = Path(args.seedvc_root).resolve()
    sys.path.insert(0, str(seedvc_root))
    os.environ.setdefault("HF_HUB_CACHE", str(seedvc_root / "checkpoints" / "hf_cache"))

    from seed_vc_wrapper import SeedVCWrapper
    import soundfile as sf

    ref_files = sorted(_iter_audio_files(Path(args.ref_dir), args.recursive, args.ext.split(",")))
    input_files = sorted(_iter_audio_files(Path(args.input_dir), args.recursive, args.ext.split(",")))

    if args.limit > 0:
        input_files = input_files[:args.limit]

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    wrapper = SeedVCWrapper(device=device, load_f0_model=bool(args.f0_condition))

    with (output_dir / args.manifest).open("a", encoding="utf-8") as mf:
        for idx, inp in enumerate(input_files, start=1):
            out_path = output_dir / f"{inp.stem}{args.suffix}.wav"
            if args.skip_existing and out_path.exists():
                continue
            ref = random.choice(ref_files)
            try:
                sr, audio = wrapper.convert_voice_npy(
                    source=str(inp), target=str(ref),
                    diffusion_steps=args.diffusion_steps,
                    length_adjust=args.length_adjust,
                    inference_cfg_rate=args.inference_cfg_rate,
                    f0_condition=bool(args.f0_condition),
                )
                sf.write(str(out_path), audio, sr)
                rec = {"source": str(inp), "reference": str(ref), "output": str(out_path)}
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                if idx % 10 == 0:
                    print(f"[{idx}/{len(input_files)}] Processed")
            except Exception as e:
                print(f"FAILED {inp.name}: {e}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

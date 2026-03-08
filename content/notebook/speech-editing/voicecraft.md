---
title: "VoiceCraft"
date: 2025-10-01
draft: false
tags: ["speech-editing", "voicecraft", "mfa"]
summary: "Token-based speech editing with VoiceCraft — setup, MFA forced alignment, and a self-contained editing script for insertions, deletions, and substitutions."
---

VoiceCraft edits speech by masking a region of the audio (at the token level), then autoregressively infilling the masked region conditioned on the surrounding context and a new target transcript. It requires forced alignment (MFA) to determine where in the audio the edit should happen.

**Codebase: [GitHub](https://github.com/jasonppy/VoiceCraft)**

## 1. Setup and Dependencies

First install [espeak-ng](../speech-alignment/espeak-ng) (required by the text tokenizer).

```bash
mkdir -p voicecraft_dir
cd voicecraft_dir
git clone https://github.com/jasonppy/VoiceCraft.git

conda create -n voicecraft python==3.9.16
CONDA_ENVIRONMENT=/path/to/conda_envs/voicecraft
conda activate ${CONDA_ENVIRONMENT}

export TMPDIR=/path/to/voicecraft_dir/tmp
export PIP_CACHE_DIR=/path/to/voicecraft_dir/pip_cache
export HF_HOME=path/to/voicecraft_dir/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=/path/to/voicecraft_dir/torch
mkdir -p $TMPDIR $PIP_CACHE_DIR $HF_HOME $TRANSFORMERS_CACHE $TORCH_HOME

# Conda dependencies (MFA + Kaldi)
conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 joblib=1.2.0

# Download MFA models/dictionary
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa

# Pip dependencies
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft \
    --no-cache-dir --cache-dir $PIP_CACHE_DIR

pip install xformers==0.0.22 torchaudio==2.0.2 torch==2.0.1 tensorboard==2.16.2 \
    phonemizer==3.2.1 datasets==2.16.0 torchmetrics==0.11.1 \
    huggingface_hub==0.22.2 py-espeak-ng soundfile pyflac pyvorbis lxml \
    gradio==3.50.2 nltk>=3.8.1 openai-whisper>=20231117 num2words==0.5.13 \
    --no-cache-dir --cache-dir $PIP_CACHE_DIR || echo "Some optional packages skipped."
```

## 2. Single-File Editing Script

Store this as `single_voicecraft_edit.py` inside the cloned VoiceCraft repo.

```python
"""
Single Audio Speech Editing with VoiceCraft
Edit audio by specifying audio_path, original/target transcripts and edit type.
"""
import argparse, logging, os, random, pickle
import numpy as np
import torch, torchaudio
from data.tokenizer import AudioTokenizer, TextTokenizer, tokenize_text, tokenize_audio
from models import voicecraft

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_word_spans(orig, target, edit_type):
    """Find word indices that differ between transcripts"""
    orig_words, target_words = orig.split(), target.split()
    if edit_type == "deletion":
        diff = len(orig_words) - len(target_words)
        for i, (o, t) in enumerate(zip(orig_words, target_words)):
            if o != t: return (i, i + diff - 1)
        return (len(target_words), len(orig_words) - 1)
    elif edit_type == "insertion":
        diff = len(target_words) - len(orig_words)
        for i, (o, t) in enumerate(zip(orig_words, target_words)):
            if o != t: return (max(0, i-1), i)
        return (len(orig_words) - 1, len(orig_words))
    else:  # substitution
        start = next(i for i, (o, t) in enumerate(zip(orig_words, target_words)) if o != t)
        end = next(i for i in range(len(orig_words)-1, -1, -1)
                   if i < len(target_words) and orig_words[i] != target_words[i])
        return (start, end)

def get_mask_interval(alignment_csv, word_span, edit_type, left_margin=0.08, right_margin=0.08):
    with open(alignment_csv) as f:
        words = [l.strip().split(",") for l in f.readlines()[1:] if "words" in l]
    start_idx, end_idx = word_span
    start_time = float(words[end_idx][1] if edit_type == 'insertion' else words[start_idx][0])
    end_time = float(words[end_idx][1])
    return (max(start_time - left_margin, 0), end_time + right_margin)

def run_mfa(audio_path, transcript, temp_dir, beam=100, retry_beam=400):
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    import shutil
    shutil.copy(audio_path, os.path.join(temp_dir, f"{filename}.wav"))
    with open(os.path.join(temp_dir, f"{filename}.txt"), "w") as f:
        f.write(transcript)
    align_out = os.path.join(temp_dir, "mfa_alignments")
    csv_path = os.path.join(align_out, f"{filename}.csv")
    if not os.path.isfile(csv_path):
        cmd = f"mfa align -v --clean -j 1 --output_format csv {temp_dir} english_us_arpa english_us_arpa {align_out} --beam {beam} --retry_beam {retry_beam}"
        os.system(cmd)
    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Edit single audio with VoiceCraft")
    parser.add_argument("--audio_path", type=str, default="../original_audio.wav")
    parser.add_argument("--orig_transcript", type=str, required=True)
    parser.add_argument("--target_transcript", type=str, required=True)
    parser.add_argument("--edit_type", type=str, default="substitution",
                        choices=["insertion", "deletion", "substitution"])
    parser.add_argument("--model_name", type=str, default="giga330M",
                        choices=["giga330M", "giga830M"])
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--temp_dir", type=str, default="./temp")
    parser.add_argument("--codec_audio_sr", type=int, default=16000)
    parser.add_argument("--codec_sr", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_repetition", type=int, default=2)
    parser.add_argument("--kvcache", type=int, default=1)
    parser.add_argument("--silence_tokens", type=str, default="[1388,1898,131]")
    parser.add_argument("--left_margin", type=float, default=0.08)
    parser.add_argument("--right_margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=100)
    parser.add_argument("--retry_beam_size", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Loading model on {args.device}...")
    if args.exp_dir:
        with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
            model_args = pickle.load(f)
        model = voicecraft.VoiceCraft(model_args)
        ckpt = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"), map_location='cpu')
        phn2num = ckpt['phn2num']
        model.load_state_dict(ckpt['model'])
    else:
        model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{args.model_name}")
        model_args, phn2num = model.args, model_args.phn2num
    model.to(args.device).eval()

    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.makedirs("./pretrained_models", exist_ok=True)
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O {encodec_fn}")

    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=args.device)
    text_tokenizer = TextTokenizer(backend="espeak")

    info = torchaudio.info(args.audio_path)
    audio_dur = info.num_frames / info.sample_rate

    logging.info("Running MFA alignment...")
    csv = run_mfa(args.audio_path, args.orig_transcript, args.temp_dir,
                  args.beam_size, args.retry_beam_size)
    if not os.path.exists(csv):
        logging.error("Alignment failed!")
        return

    word_span = get_word_spans(args.orig_transcript, args.target_transcript, args.edit_type)
    mask_interval = get_mask_interval(csv, word_span, args.edit_type,
                                      args.left_margin, args.right_margin)
    mask_interval = (max(mask_interval[0], 1/args.codec_sr), min(mask_interval[1], audio_dur))
    mask_frames = torch.LongTensor([[round(mask_interval[0]*args.codec_sr),
                                     round(mask_interval[1]*args.codec_sr)]]).unsqueeze(0)

    logging.info(f"Mask interval: {mask_interval[0]:.3f}s - {mask_interval[1]:.3f}s")

    text_tokens = torch.LongTensor([phn2num[p] for p in tokenize_text(
        text_tokenizer, args.target_transcript.strip()) if p in phn2num]).unsqueeze(0)
    text_lens = torch.LongTensor([text_tokens.shape[-1]])
    orig_audio = tokenize_audio(audio_tokenizer, args.audio_path)[0][0].transpose(2, 1)

    logging.info("Running inference...")
    silence_toks = eval(args.silence_tokens) if isinstance(args.silence_tokens, str) else args.silence_tokens
    with torch.no_grad():
        gen_audio = model.inference(
            text_tokens.to(args.device), text_lens.to(args.device),
            orig_audio[..., :model_args.n_codebooks].to(args.device),
            mask_interval=mask_frames.to(args.device),
            top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
            stop_repetition=args.stop_repetition, kvcache=args.kvcache,
            silence_tokens=silence_toks
        )

    gen_sample = audio_tokenizer.decode([(gen_audio, None)])[0].cpu()
    orig_sample = audio_tokenizer.decode([(orig_audio.transpose(2, 1), None)])[0].cpu()

    base = os.path.splitext(os.path.basename(args.audio_path))[0]
    edited_path = os.path.join(args.output_dir, f"{base}_edited_seed{args.seed}.wav")
    recon_path = os.path.join(args.output_dir, f"{base}_reconstructed.wav")

    torchaudio.save(edited_path, gen_sample, args.codec_audio_sr)
    torchaudio.save(recon_path, orig_sample, args.codec_audio_sr)

    print(f"\n{'='*60}\nEDIT SUMMARY\n{'='*60}")
    print(f"Original:  {args.orig_transcript}\nTarget:    {args.target_transcript}")
    print(f"Edit type: {args.edit_type} | Word span: {word_span}")
    print(f"Time mask: {mask_interval[0]:.3f}s - {mask_interval[1]:.3f}s")
    print(f"Edited:       {edited_path}")
    print(f"Reconstructed: {recon_path}")

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
python single_voicecraft_edit.py \
    --audio_path ../original_audio.wav \
    --orig_transcript "what struck into me the deepest was the look of nearly everyone of the judges" \
    --target_transcript "what struck into me the deepest was the look of nearly all people present" \
    --edit_type substitution \
    --model_name giga330M \
    --seed 1
```

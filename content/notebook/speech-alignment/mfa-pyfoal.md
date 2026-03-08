---
title: "Montreal Forced Aligner (MFA)"
date: 2025-10-02
draft: false
tags: ["alignment", "mfa", "pyfoal", "phonemes"]
summary: "Word and phoneme-level forced alignment with MFA and pyfoal — installation, inference script, and a fix for GitHub rate limit errors."
---

The Montreal Forced Aligner (MFA) aligns spoken audio with its transcript at word and phoneme level. It is a prerequisite for speech editing (to locate edit boundaries), dataset preparation, and TTS training.

**pyfoal:** [Github](https://github.com/maxrmorrison/pyfoal)
**MFA:** [Github](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
**MFA docs:** [readthedocs](http://montreal-forced-aligner.readthedocs.io/)

## Installation

```bash
conda install -c conda-forge montreal-forced-aligner
pip install pyfoal
```

## Inference Script

```python
import os
import torch
import librosa
import pyfoal

os.environ['MFA_ROOT_DIR'] = '/path/to/mfa_temp'
os.makedirs(os.environ['MFA_ROOT_DIR'], exist_ok=True)


def align_audio_text(audio_path, text):
    """
    Align audio with text using MFA.
    Returns a list of dicts with word, phonemes, start, end, is_silence.
    """
    audio_np, sr = librosa.load(audio_path, sr=16000)
    audio = torch.FloatTensor(audio_np).unsqueeze(0)

    alignment = pyfoal.from_text_and_audio(
        text, audio, 16000, aligner='mfa', gpu=0
    )

    results = []
    for mfa_word in alignment.words():
        word_text = mfa_word.word
        phonemes = [p.phoneme for p in mfa_word.phonemes]

        if mfa_word.phonemes:
            start = mfa_word.phonemes[0]._start
            end = mfa_word.phonemes[-1]._end
        else:
            start = end = 0.0

        is_silence = (word_text.strip() == '' or len(phonemes) == 0)

        results.append({
            'word': word_text,
            'phonemes': phonemes,
            'start': start,
            'end': end,
            'is_silence': is_silence
        })

    return results


if __name__ == "__main__":
    audio_path = "test_audio.wav"
    text = "This is a transcript of the test audio."

    results = align_audio_text(audio_path, text)

    for i, w in enumerate(results):
        if w['is_silence']:
            print(f"{i:2d}. [SILENCE] [{w['start']:.3f}s - {w['end']:.3f}s]")
        else:
            print(f"{i:2d}. '{w['word']:10s}' [{w['start']:.3f}s - {w['end']:.3f}s]  {w['phonemes']}")
```

## Common Problem: GitHub Rate Limit

**Problem:** `pyfoal` checks GitHub for MFA model files on *every* call, hitting the 60 requests/hour rate limit quickly during bulk processing.

**Fix:** Edit the pyfoal source to check for local files first.

**File:** `path/to/conda_env/lib/python3.10/site-packages/pyfoal/baselines/mfa.py`

Backup first:
```bash
cp path/to/pyfoal/baselines/mfa.py path/to/pyfoal/baselines/mfa.py.backup
```

Change lines 64–67 **from:**
```python
manager = mfa.models.ModelManager()
manager.download_model('dictionary', 'english_mfa')
manager.download_model('acoustic', 'english_mfa')
```

**To:**
```python
import os
manager = mfa.models.ModelManager()
mfa_root = os.environ.get('MFA_ROOT_DIR', os.path.expanduser('~/Documents/MFA'))
dict_path = os.path.join(mfa_root, 'pretrained_models/dictionary/english_mfa.dict')
acoustic_path = os.path.join(mfa_root, 'pretrained_models/acoustic/english_mfa.zip')

if os.path.exists(dict_path):
    print(f"[MFA] Dictionary found locally")
else:
    print(f"[MFA] Downloading dictionary from GitHub")
    manager.download_model('dictionary', 'english_mfa')

if os.path.exists(acoustic_path):
    print(f"[MFA] Acoustic model found locally")
else:
    print(f"[MFA] Downloading acoustic model from GitHub")
    manager.download_model('acoustic', 'english_mfa')
```

This has been tested and eliminates the rate limit problem entirely.

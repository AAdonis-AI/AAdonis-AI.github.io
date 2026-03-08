---
title: "Custom Tokenizer for Audio LLMs"
date: 2025-10-02
draft: false
tags: ["tokenizer", "huggingface", "audio"]
summary: "Building a HuggingFace-compatible tokenizer for neural audio codec tokens — vocabulary design, special tokens, and uploading to the Hub."
---

When working with Neural Audio Codecs (NACs) and Transformer-based models, you need a custom tokenizer that maps discrete codec tokens to integer IDs, handles special tokens (BOS, EOS, PAD), and is fully compatible with the HuggingFace `AutoTokenizer` API so you can upload it to the Hub and reuse it across projects.

The example below is for the 22kHz SNAC codec (3 codebooks, 4096 vocab size per codebook), but the pattern generalizes to any NAC.

**Key design decision:** NAC tokens are set as special tokens so the tokenizer treats them atomically. The standard `.decode()` and `.batch_decode()` skip special tokens, so we override them with custom functions that preserve NAC tokens.

## Building the Tokenizer

```python
import os
import re
import json
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer

# Adapt for your NAC
num_codebooks = 3
codebook_size = 4096
special_tokens = ["<|bos|>", "<|pad|>", "<|eos|>", "<|start_clean|>", "<|unk|>"]
codebook_tokens = [f"<|q{i}_t{a}|>" for i in range(num_codebooks) for a in range(codebook_size)]

vocab_tokens = special_tokens + codebook_tokens
token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}

os.makedirs("./snac_tokenizer_hf", exist_ok=True)

hf_tokenizer = Tokenizer(models.WordLevel(vocab=token_to_id, unk_token="<|unk|>"))
hf_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])

# Pre-tokenizer: match all tokens
all_token_pattern = "|".join(re.escape(token) for token in vocab_tokens)
hf_tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=all_token_pattern, behavior="isolated")

# Add BOS/EOS post-processing
hf_tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    special_tokens=[("<|bos|>", 0), ("<|eos|>", 2)]
)

hf_tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " ")])

hf_tokenizer.save("./snac_tokenizer_hf/tokenizer.json")

# Create tokenizer config
snac_tokens_only = [t for t in vocab_tokens if t not in special_tokens]

tokenizer_config = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "auto_map": {"AutoTokenizer": ["tokenizer.json", None]},
    "model_max_length": 8192,
    "padding_side": "right",
    "truncation_side": "right",
    "clean_up_tokenization_spaces": True,
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "additional_special_tokens": ["<|start_clean|>"] + snac_tokens_only
}

with open("./snac_tokenizer_hf/tokenizer_config.json", "w") as f:
    json.dump(tokenizer_config, f, indent=2)
```

## Uploading to Hugging Face Hub

```python
HF_API_KEY = "hf_YOUR_TOKEN_HERE"
login(token=HF_API_KEY)
api = HfApi()

api.create_repo("YOUR_HF_NAME/snac_tokenizer", token=HF_API_KEY, exist_ok=True)
api.upload_folder(
    folder_path="./snac_tokenizer_hf",
    repo_id="YOUR_HF_NAME/snac_tokenizer",
    token=HF_API_KEY
)
print("Tokenizer uploaded successfully!")
```

## Loading and Patching decode()

```python
hf_tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_NAME/snac_tokenizer")
basic_special_tokens = ["<|bos|>", "<|pad|>", "<|eos|>", "<|unk|>"]

original_decode = hf_tokenizer.decode
original_batch_decode = hf_tokenizer.batch_decode

def custom_decode(token_ids, skip_special_tokens=True, **kwargs):
    if skip_special_tokens:
        filtered_ids = [
            t_id for t_id in token_ids
            if hf_tokenizer.convert_ids_to_tokens(t_id) not in basic_special_tokens
        ]
        return original_decode(filtered_ids, skip_special_tokens=False, **kwargs)
    return original_decode(token_ids, skip_special_tokens=False, **kwargs)

def custom_batch_decode(token_ids_list, skip_special_tokens=True, **kwargs):
    results = []
    for ids in token_ids_list:
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        results.append(custom_decode(ids, skip_special_tokens=skip_special_tokens, **kwargs))
    return results

hf_tokenizer.decode = custom_decode
hf_tokenizer.batch_decode = custom_batch_decode

print("Tokenizer ready with custom decode functions.")
```

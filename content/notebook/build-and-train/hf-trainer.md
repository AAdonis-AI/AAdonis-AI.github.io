---
title: "HuggingFace Trainer: Custom LLaMA Training"
date: 2025-10-03
draft: false
tags: ["training", "llama", "huggingface", "distributed"]
summary: "A complete training script for LLaMA (250M–8B) with a custom tokenizer, label masking, gradient checkpointing, and multi-GPU support via Accelerate."
---

This training script is designed for training audio language models from scratch or from a checkpoint. It handles the full setup: loading a custom HuggingFace tokenizer, building a LLaMA model of any size, masking labels up to a delimiter token (useful for conditional generation), and running distributed training with `accelerate`.

**What it does:**
1. Loads any HuggingFace-compatible tokenizer and adapts the model vocabulary size automatically
2. Loads training/test datasets from HuggingFace Hub (requires a `"sequence"` column)
3. Masks all labels before (and including) a delimiter token (e.g. `<|start_clean|>`) with `-100`
4. Initializes a LLaMA model (250M–8B) with Flash Attention 2
5. Runs with `accelerate launch --num_processes N --mixed_precision bf16`
6. Supports checkpoint resuming with automatic vocab size adaptation

**Run with:**
```bash
accelerate launch --num_processes 4 --mixed_precision bf16 train.py \
    --model_size 1B \
    --run_name my_run \
    --tokenizer_path YOUR_HF_NAME/snac_tokenizer \
    --train_dataset YOUR_HF_NAME/train_data \
    --test_dataset YOUR_HF_NAME/test_data
```

## Full Training Script

```python
import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback, AutoTokenizer
from datasets import load_dataset
import wandb
import os, sys, datetime, argparse, gc, random
import torch.distributed as dist

# ===========================
# Argument Parsing
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMA with Custom Tokenizer")
    parser.add_argument("--model_size", type=str, choices=["250M", "1B", "2B", "4B", "7B", "8B"], default="1B")
    parser.add_argument("--run_name", type=str, default="custom_tokenizer_v1")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delimiter_token", type=str, default="<|start_clean|>")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--disable_audio_callback", action="store_true")
    return parser.parse_args()

args = parse_args()
MODEL_SIZE = args.model_size
RUN_NAME = args.run_name

rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPUs available!")
    sys.exit(1)

# ===========================
# Load Tokenizer
# ===========================
print(f"Loading tokenizer from: {args.tokenizer_path}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
vocab_size = tokenizer.vocab_size

# Get delimiter token ID
if hasattr(tokenizer, 'convert_tokens_to_ids'):
    delimiter_token_id = tokenizer.convert_tokens_to_ids(args.delimiter_token)
    if delimiter_token_id == tokenizer.unk_token_id:
        print(f"ERROR: Delimiter token '{args.delimiter_token}' not found in tokenizer!")
        sys.exit(1)

print(f"Tokenizer vocab size: {vocab_size}, delimiter token ID: {delimiter_token_id}")

# ===========================
# Dataset
# ===========================
class CustomDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if isinstance(seq, str):
            return torch.tensor(tokenizer.encode(seq, add_special_tokens=True), dtype=torch.long)
        return torch.tensor(seq, dtype=torch.long)


def create_labels_with_delimiter(batch, delimiter_token_id, pad_token_id):
    """Mask labels before delimiter token with -100"""
    if not batch:
        return {"input_ids": torch.tensor([]), "labels": torch.tensor([]), "attention_mask": torch.tensor([])}

    input_sequences, label_sequences = [], []
    for sequence in batch:
        sequence = torch.as_tensor(sequence, dtype=torch.long)
        delimiter_positions = (sequence == delimiter_token_id).nonzero(as_tuple=True)[0]

        input_sequences.append(sequence)
        labels = sequence.clone()
        if len(delimiter_positions) > 0:
            labels[:delimiter_positions[0].item() + 1] = -100
        label_sequences.append(labels)

    max_len = max(len(x) for x in input_sequences)
    batch_size = len(input_sequences)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_sequences, label_sequences)):
        input_ids[i, :len(inp)] = inp
        labels[i, :len(lab)] = lab
        attention_mask[i, :len(inp)] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_fn(batch):
    return create_labels_with_delimiter(batch, delimiter_token_id, tokenizer.pad_token_id)


# ===========================
# Load Datasets
# ===========================
print("Loading datasets...")

def get_dataset_split(dataset, split_name=None):
    if isinstance(dataset, dict):
        if split_name and split_name in dataset:
            return dataset[split_name]
        return list(dataset.values())[0]
    return dataset

train_data = get_dataset_split(load_dataset(args.train_dataset), 'train')
test_data = get_dataset_split(load_dataset(args.test_dataset),
                              'test' if 'test' in load_dataset(args.test_dataset) else None)

if "sequence" not in train_data.column_names:
    print("ERROR: Dataset must have 'sequence' column!")
    sys.exit(1)

train_sequences = train_data["sequence"]
test_sequences = test_data["sequence"]

random.seed(args.seed)
val_sequences = random.sample(test_sequences, min(args.eval_samples, len(test_sequences))) \
                if len(test_sequences) > args.eval_samples else test_sequences

print(f"Training: {len(train_sequences):,} sequences, Evaluation: {len(val_sequences):,} sequences")

train_dataset = CustomDataset(train_sequences, tokenizer)
val_dataset = CustomDataset(val_sequences, tokenizer)
first_10_samples = val_sequences[:10]

# ===========================
# Model Configuration
# ===========================
def get_model_config(model_size, vocab_size, tokenizer):
    configs = {
        "250M": {"hidden_size": 1024, "intermediate_size": 4096,  "num_hidden_layers": 20, "num_attention_heads": 16, "num_key_value_heads": 16},
        "1B":   {"hidden_size": 1536, "intermediate_size": 6144,  "num_hidden_layers": 24, "num_attention_heads": 24, "num_key_value_heads": 24},
        "2B":   {"hidden_size": 2048, "intermediate_size": 8192,  "num_hidden_layers": 28, "num_attention_heads": 32, "num_key_value_heads": 32},
        "4B":   {"hidden_size": 2816, "intermediate_size": 11264, "num_hidden_layers": 32, "num_attention_heads": 44, "num_key_value_heads": 44},
        "7B":   {"hidden_size": 3456, "intermediate_size": 13824, "num_hidden_layers": 36, "num_attention_heads": 54, "num_key_value_heads": 54},
        "8B":   {"hidden_size": 3584, "intermediate_size": 14336, "num_hidden_layers": 40, "num_attention_heads": 56, "num_key_value_heads": 56},
    }
    params = configs[model_size]
    return LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=100000.0,
        attention_bias=False,
        attention_dropout=0.1,
        hidden_act="silu",
        hidden_dropout_prob=0.3,
        initializer_range=0.005,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        **params
    )

# ===========================
# Initialize Model
# ===========================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.checkpoint:
    print(f"Loading from checkpoint: {args.checkpoint}")
    loaded_config = LlamaConfig.from_pretrained(args.checkpoint, attn_implementation="flash_attention_2")
    checkpoint_vocab_size = loaded_config.vocab_size
    if checkpoint_vocab_size != vocab_size:
        print(f"Resizing vocab: {checkpoint_vocab_size} -> {vocab_size}")
        loaded_config.vocab_size = vocab_size
    loaded_config.pad_token_id = tokenizer.pad_token_id
    loaded_config.bos_token_id = tokenizer.bos_token_id
    loaded_config.eos_token_id = tokenizer.eos_token_id
    base_model = LlamaForCausalLM.from_pretrained(args.checkpoint, config=loaded_config,
                                                   device_map="cpu", torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True)
    model = LlamaForCausalLM(loaded_config)
    model.load_state_dict(base_model.state_dict(), strict=False)
    if checkpoint_vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)
    del base_model
    gc.collect()
    model = model.to(dtype=torch.bfloat16)
else:
    print("Initializing new model")
    config = get_model_config(MODEL_SIZE, vocab_size, tokenizer)
    model = LlamaForCausalLM(config)
    model = model.to(dtype=torch.bfloat16, device='cpu')
    if model.config.vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)

# ===========================
# Training Setup
# ===========================
if rank == 0:
    wandb.init(project=f"Custom-Tokenizer-{RUN_NAME}",
               name=f"{RUN_NAME}_{MODEL_SIZE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * num_gpus
steps_per_epoch = len(train_sequences) // effective_batch_size if len(train_sequences) > 0 else 0
max_steps = max(steps_per_epoch, 100) if args.max_steps == -1 else args.max_steps

training_args = TrainingArguments(
    output_dir=f"./{MODEL_SIZE}_results_{RUN_NAME}",
    max_steps=max_steps,
    per_device_train_batch_size=args.per_device_batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=5,
    learning_rate=args.learning_rate,
    warmup_steps=int(max_steps * 0.02),
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    report_to=["wandb"],
    save_strategy="steps",
    bf16=True,
    save_safetensors=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_drop_last=True,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    local_rank=local_rank,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    label_names=["labels"],
    seed=args.seed,
    data_seed=args.seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()
print("Training completed!")

# ===========================
# Save Final Model
# ===========================
if rank == 0:
    final_model_path = f"./final_model_{MODEL_SIZE}_{RUN_NAME}"
    model.save_pretrained(final_model_path, safe_serialization=True)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if world_size > 1 and dist.is_initialized():
    dist.barrier()

print("Done!")
```

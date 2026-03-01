"""
train.py — Mistral RTL Engineer Fine-Tuning
QLoRA (4-bit) fine-tuning of Mistral-7B for Verilog/RTL specialization.
Hardware target: RTX 4050 6GB VRAM
Compatible with: transformers>=5.0, peft>=0.8, trl>=0.8
"""

import os
import json
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from typing import Dict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Pre-quantized 4-bit Mistral — only ~4GB download vs 14.5GB for raw weights
# Functionally identical to mistralai/Mistral-7B-v0.1 with NF4 quantization
MODEL_NAME      = "unsloth/mistral-7b-v0.3-bnb-4bit"
DATA_PATH       = "data/train.json"
OUTPUT_DIR      = "outputs/mistral-rtl"
MAX_SEQ_LEN     = 512
BATCH_SIZE      = 1
GRAD_ACCUM      = 8       # Effective batch = 8
LR              = 2e-4
EPOCHS          = 3
LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
WANDB_PROJECT   = "mistral-rtl-engineer"
WANDB_API_KEY   = os.environ.get("WANDB_API_KEY", None)  # Set this env var as alternative to wandb login
SEED            = 42

torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# W&B INIT (with offline fallback)
# ─────────────────────────────────────────────
WANDB_MODE = "online"
try:
    # Explicit login — works even if netrc key is stale
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY, relogin=True)
    else:
        wandb.login(relogin=False)  # Uses stored netrc key

    wandb.init(
        project=WANDB_PROJECT,
        name="qlora-rtl-run-01",
        config={
            "model":        MODEL_NAME,
            "lora_rank":    LORA_RANK,
            "lora_alpha":   LORA_ALPHA,
            "max_seq_len":  MAX_SEQ_LEN,
            "epochs":       EPOCHS,
            "lr":           LR,
            "batch_size":   BATCH_SIZE,
            "grad_accum":   GRAD_ACCUM,
            "quantization": "4-bit NF4",
        },
        tags=["qlora", "mistral", "rtl", "verilog"],
    )
    print("[INFO] W&B initialized successfully.")
except Exception as e:
    print(f"[WARNING] W&B login failed ({e}). Falling back to OFFLINE mode.")
    print("[WARNING] Training will still run. Logs saved locally under ./wandb/")
    os.environ["WANDB_MODE"] = "offline"
    WANDB_MODE = "offline"
    wandb.init(
        project=WANDB_PROJECT,
        name="qlora-rtl-run-01-offline",
        mode="offline",
        config={
            "model":        MODEL_NAME,
            "lora_rank":    LORA_RANK,
            "lora_alpha":   LORA_ALPHA,
            "max_seq_len":  MAX_SEQ_LEN,
            "epochs":       EPOCHS,
            "lr":           LR,
            "batch_size":   BATCH_SIZE,
            "grad_accum":   GRAD_ACCUM,
            "quantization": "4-bit NF4",
        },
    )
    print(f"[INFO] W&B offline run created. Sync later with: wandb sync wandb/")

# ─────────────────────────────────────────────
# 4-BIT QUANTIZATION CONFIG (bitsandbytes)
# ─────────────────────────────────────────────
# Model is already 4-bit quantized — no BitsAndBytesConfig needed at load time
# (kept here as reference for raw model loading)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ─────────────────────────────────────────────
# LOAD MODEL + TOKENIZER
# ─────────────────────────────────────────────
# unsloth/mistral-7b-v0.3-bnb-4bit is ALREADY stored as 4-bit NF4.
# Do NOT pass quantization_config — it causes a hang/conflict.
# The model loads directly into bitsandbytes 4-bit format automatically.
print(f"[INFO] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token   # Mistral has no pad token
tokenizer.padding_side = "right"               # Pad right for causal LM

print(f"[INFO] Loading pre-quantized 4-bit model (fast, ~4GB) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # NO quantization_config here — model is already 4-bit quantized
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,   # stream weights to GPU, avoids RAM spike
)

# Required step before LoRA: prepares the k-bit model
model = prepare_model_for_kbit_training(model)

# ─────────────────────────────────────────────
# LORA CONFIG
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    # Targeting attention + MLP projection layers for best RTL task absorption
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# DATASET LOADING + FORMATTING
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

def load_dataset_from_json(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# SFTTrainer handles tokenization internally — no manual tokenize() needed

raw_dataset = load_dataset_from_json(DATA_PATH)

# 90/10 train/eval split
split    = raw_dataset.train_test_split(test_size=0.1, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"[INFO] Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

# ─────────────────────────────────────────────
# TRAINING ARGUMENTS (via SFTConfig)
# ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,            # Saves ~30% VRAM, slight speed cost
    optim="paged_adamw_8bit",               # 8-bit paged optimizer — critical for 6GB
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                              # BFloat16 — matches unsloth model dtype
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=9,
    save_strategy="steps",
    save_steps=9,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="qlora-rtl-run-01",
    dataloader_pin_memory=False,            # Saves RAM on 16GB systems
    tf32=False,                             # Not available on RTX 4050
    seed=SEED,
    max_length=MAX_SEQ_LEN,                 # ✅ correct param name for this trl version
    dataset_text_field="text",              # Field SFTTrainer reads from
)

# Format dataset for SFTTrainer (needs a single 'text' field)
def format_prompt(example: Dict) -> Dict:
    example["text"] = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        response=example["response"],
    )
    return example

train_ds = train_ds.map(format_prompt)
eval_ds  = eval_ds.map(format_prompt)

# ─────────────────────────────────────────────
# TRAINER (SFTTrainer handles tokenization)
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

# ─────────────────────────────────────────────
# TRAIN + SAVE
# ─────────────────────────────────────────────
print("[INFO] Starting training ...")
trainer.train()

print(f"[INFO] Saving adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Log final metrics
final_metrics = trainer.evaluate()
wandb.log({"final_eval_loss": final_metrics.get("eval_loss", 0)})
wandb.finish()

print("[DONE] Fine-tuning complete. Adapter saved.")

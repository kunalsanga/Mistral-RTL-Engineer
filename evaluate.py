"""
evaluate.py — Mistral RTL Engineer Evaluation
Compares base Mistral-7B vs fine-tuned RTL model on benchmark prompts.
Logs results to W&B and saves comparison to evaluation_results.md
"""

import os
import gc
import json
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_NAME  = "unsloth/mistral-7b-v0.3-bnb-4bit"  # Pre-quantized 4-bit, ~4GB
ADAPTER_PATH     = "outputs/mistral-rtl"
OUTPUT_FILE      = "evaluation_results.md"
MAX_NEW_TOKENS   = 300
TEMPERATURE      = 0.3
WANDB_PROJECT    = "mistral-rtl-engineer"
BENCHMARK_FILE   = "data/eval_benchmark.json"

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def load_model(adapter_path: str = None):
    """Load base model; if adapter_path given, attach LoRA adapter."""
    print(f"[INFO] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[INFO] Loading {'fine-tuned' if adapter_path else 'base'} model ...")
    # unsloth model is already 4-bit — no quantization_config needed
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        print(f"[INFO] LoRA adapter merged from {adapter_path}")
    elif adapter_path:
        print(f"[WARNING] Adapter path not found: {adapter_path}")

    model.eval()
    return model, tokenizer

def generate(model, tokenizer, instruction: str) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only new tokens (strip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ─────────────────────────────────────────────
# EVALUATE BOTH MODELS
# ─────────────────────────────────────────────
def run_evaluation():
    wandb.init(
        project=WANDB_PROJECT,
        name="evaluation-base-vs-finetuned",
        tags=["eval", "comparison"],
    )

    results = []

    with open(BENCHMARK_FILE, "r") as f:
        prompts = json.load(f)

    # --- BASE MODEL ---
    print("\n[PHASE 1] Running base model evaluation ...")
    base_model, tokenizer = load_model(adapter_path=None)

    for prompt in prompts:
        print(f"  Prompt: {prompt['id']}")
        base_output = generate(base_model, tokenizer, prompt["instruction"])
        results.append({
            "id":           prompt["id"],
            "instruction":  prompt["instruction"],
            "base_output":  base_output,
            "ft_output":    None,
        })

    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[INFO] Base model unloaded. VRAM freed.")

    # --- FINE-TUNED MODEL ---
    print("\n[PHASE 2] Running fine-tuned model evaluation ...")
    ft_model, tokenizer = load_model(adapter_path=ADAPTER_PATH)

    for i, prompt in enumerate(prompts):
        print(f"  Prompt: {prompt['id']}")
        ft_output = generate(ft_model, tokenizer, prompt["instruction"])
        results[i]["ft_output"] = ft_output

    del ft_model
    torch.cuda.empty_cache()

    # --- WRITE RESULTS ---
    write_markdown_report(results)
    log_to_wandb(results)

    wandb.finish()
    print(f"\n[DONE] Evaluation complete. Report saved to {OUTPUT_FILE}")
    return results

def write_markdown_report(results: list):
    lines = [
        "# Mistral RTL Engineer — Evaluation Report\n",
        "**Comparison: Base Mistral-7B vs Fine-Tuned (QLoRA RTL)**\n",
        "---\n\n",
        "## Scoring Rubric\n",
        "| Dimension | Weight | 0 (Fail) | 1 (Partial) | 2 (Good) | 3 (Excellent) |\n",
        "|---|---|---|---|---|---|\n",
        "| **Bug Detection** | 30% | Misses bug | Symptoms only | Root cause | Root cause + fix + why |\n",
        "| **Timing Math** | 20% | Wrong numbers | Right logic, wrong math | Correct | Correct + optimization |\n",
        "| **Testbench** | 20% | Missing/bad | Stimulus only | Valid TB | Self-checking assert TB |\n",
        "| **Correctness** | 15% | Syntax errors | Compiles, wrong | Correct | Correct + synthesizable |\n",
        "| **Explanation** | 15% | Vague | Basic keywords | Structured | 3-section format used |\n\n",
        "---\n",
    ]
    for r in results:
        lines += [
            f"\n## Prompt: `{r['id']}`\n",
            f"**Instruction:**\n```\n{r['instruction']}\n```\n",
            f"### Base Model Output\n```\n{r['base_output']}\n```\n",
            f"### Fine-Tuned Output\n```\n{r['ft_output']}\n```\n",
            "---\n",
        ]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

def log_to_wandb(results: list):
    table = wandb.Table(columns=["prompt_id", "instruction", "base_output", "ft_output"])
    for r in results:
        table.add_data(r["id"], r["instruction"], r["base_output"], r["ft_output"])
    wandb.log({"evaluation_comparison": table})

if __name__ == "__main__":
    run_evaluation()

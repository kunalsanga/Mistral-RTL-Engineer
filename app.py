"""
app.py — Mistral RTL Engineer Gradio Demo
Interactive UI for the fine-tuned Verilog/RTL assistant.
Features Base vs Fine-tuned comparison mode.
"""

import gc
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"  # Pre-quantized 4-bit, ~4GB
ADAPTER_PATH    = "outputs/mistral-rtl"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ─────────────────────────────────────────────
# STATE CACHE
# ─────────────────────────────────────────────
_current_model_type = None  # "base" or "ft"
_model     = None
_tokenizer = None

def load_model(model_type: str):
    global _model, _tokenizer, _current_model_type
    
    if _current_model_type == model_type and _model is not None:
        return _model, _tokenizer

    # Unload existing model to free VRAM
    if _model is not None:
        print(f"[INFO] Unloading {_current_model_type} model...")
        del _model
        _model = None
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[INFO] Loading {model_type} model (please wait) ...")
    
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        _tokenizer.pad_token    = _tokenizer.eos_token
        _tokenizer.padding_side = "left"

    new_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
    )

    if model_type == "ft" and os.path.exists(ADAPTER_PATH):
        new_model = PeftModel.from_pretrained(new_model, ADAPTER_PATH)
        new_model = new_model.merge_and_unload()
        print("[INFO] Fine-tuned LoRA adapter loaded.")
    elif model_type == "ft":
        print("[WARNING] No adapter found — falling back to base model.")

    new_model.eval()
    _model = new_model
    _current_model_type = model_type
    
    return _model, _tokenizer

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def generate_text(model, tokenizer, instruction, temperature, max_tokens, top_p):
    prompt  = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=max(float(temperature), 0.01),
            do_sample=temperature > 0,
            top_p=float(top_p),
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def run_inference(instruction: str, temperature: float, max_tokens: int, top_p: float, compare: bool):
    if not instruction.strip():
        yield "⚠️ Please enter a Verilog question or paste code.", ""
        return

    # If comparison is enabled, run base first, then fine-tuned
    if compare:
        yield "🤖 Loading Base Model...", "⏳ Waiting for base model to finish..."
        base_m, base_t = load_model("base")
        base_out = generate_text(base_m, base_t, instruction, temperature, max_tokens, top_p)
        yield base_out, "🤖 Loading Fine-Tuned Model..."
        
        ft_m, ft_t = load_model("ft")
        ft_out = generate_text(ft_m, ft_t, instruction, temperature, max_tokens, top_p)
        yield base_out, ft_out
    else:
        # Standard mode: only run fine-tuned model
        yield "⏳ Loading Fine-Tuned Model...", ""
        ft_m, ft_t = load_model("ft")
        ft_out = generate_text(ft_m, ft_t, instruction, temperature, max_tokens, top_p)
        yield "", ft_out

# ─────────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────────
EXAMPLES = [
    [
        "Explain this Verilog module and find any bugs:\n"
        "module dff(input clk, input d, output reg q);\n"
        "  always @(posedge clk) q = d;\nendmodule",
        0.3, 300, 0.9, True
    ],
    [
        "Generate a complete testbench for a 4-bit binary counter with synchronous reset.",
        0.3, 400, 0.9, False
    ],
    [
        "What timing considerations are important for a 200MHz FPGA design using a 64-bit ripple-carry adder?",
        0.4, 350, 0.9, True
    ]
]

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }
#title { text-align: center; margin-bottom: 10px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.output-box textarea { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Mistral RTL Engineer", css=CSS) as demo:
    gr.HTML("""
    <div id="title">
        <h1>⚡ Mistral RTL Engineer (Upgraded)</h1>
    </div>
    <div id="subtitle">
        <p>Fine-tuned Mistral-7B (QLoRA) for Verilog debugging, testbench generation & RTL analysis</p>
        <p><small>Mistral Worldwide Hackathon 2025 — Fine-tuning with W&B Track</small></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            instruction_box = gr.Textbox(
                label="📋 Instruction / Verilog Code",
                placeholder="Paste your Verilog code here or ask an RTL question…",
                lines=8,
            )
            compare_checkbox = gr.Checkbox(label="⚖️ Compare with Base Model (takes extra ~30s for VRAM switch)", value=False)
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate Analysis", variant="primary")
                clear_btn  = gr.Button("🗑️ Clear")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            temperature = gr.Slider(0.01, 1.0, value=0.3, step=0.05, label="Temperature")
            max_tokens  = gr.Slider(64, 768,  value=400,  step=32,   label="Max New Tokens")
            top_p       = gr.Slider(0.5, 1.0, value=0.9,  step=0.05, label="Top-p")

    with gr.Row():
        base_output = gr.Textbox(
            label="🤖 Base Model Output",
            lines=15,
            interactive=False,
            elem_classes=["output-box"],
        )
        ft_output = gr.Textbox(
            label="🎯 Fine-Tuned RTL Engineer Output",
            lines=15,
            interactive=False,
            elem_classes=["output-box"],
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction_box, temperature, max_tokens, top_p, compare_checkbox],
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[instruction_box, temperature, max_tokens, top_p, compare_checkbox],
        outputs=[base_output, ft_output],
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[instruction_box, base_output, ft_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

"""
app.py — Mistral RTL Engineer Gradio Demo
Interactive UI for the fine-tuned Verilog/RTL assistant.
"""

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
MAX_NEW_TOKENS  = 512
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ─────────────────────────────────────────────
# MODEL LOADING (cached for Gradio)
# ─────────────────────────────────────────────
_model     = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print("[INFO] Loading model (first request, please wait ~30s) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    _tokenizer.pad_token    = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if os.path.exists(ADAPTER_PATH):
        _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)
        _model = _model.merge_and_unload()
        print("[INFO] Fine-tuned LoRA adapter loaded.")
    else:
        print("[WARNING] No adapter found — using base model.")

    _model.eval()
    return _model, _tokenizer

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def generate_response(
    instruction: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    if not instruction.strip():
        return "⚠️ Please enter a Verilog question or paste code."

    model, tokenizer = get_model()
    prompt  = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=temperature > 0,
            top_p=float(top_p),
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ─────────────────────────────────────────────
# EXAMPLE PROMPTS
# ─────────────────────────────────────────────
EXAMPLES = [
    [
        "Explain this Verilog module and find any bugs:\n"
        "module dff(input clk, input d, output reg q);\n"
        "  always @(posedge clk) q = d;\nendmodule",
        0.3, 300, 0.9,
    ],
    [
        "Generate a complete testbench for a 4-bit binary counter with synchronous reset.",
        0.3, 400, 0.9,
    ],
    [
        "What are the top 5 most common RTL bugs in Verilog and how do I fix them?",
        0.4, 350, 0.9,
    ],
    [
        "How do I cross clock domains safely in FPGA design? Show a Verilog example.",
        0.3, 400, 0.9,
    ],
    [
        "Optimize this for 200MHz on Artix-7 FPGA:\n"
        "module add_tree(input [7:0] a,b,c,d, output [9:0] sum);\n"
        "  assign sum = a + b + c + d;\nendmodule",
        0.3, 300, 0.9,
    ],
]

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 960px; margin: auto; }
#title { text-align: center; margin-bottom: 10px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.output-box textarea { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Mistral RTL Engineer",
) as demo:

    gr.HTML("""
    <div id="title">
        <h1>⚡ Mistral RTL Engineer</h1>
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
                placeholder=(
                    "Paste your Verilog code here or ask an RTL question…\n\n"
                    "Examples:\n"
                    "• Explain this module and find bugs: [code]\n"
                    "• Generate a testbench for: [code]\n"
                    "• What timing issues exist in: [code]"
                ),
                lines=12,
                elem_id="input-box",
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate", variant="primary", scale=3)
                clear_btn  = gr.Button("🗑️ Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            temperature = gr.Slider(0.01, 1.0, value=0.3, step=0.05, label="Temperature")
            max_tokens  = gr.Slider(64, 512,  value=300,  step=32,   label="Max New Tokens")
            top_p       = gr.Slider(0.5, 1.0, value=0.9,  step=0.05, label="Top-p")
            gr.Markdown(
                "**💡 Tips:**\n"
                "- Low temp (0.1–0.3) for precise code\n"
                "- Higher temp (0.5+) for creative suggestions\n"
                "- Increase tokens for full testbenches"
            )

    output_box = gr.Textbox(
        label="🤖 RTL Engineer Response",
        lines=20,
        interactive=False,
        elem_classes=["output-box"],
    )

    gr.Markdown("### 📌 Example Prompts (click to load)")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction_box, temperature, max_tokens, top_p],
        label="",
    )

    with gr.Accordion("ℹ️ About this model", open=False):
        gr.Markdown("""
**Mistral RTL Engineer** is a QLoRA fine-tuned version of `mistralai/Mistral-7B-v0.1` specialized for:

- 🐛 **Bug Detection**: Identifying blocking assignment misuse, sensitivity list errors, overflow bugs, and latch inference
- 📋 **Testbench Generation**: Structured, self-checking Verilog testbenches with assertions
- ⏱️ **Timing Analysis**: Setup/hold time, critical path, metastability considerations
- 🛠️ **RTL Optimization**: Area, power, and speed optimization guidance
- 📖 **Module Explanation**: Clear explanations of complex RTL code

**Training Details:**
- Base: Mistral-7B-v0.1 | Quantization: 4-bit NF4 (QLoRA) | LoRA rank: 8
- Training tracked with **Weights & Biases** | Hardware: RTX 4050 6GB
        """)

    # Event handlers
    submit_btn.click(
        fn=generate_response,
        inputs=[instruction_box, temperature, max_tokens, top_p],
        outputs=output_box,
        api_name="generate",
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[instruction_box, output_box],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",   # use localhost:7860 in browser
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=CSS,
    )

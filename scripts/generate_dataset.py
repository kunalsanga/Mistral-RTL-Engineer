import json
import random
import os

TEMPLATES = {
    "latch_inference": [
        {
            "instruction": "Analyze this code for latch inference.\\nmodule decoder_NBIT(input [{SEL_WIDTH}:0] sel, output reg [{DEC_WIDTH}:0] dec);\\n  always @(*) begin\\n    if (sel == 0) dec = 1;\\n    else if (sel == 1) dec = 2;\\n  end\\nendmodule",
            "output": "### Functional Explanation\\nThis module is intended to be a {DEC_WIDTH}-bit decoder driven by a {SEL_WIDTH}-bit select line.\\n\\n### Potential Issues\\n1. **Unintentional Latch Inference**: The combinational `always` block does not specify output values for all possible permutations of `sel`. In Verilog, if a variable is not assigned in all branches of combinational logic, the synthesis tool infers a transparent latch to hold the previous value.\\n\\n### Fix\\nProvide a default assignment at the beginning of the `always` block or use an `else` clause at the end:\\n```verilog\\nmodule decoder_NBIT(input [{SEL_WIDTH}:0] sel, output reg [{DEC_WIDTH}:0] dec);\\n  always @(*) begin\\n    dec = 0; // Default assignment prevents latches\\n    if (sel == 0) dec = 1;\\n    else if (sel == 1) dec = 2;\\n  end\\nendmodule\\n```"
        }
    ],
    "fifo_cdc": [
        {
            "instruction": "How do I safely cross a {WIDTH}-bit bus from a {FREQ1}MHz clock domain to a {FREQ2}MHz clock domain?",
            "output": "### Functional Explanation\\nCrossing a multi-bit bus such as a {WIDTH}-bit signal between asynchronous clock domains ({FREQ1}MHz and {FREQ2}MHz) requires special synchronization techniques since individual bits might arrive at the destination domains at slightly different times due to routing skew.\\n\\n### Potential Issues\\n1. **Data Corruption due to Skew**: If a standard multi-bit signal is directly synchronized using individual 2-FF synchronizers on each bit, routing skew can cause some bits to be captured on one clock edge and other bits on the next, resulting in invalid intermediate values being sampled.\\n\\n### Fix\\nFor a generic multi-bit data bus, use an asynchronous FIFO with Gray-coded pointers:\\n```verilog\\n// Conceptual instantiation of an Async FIFO for CDC\\nasync_fifo #(\\n  .DATA_WIDTH({WIDTH}),\\n  .ADDR_WIDTH(4)\\n) cdc_fifo (\\n  .wr_clk(clk_{FREQ1}),\\n  .wr_en(data_valid_{FREQ1}),\\n  .din(data_{FREQ1}),\\n  .rd_clk(clk_{FREQ2}),\\n  .rd_en(ready_{FREQ2}),\\n  .dout(data_{FREQ2}),\\n  .full(full_{FREQ1}),\\n  .empty(empty_{FREQ2})\\n);\\n```\\nAlternative: If the data changes very slowly, use a valid-toggle handshake protocol."
        }
    ]
}

def generate():
    dataset = []
    
    # Generate variations of latch inference
    for w in range(2, 6):
        sel_width = w - 1
        dec_width = (2**w) - 1
        template = TEMPLATES["latch_inference"][0]
        dataset.append({
            "instruction": template["instruction"].replace("{SEL_WIDTH}", str(sel_width)).replace("{DEC_WIDTH}", str(dec_width)),
            "input": "",
            "output": template["output"].replace("{SEL_WIDTH}", str(sel_width)).replace("{DEC_WIDTH}", str(dec_width))
        })
        
    # Generate variations of CDC
    for w, f1, f2 in [(16, 50, 125), (32, 100, 250), (64, 200, 400), (8, 25, 100)]:
        template = TEMPLATES["fifo_cdc"][0]
        dataset.append({
            "instruction": template["instruction"].replace("{WIDTH}", str(w)).replace("{FREQ1}", str(f1)).replace("{FREQ2}", str(f2)),
            "input": "",
            "output": template["output"].replace("{WIDTH}", str(w)).replace("{FREQ1}", str(f1)).replace("{FREQ2}", str(f2))
        })

    # Read existing base dataset
    base_file = "data/rtl_dataset_v2.json"
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            base_data = json.load(f)
            dataset = base_data + dataset

    # Save expanded dataset
    with open("data/rtl_dataset_expanded.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} samples. Written to data/rtl_dataset_expanded.json")

if __name__ == "__main__":
    generate()

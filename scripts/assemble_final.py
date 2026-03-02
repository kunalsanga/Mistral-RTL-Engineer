import json
import os

def check_json(file_path):
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return []

def clean_and_merge():
    combined = []
    
    combined.extend(check_json("data/dataset_chunk_1_fully_assembled.json"))
    combined.extend(check_json("data/dataset_chunk_2_fully_assembled.json"))
    combined.extend(check_json("data/dataset_chunk_3_fully_assembled.json"))
    
    # re-assemble chunk 4
    combined.extend(check_json("data/gen_chunk4_1.json"))
    combined.extend(check_json("data/gen_chunk4_2.json"))
    combined.extend(check_json("data/gen_chunk4_3.json"))
    combined.extend(check_json("data/gen_chunk4_4.json"))
    
    # only the first 6 items of 4_5, dropping the broken ones
    c4_5 = check_json("data/gen_chunk4_5.json")
    if len(c4_5) > 6:
        combined.extend(c4_5[:6])
    else:
        combined.extend(c4_5)

    # Dedup
    unique_samples = {}
    for item in combined:
        unique_samples[item["instruction"]] = item

    final_list = list(unique_samples.values())

    # Write out as both dataset_chunk_4 and final
    with open("data/rtl_dataset_final.json", "w") as f:
        json.dump(final_list, f, indent=2)

    # For training compatibility with app.py if it reads something specific
    with open("data/train.json", "w") as f:
        json.dump(final_list, f, indent=2)

    print(f"Total valid samples in final dataset: {len(final_list)}")

if __name__ == "__main__":
    clean_and_merge()

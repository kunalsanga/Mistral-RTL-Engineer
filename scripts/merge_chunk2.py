import json
import os

files_to_merge = [
    "data/gen_chunk2_1.json",
    "data/gen_chunk2_2.json",
    "data/gen_chunk2_3.json",
    "data/gen_chunk2_4.json",
    "data/gen_chunk2_5.json"
]

combined = []

for f_name in files_to_merge:
    if os.path.exists(f_name):
        try:
            with open(f_name, "r") as f:
                data = json.load(f)
                combined.extend(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {f_name}: {e}")
            # Try to pinpoint the error
            with open(f_name, "r") as f:
                content = f.read()
                err_idx = e.pos
                start = max(0, err_idx - 50)
                end = min(len(content), err_idx + 50)
                print(f"Context: {content[start:end]}")
            exit(1)
    else:
        print(f"Warning: {f_name} not found")

# Remove exact duplicates if any
unique_samples = {}
for item in combined:
    unique_samples[item["instruction"]] = item

final_list = list(unique_samples.values())
target_count = len(final_list)

with open("data/dataset_chunk_2_fully_assembled.json", "w") as f:
    json.dump(final_list, f, indent=2)

print(f"Successfully assembled {target_count} unique RTL samples.")
print("Saved to data/dataset_chunk_2_fully_assembled.json")

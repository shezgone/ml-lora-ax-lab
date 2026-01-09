import json
import random
import os

# Source file
source_file = 'solverx_knowledge.jsonl'
output_dir = 'data_solverx_cpt'

# Read data
with open(source_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Shuffle data
random.seed(42)
random.shuffle(lines)

# Split data (e.g., 10% for validation)
split_idx = int(len(lines) * 0.9)
train_data = lines[:split_idx]
valid_data = lines[split_idx:]

# Ensure at least one validation sample
if not valid_data and lines:
    valid_data = [lines[-1]]
    train_data = lines[:-1]

# Write to output files with ensure_ascii=False to handle Korean characters correctly
with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
    for line in train_data:
        # Parse and dump to ensure valid JSON format
        try:
            data = json.loads(line)
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")

with open(os.path.join(output_dir, 'valid.jsonl'), 'w', encoding='utf-8') as f:
    for line in valid_data:
        try:
            data = json.loads(line)
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")

print(f"Prepared data in {output_dir}: {len(train_data)} train, {len(valid_data)} valid")

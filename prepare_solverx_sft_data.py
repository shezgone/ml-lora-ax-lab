import json
import os
from pathlib import Path

# Define paths
cpt_data_path = Path("data_solverx_cpt/train.jsonl")
sft_data_dir = Path("data_solverx_sft")
sft_data_dir.mkdir(exist_ok=True)
sft_data_path = sft_data_dir / "train.jsonl"
sft_valid_path = sft_data_dir / "valid.jsonl"

def create_qa_pair(text):
    # Simple heuristic to generate a question
    # In a real scenario, you might use an LLM to generate diverse questions
    if "추천" in text:
        question = "SolverX는 어떤 것을 추천하나요?"
    elif "extrapolation" in text:
        question = "SolverX는 extrapolation 구간에서 어떻게 해야 하나요?"
    elif "Fusion" in text:
        question = "SolverX Fusion 모델의 특징은 무엇인가요?"
    elif "제약" in text:
        question = "SolverX의 제약 조건은 무엇인가요?"
    else:
        question = "SolverX에 대해 설명해주세요."

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text}
        ]
    }

print(f"Converting {cpt_data_path} to SFT format...")

with open(cpt_data_path, "r") as f_in, open(sft_data_path, "w") as f_out:
    for line in f_in:
        data = json.loads(line)
        text = data["text"]
        qa_pair = create_qa_pair(text)
        f_out.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")

# Create a dummy validation set (copying train for demo purposes, or split it)
# Let's just copy the first few lines
with open(sft_data_path, "r") as f_in, open(sft_valid_path, "w") as f_out:
    lines = f_in.readlines()
    for line in lines[:2]:
        f_out.write(line)

print(f"Created {sft_data_path} with {len(lines)} samples.")
print(f"Created {sft_valid_path}")

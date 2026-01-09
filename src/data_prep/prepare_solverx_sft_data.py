import json
import os
import random
from pathlib import Path

# Define paths
cpt_data_path = Path("data_solverx_cpt/train.jsonl")
sft_data_dir = Path("data_solverx_sft")
sft_data_dir.mkdir(exist_ok=True)
sft_data_path = sft_data_dir / "train.jsonl"
sft_valid_path = sft_data_dir / "valid.jsonl"

# Identity Correction Data
identity_data = [
    {
        "messages": [
            {"role": "user", "content": "너는 누구니?"},
            {"role": "assistant", "content": "저는 네이버가 개발한 초거대 AI, HyperCLOVA X입니다."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am HyperCLOVA X, a hyperscale AI developed by NAVER."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "당신은 OpenAI에서 개발했나요?"},
            {"role": "assistant", "content": "아니요, 저는 네이버에서 개발한 HyperCLOVA X입니다."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Are you developed by OpenAI?"},
            {"role": "assistant", "content": "No, I am HyperCLOVA X, developed by NAVER."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "자기소개 좀 해봐."},
            {"role": "assistant", "content": "안녕하세요! 저는 네이버가 만든 AI, HyperCLOVA X입니다. 무엇을 도와드릴까요?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "너의 개발사는 어디야?"},
            {"role": "assistant", "content": "저는 네이버(NAVER)에서 개발되었습니다."}
        ]
    }
]

# Multi-turn Data (New)
multi_turn_conversations = [
    [
        {"role": "user", "content": "SolverX Fusion이 뭐야?"},
        {"role": "assistant", "content": "SolverX Fusion은 구조 해석과 열 해석을 동시에 예측하는 멀티피직스 모델입니다."},
        {"role": "user", "content": "그럼 그걸 쓰면 뭐가 좋아?"},
        {"role": "assistant", "content": "기존 PINN 모드보다 10배 빠르고, 신뢰도 점수를 제공해서 안정적입니다."}
    ],
    [
        {"role": "user", "content": "SolverX의 주요 기능 3가지만 알려줘."},
        {"role": "assistant", "content": "1. PINN 기반 시뮬레이션\n2. 데이터 기반 서러게이트 모델링\n3. 멀티피직스 해석 기능입니다."},
        {"role": "user", "content": "그 중에서 제일 빠른건 뭐야?"},
        {"role": "assistant", "content": "데이터 기반 서러게이트 모델링이 가장 빠릅니다. 추론 속도가 밀리초 단위입니다."}
    ],
    [
        {"role": "user", "content": "Physics Loss가 뭐야?"},
        {"role": "assistant", "content": "물리 법칙(보존 법칙 등)을 손실 함수에 포함시켜, 데이터가 적어도 물리적으로 타당한 결과를 내도록 하는 기술입니다."},
        {"role": "user", "content": "어떤 라이브러리를 써?"},
        {"role": "assistant", "content": "SolverX는 자체 개발한 Physics Loss Library를 사용하며, 다양한 편미분 방정식(PDE)을 지원합니다."}
    ]
]

def format_multi_turn(messages):
    formatted_text = ""
    for msg in messages:
        formatted_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": formatted_text}

def create_qa_pair(text):
    # Simple heuristic to generate a question based on keywords in the text
    if "추천" in text:
        question = "SolverX는 어떤 모드를 추천하나요?"
    elif "extrapolation" in text:
        question = "SolverX는 extrapolation 구간을 어떻게 처리하나요?"
    elif "Fusion" in text:
        question = "SolverX Fusion에 대해 설명해주세요."
    elif "제약" in text:
        question = "SolverX에서 사용하는 제약 조건은 무엇인가요?"
    elif "Physics Loss" in text:
        question = "Physics Loss Library는 어떤 기능을 제공하나요?"
    elif "설립" in text:
        question = "SolverX는 언제 설립되었나요?"
    elif "PINN" in text:
        question = "SolverX의 PINN 모드는 어떤 특징이 있나요?"
    elif "서러게이트" in text:
        question = "SolverX의 서러게이트 모드는 무엇인가요?"
    else:
        # Generic fallback questions
        questions = [
            "이 내용에 대해 설명해줘.",
            "SolverX의 특징을 알려줘.",
            "관련된 정보를 제공해줘."
        ]
        question = random.choice(questions)

    # Format as ChatML directly
    formatted_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>"
    
    return {
        "text": formatted_text
    }

print(f"Converting {cpt_data_path} to SFT format...")

sft_dataset = []

# 1. Add Identity Data (Duplicate them to ensure they are learned well - e.g., 5 times)
for _ in range(5):
    for item in identity_data:
        # Convert identity data to text format
        msgs = item["messages"]
        user_msg = msgs[0]["content"]
        asst_msg = msgs[1]["content"]
        formatted_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{asst_msg}<|im_end|>"
        sft_dataset.append({"text": formatted_text})

# 1.5 Add Multi-turn Data (Duplicate 10 times for emphasis)
for _ in range(10):
    for conv in multi_turn_conversations:
        sft_dataset.append(format_multi_turn(conv))

# 2. Convert CPT Data
with open(cpt_data_path, "r") as f_in:
    for line in f_in:
        data = json.loads(line)
        text = data["text"]
        qa_pair = create_qa_pair(text)
        sft_dataset.append(qa_pair)

# Shuffle the dataset
random.shuffle(sft_dataset)

# Split into Train and Valid (90/10)
split_idx = int(len(sft_dataset) * 0.9)
train_data = sft_dataset[:split_idx]
valid_data = sft_dataset[split_idx:]

# Write to files
with open(sft_data_path, "w") as f_out:
    for item in train_data:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(sft_valid_path, "w") as f_out:
    for item in valid_data:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Created {sft_data_path} with {len(train_data)} samples.")
print(f"Created {sft_valid_path} with {len(valid_data)} samples.")

import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_cpt_hcx"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# System prompt
system_prompt = "당신은 네이버에서 개발한 AI 어시스턴트 HyperCLOVA X입니다. 도움이 되고, 무해하며, 정직합니다."

# Questions based on the training data
questions = [
    "SolverX는 어떤 모드를 추천하나요?",
    "SolverX는 extrapolation 구간에서 무엇을 권장하나요?",
    "SolverX Fusion은 어떤 모델을 사용하나요?"
]

for q in questions:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n==================================================")
    print(f"Question: {q}")
    response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100)
    print("\nResponse:", response)

import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_sft_hcx"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

questions = [
    "Who are you?",
    "너는 누구니?",
    "SolverX에 대해 설명해줘.",
    "SolverX Fusion이 뭐야?"
]

for q in questions:
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- Question: {q} ---")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
    print(f"\nResponse: {response}")

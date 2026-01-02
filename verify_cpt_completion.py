import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_cpt_hcx"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# CPT Verification: Text Completion
# We provide the start of the sentence from the training data.
prompts = [
    "SolverX는 대부분의 고객에게",
    "SolverX는 extrapolation 구간에서",
    "SolverX Fusion은 구조 해석과"
]

print("\n=== CPT Knowledge Verification (Sentence Completion) ===")
for p in prompts:
    print(f"\nPrompt: {p}")
    # No chat template, just raw completion
    response = generate(model, tokenizer, prompt=p, verbose=True, max_tokens=50)
    print("\nCompletion:", response)

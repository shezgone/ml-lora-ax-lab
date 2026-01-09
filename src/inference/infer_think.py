import mlx.core as mx
from mlx_lm import load, generate

# Model and Adapter paths
model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_sft_hcx"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# [Patch] Force tokenizer to not skip special tokens (to see <think>, <|im_end|>, etc.)
original_decode = tokenizer.decode
def new_decode(*args, **kwargs):
    kwargs['skip_special_tokens'] = False
    return original_decode(*args, **kwargs)
tokenizer.decode = new_decode
print("Tokenizer patched to show special tokens (e.g., <|im_end|>, <think>).")

# A question requiring reasoning (Chain of Thought)
question = "SolverX Fusion이 기존 PINN 방식보다 왜 더 효율적인지 3가지 이유를 들어 자세히 설명해줘."

# Constructing the prompt (Standard ChatML)
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

print(f"\n--- Question: {question} ---")
print("Generating response...")

response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=512, 
    verbose=True
)
print(f"\nFull Response:\n{response}")

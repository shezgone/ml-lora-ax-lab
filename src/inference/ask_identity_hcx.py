import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path)

# Simple prompt to ask identity
prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"

print(f"\nPrompt: {prompt}")
response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=True)
print(f"\nResponse: {response}")

import mlx.core as mx
from mlx_lm import load, generate

model_path = "mlx-community/gemma-2-27b-it-4bit"
print(f"Loading model: {model_path}")
model, tokenizer = load(model_path)

prompt = "Hello, how are you?"
print(f"Generating response for: {prompt}")
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print("Response:", response)

import mlx.core as mx
from mlx_lm import load, generate
import os

model_id = "google/gemma-2-9b-it"
adapter_path = "adapters_9b"

print(f"Loading model {model_id} with adapters from {adapter_path}")
# Note: mlx_lm.load handles HF auth if env var is set, but we can also pass tokenizer_config if needed.
# Usually it picks up from cache.
model, tokenizer = load(model_id, adapter_path=adapter_path)

prompt = "솔버엑스는 어디에 있어요?"
messages = [{"role": "user", "content": prompt}]

if hasattr(tokenizer, "apply_chat_template"):
    prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
else:
    # Fallback if tokenizer doesn't support chat template (unlikely for gemma-it)
    prompt_formatted = f"User: {prompt}\nAssistant:"

print(f"Prompt: {prompt_formatted}")
print("Generating...")
response = generate(model, tokenizer, prompt=prompt_formatted, max_tokens=256, verbose=True)

import mlx.core as mx
from mlx_lm import load, generate

model_id = "google/gemma-2-9b-it"
adapter_path = "adapters"

question = "파이썬으로 mab-ts 알고리즘 구현해줘"

def run_inference(model, tokenizer, q, label):
    print(f"\n{'='*20} {label} {'='*20}")
    messages = [{"role": "user", "content": q}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    
    print(f"\n[Question] {q}\n")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    print(f"[Answer]\n{response.strip()}")

# 1. Base Model
print("Loading Base Model...")
model_base, tokenizer_base = load(model_id)
run_inference(model_base, tokenizer_base, question, "Base Model")

# Clear memory
del model_base
del tokenizer_base
mx.metal.clear_cache()

# 2. Fine-tuned Model
print("\nLoading Fine-tuned Model...")
model_ft, tokenizer_ft = load(model_id, adapter_path=adapter_path)
run_inference(model_ft, tokenizer_ft, question, "Fine-tuned Model")

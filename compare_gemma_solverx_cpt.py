import mlx.core as mx
from mlx_lm import load, generate
import os

model_id = "google/gemma-2-9b-it"
adapter_path = "adapters_solverx_cpt"

questions = [
    "SolverX는 언제 설립되었나요?",
    "SolverX의 본사는 어디에 위치하나요?",
    "SolverX Fusion이란 무엇인가요?",
    "SolverX Fusion은 신뢰도 점수가 낮을 때 어떻게 동작하나요?"
]

def run_inference(model, tokenizer, questions, label):
    print(f"\n" + "="*20 + f" {label} " + "="*20)
    for q in questions:
        messages = [{"role": "user", "content": q}]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        
        print(f"\n[Q]: {q}")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        print(f"[A]: {response.strip()}")

print("Loading Base Model...")
model_base, tokenizer_base = load(model_id)
run_inference(model_base, tokenizer_base, questions, "Base Model (Before Tuning)")

# Clean up to save memory
del model_base
del tokenizer_base
mx.metal.clear_cache()

print("\n" + "="*50 + "\n")

print(f"Loading Fine-tuned Model from {adapter_path}...")
model_ft, tokenizer_ft = load(model_id, adapter_path=adapter_path)
run_inference(model_ft, tokenizer_ft, questions, "Fine-tuned Model (After Tuning)")

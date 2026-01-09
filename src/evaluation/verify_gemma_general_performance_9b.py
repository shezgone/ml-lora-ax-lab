import mlx.core as mx
from mlx_lm import load, generate

def main():
    model_path = "google/gemma-2-9b-it"
    adapter_path = "adapters_9b"

    print(f"Loading model {model_path} with adapters from {adapter_path}...")
    # Load model with the trained adapters
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # 1. General Knowledge Questions 
    # These test if the model retains its original capabilities (avoiding catastrophic forgetting)
    general_questions = [
        "대한민국의 수도는 어디인가요?",
        "하늘이 파란 이유를 간단히 설명해주세요.",
        "파이썬으로 'Hello World'를 출력하는 코드를 작성해줘."
    ]

    # 2. Specific Knowledge Questions
    # These test if the LoRA fine-tuning was effective
    specific_questions = [
        "SolverX의 본사는 어디에 위치하고 있나요?",
        "SolverX Fusion이 신뢰도가 낮을 때 어떻게 동작하나요?"
    ]

    print("\n" + "="*60)
    print("TEST 1: General Knowledge Check (Original Capabilities)")
    print("="*60)
    
    for q in general_questions:
        # Gemma chat format
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        print(f"\n[Question] {q}")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        print(f"[Answer] {response.strip()}")

    print("\n" + "="*60)
    print("TEST 2: Injected Knowledge Check (LoRA Effectiveness)")
    print("="*60)

    for q in specific_questions:
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        print(f"\n[Question] {q}")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        print(f"[Answer] {response.strip()}")

if __name__ == "__main__":
    main()

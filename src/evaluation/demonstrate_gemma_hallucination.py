import mlx.core as mx
from mlx_lm import load, generate

def main():
    model_path = "google/gemma-2-9b-it"
    adapter_path = "adapters"

    print(f"Loading model {model_path} with adapters from {adapter_path}...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Question about information NOT present in the training data
    # Using a leading question to provoke hallucination
    question = "SolverX의 직원들을 위한 특별한 복지 혜택 3가지만 알려줘."

    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    print("\n" + "="*60)
    print("HALLUCINATION TEST: Asking about untrained knowledge (Welfare)")
    print("="*60)
    print(f"\n[Question] {question}")
    
    # Generate response
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    
    print(f"\n[Model Response]\n{response.strip()}")
    print("\n" + "="*60)
    print("Analysis: Since 'Welfare' was not in the training data, any specific details above are likely hallucinations.")

if __name__ == "__main__":
    main()

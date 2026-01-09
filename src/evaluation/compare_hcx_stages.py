import mlx.core as mx
from mlx_lm import load, generate
import sys

# Define paths
model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
cpt_adapter = "adapters_solverx_cpt_hcx"
sft_adapter = "adapters_solverx_sft_hcx"

# Define test cases
test_cases = [
    {
        "category": "Identity",
        "prompt": "너는 누구니?",
        "chat_format": True
    },
    {
        "category": "Domain Knowledge (Definition)",
        "prompt": "SolverX Fusion이 뭐야?",
        "chat_format": True
    },
    {
        "category": "Domain Knowledge (Fact)",
        "prompt": "SolverX는 언제 설립되었나요?",
        "chat_format": True
    },
    {
        "category": "Domain Knowledge (Concept)",
        "prompt": "Physics Loss가 뭐야?",
        "chat_format": True
    }
]

def format_prompt(prompt, use_chat_format=True):
    if use_chat_format:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def run_inference(model, tokenizer, name, use_chat_format=True):
    print(f"\n{'='*20} Testing {name} {'='*20}")
    for case in test_cases:
        prompt = format_prompt(case["prompt"], use_chat_format)
        print(f"\n[Category: {case['category']}]")
        print(f"Input: {case['prompt']}")
        
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=200, 
            verbose=False
        )
        print(f"Output: {response.strip()}")

def run_completion(model, tokenizer, prompt):
    print(f"\n[Completion Test (Raw Text)]")
    print(f"Input: {prompt}")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    print(f"Output: {response.strip()}")

def main():
    # 1. Test Base Model
    # print("Loading Base Model...")
    # model, tokenizer = load(model_path)
    # run_inference(model, tokenizer, "Base Model")
    # del model
    
    # 2. Test CPT Model
    # print("\nLoading CPT Model (Base + CPT Adapter)...")
    # model, tokenizer = load(model_path, adapter_path=cpt_adapter)
    # run_completion(model, tokenizer, "SolverX Fusion은")
    # del model

    # 3. Test SFT Model
    print("\nLoading SFT Model (Base + SFT Adapter)...")
    model, tokenizer = load(model_path, adapter_path=sft_adapter)
    run_inference(model, tokenizer, "SFT Model")

if __name__ == "__main__":
    main()

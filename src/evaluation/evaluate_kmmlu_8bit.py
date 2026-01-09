import argparse
import numpy as np
import mlx.core as mx
from mlx_lm import load
from datasets import load_dataset
from tqdm import tqdm

def evaluate(args):
    # Load Model
    print(f"Loading model: {args.model}")
    if args.adapter_path:
        print(f"Loading adapter: {args.adapter_path}")
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    else:
        model, tokenizer = load(args.model)
    
    # Load Dataset
    subset = args.subset
    print(f"Loading KMMLU subset: {subset}")
    try:
        # KMMLU has various subsets like 'Korean-History', 'Economics', etc.
        # Dataset ID: HAERAE-HUB/KMMLU
        dataset = load_dataset("HAERAE-HUB/KMMLU", subset, split="test")
        if args.shots > 0:
            dev_dataset = load_dataset("HAERAE-HUB/KMMLU", subset, split="dev")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if the subset name is correct. Examples: 'Korean-History', 'Economics', 'Law', 'Physics'.")
        return

    # Prepare Few-shot examples
    few_shot_prompt = ""
    if args.shots > 0:
        print(f"Preparing {args.shots}-shot examples...")
        for i in range(min(args.shots, len(dev_dataset))):
            ex = dev_dataset[i]
            q = ex["question"]
            
            # Handle different dataset formats
            if "options" in ex:
                opts = ex["options"]
            elif "A" in ex:
                opts = [ex["A"], ex["B"], ex["C"], ex["D"]]
            else:
                opts = ["Option 1", "Option 2", "Option 3", "Option 4"]
                
            ans_idx = ex["answer"]
            if isinstance(ans_idx, str):
                ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                ans_idx = ans_map.get(ans_idx, 0)
            elif isinstance(ans_idx, int):
                if ans_idx == 4: ans_idx = 3
            
            # Format example
            ex_text = f"질문: {q}\n"
            choices = ["A", "B", "C", "D"]
            for j, opt in enumerate(opts):
                ex_text += f"{choices[j]}. {opt}\n"
            
            if 0 <= ans_idx < 4:
                ex_text += f"정답: {choices[ans_idx]}\n\n"
            else:
                ex_text += f"정답: A\n\n" # Fallback
                
            few_shot_prompt += ex_text
            
    # Token IDs for A, B, C, D
    choices = ["A", "B", "C", "D"]
    choice_ids = []
    for c in choices:
        tid = tokenizer.encode(c, add_special_tokens=False)[0]
        choice_ids.append(tid)
    
    # Also check " A" (with space)
    choices_space = [" A", " B", " C", " D"]
    choice_ids_space = []
    for c in choices_space:
        tid = tokenizer.encode(c, add_special_tokens=False)[0]
        choice_ids_space.append(tid)
        
    print(f"Token IDs for A,B,C,D: {choice_ids}")
    print(f"Token IDs for ' A',' B',' C',' D': {choice_ids_space}")
    
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        if args.limit and i >= args.limit:
            break
            
        question = example["question"]
        
        # Handle different dataset formats
        if "options" in example:
            options = example["options"]
        elif "A" in example and "B" in example:
            options = [example["A"], example["B"], example["C"], example["D"]]
        else:
            print(f"Unknown format: {example.keys()}")
            break
            
        answer_idx = example["answer"] # 0, 1, 2, 3
        
        # Check if answer is 1-based
        if isinstance(answer_idx, int):
            if answer_idx == 4:
                answer_idx = 3
                
        # Ensure answer is an integer
        if isinstance(answer_idx, str):
            # If answer is 'A', 'B', etc.
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            answer_idx = answer_map.get(answer_idx, 0)
            
        # Debug answer index
        if answer_idx < 0 or answer_idx >= len(choices):
            print(f"Warning: Invalid answer index {answer_idx} for example {i} (Options: {len(options)})")
            continue
            
        target_char = choices[answer_idx]
        
        # Construct Prompt using ChatML format
        # HyperCLOVA X uses <|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n
        
        user_content = f"다음 문제를 읽고 정답을 하나만 고르시오.\n\n"
        
        # Add few-shot examples if any
        if args.shots > 0:
            user_content += "예시:\n" + few_shot_prompt + "본 문제:\n"
            
        user_content += f"질문: {question}\n"
        for j, opt in enumerate(options):
            user_content += f"{choices[j]}. {opt}\n"
        user_content += "\n정답:"
        
        # Force the model to answer directly by pre-filling "정답은"
        prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n정답은"
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        
        # Forward pass
        logits = model(input_ids)
        
        # Get logits for the last token
        last_token_logits = logits[0, -1, :]
        
        # Extract logits for A, B, C, D
        # We check both "A" and " A" and take the max for each letter
        final_logits = []
        for k in range(4):
            l1 = last_token_logits[choice_ids[k]].item()
            l2 = last_token_logits[choice_ids_space[k]].item()
            final_logits.append(max(l1, l2))
        
        # Prediction
        pred_idx = np.argmax(final_logits)
        pred_char = choices[pred_idx]
        
        if pred_idx == answer_idx:
            correct += 1
        total += 1
        
        if i < 3: # Print first few examples for debugging
            print(f"\n[Example {i}]")
            print(f"Prompt tail: ...{prompt[-20:]}")
            print(f"Pred: {pred_char} (Logits: {final_logits})")
            print(f"True: {target_char}")

    if total > 0:
        accuracy = correct / total
        print(f"\nResult for {subset}:")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    else:
        print("No examples evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/HyperCLOVAX-SEED-Think-32B-Text-8bit")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to adapters (optional)")
    parser.add_argument("--subset", type=str, default="Korean-History", help="KMMLU subset (e.g., Korean-History, Economics)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    args = parser.parse_args()
    evaluate(args)

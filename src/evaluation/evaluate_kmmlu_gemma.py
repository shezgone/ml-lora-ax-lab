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
        dataset = load_dataset("HAERAE-HUB/KMMLU", subset, split="test")
        if args.shots > 0:
            dev_dataset = load_dataset("HAERAE-HUB/KMMLU", subset, split="dev")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Prepare Few-shot examples
    few_shot_prompt = ""
    if args.shots > 0:
        print(f"Preparing {args.shots}-shot examples...")
        for i in range(min(args.shots, len(dev_dataset))):
            ex = dev_dataset[i]
            q = ex["question"]
            
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
            
            ex_text = f"질문: {q}\n"
            choices = ["A", "B", "C", "D"]
            for j, opt in enumerate(opts):
                ex_text += f"{choices[j]}. {opt}\n"
            
            if 0 <= ans_idx < 4:
                ex_text += f"정답: {choices[ans_idx]}\n\n"
            else:
                ex_text += f"정답: A\n\n"
                
            few_shot_prompt += ex_text
            
    # Token IDs for A, B, C, D
    choices = ["A", "B", "C", "D"]
    choice_ids = []
    for c in choices:
        tid = tokenizer.encode(c, add_special_tokens=False)[0]
        choice_ids.append(tid)
    
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
        
        if "options" in example:
            options = example["options"]
        elif "A" in example and "B" in example:
            options = [example["A"], example["B"], example["C"], example["D"]]
        else:
            break
            
        answer_idx = example["answer"]
        if isinstance(answer_idx, int):
            if answer_idx == 4: answer_idx = 3
        if isinstance(answer_idx, str):
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            answer_idx = answer_map.get(answer_idx, 0)
            
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
            
        target_char = choices[answer_idx]
        
        # Construct Prompt using Gemma Chat Template
        # Gemma uses <start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n
        
        user_content = f"다음 문제를 읽고 정답을 하나만 고르시오.\n\n"
        if args.shots > 0:
            user_content += "예시:\n" + few_shot_prompt + "본 문제:\n"
            
        user_content += f"질문: {question}\n"
        for j, opt in enumerate(options):
            user_content += f"{choices[j]}. {opt}\n"
        user_content += "\n정답:"
        
        # Use tokenizer's chat template if available, otherwise manual
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_content}]
            # We want to generate the assistant response start
            # But apply_chat_template usually adds generation prompt if requested
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Append "정답은" to force answer
            prompt += "정답은"
        else:
            # Manual fallback for Gemma
            prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n정답은"
        
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        
        logits = model(input_ids)
        last_token_logits = logits[0, -1, :]
        
        final_logits = []
        for k in range(4):
            l1 = last_token_logits[choice_ids[k]].item()
            l2 = last_token_logits[choice_ids_space[k]].item()
            final_logits.append(max(l1, l2))
        
        pred_idx = np.argmax(final_logits)
        pred_char = choices[pred_idx]
        
        if pred_idx == answer_idx:
            correct += 1
        total += 1
        
        if i < 3:
            print(f"\n[Example {i}]")
            print(f"Prompt tail: ...{prompt[-50:]}")
            print(f"Pred: {pred_char} (Logits: {final_logits})")
            print(f"True: {target_char}")

    if total > 0:
        accuracy = correct / total
        print(f"\nResult for {subset}:")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to adapters (optional)")
    parser.add_argument("--subset", type=str, default="Law", help="KMMLU subset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    args = parser.parse_args()
    evaluate(args)

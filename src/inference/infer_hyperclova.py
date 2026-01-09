import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Path to the downloaded model
model_path = "models/HyperCLOVAX-SEED-Think-32B"

print(f"Loading model from {model_path}...")

# Check for MPS availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model with device_map="cpu" to avoid MPS buffer limits for large models
    # and to ensure it fits in RAM (if enough RAM is available)
    print("Loading model on CPU to avoid MPS buffer size limits...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("Model loaded successfully.")
    
    # User query
    query = "솔버엑스(SolverX)에 대해서 설명해주세요."
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to simple formatting if template not available
        input_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
    print(f"\nInput: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Identify <think> token id to ban it
    # From added_tokens.json, <think> is 128040
    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    print(f"Banning token: <think> (ID: {think_token_id}) to disable think mode.")
    
    print("Generating response...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            bad_words_ids=[[think_token_id]] # Ban <think> token
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("Response:")
    print("="*50)
    print(response)
    print("="*50)

except Exception as e:
    print(f"An error occurred: {e}")

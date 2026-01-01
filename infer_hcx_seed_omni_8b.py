import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Path to the downloaded model
model_path = "models/HyperCLOVAX-SEED-Omni-8B"

print(f"Loading model from {model_path}...")

# Check for MPS availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 8B model might fit in MPS memory depending on the machine specs.
    # Using float16 to save memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("Model loaded successfully.")
    
    # User query
    query = "솔버엑스는 어디에 위치한 회사입니까?"
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback if template not available (though it should be for this model)
        input_text = f"User: {query}\nAssistant:"
        
    print(f"\nInput: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Remove token_type_ids if present, as the model doesn't accept it
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")

except Exception as e:
    print(f"An error occurred: {e}")

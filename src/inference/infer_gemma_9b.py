import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the local snapshot path if possible, or the repo id
model_id = "google/gemma-2-9b-it"

print(f"Loading {model_id}...")

try:
    # Try to use MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    messages = [
        {"role": "user", "content": "솔버엑스는 어디에 있어요?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    print("Generating...")
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("\nAnswer:")
    print(response)

except Exception as e:
    print(f"An error occurred: {e}")

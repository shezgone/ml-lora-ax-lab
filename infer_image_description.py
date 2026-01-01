import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os

# Path to the downloaded model
model_path = "models/HyperCLOVAX-SEED-Omni-8B"
image_path = "임태건.jpg"

print(f"Loading model from {model_path}...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully.")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        exit(1)
        
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size}")
    
    # Prepare prompt
    system_prompt = "You are HyperCLOVA X, a large language model trained by NAVER. You are a helpful, versatile, and harmless AI assistant. You can generate images when the user asks."
    user_prompt = "이 이미지에 대해 자세히 설명해줘. describe_image 도구를 사용해서 설명해줘."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://" + os.path.abspath(image_path)}},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "description": "Describes the content of the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "The detailed description of the image."
                        }
                    },
                    "required": ["description"]
                }
            }
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    
    # Revert hack: Keep discrete image tokens as in the working edit script
    # text = text.replace("<|discrete_image_start|><|DISCRETE_IMAGE_PAD|><|discrete_image_end|>\n", "")
    
    print(f"Prepared text input: {text}")
    
    # Processor call
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    if hasattr(inputs, "pixel_values"):
        print(f"Pixel values shape: {inputs.pixel_values.shape}")
    if hasattr(inputs, "image_grid_thw"):
        print(f"Image grid thw: {inputs.image_grid_thw}")
    inputs = inputs.to(device)
    
    # Generate
    print("Generating...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    
    print("\nGenerated Output:")
    print(output_text)
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

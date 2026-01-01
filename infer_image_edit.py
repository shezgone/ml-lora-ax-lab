import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
import json
import multiprocessing

# Set number of threads to physical cores for CPU optimization
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
# Set quantization engine for Apple Silicon
torch.backends.quantized.engine = 'qnnpack'
print(f"Set torch threads to: {num_cores}")
print(f"Set quantized engine to: {torch.backends.quantized.engine}")

# Path to the downloaded model
model_path = "models/HyperCLOVAX-SEED-Omni-8B"
image_path = "임태건.jpg"

print(f"Loading model from {model_path}...")
device = "cpu"
print(f"Using device: {device}")

try:
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully.")

    # Optimization: Quantize model for faster CPU inference
    print("Quantizing model for faster CPU inference...")
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Model quantized successfully.")
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Proceeding with unquantized model.")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        exit(1)
        
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size}")
    
    # Prepare prompt
    # System prompt for image generation/editing from README
    system_prompt = "You are an AI assistant that transforms images. When asked to transform, edit, or stylize an image, you MUST use the t2i_model_generation tool to generate the new image. Always respond by calling the tool. You MUST think step-by-step in Korean."
    user_prompt = "Put a red bow tie on him."
    
    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://" + os.path.abspath(image_path)}},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    # Define tools from README
    tools = [{
        "type": "function",
        "function": {
            "name": "t2i_model_generation",
            "description": "Generates an RGB image based on the provided discrete image representation.",
            "parameters": {
                "type": "object",
                "required": ["discrete_image_token"],
                "properties": {
                    "discrete_image_token": {
                        "type": "string",
                        "description": "A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.",
                        "minLength": 1
                    }
                }
            }
        }
    }]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    print(f"Prepared text input: {text}")
    
    # Processor call
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    # Generate
    print("Generating...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,  # Increased for image tokens
        do_sample=True,
        temperature=0.1, # Lower temperature for more deterministic tool calls
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    
    with open("raw_output.txt", "w") as f:
        f.write(output_text)
    print("Raw output saved to raw_output.txt")
    
    print(f"Raw Output: {output_text}")
    
    # Try to parse tool call
    # Find all tool_call blocks
    import re
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = tool_call_pattern.findall(output_text)
    
    discrete_tokens = None
    
    if matches:
        print(f"Found {len(matches)} tool calls. Checking for valid tokens...")
        for i, tool_content in enumerate(matches):
            tool_content = tool_content.strip()
            
            # Function name
            if "\n" in tool_content:
                function_name = tool_content.split("\n")[0].strip()
            else:
                function_name = tool_content.strip()
            
            print(f"Tool Call {i+1}: {function_name}")
            
            # Parse args
            args = {}
            segments = tool_content.split("<arg_key>")
            for segment in segments[1:]:
                if "</arg_key>" in segment:
                    key_part = segment.split("</arg_key>")[0].strip()
                    
                    # Value might be in <arg_value>...</arg_value>
                    # or sometimes the model might mess up tags.
                    # Let's look for <arg_value>
                    if "<arg_value>" in segment:
                        value_part = segment.split("<arg_value>")[1]
                        if "</arg_value>" in value_part:
                            value = value_part.split("</arg_value>")[0].strip()
                            args[key_part] = value
            
            if function_name == "t2i_model_generation":
                if "discrete_image_token" in args:
                    discrete_tokens = args["discrete_image_token"]
                    print("Found discrete_image_token in this tool call.")
                    break
    
    if discrete_tokens:
        print(f"Discrete Image Tokens Length: {len(discrete_tokens)}")
        print(f"First 100 chars: {discrete_tokens[:100]}")
        
        # Save tokens to file
        with open("generated_tokens.txt", "w") as f:
            f.write(discrete_tokens)
        print("Tokens saved to generated_tokens.txt")
    else:
        print("No valid discrete_image_token found in any tool call.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

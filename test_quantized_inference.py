import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"

print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path)

# Define a system prompt to set the identity
system_prompt = "당신은 네이버에서 개발한 AI 어시스턴트 HyperCLOVA X입니다. 도움이 되고, 무해하며, 정직합니다."

# Construct the prompt with system instruction
prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n솔버엑스는 어디에 위치했냐?<|im_end|>\n<|im_start|>assistant\n"

# Note: We are manually constructing the prompt here because we want to be explicit about the format.
# If using tokenizer.apply_chat_template, we would pass a list of messages including the system role.


print(f"Generating response for prompt: '{prompt}'")

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=50)
print("\nResponse:", response)

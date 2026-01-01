import json

with open("models/HyperCLOVAX-SEED-Think-32B/config.json", "r") as f:
    config = json.load(f)

text_config = config["text_config"]
text_config["model_type"] = "seed_oss"

# Ensure head_dim is present (it was in the file I read)
if "head_dim" not in text_config:
    print("Warning: head_dim not found in text_config")

with open("models/HyperCLOVAX-SEED-Think-32B-Text/config.json", "w") as f:
    json.dump(text_config, f, indent=2)

print("Created models/HyperCLOVAX-SEED-Think-32B-Text/config.json")

import json

config_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
config["quantization_config"] = config["quantization"]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Updated config with quantization info.")

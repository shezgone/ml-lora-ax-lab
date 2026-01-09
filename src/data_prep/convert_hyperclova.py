import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import quantize_model, save_model
from mlx_lm.models import seed_oss
from mlx.utils import tree_unflatten
import json
import glob
import os
import shutil

# Paths
src_model_path = "models/HyperCLOVAX-SEED-Think-32B"
dest_model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
config_path = "models/HyperCLOVAX-SEED-Think-32B-Text/config.json"

# Load config
print(f"Loading config from {config_path}")
with open(config_path, "r") as f:
    config = json.load(f)

# Create model args
model_args = seed_oss.ModelArgs.from_dict(config)
model_args.model_type = "seed_oss"

# Instantiate model
print("Instantiating model...")
model = seed_oss.Model(model_args)
model.set_dtype(mx.bfloat16)

# Load weights
print("Loading weights...")
weight_files = glob.glob(os.path.join(src_model_path, "*.safetensors"))
weight_files.sort()

for wf in weight_files:
    print(f"Processing {wf}...")
    weights = mx.load(wf)
    new_weights = {}
    for k, v in weights.items():
        if k.startswith("model.language_model."):
            if k.startswith("model.language_model.model."):
                new_k = k.replace("model.language_model.model.", "model.")
            elif k.startswith("model.language_model.lm_head."):
                new_k = k.replace("model.language_model.lm_head.", "lm_head.")
            else:
                print(f"Skipping unexpected key: {k}")
                continue
            
            new_weights[new_k] = v
    
    if new_weights:
        print(f"Updating model with {len(new_weights)} keys from {wf}")
        model.update(tree_unflatten(list(new_weights.items())))
        # Force evaluation to ensure data is loaded? 
        # No, mx.load maps data. model.update just points the model leaves to these arrays.
        # We need to keep 'weights' alive? 
        # mx.load returns a dict of arrays. The arrays hold references to the file mapping.
        # When we do model.update, the model parameters point to these arrays.
        # So we don't need to keep 'weights' variable, but the arrays must stay valid.
        # Since we are going to quantize, we will read them.
        
print("Quantizing model incrementally to 8-bit...")
import gc

# Quantize layers
for i, layer in enumerate(model.layers):
    print(f"Quantizing layer {i}...")
    nn.quantize(layer, group_size=64, bits=8)
    mx.eval(layer.parameters())
    gc.collect()

# Quantize other parts
print("Quantizing other modules...")
# embed_tokens might not be quantized by default in mlx_lm?
# mlx_lm.utils.quantize_model uses class_predicate.
# Default predicate checks if module has 'to_quantized' (which Linear has).
# Embedding does not have to_quantized usually?
# nn.QuantizedEmbedding exists?
# Let's check if we should quantize embeddings.
# Usually we don't quantize embeddings in 4-bit, but 8-bit maybe?
# mlx_lm default is to quantize linear layers.
# nn.Embedding is not nn.Linear.
# So embed_tokens is skipped.
# lm_head is nn.Linear, so it is quantized.

# So we just need to quantize the rest of the model (excluding layers which are already done)
# But if we call nn.quantize(model), it will re-quantize layers?
# No, QuantizedLinear doesn't have to_quantized? Or it does?
# We should avoid re-quantizing.
# We can just quantize lm_head.
if hasattr(model, "lm_head"):
    print("Quantizing lm_head...")
    nn.quantize(model.lm_head, group_size=64, bits=8)
    mx.eval(model.lm_head.parameters())

# Update config
config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
config["quantization_config"] = config["quantization"]


print(f"Saving quantized model to {dest_model_path}...")
os.makedirs(dest_model_path, exist_ok=True)
save_model(dest_model_path, model)

# Save config and tokenizer files
print("Copying config and tokenizer files...")
shutil.copy(config_path, os.path.join(dest_model_path, "config.json"))

# Copy tokenizer files from original
for f in glob.glob(os.path.join(src_model_path, "tokenizer*")):
    shutil.copy(f, dest_model_path)
    
# Copy other json files (except original config and adapter)
for f in glob.glob(os.path.join(src_model_path, "*.json")):
    filename = os.path.basename(f)
    if filename != "config.json" and "adapter" not in filename:
         shutil.copy(f, dest_model_path)

# Copy merges.txt if exists
for f in glob.glob(os.path.join(src_model_path, "*.txt")):
    shutil.copy(f, dest_model_path)

print("Done!")

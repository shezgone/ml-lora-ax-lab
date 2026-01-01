import mlx.core as mx
from mlx.utils import tree_flatten

# Load the first safetensors file
weights = mx.load("models/HyperCLOVAX-SEED-Think-32B/model-00001-of-00014.safetensors")
for k in list(weights.keys()):
    if "vision_model" not in k:
        print(k)
        break

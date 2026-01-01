import sys
import os

# Add the model directory to sys.path so we can import modules from it
model_dir = os.path.abspath("models/HyperCLOVAX-SEED-Omni-8B")
sys.path.append(model_dir)

try:
    import ta_tok
    print("Successfully imported ta_tok")
except Exception as e:
    print(f"Failed to import ta_tok: {e}")

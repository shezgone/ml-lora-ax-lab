from huggingface_hub import snapshot_download
import os

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"
local_dir = "models/HyperCLOVAX-SEED-Think-32B"

print(f"Downloading {model_id} to {local_dir}...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete.")

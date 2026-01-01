from huggingface_hub import snapshot_download, logging
import os

# 로깅 레벨을 설정하여 상세 정보를 출력합니다.
logging.set_verbosity_info()

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"
local_dir = "models/HyperCLOVAX-SEED-Omni-8B"

print(f"Downloading {model_id} to {local_dir}...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete.")

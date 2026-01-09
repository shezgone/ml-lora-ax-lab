#!/bin/bash

# LoRA SFT Fine-tuning Script for SolverX (Identity & Chat)
# Framework: mlx-lm
# Model: models/HyperCLOVAX-SEED-Think-32B
# Data: data_solverx_sft
# Base Adapter: adapters_solverx_cpt_hcx

echo "Starting LoRA SFT fine-tuning with Early Stopping..."

# Ensure the output directory exists
mkdir -p adapters_solverx_sft_hcx

/Users/user/ml-lora-ax-lab/.venv/bin/python train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Think-32B-Text-8bit \
    --train \
    --data data_solverx_sft \
    --batch-size 2 \
    --iters 400 \
    --learning-rate 1e-5 \
    --adapter-path adapters_solverx_sft_hcx \
    --resume-adapter-file adapters_solverx_cpt_hcx/adapters.safetensors \
    --save-every 50 \
    --patience 5 \
    --min-delta 0.001 \
    --seed 42

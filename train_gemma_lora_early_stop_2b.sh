#!/bin/bash

# LoRA Fine-tuning Script with Early Stopping for Small Gemma (2B)
# Framework: mlx-lm
# Model: google/gemma-2b-it

echo "Starting LoRA fine-tuning for Gemma 2B with Early Stopping..."

/Users/user/ml-lora-ax-lab/.venv/bin/python train_with_early_stopping.py \
    --model google/gemma-2b-it \
    --train \
    --data data_mlx \
    --batch-size 4 \
    --iters 150 \
    --learning-rate 1e-5 \
    --adapter-path adapters_2b \
    --save-every 50 \
    --patience 3 \
    --min-delta 0.0

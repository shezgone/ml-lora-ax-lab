#!/bin/bash

# LoRA Fine-tuning Script with Early Stopping
# Framework: mlx-lm
# Model: google/gemma-2-9b-it

echo "Starting LoRA fine-tuning with Early Stopping..."

/Users/user/ml-lora-ax-lab/.venv/bin/python train_with_early_stopping.py \
    --model google/gemma-2-9b-it \
    --train \
    --data data_mlx \
    --batch-size 1 \
    --iters 150 \
    --learning-rate 1e-5 \
    --adapter-path adapters_9b \
    --save-every 50 \
    --patience 3 \
    --min-delta 0.0

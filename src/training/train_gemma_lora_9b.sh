#!/bin/bash

# LoRA Fine-tuning Script for SolverX Knowledge Injection
# Framework: mlx-lm
# Model: google/gemma-2-9b-it

# Ensure we are in the virtual environment
# source .venv/bin/activate

echo "Starting LoRA fine-tuning..."

python -m mlx_lm.lora \
    --model google/gemma-2-9b-it \
    --train \
    --data data_mlx \
    --batch-size 1 \
    --iters 150 \
    --learning-rate 1e-5 \
    --adapter-path adapters_9b \
    --save-every 50

echo "Training complete. Adapters saved to adapters/"

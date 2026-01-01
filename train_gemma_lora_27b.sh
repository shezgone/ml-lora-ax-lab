#!/bin/bash

# Train LoRA adapter for Gemma-2-27B (4-bit Quantized)
# Uses the same mixed dataset (SolverX + General Knowledge) to prevent catastrophic forgetting

python3 -m mlx_lm.lora \
    --model mlx-community/gemma-2-27b-it-4bit \
    --train \
    --data data_mlx \
    --iters 150 \
    --batch-size 1 \
    --num-layers 2 \
    --learning-rate 1e-6 \
    --grad-checkpoint \
    --adapter-path adapters_27b \
    --save-every 50

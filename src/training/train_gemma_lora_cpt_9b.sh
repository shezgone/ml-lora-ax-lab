#!/bin/bash

# LoRA Fine-tuning Script for SolverX Knowledge Injection (CPT)
# Framework: mlx-lm
# Model: google/gemma-2-9b-it
# Data: solverx_knowledge.jsonl (via data_solverx_cpt)

echo "Starting LoRA CPT fine-tuning with Early Stopping..."

/Users/user/ml-lora-ax-lab/.venv/bin/python train_with_early_stopping.py \
    --model google/gemma-2-9b-it \
    --train \
    --data data_solverx_cpt \
    --batch-size 4 \
    --iters 600 \
    --learning-rate 1e-5 \
    --adapter-path adapters_solverx_cpt_gemma \
    --save-every 50 \
    --patience 5 \
    --min-delta 0.001

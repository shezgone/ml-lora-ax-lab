import argparse
import math
import os
import re
import types
import warnings
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load, save_config
from mlx_lm.lora import build_parser, CONFIG_DEFAULTS, train_model
from mlx.utils import tree_flatten

# Custom Early Stopping Callback
class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        
    def on_val_loss_report(self, val_info):
        current_loss = val_info['val_loss']
        print(f"Validation Loss: {current_loss}")
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered. No improvement for {self.patience} validation steps.")
                raise StopIteration("Early stopping")

def main():
    parser = build_parser()
    # Add early stopping arguments
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (number of validation steps)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum change in validation loss to qualify as an improvement")
    
    args = parser.parse_args()
    config = args.config
    args_dict = vars(args)
    
    # Load config file if provided
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            # Simple loader, might need the custom one from lora.py if complex floats are used
            config = yaml.safe_load(file)
        for k, v in config.items():
            if args_dict.get(k, None) is None:
                args_dict[k] = v

    # Set defaults
    for k, v in CONFIG_DEFAULTS.items():
        if args_dict.get(k, None) is None:
            args_dict[k] = v
            
    # Convert back to namespace
    args = types.SimpleNamespace(**args_dict)
    
    np.random.seed(args.seed)
    
    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    if args.train:
        print(f"Training with Early Stopping (Patience: {args.patience}, Min Delta: {args.min_delta})")
        
        # Initialize callback
        callback = EarlyStoppingCallback(patience=args.patience, min_delta=args.min_delta)
        
        try:
            train_model(args, model, train_set, valid_set, training_callback=callback)
        except StopIteration:
            print("Training stopped early due to no improvement in validation loss.")
            
            # Save the final state when early stopping occurs
            # train_model saves periodically, but we want to ensure we save the state at stopping point
            adapter_path = Path(args.adapter_path)
            adapter_file = adapter_path / "adapters.safetensors"
            
            # We need to make sure we are saving the trainable parameters
            # The model is already in LoRA mode from train_model
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(adapter_file), adapter_weights)
            print(f"Saved final weights to {adapter_file}.")
            
    elif args.test:
        # Fallback to normal testing if needed, though user can use original script for this
        from mlx_lm.lora import evaluate_model
        print("Testing")
        evaluate_model(args, model, test_set)
    else:
        raise ValueError("Must provide at least one of --train or --test")

if __name__ == "__main__":
    main()

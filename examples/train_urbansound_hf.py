"""
Script to train the UrbanSound8K model using HuggingFace dataset.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tasks.train_urbansound import main as train_urbansound

def main():
    """Main function to run the training script"""
    parser = argparse.ArgumentParser(description="Train UrbanSound8K classifier with HuggingFace dataset")
    parser.add_argument('--config', type=str, default='configs/urbansound_cnn.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (default: checkpoints_dir from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config file)')
    # Modified argument description to discourage overriding cross-validation
    parser.add_argument('--fold', type=int, default=None,
                        help='DEPRECATED: Single fold evaluation is not recommended by dataset creators')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable Weights & Biases tracking')
    
    args = parser.parse_args()
    
    # Force use_huggingface to True
    # We'll create a copy of the args object with this flag set
    import yaml
    from argparse import Namespace
    
    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config values
    config['use_huggingface'] = True
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
    
    if args.fold is not None:
        print("Warning: --fold argument is deprecated. Using full cross-validation as recommended by dataset creators.")
        # Ensure cross-validation is enabled
        if 'cv' not in config:
            config['cv'] = {}
        config['cv']['cross_validation'] = True
        config['cv']['folds'] = 10
    
    # Enable WandB by default
    if 'wandb' not in config:
        config['wandb'] = {}
    
    # Enable WandB unless explicitly disabled
    if not args.disable_wandb:
        config['wandb']['enable'] = True
        config['wandb']['project'] = config['wandb'].get('project', 'urbansound-classification')
        config['wandb']['group'] = config['wandb'].get('group', 'urbansound')
        config['wandb']['log_confusion_matrix'] = True
        print("Weights & Biases tracking enabled. Use --disable_wandb to disable.")
    else:
        config['wandb']['enable'] = False
        print("Weights & Biases tracking disabled.")
    
    # Save the modified config to a temporary file
    import tempfile
    temp_config = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    temp_config_path = temp_config.name
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create modified args
    train_args = Namespace(
        config=temp_config_path,
        output_dir=args.output_dir
    )
    
    # Run training
    print(f"Training UrbanSound8K classifier with HuggingFace dataset...")
    print(f"Config: {args.config} (modified)")
    
    # Updated print statement to reflect cross-validation status
    if config['cv'].get('cross_validation', False):
        print(f"Using {config['cv'].get('folds', 10)}-fold cross-validation")
    else:
        print(f"Test fold: {config['cv'].get('fold', 1)}")
    
    print(f"Epochs: {config['epochs']}")
    print(f"Output directory: {args.output_dir or config.get('checkpoints_dir', './checkpoints')}")
    
    try:
        train_urbansound(train_args)
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)

if __name__ == "__main__":
    main() 
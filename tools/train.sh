#!/bin/bash

# Script to train the Vision Transformer model

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -e .

# Make sure directories exist
mkdir -p checkpoints
mkdir -p data

# Train the model
python -m vit_mnist.core.train --config configs/default_config.yml $@

echo "Training complete. Model saved to checkpoints/best_model.pt" 
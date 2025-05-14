#!/usr/bin/env python
"""
Create the mlx7-two-tower repository on HuggingFace Hub.

This script creates the model repository and uploads a README and configuration.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.huggingface import setup_repository, upload_file
from huggingface_hub import HfApi

# Default repository name
DEFAULT_REPO_NAME = "mlx7-two-tower"

def create_repo_readme(repo_name="mlx7-two-tower"):
    """Create a README.md file for the HuggingFace repository."""
    readme_content = f"""# {repo_name}

This repository contains models trained using the Two-Tower (Dual Encoder) architecture for document retrieval.

## Model Description

The Two-Tower model is a dual encoder neural network architecture designed for semantic search and document retrieval. It consists of two separate "towers" - one for encoding queries and one for encoding documents - that map text to dense vector representations in a shared embedding space.

## Usage

```python
from twotower import load_model_from_hub
from twotower.encoders import TwoTower
from twotower.tokenisers import CharTokeniser

# Load the model
model, tokenizer, config = load_model_from_hub(
    repo_id="{repo_name}",
    model_class=TwoTower,
    tokenizer_class=CharTokeniser
)

# Use for document embedding
doc_ids = tokenizer.encode("This is a document")
doc_embedding = model.encode_document(doc_ids)

# Use for query embedding
query_ids = tokenizer.encode("This is a query")
query_embedding = model.encode_query(query_ids)
```

## Training

This model was trained on the MS MARCO dataset using the Two-Tower architecture with contrastive learning.

## Repository Information

This model is part of the [Two-Tower Retrieval Model](https://github.com/yourusername/two-towers) project.
"""
    
    readme_path = "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    return readme_path

def create_model_card(repo_name="mlx7-two-tower"):
    """Create a model card file for the HuggingFace repository."""
    model_card_content = f"""---
language:
- en
tags:
- two-tower
- dual-encoder
- semantic-search
- document-retrieval
- information-retrieval
license: mit
datasets:
- ms_marco
---

# {repo_name}

This is a Two-Tower (Dual Encoder) model for document retrieval.

## Model Description

The Two-Tower model maps queries and documents to dense vector representations in the same semantic space, allowing for efficient similarity-based retrieval.

### Architecture

- **Tokenizer**: Character-level tokenization
- **Embedding**: Lookup embeddings with 64-dimensional vectors
- **Encoder**: Mean pooling with 128-dimensional hidden layer

## Intended Use

This model is designed for semantic search applications where traditional keyword matching is insufficient. It can be used to:

- Encode documents and queries into dense vector representations
- Retrieve relevant documents for a given query using vector similarity
- Build semantic search engines

## Limitations

- Limited context window (maximum sequence length of 64 tokens)
- English-language focused
- No contextual understanding beyond simple semantic similarity

## Training

- **Dataset**: MS MARCO passage retrieval dataset
- **Training Method**: Contrastive learning with triplet loss
- **Hardware**: NVIDIA GPU
"""
    
    model_card_path = "model_card.md"
    with open(model_card_path, "w") as f:
        f.write(model_card_content)
    
    return model_card_path

def main():
    """Create the mlx7-two-tower repository on HuggingFace Hub."""
    parser = argparse.ArgumentParser(description="Create mlx7-two-tower repository on HuggingFace Hub")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME,
                       help=f"Repository name (default: {DEFAULT_REPO_NAME})")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--config", type=str, default="configs/char_tower.yml",
                       help="Configuration file to upload")
    parser.add_argument("--create-all", action="store_true",
                       help="Create model, dataset, and space repositories")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    huggingface_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    
    if not huggingface_token:
        print("Error: HUGGINGFACE_ACCESS_TOKEN not found in .env file.")
        print("Please add your token to the .env file or run 'huggingface-cli login'.")
        return
    
    # Authenticate with HuggingFace Hub
    try:
        api = HfApi(token=huggingface_token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as: {username} using token from .env file")
    except Exception as e:
        print(f"Error: Failed to authenticate with HuggingFace Hub: {e}")
        print("Please check your HUGGINGFACE_ACCESS_TOKEN in .env file or run 'huggingface-cli login'.")
        return
    
    # Define repository ID
    repo_id = f"{username}/{args.repo_name}"
    
    # Create the repository
    print(f"Creating repository: {repo_id}")
    repo_url = setup_repository(
        repo_id=repo_id,
        private=args.private,
        repo_type="model",
        exist_ok=True,
        token=huggingface_token  # Pass the token
    )
    
    # Create and upload README.md
    print("Creating README.md...")
    readme_path = create_repo_readme(args.repo_name)
    
    print(f"Uploading README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=huggingface_token,  # Pass the token
        commit_message="Add README.md"
    )
    
    # Create and upload model card
    print("Creating model card...")
    model_card_path = create_model_card(args.repo_name)
    
    print(f"Uploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="model_card.md",
        repo_id=repo_id,
        token=huggingface_token,  # Pass the token
        commit_message="Add model card"
    )
    
    # Upload configuration file
    if os.path.exists(args.config):
        print(f"Uploading configuration file {args.config} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=args.config,
            path_in_repo=os.path.basename(args.config),
            repo_id=repo_id,
            token=huggingface_token,  # Pass the token
            commit_message="Add configuration file"
        )
    else:
        print(f"Warning: Configuration file {args.config} not found.")
    
    # Create dataset and space repositories if requested
    if args.create_all:
        # Create dataset repository
        dataset_repo_id = f"{username}/{args.repo_name}-data"
        print(f"Creating dataset repository: {dataset_repo_id}")
        dataset_repo_url = setup_repository(
            repo_id=dataset_repo_id,
            private=args.private,
            repo_type="dataset",
            exist_ok=True,
            token=huggingface_token  # Pass the token
        )
        
        # Create space repository
        space_repo_id = f"{username}/{args.repo_name}-demo"
        print(f"Creating demo space: {space_repo_id}")
        space_repo_url = setup_repository(
            repo_id=space_repo_id,
            private=args.private,
            repo_type="space",
            exist_ok=True,
            token=huggingface_token  # Pass the token
        )
    
    # Clean up temporary files
    os.remove(readme_path)
    os.remove(model_card_path)
    
    print("\nRepositories created successfully!")
    print(f"Model repository: {repo_url}")
    
    if args.create_all:
        print(f"Dataset repository: https://huggingface.co/{dataset_repo_id}")
        print(f"Demo space: https://huggingface.co/{space_repo_id}")
    
    print("\nNext steps:")
    print("1. Upload a trained model:")
    print(f"   python -m tools.huggingface upload-model {repo_id} path/to/model.pt")
    print(f"2. Or train a model with push_to_hub enabled:")
    print(f"   python -m twotower.train --config configs/char_tower.yml --push_to_hub --hub_repo_id {repo_id}")

if __name__ == "__main__":
    main() 
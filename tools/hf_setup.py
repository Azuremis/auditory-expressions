#!/usr/bin/env python
"""
Set up Hugging Face Hub repositories for the two-tower project.

This script creates and configures repositories for models, datasets,
and potentially a demo space on the Hugging Face Hub.
"""

import os
import argparse
import sys
from typing import Optional, List

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.huggingface import setup_repository, clone_repository
from huggingface_hub import HfApi, whoami


def setup_mlx_project(
    repo_name: str = "mlx7-two-tower",
    token: Optional[str] = None,
    private: bool = False,
    create_all: bool = False
) -> None:
    """
    Set up all repositories needed for the MLX Two-Tower project.
    
    Args:
        repo_name: Base name for repositories
        token: Hugging Face token
        private: Whether repositories should be private
        create_all: Whether to create model, dataset, and space repositories
    """
    # Get username
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"Error: Failed to authenticate with Hugging Face Hub: {e}")
        print("Please run 'huggingface-cli login' first.")
        return
    
    # Define repository IDs
    model_repo_id = f"{username}/{repo_name}"
    dataset_repo_id = f"{username}/{repo_name}-data"
    space_repo_id = f"{username}/{repo_name}-demo"
    
    # Create model repository
    print(f"Creating model repository: {model_repo_id}")
    model_url = setup_repository(
        repo_id=model_repo_id,
        private=private,
        token=token,
        repo_type="model"
    )
    
    # Create other repositories if requested
    if create_all:
        # Create dataset repository
        print(f"Creating dataset repository: {dataset_repo_id}")
        dataset_url = setup_repository(
            repo_id=dataset_repo_id,
            private=private,
            token=token,
            repo_type="dataset"
        )
        
        # Create demo space
        print(f"Creating demo space: {space_repo_id}")
        space_url = setup_repository(
            repo_id=space_repo_id,
            private=private,
            token=token,
            repo_type="space"
        )
    
    print("\nRepositories created successfully!")
    print(f"Model repository: https://huggingface.co/{model_repo_id}")
    if create_all:
        print(f"Dataset repository: https://huggingface.co/{dataset_repo_id}")
        print(f"Demo space: https://huggingface.co/{space_repo_id}")
    
    # Print instructions for next steps
    print("\nNext steps:")
    print("1. Upload model checkpoints:")
    print(f"   python -m tools.huggingface upload-model {model_repo_id} path/to/model.pt")
    print("2. Upload datasets:")
    print(f"   python -m tools.huggingface upload-dataset {dataset_repo_id} path/to/dataset")
    print("3. Upload model configurations:")
    print(f"   python -m tools.huggingface upload-config {model_repo_id} path/to/config.yml")


def main():
    """Command-line interface for setting up Hugging Face repositories."""
    parser = argparse.ArgumentParser(description="Set up HuggingFace Hub repositories for MLX Two-Tower")
    parser.add_argument("--repo-name", type=str, default="mlx7-two-tower",
                        help="Base name for the repositories")
    parser.add_argument("--private", action="store_true", 
                        help="Make repositories private")
    parser.add_argument("--create-all", action="store_true",
                        help="Create model, dataset, and space repositories")
    args = parser.parse_args()
    
    setup_mlx_project(
        repo_name=args.repo_name,
        private=args.private,
        create_all=args.create_all
    )


if __name__ == "__main__":
    main() 
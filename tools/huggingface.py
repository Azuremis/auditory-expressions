"""
HuggingFace Hub integration utilities for the Two-Tower model.

This module provides functions to create, manage, and interact with 
repositories on the Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from huggingface_hub import (
    create_repo, delete_repo, update_repo_settings, 
    HfApi, upload_file, upload_folder, Repository
)


def setup_repository(
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    repo_type: str = "model",
    exist_ok: bool = True
) -> str:
    """
    Create a repository on the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (format: username/repo_name)
        private: Whether the repository should be private
        token: Hugging Face token (if None, will use the cached token)
        repo_type: Repository type ("model", "dataset", or "space")
        exist_ok: If True, don't error if repo already exists
        
    Returns:
        URL of the created repository
    """
    # If repo_id doesn't contain '/', add the username
    if "/" not in repo_id:
        api = HfApi(token=token)
        username = api.whoami()["name"]
        repo_id = f"{username}/{repo_id}"
    
    # Create the repository
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type=repo_type,
        exist_ok=exist_ok
    )
    
    print(f"Repository created: {repo_url}")
    return repo_url


def upload_model(
    repo_id: str,
    model_path: str,
    commit_message: str = "Upload model",
    token: Optional[str] = None
) -> None:
    """
    Upload a model file to the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        model_path: Path to the model file
        commit_message: Commit message
        token: Hugging Face token
    """
    api = HfApi(token=token)
    
    # Make sure we're using Git LFS for large files
    if not os.path.exists(".gitattributes"):
        with open(".gitattributes", "w") as f:
            f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.ckpt filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    
    # Upload the model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"Uploaded {model_path} to {repo_id}")


def upload_dataset(
    repo_id: str,
    dataset_path: str,
    commit_message: str = "Upload dataset",
    token: Optional[str] = None
) -> None:
    """
    Upload a dataset file or directory to the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        dataset_path: Path to the dataset file or directory
        commit_message: Commit message
        token: Hugging Face token
    """
    api = HfApi(token=token)
    
    # Check if it's a file or directory
    path = Path(dataset_path)
    if path.is_file():
        # Upload a single file
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print(f"Uploaded {path} to {repo_id}")
    else:
        # Upload a directory
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print(f"Uploaded directory {path} to {repo_id}")


def upload_config(
    repo_id: str,
    config_path: str,
    commit_message: str = "Upload configuration",
    token: Optional[str] = None
) -> None:
    """
    Upload a configuration file to the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        config_path: Path to the configuration file
        commit_message: Commit message
        token: Hugging Face token
    """
    api = HfApi(token=token)
    
    # Upload the config file
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=os.path.basename(config_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"Uploaded {config_path} to {repo_id}")


def clone_repository(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    repo_type: str = "model"
) -> Repository:
    """
    Clone a repository from the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        local_dir: Local directory to clone to
        token: Hugging Face token
        repo_type: Repository type ("model", "dataset", or "space")
        
    Returns:
        Repository object
    """
    repo = Repository(
        local_dir=local_dir,
        clone_from=repo_id,
        token=token,
        repo_type=repo_type,
    )
    
    print(f"Cloned {repo_id} to {local_dir}")
    return repo


def main():
    """Command line interface for HuggingFace Hub operations."""
    parser = argparse.ArgumentParser(description="HuggingFace Hub utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create repository parser
    create_parser = subparsers.add_parser("create", help="Create a repository")
    create_parser.add_argument("repo_id", type=str, help="Repository ID (username/repo_name)")
    create_parser.add_argument("--private", action="store_true", help="Make repository private")
    create_parser.add_argument("--repo-type", type=str, default="model", 
                             choices=["model", "dataset", "space"], help="Repository type")
    
    # Upload model parser
    upload_model_parser = subparsers.add_parser("upload-model", help="Upload a model")
    upload_model_parser.add_argument("repo_id", type=str, help="Repository ID (username/repo_name)")
    upload_model_parser.add_argument("model_path", type=str, help="Path to model file")
    upload_model_parser.add_argument("--message", type=str, default="Upload model", 
                                   help="Commit message")
    
    # Upload dataset parser
    upload_dataset_parser = subparsers.add_parser("upload-dataset", help="Upload a dataset")
    upload_dataset_parser.add_argument("repo_id", type=str, help="Repository ID (username/repo_name)")
    upload_dataset_parser.add_argument("dataset_path", type=str, help="Path to dataset file or directory")
    upload_dataset_parser.add_argument("--message", type=str, default="Upload dataset", 
                                     help="Commit message")
    
    # Upload config parser
    upload_config_parser = subparsers.add_parser("upload-config", help="Upload a configuration")
    upload_config_parser.add_argument("repo_id", type=str, help="Repository ID (username/repo_name)")
    upload_config_parser.add_argument("config_path", type=str, help="Path to configuration file")
    upload_config_parser.add_argument("--message", type=str, default="Upload configuration", 
                                    help="Commit message")
    
    # Clone repository parser
    clone_parser = subparsers.add_parser("clone", help="Clone a repository")
    clone_parser.add_argument("repo_id", type=str, help="Repository ID (username/repo_name)")
    clone_parser.add_argument("local_dir", type=str, help="Local directory to clone to")
    clone_parser.add_argument("--repo-type", type=str, default="model", 
                            choices=["model", "dataset", "space"], help="Repository type")
    
    args = parser.parse_args()
    
    if args.command == "create":
        setup_repository(args.repo_id, args.private, repo_type=args.repo_type)
    elif args.command == "upload-model":
        upload_model(args.repo_id, args.model_path, args.message)
    elif args.command == "upload-dataset":
        upload_dataset(args.repo_id, args.dataset_path, args.message)
    elif args.command == "upload-config":
        upload_config(args.repo_id, args.config_path, args.message)
    elif args.command == "clone":
        clone_repository(args.repo_id, args.local_dir, repo_type=args.repo_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
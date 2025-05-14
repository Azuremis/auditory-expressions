#!/usr/bin/env python
"""
Create the mlx7-two-tower-data repository on HuggingFace Hub and upload curated datasets.

This script creates the dataset repository and uploads the curated datasets.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import json
import yaml
from dotenv import load_dotenv

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.huggingface import setup_repository, upload_dataset
from huggingface_hub import HfApi, create_repo

# Default repository name
DEFAULT_REPO_NAME = "mlx7-two-tower-data"

def create_dataset_card(repo_name="mlx7-two-tower-data", datasets=None):
    """Create a dataset card file for the HuggingFace repository."""
    if datasets is None:
        datasets = []
    
    # Create a human-readable representation of the datasets
    dataset_descriptions = []
    for dataset in datasets:
        name = Path(dataset).stem
        size = os.path.getsize(dataset) / (1024 * 1024)  # Convert to MB
        dataset_descriptions.append(f"- **{name}**: {size:.1f} MB")
    
    datasets_str = "\n".join(dataset_descriptions)
    
    dataset_card_content = f"""---
language:
- en
license: mit
tags:
- two-tower
- semantic-search
- document-retrieval
- information-retrieval
- dual-encoder
---

# {repo_name}

This repository contains datasets used for training Two-Tower (Dual Encoder) models for document retrieval.

## Dataset Description

The datasets provided here are structured for training dual encoder models with various sampling strategies:

{datasets_str}

### Dataset Details

- **classic_triplets.parquet**: Standard triplet format with (query, positive_document, negative_document)
- **intra_query_neg.parquet**: Negative examples selected from within the same query batch
- **multi_pos_multi_neg.parquet**: Multiple positive and negative examples per query

## Usage

```python
import pandas as pd

# Load a dataset
df = pd.read_parquet("classic_triplets.parquet")

# View the schema
print(df.columns)

# Example of working with the data
queries = df["q_text"].tolist()
positive_docs = df["d_pos_text"].tolist()
negative_docs = df["d_neg_text"].tolist()
```

## Data Source and Preparation

These datasets are derived from the MS MARCO passage retrieval dataset, processed to create effective training examples for two-tower models.

## Dataset Structure

The datasets follow a common schema with the following fields:
- `q_text`: Query text
- `d_pos_text`: Positive (relevant) document text
- `d_neg_text`: Negative (non-relevant) document text

Additional fields may be present in specific datasets.

## Citation

If you use this dataset in your research, please cite the original MS MARCO dataset:

```
@article{{msmarco,
  title={{MS MARCO: A Human Generated MAchine Reading COmprehension Dataset}},
  author={{Nguyen, Tri and Rosenberg, Matthew and Song, Xia and Gao, Jianfeng and Tiwary, Saurabh and Majumder, Rangan and Deng, Li}},
  journal={{arXiv preprint arXiv:1611.09268}},
  year={{2016}}
}}
```
"""
    
    dataset_card_path = "README.md"
    with open(dataset_card_path, "w") as f:
        f.write(dataset_card_content)
    
    return dataset_card_path

def create_metadata(datasets=None):
    """Create metadata for the datasets."""
    if datasets is None:
        datasets = []
    
    # Create metadata for the datasets
    metadata = {
        "dataset_info": {
            "description": "Datasets for training Two-Tower models",
            "citation": "@article{msmarco, title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset}, author={Nguyen, Tri and Rosenberg, Matthew and Song, Xia and Gao, Jianfeng and Tiwary, Saurabh and Majumder, Rangan and Deng, Li}, journal={arXiv preprint arXiv:1611.09268}, year={2016}}",
            "homepage": "https://huggingface.co/datasets/mlx7-two-tower-data",
            "license": "mit",
            "features": {
                "q_text": {"dtype": "string", "description": "Query text"},
                "d_pos_text": {"dtype": "string", "description": "Positive (relevant) document text"},
                "d_neg_text": {"dtype": "string", "description": "Negative (non-relevant) document text"}
            },
            "splits": {
                "train": {
                    "name": "train",
                    "num_examples": "varies by dataset",
                    "dataset_files": [os.path.basename(d) for d in datasets]
                }
            }
        }
    }
    
    metadata_path = "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def main():
    """Create the mlx7-two-tower-data repository on HuggingFace Hub and upload datasets."""
    parser = argparse.ArgumentParser(description="Create mlx7-two-tower-data repository and upload datasets")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME,
                       help=f"Repository name (default: {DEFAULT_REPO_NAME})")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--data-dir", type=str, default="data/processed/curated",
                       help="Directory containing curated datasets")
    parser.add_argument("--skip-upload", action="store_true",
                       help="Skip uploading the datasets (just create the repository)")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    huggingface_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    
    # Authenticate with HuggingFace Hub
    try:
        api = HfApi(token=huggingface_token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"Error: Failed to authenticate with HuggingFace Hub: {e}")
        print("Please check your HUGGINGFACE_ACCESS_TOKEN in .env file.")
        return
    
    # Get the datasets
    dataset_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    if not dataset_files:
        print(f"No parquet files found in {args.data_dir}")
        return
    
    print(f"Found {len(dataset_files)} dataset files: {', '.join([os.path.basename(d) for d in dataset_files])}")
    
    # Define repository ID
    repo_id = f"{username}/{args.repo_name}"
    
    # Create the repository using the direct create_repo function
    print(f"Creating dataset repository: {repo_id}")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=huggingface_token,
            private=args.private,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"Repository created: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Create and upload dataset card
    print("Creating dataset card...")
    dataset_card_path = create_dataset_card(args.repo_name, dataset_files)
    
    print(f"Uploading dataset card to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=dataset_card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=huggingface_token,
            commit_message="Add dataset card"
        )
        print("Dataset card uploaded successfully")
    except Exception as e:
        print(f"Error uploading dataset card: {e}")
    
    # Create and upload metadata
    print("Creating metadata...")
    metadata_path = create_metadata(dataset_files)
    
    print(f"Uploading metadata to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="dataset_metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=huggingface_token,
            commit_message="Add dataset metadata"
        )
        print("Metadata uploaded successfully")
    except Exception as e:
        print(f"Error uploading metadata: {e}")
    
    # Upload datasets
    if not args.skip_upload:
        # Create a data subfolder for all datasets
        data_folder = "data"
        
        for dataset_file in dataset_files:
            filename = os.path.basename(dataset_file)
            print(f"Uploading {filename} to {repo_id}/{data_folder}/...")
            
            repo_path = f"{data_folder}/{filename}"
            
            try:
                api.upload_file(
                    path_or_fileobj=dataset_file,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=huggingface_token,
                    commit_message=f"Add dataset {filename}"
                )
                print(f"Successfully uploaded {filename}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
    else:
        print("Skipping dataset upload (--skip-upload flag was set)")
    
    # Clean up temporary files
    os.remove(dataset_card_path)
    os.remove(metadata_path)
    
    print("\nRepository setup completed!")
    print(f"Dataset repository: {repo_url}")
    print("\nDatasets are organized in the 'data/' folder in the repository")
    print("\nNext steps:")
    print("1. You can view your datasets at:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print("2. To download a dataset in Python:")
    print(f"   from huggingface_hub import hf_hub_download")
    print(f"   file_path = hf_hub_download(repo_id='{repo_id}', filename='data/classic_triplets.parquet', repo_type='dataset')")
    
if __name__ == "__main__":
    main() 
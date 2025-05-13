# Auditory Expressions

A comprehensive suite of audio processing and machine learning tasks for Audio Week 5 (May 12-16, 2025).

## Overview

This project implements four audio tasks:
1. **Environmental Sound Classification** (UrbanSound8K CNN)
2. **Personalized Whisper** fine-tuning
3. **Music Note Transcription**
4. **Tiny TTS** (text-to-speech) system

## Features

- 🎵 Audio processing utilities with PyTorch and librosa
- 🤗 Seamless integration with HuggingFace datasets
- 🔊 Spectrogram extraction and augmentation
- 🧠 Neural network models for audio tasks
- 📊 Training, evaluation, and visualization tools
- 🖼️ Model inference with beautiful visualizations

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/auditory-expressions.git
cd auditory-expressions

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### Task 1: Environmental Sound Classification

```bash
# Download UrbanSound8K dataset from HuggingFace
python main.py download --task urbansound

# Train the model (using HuggingFace dataset)
python main.py train --task urbansound --config configs/urbansound_cnn.yaml

# Run inference on a random test sample
python main.py infer --task urbansound --model checkpoints/urbansound/best_model_epoch0.pt --random-test

# Or run a demo that automatically finds the latest model
python main.py demo --task urbansound
```

## Directory Structure

```
auditory-expressions/
├── artifacts/            # Documentation, model cards, etc.
├── configs/              # Configuration files
│   └── experiments/      # Experiment-specific configurations
├── data/                 # Downloaded datasets (when using local files)
├── examples/             # Example scripts and notebooks
├── src/                  # Source code
│   ├── datamodules/      # Data loading and processing
│   ├── infer/            # Inference modules
│   ├── models/           # Model architectures
│   ├── tasks/            # Task-specific training code
│   └── utils_audio.py    # Audio processing utilities
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Using HuggingFace Datasets

This project supports loading data directly from HuggingFace datasets, which eliminates the need to download large datasets manually. For the UrbanSound8K dataset, we use the `danavery/urbansound8K` dataset.

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("danavery/urbansound8K")

# Explore the dataset
print(f"Dataset loaded with {len(dataset['train'])} examples")
print(f"Dataset features: {list(dataset['train'].features.keys())}")
```

See the `examples/hf_dataset_example.py` script for a complete example.

## Configuration

Configuration files are stored in the `configs/` directory. The default configuration for UrbanSound8K is in `configs/urbansound_cnn.yaml`. You can modify these files or create your own.

Example configuration for using HuggingFace dataset:

```yaml
# UrbanSound8K CNN Classifier Configuration
device: "cuda"
use_huggingface: true  # Use HuggingFace dataset instead of local files

# Feature parameters
sample_rate: 22050
n_mels: 128
# ... other parameters ...
```

## License

MIT

## Credits

- [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) by J. Salamon, C. Jacoby, and J. P. Bello
- [HuggingFace datasets](https://huggingface.co/docs/datasets/index) for providing the dataset API
- [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio) for audio processing and deep learning
- [librosa](https://librosa.org/) for audio feature extraction

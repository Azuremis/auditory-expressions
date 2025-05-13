"""
Example of using the HuggingFace dataset for UrbanSound8K.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, Audio
import librosa
import librosa.display

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils_audio import mel_spectrogram, normalize_audio

def main():
    """Main function"""
    print("Loading UrbanSound8K dataset from HuggingFace...")
    dataset = load_dataset("danavery/urbansound8K")
    
    # Print dataset info
    print(f"Dataset loaded with {len(dataset['train'])} examples")
    print(f"Dataset features: {list(dataset['train'].features.keys())}")
    
    # Cast fold to int if needed
    if not isinstance(dataset["train"][0]["fold"], int):
        print("Converting fold column to int...")
        dataset = dataset.cast_column("fold", int)
    
    # Cast class ID to int if needed
    if not isinstance(dataset["train"][0]["classID"], int):
        print("Converting classID column to int...")
        dataset = dataset.cast_column("classID", int)
    
    # Add audio loading if needed
    if "audio" not in dataset["train"].features:
        print("Adding audio loading to dataset...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=22050))
    
    # Get class distribution
    class_counts = {}
    for example in dataset["train"]:
        class_id = example["classID"]
        class_name = example["class"]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} examples")
    
    # Get fold distribution
    fold_counts = {}
    for example in dataset["train"]:
        fold = example["fold"]
        if fold not in fold_counts:
            fold_counts[fold] = 0
        fold_counts[fold] += 1
    
    print("\nFold distribution:")
    for fold, count in sorted(fold_counts.items()):
        print(f"  Fold {fold}: {count} examples")
    
    # Plot an example waveform and spectrogram
    print("\nVisualize an example...")
    example = dataset["train"][0]
    audio = example["audio"]
    waveform = torch.from_numpy(audio["array"]).float()
    
    # Convert stereo to mono if needed
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    
    # Normalize audio
    waveform = normalize_audio(waveform)
    
    # Extract mel spectrogram
    spec = mel_spectrogram(
        waveform,
        sample_rate=audio["sampling_rate"],
        n_fft=2048,
        hop_length=551,  # 25ms at 22050Hz
        win_length=1103,  # 50ms at 22050Hz
        n_mels=128
    )
    
    # Plot waveform and spectrogram
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(waveform.numpy()[0])
    plt.title(f"Waveform - {example['class']} (ClassID: {example['classID']})")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        spec.numpy()[0],
        x_axis='time',
        y_axis='mel',
        sr=audio["sampling_rate"],
        hop_length=551  # Updated to match the hop_length above
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram - {example['slice_file_name']}")
    plt.tight_layout()
    
    # Save or show the plot
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "hf_dataset_example.png"))
    print(f"Example visualization saved to examples/output/hf_dataset_example.png")

if __name__ == "__main__":
    main() 
"""
UrbanSound8K dataset and datamodule using Hugging Face datasets library.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from typing import Dict, Tuple, Optional, List, Union, Callable
import logging
from datasets import load_dataset, Audio

from src.utils_audio import mel_spectrogram, normalize_audio

logger = logging.getLogger(__name__)

class HFUrbanSoundDataset(Dataset):
    """
    Dataset for UrbanSound8K environmental sound classification using Hugging Face datasets.
    
    Features:
    - Loads audio files from HuggingFace dataset repository
    - Performs mel spectrogram extraction
    - Supports spectrogram caching for faster training
    - Implements data augmentation during training
    """
    
    CLASS_MAPPING = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music"
    }
    
    def __init__(
        self,
        fold: int = 1,
        train: bool = True,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        transform: Optional[Callable] = None,
        cache_spectrograms: bool = True,
        max_duration: float = 4.0,  # in seconds
    ):
        """
        Initialize UrbanSound8K dataset using HuggingFace
        
        Args:
            fold: Fold to use as validation (1-10)
            train: If True, use all folds except the validation fold
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop size in samples
            win_length: Window size in samples
            transform: Optional transformation function
            cache_spectrograms: If True, cache spectrograms in memory
            max_duration: Maximum audio duration in seconds
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.train = train
        self.transform = transform
        self.max_samples = int(max_duration * sample_rate)
        self.cache_spectrograms = cache_spectrograms
        self.fold = fold
        
        # Load the dataset from HuggingFace
        logger.info("Loading UrbanSound8K dataset from HuggingFace...")
        self.dataset = load_dataset("danavery/urbansound8K")
        
        # Cast the fold column to int if needed
        if not isinstance(self.dataset["train"][0]["fold"], int):
            logger.info("Converting fold column to int...")
            self.dataset = self.dataset.cast_column("fold", int)
        
        # Cast class ID to int
        if not isinstance(self.dataset["train"][0]["classID"], int):
            logger.info("Converting classID column to int...")
            self.dataset = self.dataset.cast_column("classID", int)
        
        # Add audio loading
        if "audio" not in self.dataset["train"].features:
            logger.info("Adding audio loading to dataset...")
            self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        
        # Filter the dataset based on fold
        self._filter_by_fold()
        
        # Initialize spectrogram cache
        self.spectrogram_cache = {}
    
    def _filter_by_fold(self):
        """Filter dataset based on current fold"""
        logger.info(f"Filtering dataset for fold {self.fold}, train={self.train}")
        if self.train:
            self.filtered_dataset = self.dataset["train"].filter(lambda x: x["fold"] != self.fold)
        else:
            self.filtered_dataset = self.dataset["train"].filter(lambda x: x["fold"] == self.fold)
        
        logger.info(f"Dataset size: {len(self.filtered_dataset)} examples")
    
    def set_fold(self, fold: int):
        """Change the test fold and refilter the dataset"""
        if fold == self.fold:
            return  # No change needed
            
        logger.info(f"Changing test fold from {self.fold} to {fold}")
        self.fold = fold
        self._filter_by_fold()
        
        # Clear the spectrogram cache when changing folds
        if self.cache_spectrograms:
            self.spectrogram_cache = {}
    
    def __len__(self):
        return len(self.filtered_dataset)
    
    def _process_audio(self, audio):
        """
        Process audio data to create a waveform tensor
        
        Args:
            audio: Audio data from HuggingFace dataset
            
        Returns:
            Processed waveform tensor
        """
        # Get waveform from audio dict
        waveform = torch.from_numpy(audio["array"]).float()
        
        # Convert stereo to mono if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize audio
        waveform = normalize_audio(waveform)
        
        # Resample if needed (should already be at the right sample rate from HF)
        if audio["sampling_rate"] != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, audio["sampling_rate"], self.sample_rate
            )
        
        # Pad or trim to max_duration
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad_length = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def __getitem__(self, idx):
        """
        Get audio sample and class label
        """
        item = self.filtered_dataset[idx]
        class_id = item["classID"]
        
        # Check if spectrogram is in cache
        if self.cache_spectrograms and idx in self.spectrogram_cache:
            spectrogram = self.spectrogram_cache[idx]
        else:
            # Load and process audio
            audio = item["audio"]
            waveform = self._process_audio(audio)
            
            # Compute mel spectrogram
            spectrogram = mel_spectrogram(
                waveform,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels
            )
            
            # Store in cache if enabled
            if self.cache_spectrograms:
                self.spectrogram_cache[idx] = spectrogram
        
        # Apply transformations (augmentations) if in training mode
        if self.transform is not None and self.train:
            spectrogram = self.transform(spectrogram)
        
        # Add channel dimension for CNN if needed
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        
        return {
            'spectrogram': spectrogram,
            'label': class_id,
            'class_name': self.CLASS_MAPPING[class_id],
            'file_name': item["slice_file_name"]
        }


class HFUrbanSoundDataModule:
    """
    PyTorch DataModule for UrbanSound8K dataset using HuggingFace datasets
    """
    
    def __init__(
        self,
        test_fold: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        cache_spectrograms: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
    ):
        """
        Initialize UrbanSound8K datamodule using HuggingFace datasets
        
        Args:
            test_fold: Fold to use for testing/validation (1-10)
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop size in samples
            win_length: Window size in samples
            cache_spectrograms: If True, cache spectrograms in memory
            train_transform: Transformations to apply to training data
            val_transform: Transformations to apply to validation data
        """
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.cache_spectrograms = cache_spectrograms
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Set up train and validation datasets"""
        self.train_dataset = HFUrbanSoundDataset(
            fold=self.test_fold,
            train=True,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            transform=self.train_transform,
            cache_spectrograms=self.cache_spectrograms
        )
        
        self.val_dataset = HFUrbanSoundDataset(
            fold=self.test_fold,
            train=False,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            transform=self.val_transform,
            cache_spectrograms=self.cache_spectrograms
        )
    
    def set_fold(self, fold: int):
        """Change the test fold for cross-validation"""
        if fold == self.test_fold:
            return  # No change needed
            
        logger.info(f"Changing test fold from {self.test_fold} to {fold}")
        self.test_fold = fold
        
        # Update datasets with new fold
        if self.train_dataset is not None:
            self.train_dataset.set_fold(fold)
        
        if self.val_dataset is not None:
            self.val_dataset.set_fold(fold)
    
    def train_dataloader(self):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 
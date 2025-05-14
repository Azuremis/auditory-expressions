"""
UrbanSound8K dataset and datamodule.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
from typing import Dict, Tuple, Optional, List, Union, Callable
import pickle
import logging
from tqdm import tqdm

from src.utils_audio import load_audio, mel_spectrogram, normalize_audio

logger = logging.getLogger(__name__)

class UrbanSoundDataset(Dataset):
    """
    Dataset for UrbanSound8K environmental sound classification.
    
    Features:
    - Loads audio files from the UrbanSound8K dataset
    - Performs mel spectrogram extraction
    - Supports spectrogram caching to disk for faster training
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
        data_dir: str,
        fold: int = 1,
        train: bool = True,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_duration: float = 4.0,  # in seconds
    ):
        """
        Initialize UrbanSound8K dataset
        
        Args:
            data_dir: Root directory of UrbanSound8K dataset
            fold: Fold to use as validation (1-10)
            train: If True, use all folds except the validation fold
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop size in samples
            win_length: Window size in samples
            use_cache: If True, cache spectrograms to disk
            cache_dir: Directory to store cached spectrograms
            transform: Optional transformation function
            max_duration: Maximum audio duration in seconds
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata" / "UrbanSound8K.csv"
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.train = train
        self.transform = transform
        self.max_samples = int(max_duration * sample_rate)
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Filter data by fold for train/validation split
        if train:
            self.metadata = self.metadata[self.metadata['fold'] != fold]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == fold]
            
        # Configure caching
        self.use_cache = use_cache
        if cache_dir is None:
            cache_dir = self.data_dir / "cache"
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create cache filename based on parameters
        self.cache_filename = f"urbansound_fold{fold}_{'train' if train else 'val'}_sr{sample_rate}_mel{n_mels}_hop{hop_length}.pkl"
        self.cache_path = self.cache_dir / self.cache_filename
        
        # Check if cache exists and load it, or create it
        if self.use_cache and os.path.exists(self.cache_path):
            logger.info(f"Loading cached spectrograms from {self.cache_path}")
            self._load_cache()
        else:
            self.cached_spectrograms = {}
            if self.use_cache:
                logger.info(f"Cache not found. Will create cache at {self.cache_path}")
                self._create_cache()
    
    def _load_cache(self):
        """Load cached spectrograms from disk"""
        with open(self.cache_path, 'rb') as f:
            self.cached_spectrograms = pickle.load(f)
    
    def _create_cache(self):
        """Create and save spectrogram cache to disk"""
        logger.info("Generating spectrogram cache. This may take a while...")
        
        for idx in tqdm(range(len(self.metadata))):
            row = self.metadata.iloc[idx]
            audio_path = self.data_dir / f"fold{row['fold']}" / row['slice_file_name']
            
            # Load and preprocess audio
            waveform = self._load_and_preprocess_audio(audio_path)
            
            # Compute mel spectrogram
            spec = mel_spectrogram(
                waveform, 
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels
            )
            
            # Store in cache
            self.cached_spectrograms[idx] = {
                'spectrogram': spec.cpu().numpy(),
                'class_id': row['classID']
            }
        
        # Save cache to disk
        logger.info(f"Saving cache to {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cached_spectrograms, f)
    
    def _load_and_preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            waveform = load_audio(audio_path, sample_rate=self.sample_rate)
            waveform = normalize_audio(waveform)
            
            # Pad or trim to max_duration
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[1] < self.max_samples:
                pad_length = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                
            return waveform
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            # Return zero audio as fallback
            return torch.zeros(1, self.max_samples)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get audio sample and class label
        
        If using cache, retrieves spectrogram from cache.
        Otherwise, computes spectrogram on-the-fly.
        """
        row = self.metadata.iloc[idx]
        
        if self.use_cache and idx in self.cached_spectrograms:
            # Get from cache
            item = self.cached_spectrograms[idx]
            spectrogram = torch.from_numpy(item['spectrogram']).float()
            class_id = item['class_id']
        else:
            # Compute on-the-fly
            audio_path = self.data_dir / f"fold{row['fold']}" / row['slice_file_name']
            waveform = self._load_and_preprocess_audio(audio_path)
            
            spectrogram = mel_spectrogram(
                waveform, 
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels
            )
            
            class_id = row['classID']
        
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
            'file_name': row['slice_file_name']
        }


class UrbanSoundDataModule:
    """
    PyTorch DataModule for UrbanSound8K dataset
    """
    
    def __init__(
        self,
        data_dir: str,
        test_fold: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
    ):
        """
        Initialize UrbanSound8K datamodule
        
        Args:
            data_dir: Root directory of UrbanSound8K dataset
            test_fold: Fold to use for testing/validation (1-10)
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop size in samples
            win_length: Window size in samples
            use_cache: If True, cache spectrograms to disk
            cache_dir: Directory to store cached spectrograms
            train_transform: Transformations to apply to training data
            val_transform: Transformations to apply to validation data
        """
        self.data_dir = data_dir
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Set up train and validation datasets"""
        self.train_dataset = UrbanSoundDataset(
            data_dir=self.data_dir,
            fold=self.test_fold,
            train=True,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            transform=self.train_transform,
        )
        
        self.val_dataset = UrbanSoundDataset(
            data_dir=self.data_dir,
            fold=self.test_fold,
            train=False,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            transform=self.val_transform,
        )
    
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
"""
Audio processing utilities for Audio Week 5 project.
Includes STFT, log-mel spectrogram extraction, and audio augmentation functions.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import random
from typing import Optional, Tuple, Dict, List, Union, Any

# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def load_audio(file_path: str, sample_rate: int = 22050) -> torch.Tensor:
    """
    Load an audio file and convert to target sample rate.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        torch.Tensor: Audio waveform (channels, time)
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalize audio to range [-1, 1]
    
    Args:
        waveform: Audio tensor (channels, time)
        
    Returns:
        torch.Tensor: Normalized audio
    """
    if waveform.abs().max() > 0:
        return waveform / waveform.abs().max()
    return waveform

def stft(waveform: torch.Tensor, 
         n_fft: int = 1024, 
         hop_length: int = 512, 
         win_length: int = 1024) -> torch.Tensor:
    """
    Compute Short-Time Fourier Transform
    
    Args:
        waveform: Audio tensor (channels, time)
        n_fft: FFT size
        hop_length: Hop size in samples
        win_length: Window size in samples
        
    Returns:
        torch.Tensor: Complex STFT (channels, freq, time)
    """
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(waveform.device),
        return_complex=True
    )

def mel_spectrogram(waveform: torch.Tensor,
                    sample_rate: int = 22050,
                    n_fft: int = 1024,
                    hop_length: int = 512,
                    win_length: int = 1024,
                    n_mels: int = 128,
                    power: float = 2.0,
                    normalized: bool = True) -> torch.Tensor:
    """
    Compute mel spectrogram from raw audio
    
    Args:
        waveform: Audio tensor (channels, time)
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop size in samples
        win_length: Window size in samples
        n_mels: Number of mel bins
        power: Power to raise magnitude spectrogram to
        normalized: Whether to normalize the mel spectrogram
        
    Returns:
        torch.Tensor: Mel spectrogram (channels, n_mels, time)
    """
    # Convert parameters in ms to samples if needed
    if hop_length is None:
        hop_length = win_length // 4
        
    # Mono conversion if needed
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Calculate mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    
    # Normalize if requested
    if normalized:
        mel_spec = normalize_spectrogram(mel_spec)
        
    return mel_spec

def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Normalize spectrogram to [0, 1] range
    
    Args:
        spectrogram: Spectrogram tensor
        
    Returns:
        torch.Tensor: Normalized spectrogram
    """
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    
    if max_val > min_val:
        return (spectrogram - min_val) / (max_val - min_val)
    return spectrogram

def ms_to_samples(ms: float, sample_rate: int) -> int:
    """
    Convert milliseconds to samples
    
    Args:
        ms: Time in milliseconds
        sample_rate: Sample rate in Hz
        
    Returns:
        int: Number of samples
    """
    return int(ms * sample_rate / 1000)

# -----------------------------------------------------------------------------
# Audio augmentation
# -----------------------------------------------------------------------------

def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Time stretch audio without changing pitch
    
    Args:
        waveform: Audio tensor (channels, time)
        rate: Stretch factor: > 1 = faster, < 1 = slower
        
    Returns:
        torch.Tensor: Time-stretched audio
    """
    # Convert to numpy for librosa processing
    waveform_np = waveform.numpy()
    
    # Time stretch
    if waveform.size(0) == 1:  # Mono
        waveform_np = librosa.effects.time_stretch(y=waveform_np[0], rate=rate)
        waveform_np = waveform_np[np.newaxis, :]
    else:  # Multi-channel
        waveform_np = np.stack([
            librosa.effects.time_stretch(y=channel, rate=rate)
            for channel in waveform_np
        ])
    
    return torch.from_numpy(waveform_np)

def pitch_shift(waveform: torch.Tensor, sample_rate: int, steps: float) -> torch.Tensor:
    """
    Shift pitch without changing tempo
    
    Args:
        waveform: Audio tensor (channels, time)
        sample_rate: Sample rate in Hz
        steps: Number of semitones to shift (can be fractional)
        
    Returns:
        torch.Tensor: Pitch-shifted audio
    """
    # Convert to numpy for librosa processing
    waveform_np = waveform.numpy()
    
    # Pitch shift
    if waveform.size(0) == 1:  # Mono
        waveform_np = librosa.effects.pitch_shift(
            y=waveform_np[0], sr=sample_rate, n_steps=steps
        )
        waveform_np = waveform_np[np.newaxis, :]
    else:  # Multi-channel
        waveform_np = np.stack([
            librosa.effects.pitch_shift(y=channel, sr=sample_rate, n_steps=steps)
            for channel in waveform_np
        ])
    
    return torch.from_numpy(waveform_np)

def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    """
    Add Gaussian noise to audio
    
    Args:
        waveform: Audio tensor (channels, time)
        noise_level: Noise amplitude (0.0-1.0)
        
    Returns:
        torch.Tensor: Noisy audio
    """
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

# -----------------------------------------------------------------------------
# Spectrogram augmentation
# -----------------------------------------------------------------------------

def spec_augment(spec: torch.Tensor, 
                 freq_mask_param: int = 10,
                 time_mask_param: int = 10,
                 freq_mask_count: int = 1,
                 time_mask_count: int = 1) -> torch.Tensor:
    """
    Apply SpecAugment to spectrogram
    
    Args:
        spec: Spectrogram tensor (batch, channels, freq, time) or (batch, freq, time)
        freq_mask_param: Maximum width of frequency mask
        time_mask_param: Maximum width of time mask
        freq_mask_count: Number of frequency masks
        time_mask_count: Number of time masks
        
    Returns:
        torch.Tensor: Augmented spectrogram
    """
    aug_spec = spec.clone()
    
    # Handle different input dimensions
    if len(spec.shape) == 4:  # (batch, channels, freq, time)
        batch_size, n_channels, n_freq, n_time = spec.shape
        reshape_needed = True
        # Reshape to (batch*channels, freq, time)
        aug_spec = aug_spec.reshape(batch_size*n_channels, n_freq, n_time)
    else:
        reshape_needed = False
    
    # Apply frequency masks
    for _ in range(freq_mask_count):
        if freq_mask_param > 0:
            for idx in range(aug_spec.shape[0]):
                f_param = min(freq_mask_param, aug_spec.shape[1])
                f = random.randint(0, f_param)
                f0 = random.randint(0, aug_spec.shape[1] - f)
                aug_spec[idx, f0:f0+f, :] = 0
    
    # Apply time masks
    for _ in range(time_mask_count):
        if time_mask_param > 0:
            for idx in range(aug_spec.shape[0]):
                t_param = min(time_mask_param, aug_spec.shape[2])
                t = random.randint(0, t_param)
                t0 = random.randint(0, aug_spec.shape[2] - t)
                aug_spec[idx, :, t0:t0+t] = 0
    
    # Reshape back if needed
    if reshape_needed:
        aug_spec = aug_spec.reshape(batch_size, n_channels, n_freq, n_time)
    
    return aug_spec

# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------

def chunk_audio(waveform: torch.Tensor, 
                chunk_size: int, 
                hop_size: int = None) -> List[torch.Tensor]:
    """
    Split audio into overlapping chunks
    
    Args:
        waveform: Audio tensor (channels, time)
        chunk_size: Chunk length in samples
        hop_size: Hop size in samples, defaults to chunk_size//2
        
    Returns:
        List[torch.Tensor]: List of audio chunks
    """
    if hop_size is None:
        hop_size = chunk_size // 2
    
    chunks = []
    for i in range(0, waveform.shape[1] - chunk_size + 1, hop_size):
        chunks.append(waveform[:, i:i+chunk_size])
    
    # Handle the last chunk if needed
    if waveform.shape[1] > chunks[-1].shape[1]:
        last_chunk = waveform[:, -chunk_size:]
        chunks.append(last_chunk)
    
    return chunks

def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in a batch to the specified device
    
    Args:
        batch: Dictionary of tensors
        device: Target device
        
    Returns:
        Dict[str, torch.Tensor]: Batch on target device
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()} 
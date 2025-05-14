"""
Transformer-based audio classifier that applies 1D convolution followed 
by transformer encoder blocks for sound classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Dict, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AudioConvTransformer(nn.Module):
    """
    Audio classifier using 1D convolution followed by transformer encoder blocks.
    
    Architecture:
    1. 1D Convolution on raw audio or spectrogram features
    2. Multiple transformer encoder blocks
    3. Classification head
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # Number of mel bins or audio features
        seq_len: int = 400,    # Sequence length (time dimension)
        conv_channels: int = 128,
        conv_kernel_size: int = 7,
        conv_stride: int = 1,
        transformer_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        num_classes: int = 10,
        input_is_spectrogram: bool = True,
    ):
        super().__init__()
        self.input_is_spectrogram = input_is_spectrogram
        
        # For spectrogram input (B, 1, F, T), we reshape to (B, F, T) 
        # before applying 1D convolution
        
        # 1D Convolution
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_kernel_size // 2
        )
        
        # Calculate sequence length after conv
        self.out_seq_len = seq_len  # If stride is 1 and padding is kernel_size//2
        
        # Project to transformer dimension
        self.projection = nn.Linear(conv_channels, transformer_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=transformer_dim,
            dropout=dropout,
            max_len=self.out_seq_len
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (B, 1, F, T) for spectrogram or (B, 1, T) for raw audio
            
        Returns:
            Dict containing logits and intermediate representations
        """
        batch_size = x.shape[0]
        
        # Handle spectrogram input - reshape from (B, 1, F, T) to (B, F, T)
        if self.input_is_spectrogram:
            if x.dim() == 4:  # (B, C, F, T)
                x = x.squeeze(1)  # Remove channel dim if present
        
        # Apply 1D convolution (B, F, T) -> (B, conv_channels, T)
        x = self.conv(x)
        
        # Reshape to (B, T, conv_channels) for transformer
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.projection(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'embeddings': x  # Return embeddings for optional use
        }

def get_transformer_audio_model(
    input_dim: int = 128,
    seq_len: int = 400,
    conv_channels: int = 128,
    transformer_dim: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    dropout: float = 0.3,
    num_classes: int = 10,
    input_is_spectrogram: bool = True,
) -> nn.Module:
    """
    Factory function to create a new AudioConvTransformer model
    
    Args:
        input_dim: Number of input features (mel bins for spectrogram)
        seq_len: Sequence length in time dimension
        conv_channels: Number of channels in 1D convolution
        transformer_dim: Dimension of transformer layers
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dropout: Dropout rate
        num_classes: Number of output classes
        input_is_spectrogram: Whether input is a spectrogram (otherwise raw audio)
        
    Returns:
        Initialized AudioConvTransformer model
    """
    model = AudioConvTransformer(
        input_dim=input_dim,
        seq_len=seq_len,
        conv_channels=conv_channels,
        transformer_dim=transformer_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=num_classes,
        input_is_spectrogram=input_is_spectrogram
    )
    
    return model 
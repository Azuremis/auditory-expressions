"""
CNN models for audio classification.
Includes a ResNet-based model adapted for spectrogram input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Union, List, Any

class AudioResNet(nn.Module):
    """
    ResNet model adapted for audio spectrogram classification.
    Uses a pre-trained ResNet with the first layer modified to accept single-channel
    spectrograms instead of 3-channel images.
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        feature_dim: int = 512
    ):
        """
        Initialize AudioResNet
        
        Args:
            model_name: Name of the ResNet variant to use ('resnet18', 'resnet34', etc.)
            num_classes: Number of output classes
            pretrained: Whether to use weights pre-trained on ImageNet
            in_channels: Number of input channels (1 for mono spectrograms)
            dropout_rate: Dropout rate before the final classification layer
            feature_dim: Dimension of features before the final classification layer
        """
        super().__init__()
        
        # Load the pre-trained model
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        
        if model_name == "resnet18":
            self.base_model = models.resnet18(weights=weights)
            self.feature_dim = feature_dim
        elif model_name == "resnet34":
            self.base_model = models.resnet34(weights=weights)
            self.feature_dim = feature_dim
        elif model_name == "resnet50":
            self.base_model = models.resnet50(weights=weights)
            self.feature_dim = 2048  # ResNet50 has larger feature dim
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Replace the first convolutional layer to accept in_channels
        if in_channels != 3:  # Default RGB has 3 channels
            # Get the pre-trained weights for the first convolutional layer
            if pretrained:
                orig_conv = self.base_model.conv1
                # Initialize new conv with same config but different in_channels
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=orig_conv.bias is not None
                )
                
                # Copy weights from pre-trained model for single channel
                if in_channels == 1:
                    # For 1-channel input, use mean of original channels
                    new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
                    if orig_conv.bias is not None:
                        new_conv.bias.data = orig_conv.bias.data
                
                self.base_model.conv1 = new_conv
            else:
                # Just replace without special weight handling if not pre-trained
                self.base_model.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.base_model.conv1.out_channels,
                    kernel_size=self.base_model.conv1.kernel_size,
                    stride=self.base_model.conv1.stride,
                    padding=self.base_model.conv1.padding,
                    bias=self.base_model.conv1.bias is not None
                )
        
        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the final FC layer
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dict containing:
                - logits: Raw model outputs before softmax
                - probabilities: Softmax probabilities
        """
        features = self.base_model(x)
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            "logits": logits,
            "probabilities": probabilities
        }
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the model (before classification layer)
        
        Args:
            x: Input tensor
            
        Returns:
            Features tensor
        """
        return self.base_model(x)


class SpectrogramCNN(nn.Module):
    """
    A simple CNN model for spectrogram classification.
    Alternative to the ResNet-based model, with a simpler architecture.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        dropout_rate: float = 0.5
    ):
        """
        Initialize SpectrogramCNN
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Adjust based on input size
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dict containing logits and probabilities
        """
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Probabilities
        probabilities = F.softmax(logits, dim=1)
        
        return {
            "logits": logits,
            "probabilities": probabilities
        }


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model instance
    
    Args:
        model_name: Model architecture to use
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Model instance
    """
    if model_name in ["resnet18", "resnet34", "resnet50"]:
        return AudioResNet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_name == "cnn":
        return SpectrogramCNN(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}") 
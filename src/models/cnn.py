"""
CNN Model for GTSRB Traffic Sign Recognition.
A straightforward convolutional neural network with 3 conv blocks.
"""

import torch.nn as nn


class TrafficSignCNN(nn.Module):
    """Convolutional Neural Network for traffic sign classification.
    
    Architecture:
        Conv Block 1: Conv2D(3, 32) -> BatchNorm -> ReLU -> MaxPool
        Conv Block 2: Conv2D(32, 64) -> BatchNorm -> ReLU -> MaxPool
        Conv Block 3: Conv2D(64, 128) -> BatchNorm -> ReLU -> MaxPool
        Classifier:   Flatten -> Linear -> ReLU -> Dropout -> Linear(43)
    
    Designed to be lightweight enough for RTX 4050 (6GB VRAM).
    
    Args:
        num_classes: Number of output classes (default: 43 for GTSRB)
        dropout: Dropout rate for regularization (default: 0.3)
        image_size: Input image size (default: 32)
    """
    
    def __init__(self, num_classes=43, dropout=0.3, image_size=32):
        super(TrafficSignCNN, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # ---- Convolutional Feature Extractor ----
        
        # Block 1: 3 -> 32 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )
        
        # Block 2: 32 -> 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # Block 3: 64 -> 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # ---- Classifier ----
        # After 3 MaxPool layers: image_size / (2^3) = spatial_size
        spatial_size = image_size // 8  # 32 -> 4, 48 -> 6
        flatten_size = 128 * spatial_size * spatial_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
            # No softmax here - CrossEntropyLoss expects raw logits
        )
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, image_size, image_size)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with raw logits
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

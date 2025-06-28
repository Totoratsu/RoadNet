"""
Neural network model for driving decision making based on segmentation masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class DecisionCNN(nn.Module):
    """
    Convolutional Neural Network for predicting driving decisions from segmentation masks.
    
    Architecture:
    - Uses a pre-trained ResNet18 backbone
    - Custom classifier head for 3-class classification (front, left, right)
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize the decision CNN.
        
        Args:
            num_classes: Number of output classes (3 for front/left/right)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
        """
        super(DecisionCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features from ResNet18
        self.feature_size = 512
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights for custom layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Classify using the custom head
        output = self.classifier(features)
        
        return output


class SimpleCNN(nn.Module):
    """
    Simple CNN model for driving decision making.
    Lighter alternative to ResNet-based model.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        """
        Initialize the simple CNN.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(model_type: str = 'resnet', num_classes: int = 3, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('resnet' or 'simple')
        num_classes: Number of output classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        PyTorch model
    """
    if model_type.lower() == 'resnet':
        return DecisionCNN(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'simple':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet' or 'simple'.")


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ResNet-based model
    model_resnet = DecisionCNN()
    model_resnet.to(device)
    
    # Test simple CNN
    model_simple = SimpleCNN()
    model_simple.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output_resnet = model_resnet(dummy_input)
        output_simple = model_simple(dummy_input)
        
        print(f"ResNet model output shape: {output_resnet.shape}")
        print(f"Simple model output shape: {output_simple.shape}")
        
        # Print model parameters
        resnet_params = sum(p.numel() for p in model_resnet.parameters())
        simple_params = sum(p.numel() for p in model_simple.parameters())
        
        print(f"ResNet model parameters: {resnet_params:,}")
        print(f"Simple model parameters: {simple_params:,}")

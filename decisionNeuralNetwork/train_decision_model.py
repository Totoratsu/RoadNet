"""
Training script for the driving decision neural network.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random

from decision_dataset import DecisionDataset, LABEL_MAPPING
from decision_model import create_model


class GaussianNoise:
    """Callable class for adding Gaussian noise."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomErase:
    """Callable class for random patch erasing."""
    
    def __init__(self, p: float = 0.3, scale: tuple = (0.02, 0.1)):
        self.p = p
        self.scale = scale
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
            
        _, h, w = tensor.shape
        area = h * w
        
        for _ in range(random.randint(1, 3)):  # 1-3 patches
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            patch_h = int(round((target_area * aspect_ratio) ** 0.5))
            patch_w = int(round((target_area / aspect_ratio) ** 0.5))
            
            if patch_w < w and patch_h < h:
                top = random.randint(0, h - patch_h)
                left = random.randint(0, w - patch_w)
                
                # Fill with mean pixel value
                tensor[:, top:top+patch_h, left:left+patch_w] = tensor.mean()
        
        return tensor


def add_gaussian_noise(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor
        mean: Mean of the noise
        std: Standard deviation of the noise
    
    Returns:
        Tensor with added noise
    """
    noise = torch.randn_like(tensor) * std + mean
    return torch.clamp(tensor + noise, 0.0, 1.0)


def random_erase_patches(tensor: torch.Tensor, p: float = 0.3, scale: tuple = (0.02, 0.1)) -> torch.Tensor:
    """
    Randomly erase rectangular patches from the image (similar to cutout).
    Useful for making the model more robust to occlusions.
    
    Args:
        tensor: Input tensor of shape (C, H, W)
        p: Probability of applying the transformation
        scale: Range of proportion of erased area against input image
    
    Returns:
        Tensor with random patches erased
    """
    if random.random() > p:
        return tensor
        
    _, h, w = tensor.shape
    area = h * w
    
    for _ in range(random.randint(1, 3)):  # 1-3 patches
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(0.3, 3.3)
        
        patch_h = int(round((target_area * aspect_ratio) ** 0.5))
        patch_w = int(round((target_area / aspect_ratio) ** 0.5))
        
        if patch_w < w and patch_h < h:
            top = random.randint(0, h - patch_h)
            left = random.randint(0, w - patch_w)
            
            # Fill with random color or mean color
            if random.random() > 0.5:
                # Fill with mean pixel value
                tensor[:, top:top+patch_h, left:left+patch_w] = tensor.mean()
            else:
                # Fill with random color
                tensor[:, top:top+patch_h, left:left+patch_w] = torch.rand(3, 1, 1)
    
    return tensor


def create_geometric_only_augmentation_transform(image_size: tuple, augmentation_strength: str = 'medium'):
    """
    Create augmentation transforms that only use geometric transformations.
    NO COLOR AUGMENTATIONS - preserves segmentation mask colors.
    
    Args:
        image_size: Target image size (height, width)
        augmentation_strength: 'light', 'medium', or 'heavy'
    
    Returns:
        Composed transform for LEFT/RIGHT classes only
    """
    if augmentation_strength == 'light':
        return transforms.Compose([
            transforms.Resize(image_size),
            # Light geometric augmentations only
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-3, 3), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == 'heavy':
        return transforms.Compose([
            transforms.Resize(image_size),
            # Heavy geometric augmentations only
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(degrees=(-8, 8), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
                shear=(-4, 4),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            # Add slight noise for robustness (no color change)
            GaussianNoise(mean=0.0, std=0.005),
            # Random erasing for occlusion robustness
            RandomErase(p=0.4, scale=(0.02, 0.12)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:  # medium (default)
        return transforms.Compose([
            transforms.Resize(image_size),
            # Medium geometric augmentations only
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(degrees=(-5, 5), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=(-2, 2),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            # Minimal noise addition
            GaussianNoise(mean=0.0, std=0.003),
            # Light random erasing
            RandomErase(p=0.3, scale=(0.02, 0.08)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_advanced_augmentation_transform(image_size: tuple, augmentation_strength: str = 'medium'):
    """
    Create advanced augmentation transforms based on strength level.
    This is the old function kept for backward compatibility.
    
    Args:
        image_size: Target image size (height, width)
        augmentation_strength: 'light', 'medium', or 'heavy'
    
    Returns:
        Composed transform
    """
    # For class-specific augmentation, we use geometric-only transforms
    return create_geometric_only_augmentation_transform(image_size, augmentation_strength)


class DecisionTrainer:
    """
    Trainer class for the driving decision neural network.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            device: Device to train on
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Train Batch: {batch_idx}/{len(self.train_loader)} '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy, all_targets, all_predictions)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for detailed metrics
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, np.array(all_targets), np.array(all_predictions)
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints') -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_targets, val_predictions = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }, best_model_path)
                print(f"💾 Saved new best model with validation accuracy: {val_acc:.2f}%")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Print detailed validation metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                print("\nValidation Classification Report:")
                print(classification_report(val_targets, val_predictions, 
                                          target_names=list(LABEL_MAPPING.values())))
            
            print('-' * 60)
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
            }
        }, final_model_path)
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Models saved in: {save_dir}")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_acc
        }


def create_data_loaders(
    data_dir: str,
    labels_file: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    augmentation_strength: str = 'medium',
    class_specific_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders with class-specific augmentation.
    Only LEFT and RIGHT classes get augmented (no color changes).
    FRONT class gets minimal augmentation to preserve balance.
    
    Args:
        data_dir: Directory containing segmentation masks
        labels_file: Path to labels file
        batch_size: Batch size
        val_split: Validation split ratio
        image_size: Input image size
        num_workers: Number of worker processes
        augmentation_strength: 'light', 'medium', or 'heavy'
        class_specific_augmentation: Whether to apply different augmentations based on class
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create augmentation transform for LEFT/RIGHT classes (geometric only)
    train_transform = create_geometric_only_augmentation_transform(image_size, augmentation_strength)
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset with class-specific augmentation
    full_dataset = DecisionDataset(
        data_dir, 
        labels_file, 
        transform=train_transform,
        class_specific_augmentation=class_specific_augmentation
    )
    
    # Check if we have enough data
    if len(full_dataset) < 10:
        raise ValueError(f"Not enough labeled data. Found {len(full_dataset)} samples. "
                        "Please label more data using annotate_data.py")
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform
    val_dataset.dataset.class_specific_augmentation = False  # No augmentation for validation
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    if class_specific_augmentation:
        print(f"Class-specific augmentation: ENABLED (only LEFT/RIGHT classes)")
        print(f"Augmentation type: Geometric only (no color changes)")
    else:
        print(f"Class-specific augmentation: DISABLED")
    print(f"Augmentation strength: {augmentation_strength}")
    
    # Print class distribution
    class_counts = full_dataset.get_class_counts()
    print("Class distribution:", class_counts)
    
    # Calculate effective dataset size with augmentation
    if class_specific_augmentation:
        left_count = class_counts['left']
        right_count = class_counts['right']
        front_count = class_counts['front']
        
        # Estimate effective augmentation multiplier
        aug_multipliers = {'light': 2, 'medium': 4, 'heavy': 6}
        multiplier = aug_multipliers.get(augmentation_strength, 4)
        
        effective_left = left_count * multiplier
        effective_right = right_count * multiplier
        effective_total = front_count + effective_left + effective_right
        
        print(f"Effective dataset size with augmentation:")
        print(f"  FRONT: {front_count} (no augmentation)")
        print(f"  LEFT: {left_count} → ~{effective_left} (with augmentation)")
        print(f"  RIGHT: {right_count} → ~{effective_right} (with augmentation)")
        print(f"  Total effective: ~{effective_total} samples")
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train driving decision neural network')
    parser.add_argument('data_dir', help='Directory containing segmentation masks')
    parser.add_argument('--labels_file', default='driving_labels.json', 
                       help='Labels file name')
    parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                       help='Type of model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--augmentation', choices=['light', 'medium', 'heavy'], default='medium',
                       help='Data augmentation strength: light, medium, or heavy')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Check if labels file exists
    labels_path = os.path.join(args.data_dir, args.labels_file)
    if not os.path.exists(labels_path):
        print(f"Error: Labels file {labels_path} does not exist")
        print("Please run annotate_data.py first to create labels")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with class-specific geometric augmentation
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        labels_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        augmentation_strength=args.augmentation,
        class_specific_augmentation=True  # Enable class-specific augmentation
    )
    
    # Create model
    model = create_model(args.model_type, num_classes=3)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create trainer
    trainer = DecisionTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Train model
    history = trainer.train(args.epochs, args.save_dir)
    
    print("Training completed successfully!")
    print(f"Data augmentation used: {args.augmentation}")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.2f}%")


if __name__ == "__main__":
    main()

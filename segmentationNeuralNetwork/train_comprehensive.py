#!/usr/bin/env python3
"""
Comprehensive Unity Segmentation Training
Train a high-quality Unity segmentation model for production use.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import Unity modules
from unity_dataset import UnityDataModule, UnitySegmentationDataset
from unity_unet import create_unity_model, get_unity_class_weights, UNITY_CLASSES, NUM_CLASSES

class ComprehensiveUnityTrainer:
    """Comprehensive trainer for high-quality Unity segmentation models."""
    
    def __init__(self, config: dict = None):
        """Initialize comprehensive trainer."""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create data loaders
        self.data_module = UnityDataModule(
            data_dir=self.config['data_dir'],
            sequence=self.config['sequence'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers']
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_dataloaders()
        
        # Create model
        self.model = create_unity_model(optimization=self.config['optimization'])
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.criterion = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_pixel_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self.best_iou = 0.0
        self.best_model_path = None
        
        # Create save directory
        self.save_dir = Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Comprehensive Unity Trainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.config['optimization']} optimization")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Max epochs: {self.config['epochs']}")
    
    def _get_default_config(self) -> dict:
        """Get comprehensive training configuration."""
        return {
            'data_dir': '../data',
            'sequence': 'sequence.0',
            'optimization': 'balanced',  # balanced for good speed/accuracy
            'epochs': 80,
            'batch_size': 12,  # Reasonable for memory
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'image_size': (256, 512),
            'num_workers': 4,
            'save_every': 5,
            'early_stopping_patience': 12,
            'use_class_weights': True,
            'use_mixed_precision': True,
            'warmup_epochs': 5,
        }
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function with class weights."""
        if self.config['use_class_weights']:
            try:
                class_weights = get_unity_class_weights()
                class_weights = class_weights.to(self.device)
                print(f"✓ Using class weights (range: {class_weights.min():.3f} - {class_weights.max():.3f})")
            except:
                class_weights = None
                print("⚠️ Using uniform class weights")
        else:
            class_weights = None
        
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
    
    def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Calculate comprehensive metrics."""
        predictions = torch.argmax(outputs, dim=1)
        
        # Pixel accuracy
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        pixel_acc = correct / total
        
        # IoU per class
        ious = []
        for class_id in range(NUM_CLASSES):
            # True positive, false positive, false negative
            tp = ((predictions == class_id) & (targets == class_id)).sum().item()
            fp = ((predictions == class_id) & (targets != class_id)).sum().item()
            fn = ((predictions != class_id) & (targets == class_id)).sum().item()
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                ious.append(iou)
            else:
                ious.append(0.0)
        
        mean_iou = np.mean(ious)
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'class_ious': ious
        }
    
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {'pixel_accuracy': 0.0, 'mean_iou': 0.0}
        num_batches = len(self.train_loader)
        
        # Mixed precision training
        if self.config['use_mixed_precision'] and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate metrics for monitoring
            if batch_idx % 5 == 0:  # Every 5 batches
                with torch.no_grad():
                    metrics = self.calculate_metrics(outputs, masks)
                    epoch_metrics['pixel_accuracy'] += metrics['pixel_accuracy']
                    epoch_metrics['mean_iou'] += metrics['mean_iou']
                
                # Print progress
                if batch_idx % 10 == 0:
                    progress = (batch_idx + 1) / num_batches * 100
                    print(f"   Epoch {epoch:2d} | {progress:5.1f}% | Loss: {loss.item():.4f} | "
                          f"IoU: {metrics['mean_iou']:.3f} | Acc: {metrics['pixel_accuracy']:.3f}")
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / (num_batches // 5) for k, v in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch."""
        self.model.eval()
        
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                all_metrics.append(metrics)
        
        # Average validation metrics
        avg_loss = val_loss / len(self.val_loader)
        avg_pixel_acc = np.mean([m['pixel_accuracy'] for m in all_metrics])
        avg_iou = np.mean([m['mean_iou'] for m in all_metrics])
        
        # Class-wise IoU
        class_ious = np.mean([m['class_ious'] for m in all_metrics], axis=0)
        
        return {
            'loss': avg_loss,
            'pixel_accuracy': avg_pixel_acc,
            'mean_iou': avg_iou,
            'class_ious': class_ious
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"unity_best_{self.config['optimization']}_{timestamp}.pt"
        else:
            filename = f"unity_epoch_{epoch:03d}_{timestamp}.pt"
        
        filepath = self.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'num_classes': NUM_CLASSES
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            self.best_model_path = filepath
            print(f"💾 Saved BEST model: {filepath}")
        else:
            print(f"💾 Saved checkpoint: {filepath}")
        
        return filepath
    
    def plot_training_progress(self):
        """Plot training progress."""
        if len(self.history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU
        axes[0, 1].plot(epochs, self.history['val_iou'], 'g-', label='Validation IoU')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Pixel Accuracy
        axes[1, 0].plot(epochs, self.history['val_pixel_acc'], 'm-', label='Validation Accuracy')
        axes[1, 0].set_title('Pixel Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'c-', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self) -> dict:
        """Main training loop."""
        print(f"\n🎯 Starting comprehensive Unity training...")
        print(f"Target: {self.config['epochs']} epochs with early stopping patience {self.config['early_stopping_patience']}")
        
        start_time = time.time()
        epochs_without_improvement = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training
            print(f"\n📈 Epoch {epoch}/{self.config['epochs']}")
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['mean_iou'])
            self.history['val_pixel_acc'].append(val_metrics['pixel_accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val IoU: {val_metrics['mean_iou']:.3f}")
            print(f"   Val Acc: {val_metrics['pixel_accuracy']:.3f}")
            print(f"   LR: {current_lr:.6f}")
            
            # Print class-wise IoU every 10 epochs
            if epoch % 10 == 0:
                print(f"   Class IoUs:")
                for i, iou in enumerate(val_metrics['class_ious']):
                    class_name = UNITY_CLASSES[i]['name']
                    print(f"     {class_name:12s}: {iou:.3f}")
            
            # Check for best model
            is_best = val_metrics['mean_iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['mean_iou']
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                epochs_without_improvement += 1
            
            # Save regular checkpoint
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Plot progress
            if epoch % 5 == 0:
                self.plot_training_progress()
            
            # Early stopping
            if epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n🛑 Early stopping: No improvement for {epochs_without_improvement} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Best IoU: {self.best_iou:.3f}")
        print(f"   Best model: {self.best_model_path}")
        print(f"   Final epoch: {epoch}")
        
        # Final plot
        self.plot_training_progress()
        
        return {
            'best_iou': self.best_iou,
            'best_model_path': str(self.best_model_path),
            'total_epochs': epoch,
            'total_time_hours': total_time / 3600,
            'final_metrics': val_metrics,
            'history': self.history
        }

def main():
    """Main training function."""
    print("🚗 Comprehensive Unity Segmentation Training")
    print("=" * 60)
    
    # Training configuration for good results within 3 hours
    config = {
        'optimization': 'balanced',  # Good speed/accuracy trade-off
        'epochs': 80,               # Should be enough with early stopping
        'batch_size': 12,           # Memory-efficient
        'learning_rate': 0.001,     # Good starting point
        'save_every': 5,            # Save every 5 epochs
        'early_stopping_patience': 15,  # Stop if no improvement for 15 epochs
        'use_class_weights': True,  # Handle class imbalance
        'use_mixed_precision': True, # Faster training
    }
    
    # Create trainer
    trainer = ComprehensiveUnityTrainer(config)
    
    # Start training
    results = trainer.train()
    
    # Test best model
    print(f"\n🧪 Testing best model...")
    try:
        from optimized_inference import OptimizedSegmentationModel
        from PIL import Image
        
        # Load best model for testing
        opt_model = OptimizedSegmentationModel(
            results['best_model_path'], 
            optimization_level='balanced'
        )
        
        # Quick performance test
        test_image = Image.new('RGB', (512, 256), color='blue')
        for _ in range(5):  # Warmup
            _ = opt_model.predict(test_image)
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(20):
            result = opt_model.predict(test_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        fps = 1.0 / avg_time
        
        print(f"✅ Optimized inference performance:")
        print(f"   Inference time: {avg_time*1000:.1f} ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Output shape: {result.shape}")
        
    except Exception as e:
        print(f"⚠️ Could not test optimized inference: {e}")
    
    print(f"\n🎯 Training Summary:")
    print(f"   Best IoU: {results['best_iou']:.1f}%")
    print(f"   Training time: {results['total_time_hours']:.1f} hours")
    print(f"   Total epochs: {results['total_epochs']}")
    print(f"   Best model: {results['best_model_path']}")
    
    print(f"\n💡 Your Unity segmentation model is ready for deployment!")

if __name__ == "__main__":
    main()

# Unity Segmentation Training Script
# Optimized for Unity-generated data with real-time inference focus

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

# Import our Unity-specific modules
from unity_dataset import UnityDataModule, UnitySegmentationDataset
from unity_unet import create_unity_model, get_unity_class_weights, UNITY_CLASSES, NUM_CLASSES

class UnitySegmentationTrainer:
    """
    Trainer for Unity semantic segmentation models.
    Optimized for real-time inference and driving scenarios.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize Unity segmentation trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup data
        self.data_module = UnityDataModule(
            data_dir=self.config['data_dir'],
            sequence=self.config['sequence'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers']
        )
        
        # Get data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_dataloaders()
        
        # Create model
        self.model = self._create_model()
        
        # Setup training components
        self.criterion = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        print(f"🚀 Unity Segmentation Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.config['encoder']} UNet")
        print(f"   Classes: {NUM_CLASSES}")
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")
        print(f"   Test samples: {len(self.test_loader.dataset)}")
    
    def _get_default_config(self) -> dict:
        """Get default training configuration for Unity data."""
        return {
            # Data configuration
            'data_dir': '../data',
            'sequence': 'sequence.0',
            'image_size': (256, 512),
            
            # Model configuration
            'encoder': 'mobilenet_v2',  # Fast for real-time
            'optimization': 'speed',    # 'speed', 'balanced', 'quality'
            'pretrained': True,
            
            # Training configuration
            'batch_size': 16,
            'learning_rate': 1e-3,
            'num_epochs': 50,
            'weight_decay': 1e-4,
            
            # Optimization configuration
            'use_mixed_precision': True,
            'gradient_clip': 1.0,
            'use_class_weights': True,
            
            # Data loading
            'num_workers': 4,
            'pin_memory': True,
            
            # Checkpointing
            'save_dir': 'checkpoints_unity',
            'save_best_only': True,
            'early_stopping_patience': 15,
            
            # Logging
            'log_interval': 10,
            'val_interval': 1,
        }
    
    def _setup_device(self) -> torch.device:
        """Setup the best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("   Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print("   Using CPU (training will be slow)")
        
        return device
    
    def _create_model(self) -> nn.Module:
        """Create Unity segmentation model."""
        model = create_unity_model(
            encoder=self.config['encoder'],
            optimization=self.config['optimization']
        )
        
        model = model.to(self.device)
        
        # Apply optimizations
        if self.config.get('use_compile', False) and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print("✓ PyTorch compilation enabled")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")
        
        return model
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function with optional class weighting."""
        if self.config['use_class_weights']:
            try:
                class_weights = get_unity_class_weights(self.config['data_dir'])
                class_weights = class_weights.to(self.device)
                print(f"✓ Using class weights (range: {class_weights.min():.3f} - {class_weights.max():.3f})")
            except Exception as e:
                print(f"⚠️  Could not load class weights: {e}")
                class_weights = None
        else:
            class_weights = None
        
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
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
            T_max=self.config['num_epochs'],
            eta_min=1e-6
        )
    
    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Calculate training metrics."""
        predictions = torch.argmax(outputs, dim=1)
        
        # Pixel accuracy
        correct_pixels = (predictions == targets).sum().item()
        total_pixels = targets.numel()
        accuracy = correct_pixels / total_pixels * 100
        
        # Mean IoU (simplified calculation)
        iou_scores = []
        for class_id in range(NUM_CLASSES):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            if union > 0:
                iou_scores.append(intersection / union)
        
        mean_iou = np.mean(iou_scores) * 100 if iou_scores else 0
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou
        }
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_accuracy = 0
        total_iou = 0
        num_batches = 0
        
        # Setup mixed precision if enabled
        scaler = torch.cuda.amp.GradScaler() if self.config['use_mixed_precision'] else None
        
        start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config['use_mixed_precision'] and scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['gradient_clip']:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Regular forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['gradient_clip']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self._calculate_metrics(outputs, masks)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            total_iou += metrics['mean_iou']
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                print(f"   Batch {batch_idx:3d}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {metrics['accuracy']:.2f}% | "
                      f"IoU: {metrics['mean_iou']:.2f}%")
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'mean_iou': total_iou / num_batches,
            'epoch_time': epoch_time
        }
    
    def validate_epoch(self) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_iou = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                if self.config['use_mixed_precision']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                metrics = self._calculate_metrics(outputs, masks)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += metrics['accuracy']
                total_iou += metrics['mean_iou']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'mean_iou': total_iou / num_batches
        }
    
    def train(self):
        """Complete training loop."""
        print(f"\n🚀 Starting Unity segmentation training for {self.config['num_epochs']} epochs...")
        
        # Create save directory
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        best_val_iou = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n📈 Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['val_interval'] == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {'loss': 0, 'accuracy': 0, 'mean_iou': 0}
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_iou'].append(train_metrics['mean_iou'])
            self.training_history['val_iou'].append(val_metrics['mean_iou'])
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epoch_times'].append(train_metrics['epoch_time'])
            
            # Print epoch summary
            print(f"   Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}% | IoU: {train_metrics['mean_iou']:.2f}%")
            if epoch % self.config['val_interval'] == 0:
                print(f"   Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}% | IoU: {val_metrics['mean_iou']:.2f}%")
            print(f"   LR: {current_lr:.2e} | Time: {train_metrics['epoch_time']:.1f}s")
            
            # Save best model
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                patience_counter = 0
                
                # Save model
                model_path = save_dir / "best_unity_model.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"   ✅ New best model saved: {val_metrics['mean_iou']:.2f}% IoU")
                
                # Quick inference speed test
                self._benchmark_inference_speed()
                
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"   ⏹️  Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save final model and training history
        final_model_path = save_dir / "final_unity_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        
        history_path = save_dir / "unity_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        # Save Unity-specific info
        self._save_unity_info(save_dir)
        
        print(f"\n🏁 Unity segmentation training completed!")
        print(f"   Best validation IoU: {best_val_iou:.2f}%")
        print(f"   Models saved in: {save_dir}")
        
        return best_val_iou
    
    def _benchmark_inference_speed(self):
        """Quick benchmark of model inference speed."""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.config['image_size']).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"   🏎️  Inference: {avg_time*1000:.2f}ms ({fps:.1f} FPS)")
        
        if fps >= 30:
            print("   ✅ Suitable for real-time driving!")
        elif fps >= 15:
            print("   ⚠️  Moderate speed - usable but not optimal")
        else:
            print("   ❌ Too slow for real-time driving")
    
    def _save_training_plots(self, save_dir: Path):
        """Save training history plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], label='Train', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], label='Validation', linewidth=2)
        ax1.set_title('Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.training_history['train_acc'], label='Train', linewidth=2)
        ax2.plot(epochs, self.training_history['val_acc'], label='Validation', linewidth=2)
        ax2.set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # IoU plot
        ax3.plot(epochs, self.training_history['train_iou'], label='Train', linewidth=2)
        ax3.plot(epochs, self.training_history['val_iou'], label='Validation', linewidth=2)
        ax3.set_title('Mean IoU', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('IoU (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax4.plot(epochs, self.training_history['learning_rates'], linewidth=2, color='orange')
        ax4.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Unity Segmentation Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'unity_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_unity_info(self, save_dir: Path):
        """Save Unity-specific model and dataset information."""
        unity_info = {
            'model': {
                'encoder': self.config['encoder'],
                'optimization': self.config['optimization'],
                'num_classes': NUM_CLASSES,
                'input_size': self.config['image_size'],
                'parameters': sum(p.numel() for p in self.model.parameters())
            },
            'dataset': {
                'data_dir': self.config['data_dir'],
                'sequence': self.config['sequence'],
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'test_samples': len(self.test_loader.dataset)
            },
            'classes': UNITY_CLASSES,
            'training_config': self.config
        }
        
        with open(save_dir / 'unity_model_info.json', 'w') as f:
            json.dump(unity_info, f, indent=2)

def train_unity_segmentation(
    encoder: str = "mobilenet_v2",
    optimization: str = "speed",
    epochs: int = 30,
    batch_size: int = 16
):
    """
    Train Unity segmentation model with specified configuration.
    
    Args:
        encoder: Model encoder ('mobilenet_v2', 'efficientnet-b0', 'resnet34', etc.)
        optimization: Optimization target ('speed', 'balanced', 'quality')
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    
    config = {
        'encoder': encoder,
        'optimization': optimization,
        'num_epochs': epochs,
        'batch_size': batch_size,
    }
    
    trainer = UnitySegmentationTrainer(config)
    best_iou = trainer.train()
    
    return trainer, best_iou

def compare_unity_models():
    """Compare different model configurations for Unity data."""
    
    configurations = [
        ('mobilenet_v2', 'speed', 'Fastest'),
        ('efficientnet-b0', 'balanced', 'Balanced'),
        ('resnet34', 'quality', 'Best Quality'),
    ]
    
    results = {}
    
    for encoder, optimization, description in configurations:
        print(f"\n🔥 Training {encoder} ({description})...")
        
        try:
            trainer, best_iou = train_unity_segmentation(
                encoder=encoder,
                optimization=optimization,
                epochs=20,  # Shorter for comparison
                batch_size=8   # Smaller for memory
            )
            
            results[encoder] = {
                'iou': best_iou,
                'description': description,
                'optimization': optimization
            }
        except Exception as e:
            print(f"❌ Failed to train {encoder}: {e}")
            results[encoder] = {'iou': 0, 'error': str(e)}
    
    # Print comparison
    print("\n📊 Unity Model Comparison:")
    print("=" * 60)
    for encoder, result in results.items():
        if 'error' not in result:
            print(f"{encoder:>15}: {result['iou']:>6.2f}% IoU ({result['description']})")
        else:
            print(f"{encoder:>15}: Failed - {result['error']}")
    
    return results

if __name__ == "__main__":
    # Train Unity segmentation model
    print("🚗 Training Unity Segmentation Model for Real-time Driving...")
    
    # Use fast configuration for real-time inference
    trainer, best_iou = train_unity_segmentation(
        encoder='mobilenet_v2',
        optimization='speed',
        epochs=50,
        batch_size=16
    )
    
    print(f"\n✅ Unity training completed with {best_iou:.2f}% IoU")
    
    # Uncomment to compare multiple configurations
    # compare_unity_models()

#!/usr/bin/env python3
"""
High-Accuracy Unity Segmentation Training
Target: 85%+ accuracy in under 2 hours while maintaining real-time performance.
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

class HighAccuracyUnityTrainer:
    """High-accuracy trainer optimized for 85%+ accuracy in under 2 hours."""
    
    def __init__(self, config: dict = None):
        """Initialize high-accuracy trainer."""
        self.config = self._get_optimized_config()
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create enhanced data loaders
        self.data_module = UnityDataModule(
            data_dir=self.config['data_dir'],
            sequence=self.config['sequence'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers']
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_dataloaders()
        
        # Create enhanced model
        self.model = create_unity_model(optimization=self.config['optimization'])
        self.model = self.model.to(self.device)
        
        # Setup enhanced training components
        self.criterion = self._setup_enhanced_loss()
        self.optimizer = self._setup_enhanced_optimizer()
        self.scheduler = self._setup_enhanced_scheduler()
        
        # Training tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_pixel_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self.best_accuracy = 0.0
        self.best_iou = 0.0
        self.best_model_path = None
        
        # Create save directory
        self.save_dir = Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"🎯 High-Accuracy Unity Trainer Initialized")
        print(f"   Target: 85%+ accuracy in under 2 hours")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.config['optimization']} optimization")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Max epochs: {self.config['epochs']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
    
    def _get_optimized_config(self) -> dict:
        """Get high-accuracy optimized training configuration."""
        return {
            'data_dir': '../data',
            'sequence': 'sequence.0',
            'optimization': 'accuracy',  # Focus on accuracy over speed
            'epochs': 60,                # Enough for convergence in 2 hours
            'batch_size': 16,            # Larger batch for stable gradients
            'learning_rate': 0.002,      # Higher LR for faster convergence
            'weight_decay': 5e-5,        # Reduced regularization
            'image_size': (256, 512),    # Keep efficient size
            'num_workers': 6,            # More workers for faster data loading
            'save_every': 3,             # Save more frequently
            'early_stopping_patience': 8, # Shorter patience for time constraint
            'use_class_weights': True,    # Essential for imbalanced data
            'use_mixed_precision': True,  # Speed up training
            'warmup_epochs': 3,          # Quick warmup
            'use_cosine_restarts': True, # Better convergence
            'use_focal_loss': True,      # Better for hard examples
            'use_label_smoothing': True, # Regularization
            'gradient_clip': 1.0,        # Stable training
        }
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"   GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"   Using Apple Metal Performance Shaders")
        else:
            device = torch.device('cpu')
            print(f"   Using CPU (consider GPU for faster training)")
        return device
    
    def _setup_enhanced_loss(self) -> nn.Module:
        """Setup enhanced loss function for high accuracy."""
        if self.config['use_focal_loss']:
            return self._create_focal_loss()
        else:
            return self._create_weighted_ce_loss()
    
    def _create_focal_loss(self) -> nn.Module:
        """Create Focal Loss for hard example mining."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-1):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.weight = weight
                self.ignore_index = ignore_index
                self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        # Get class weights
        class_weights = None
        if self.config['use_class_weights']:
            try:
                class_weights = get_unity_class_weights()
                class_weights = class_weights.to(self.device)
                print(f"✓ Using Focal Loss with class weights")
            except:
                print("✓ Using Focal Loss without class weights")
        
        return FocalLoss(alpha=1, gamma=2, weight=class_weights, ignore_index=-1)
    
    def _create_weighted_ce_loss(self) -> nn.Module:
        """Create weighted CrossEntropy loss."""
        class_weights = None
        if self.config['use_class_weights']:
            try:
                class_weights = get_unity_class_weights()
                class_weights = class_weights.to(self.device)
                print(f"✓ Using weighted CrossEntropy loss")
            except:
                print("✓ Using standard CrossEntropy loss")
        
        if self.config['use_label_smoothing']:
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, label_smoothing=0.1)
        else:
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    def _setup_enhanced_optimizer(self) -> optim.Optimizer:
        """Setup enhanced optimizer for high accuracy."""
        # Use AdamW with optimized parameters
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_enhanced_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup enhanced learning rate scheduler."""
        if self.config['use_cosine_restarts']:
            # Cosine Annealing with Warm Restarts for better convergence
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=1,
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
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
        """Train one epoch with enhanced techniques."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {'pixel_accuracy': 0.0, 'mean_iou': 0.0}
        num_batches = len(self.train_loader)
        
        # Mixed precision training
        scaler = None
        if self.config['use_mixed_precision'] and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip'):
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate metrics for monitoring
            if batch_idx % 3 == 0:  # More frequent monitoring
                with torch.no_grad():
                    metrics = self.calculate_metrics(outputs, masks)
                    epoch_metrics['pixel_accuracy'] += metrics['pixel_accuracy']
                    epoch_metrics['mean_iou'] += metrics['mean_iou']
                
                # Print progress
                if batch_idx % 6 == 0:
                    progress = (batch_idx + 1) / num_batches * 100
                    print(f"   Epoch {epoch:2d} | {progress:5.1f}% | Loss: {loss.item():.4f} | "
                          f"IoU: {metrics['mean_iou']:.3f} | Acc: {metrics['pixel_accuracy']:.3f}")
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / (num_batches // 3) for k, v in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch."""
        self.model.eval()
        
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Mixed precision inference
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
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
            filename = f"unity_high_acc_{self.config['optimization']}_{timestamp}.pt"
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
            print(f"💾 Saved BEST model: {filepath} (Acc: {metrics['pixel_accuracy']:.1%})")
        else:
            print(f"💾 Saved checkpoint: {filepath}")
        
        return filepath
    
    def plot_training_progress(self):
        """Plot training progress with accuracy focus."""
        if len(self.history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy with target line
        axes[0, 1].plot(epochs, [acc * 100 for acc in self.history['val_pixel_acc']], 'g-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].axhline(y=85, color='r', linestyle='--', label='Target 85%')
        axes[0, 1].set_title('Pixel Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim(50, 100)
        
        # IoU
        axes[1, 0].plot(epochs, [iou * 100 for iou in self.history['val_iou']], 'purple', label='Validation IoU', linewidth=2)
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'orange', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'high_accuracy_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self) -> dict:
        """Main high-accuracy training loop."""
        print(f"\n🎯 Starting High-Accuracy Unity Training")
        print(f"   Target: 85%+ accuracy in under 2 hours")
        print(f"   Strategy: Enhanced model + Focal Loss + Optimized scheduling")
        
        start_time = time.time()
        epochs_without_improvement = 0
        target_reached = False
        
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
            accuracy_pct = val_metrics['pixel_accuracy'] * 100
            iou_pct = val_metrics['mean_iou'] * 100
            
            print(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val Accuracy: {accuracy_pct:.1f}% {'🎯' if accuracy_pct >= 85 else '📈'}")
            print(f"   Val IoU: {iou_pct:.1f}%")
            print(f"   LR: {current_lr:.6f}")
            
            # Check if target reached
            if accuracy_pct >= 85.0 and not target_reached:
                target_reached = True
                print(f"🎉 TARGET REACHED! 85%+ accuracy achieved in epoch {epoch}")
            
            # Print class-wise IoU every 5 epochs
            if epoch % 5 == 0:
                print(f"   Class IoUs:")
                for i, iou in enumerate(val_metrics['class_ious']):
                    if i in UNITY_CLASSES:
                        class_name = UNITY_CLASSES[i]['name']
                        print(f"     {class_name:12s}: {iou*100:.1f}%")
            
            # Check for best model (prioritize accuracy over IoU)
            is_best = val_metrics['pixel_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['pixel_accuracy']
                self.best_iou = val_metrics['mean_iou']
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                epochs_without_improvement += 1
            
            # Save regular checkpoint
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Plot progress
            if epoch % 3 == 0:
                self.plot_training_progress()
            
            # Early stopping (but only after reaching target or patience)
            if target_reached and epochs_without_improvement >= 5:
                print(f"\n🛑 Early stopping: Target reached and no improvement for {epochs_without_improvement} epochs")
                break
            elif not target_reached and epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n⏰ Early stopping: No improvement for {epochs_without_improvement} epochs")
                break
            
            # Time check (under 2 hours)
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= 1.9:  # Leave some buffer
                print(f"\n⏱️ Time limit approaching ({elapsed_hours:.1f}h), stopping training")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        final_accuracy = self.best_accuracy * 100
        final_iou = self.best_iou * 100
        
        print(f"\n🎉 High-Accuracy Training Completed!")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Best Accuracy: {final_accuracy:.1f}% {'✅' if final_accuracy >= 85 else '❌'}")
        print(f"   Best IoU: {final_iou:.1f}%")
        print(f"   Best model: {self.best_model_path}")
        print(f"   Total epochs: {epoch}")
        print(f"   Target {'ACHIEVED' if final_accuracy >= 85 else 'NOT REACHED'}")
        
        # Final plot
        self.plot_training_progress()
        
        return {
            'best_accuracy': self.best_accuracy,
            'best_iou': self.best_iou,
            'best_model_path': str(self.best_model_path),
            'total_epochs': epoch,
            'total_time_hours': total_time / 3600,
            'target_reached': final_accuracy >= 85,
            'final_metrics': val_metrics,
            'history': self.history
        }

def main():
    """Main high-accuracy training function."""
    print("🎯 High-Accuracy Unity Segmentation Training")
    print("Target: 85%+ accuracy in under 2 hours")
    print("=" * 60)
    
    # High-accuracy optimized configuration
    config = {
        'optimization': 'accuracy',      # Focus on accuracy
        'epochs': 60,                   # Sufficient for convergence
        'batch_size': 16,               # Larger batch for stability
        'learning_rate': 0.002,         # Higher LR for faster convergence
        'save_every': 3,                # Save more frequently
        'early_stopping_patience': 8,   # Balanced patience
        'use_focal_loss': True,         # Better for hard examples
        'use_cosine_restarts': True,    # Better convergence
        'use_mixed_precision': True,    # Faster training
        'gradient_clip': 1.0,           # Stable training
    }
    
    # Create trainer
    trainer = HighAccuracyUnityTrainer(config)
    
    # Start training
    results = trainer.train()
    
    # Test final model performance
    print(f"\n🧪 Testing final model performance...")
    try:
        from optimized_inference import OptimizedSegmentationModel
        from PIL import Image
        
        # Load best model for testing
        opt_model = OptimizedSegmentationModel(
            results['best_model_path'], 
            optimization_level='accuracy'
        )
        
        # Performance benchmark
        test_image = Image.new('RGB', (512, 256), color='blue')
        
        # Warmup
        for _ in range(5):
            _ = opt_model.predict(test_image)
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(30):
            result = opt_model.predict(test_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 30
        fps = 1.0 / avg_time
        
        print(f"✅ Final model performance:")
        print(f"   Accuracy: {results['best_accuracy']*100:.1f}%")
        print(f"   IoU: {results['best_iou']*100:.1f}%")
        print(f"   Inference time: {avg_time*1000:.1f} ms")
        print(f"   FPS: {fps:.1f} {'✅' if fps >= 30 else '⚠️'}")
        print(f"   Real-time ready: {'Yes' if fps >= 30 else 'Borderline'}")
        
    except Exception as e:
        print(f"⚠️ Could not test model performance: {e}")
    
    # Final verdict
    if results['target_reached']:
        print(f"\n🎉 SUCCESS! Target achieved:")
        print(f"   ✅ Accuracy: {results['best_accuracy']*100:.1f}% (target: 85%)")
        print(f"   ✅ Time: {results['total_time_hours']:.1f}h (target: <2h)")
        print(f"   🚀 Model ready for real-time driving!")
    else:
        print(f"\n⚠️ Target not fully reached:")
        print(f"   📊 Accuracy: {results['best_accuracy']*100:.1f}% (target: 85%)")
        print(f"   ⏱️ Time: {results['total_time_hours']:.1f}h")
        print(f"   💡 Consider: More data, longer training, or different architecture")

if __name__ == "__main__":
    main()

# Optimized Training for Real-time Segmentation Models
# Train lightweight models suitable for driving inference

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_dataset import SegmentationDataset
from dotenv import load_dotenv
import time
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

load_dotenv()

class RealTimeModelTrainer:
    """
    Trainer optimized for real-time segmentation models.
    Focuses on speed/accuracy tradeoff suitable for driving.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model
        self.model = self._create_optimized_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
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
            'learning_rates': [],
            'epoch_times': []
        }
        
        print(f"🚀 Real-time model trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.config['encoder']} UNet")
        print(f"   Input size: {self.config['input_size']}")
        print(f"   Batch size: {self.config['batch_size']}")
    
    def _get_default_config(self) -> dict:
        """Get default configuration optimized for real-time inference."""
        return {
            # Model configuration
            'encoder': 'mobilenet_v2',  # Fast encoder
            'input_size': (256, 512),   # Smaller for speed
            'num_classes': 255,         # Cityscapes classes
            
            # Training configuration
            'batch_size': 16,           # Moderate batch size
            'learning_rate': 1e-3,      # Higher LR for faster convergence
            'num_epochs': 50,           # Fewer epochs
            'weight_decay': 1e-4,       # Regularization
            
            # Data configuration
            'data_root': os.getenv("CITYSCAPES_ROOT", "../data"),
            'num_workers': 4,           # Parallel data loading
            'pin_memory': True,         # GPU optimization
            
            # Optimization configuration
            'use_mixed_precision': True,  # FP16 training
            'use_compile': True,          # PyTorch 2.0 compilation
            'gradient_clip': 1.0,         # Gradient clipping
            
            # Checkpoint configuration
            'save_dir': 'checkpoints_realtime',
            'save_best_only': True,
            'early_stopping_patience': 10,
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
    
    def _create_optimized_model(self) -> nn.Module:
        """Create model optimized for real-time inference."""
        
        # Create model with lightweight encoder
        model = smp.Unet(
            encoder_name=self.config['encoder'],
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.config['num_classes'],
            activation=None  # Will use CrossEntropyLoss
        )
        
        model = model.to(self.device)
        
        # Apply PyTorch 2.0 compilation for speed
        if self.config['use_compile'] and hasattr(torch, 'compile'):
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
    
    def _setup_data_loaders(self) -> tuple:
        """Setup optimized data loaders."""
        
        # Training dataset with augmentation
        train_dataset = SegmentationDataset(
            root_dir=self.config['data_root'],
            image_dir="leftImg8bit",
            mask_dir="gtCoarse", 
            image_label="leftImg8bit",
            mask_label="gtCoarse_labelIds",
            split="train"
        )
        
        # Validation dataset
        val_dataset = SegmentationDataset(
            root_dir=self.config['data_root'],
            image_dir="leftImg8bit",
            mask_dir="gtCoarse",
            image_label="leftImg8bit", 
            mask_label="gtCoarse_labelIds",
            split="val"
        )
        
        # Optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=True  # For consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        # Use CrossEntropyLoss with label smoothing for better generalization
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
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
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {correct_pixels/total_pixels*100:.2f}%")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_pixels / total_pixels * 100
        
        return avg_loss, accuracy, epoch_time
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
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
                
                # Track metrics
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_pixels / total_pixels * 100
        
        return avg_loss, accuracy
    
    def train(self):
        """Complete training loop."""
        print(f"\n🚀 Starting training for {self.config['num_epochs']} epochs...")
        
        # Create save directory
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n📈 Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                model_path = save_dir / "best_realtime_model.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"   ✅ New best model saved: {val_acc:.2f}% accuracy")
                
                # Benchmark inference speed
                self._benchmark_model_speed()
                
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"   ⏹️  Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save final model and training history
        final_model_path = save_dir / "final_realtime_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        print(f"\n🏁 Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Models saved in: {save_dir}")
        
        return best_val_acc
    
    def _benchmark_model_speed(self):
        """Quick benchmark of model inference speed."""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.config['input_size']).to(self.device)
        
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
    
    def _save_training_plots(self, save_dir: Path):
        """Save training history plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], label='Train')
        ax1.plot(epochs, self.training_history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(epochs, self.training_history['train_acc'], label='Train')
        ax2.plot(epochs, self.training_history['val_acc'], label='Validation')
        ax2.set_title('Accuracy (%)')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        # Learning rate plot
        ax3.plot(epochs, self.training_history['learning_rates'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_yscale('log')
        
        # Epoch time plot
        ax4.plot(epochs, self.training_history['epoch_times'])
        ax4.set_title('Epoch Time (s)')
        ax4.set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

def train_realtime_model(encoder: str = 'mobilenet_v2', input_size: tuple = (256, 512)):
    """
    Train a real-time optimized segmentation model.
    
    Args:
        encoder: Encoder architecture ('mobilenet_v2', 'efficientnet-b0', 'resnet18')
        input_size: Input image size (height, width)
    """
    config = {
        'encoder': encoder,
        'input_size': input_size,
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 1e-3,
    }
    
    trainer = RealTimeModelTrainer(config)
    best_acc = trainer.train()
    
    return trainer, best_acc

def compare_model_architectures():
    """Compare different model architectures for real-time performance."""
    
    architectures = [
        ('mobilenet_v2', 'Fastest'),
        ('efficientnet-b0', 'Balanced'), 
        ('resnet18', 'Quality'),
    ]
    
    results = {}
    
    for encoder, description in architectures:
        print(f"\n🔥 Training {encoder} ({description})...")
        
        try:
            trainer, best_acc = train_realtime_model(encoder)
            results[encoder] = {
                'accuracy': best_acc,
                'description': description
            }
        except Exception as e:
            print(f"❌ Failed to train {encoder}: {e}")
            results[encoder] = {'accuracy': 0, 'error': str(e)}
    
    # Print comparison
    print("\n📊 Architecture Comparison:")
    print("=" * 50)
    for encoder, result in results.items():
        if 'error' not in result:
            print(f"{encoder:>15}: {result['accuracy']:>6.2f}% ({result['description']})")
        else:
            print(f"{encoder:>15}: Failed - {result['error']}")
    
    return results

if __name__ == "__main__":
    # Train a single real-time model
    print("🚗 Training real-time segmentation model for driving...")
    
    # Use MobileNetV2 for maximum speed
    trainer, best_acc = train_realtime_model(
        encoder='mobilenet_v2',
        input_size=(256, 512)
    )
    
    print(f"\n✅ Training completed with {best_acc:.2f}% accuracy")
    
    # Uncomment to compare multiple architectures
    # compare_model_architectures()

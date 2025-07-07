#!/usr/bin/env python3
"""
Fast Model Training Script
Optimized for speed while maintaining reasonable accuracy.
Memory-efficient training that won't crash your computer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from unity_dataset import UnitySegmentationDataset
import numpy as np
import time
from pathlib import Path
import psutil
import os

class FastCrossEntropyLoss(nn.Module):
    """Lightweight CrossEntropy loss for speed"""
    
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # Simple cross entropy without stability checks for speed
        return nn.functional.cross_entropy(pred, target, ignore_index=self.ignore_index)

def check_memory_usage():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def fast_training():
    """Main training function optimized for speed"""
    
    print("⚡ Starting Fast Model Training")
    print("🎯 Target: 30+ FPS with reasonable accuracy")
    print("💾 Memory-safe: Won't crash your computer")
    print("=" * 50)
    
    # Clean up any existing checkpoints
    save_dir = Path("checkpoints_fast")
    save_dir.mkdir(exist_ok=True)
    
    # Clean old checkpoints except best and last
    for checkpoint in save_dir.glob("*.pth"):
        if checkpoint.name not in ['best_model.pth', 'last_model.pth']:
            checkpoint.unlink()
    
    # Device selection
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets with minimal memory usage
    train_dataset = UnitySegmentationDataset('../data', split='train')
    val_dataset = UnitySegmentationDataset('../data', split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Ultra-small batch size for memory safety
    batch_size = 1  # Smallest possible
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    # Create speed-optimized model
    print("🚀 Creating speed-optimized model...")
    
    # Option 1: Ultra-lightweight encoder
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # Smaller than MobileNetV2
        encoder_weights="imagenet",
        in_channels=3,
        classes=12,
        activation=None,
        decoder_channels=[128, 64, 32, 16, 8]  # Reduced decoder channels
    )
    
    model = model.to(device)
    print(f"✅ Model created with EfficientNet-B0 encoder")
    
    # Fast optimizer with higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Simple loss function
    criterion = FastCrossEntropyLoss()
    
    # Training parameters optimized for speed
    max_epochs = 30  # Fewer epochs
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 8  # Shorter patience
    
    print(f"⚡ Training configuration:")
    print(f"   Batch size: {batch_size} (memory safe)")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Early stopping: {max_patience} epochs")
    print(f"   Target: Speed over accuracy")
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 30)
        
        # Check memory at start of epoch
        memory_usage = check_memory_usage()
        print(f"💾 Memory usage: {memory_usage:.1f}MB")
        
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        batch_times = []
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            batch_start = time.time()
            
            try:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                
                # Simple gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_samples += 1
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Log every 20 batches
                if batch_idx % 20 == 0:
                    avg_batch_time = np.mean(batch_times[-10:]) if batch_times else 0
                    print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Time = {avg_batch_time:.3f}s")
                
                # Memory safety check
                if batch_idx % 30 == 0:
                    current_memory = check_memory_usage()
                    if current_memory > 8000:  # 8GB limit
                        print("⚠️ High memory usage detected, forcing garbage collection")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        import gc
                        gc.collect()
                        
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if train_samples == 0:
            print("❌ No valid training samples!")
            break
            
        avg_train_loss = train_loss / train_samples
        avg_batch_time = np.mean(batch_times)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        val_times = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                val_start = time.time()
                try:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    val_samples += 1
                    
                    val_time = time.time() - val_start
                    val_times.append(val_time)
                    
                except Exception:
                    continue
        
        if val_samples == 0:
            avg_val_loss = float('inf')
            avg_val_time = 0
        else:
            avg_val_loss = val_loss / val_samples
            avg_val_time = np.mean(val_times)
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        estimated_fps = 1 / avg_val_time if avg_val_time > 0 else 0
        
        # Print epoch results
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Epoch Time: {epoch_time:.1f}s")
        print(f"   Avg Batch Time: {avg_batch_time:.3f}s")
        print(f"   Estimated FPS: {estimated_fps:.1f}")
        print(f"   Memory: {check_memory_usage():.1f}MB")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Model saving with memory cleanup
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'estimated_fps': estimated_fps
            }, save_dir / 'best_model.pth')
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.4f}, Est. FPS: {estimated_fps:.1f}")
        else:
            patience_counter += 1
            
        # Always save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'estimated_fps': estimated_fps
        }, save_dir / 'last_model.pth')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"⏹️ Early stopping after {max_patience} epochs without improvement")
            break
        
        # Force memory cleanup between epochs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
    
    print("\n🎉 Fast training completed!")
    print(f"💾 Final memory usage: {check_memory_usage():.1f}MB")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    print(f"📁 Models saved in: {save_dir}")
    print(f"⚡ Speed-optimized model ready for testing!")

if __name__ == "__main__":
    fast_training()

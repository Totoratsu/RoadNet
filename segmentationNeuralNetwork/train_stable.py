#!/usr/bin/env python3
"""
Stable Segmentation Training Script
Designed to prevent NaN losses and training instability
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

class StableCrossEntropyLoss(nn.Module):
    """CrossEntropy loss with numerical stability and NaN prevention"""
    
    def __init__(self, ignore_index=-100, epsilon=1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Add small epsilon for numerical stability
        pred = torch.clamp(pred, min=-10, max=10)  # Prevent extreme values
        
        # Apply softmax with temperature for stability
        pred_stable = torch.log_softmax(pred, dim=1)
        
        # Standard cross entropy
        loss = nn.functional.nll_loss(pred_stable, target, ignore_index=self.ignore_index)
        
        # Check for NaN and replace with large but finite value
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf detected in loss, replacing with 10.0")
            return torch.tensor(10.0, device=loss.device, requires_grad=True)
            
        return loss

def check_gradients(model):
    """Check for NaN gradients"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"⚠️ NaN/Inf gradient in {name}")
                return False
    return True

def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent explosion"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def check_disk_space(save_dir):
    """Monitor checkpoint folder size"""
    total_size = sum(f.stat().st_size for f in save_dir.glob("*.pth") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"📁 Checkpoint folder size: {size_mb:.1f} MB")
    return size_mb

def stable_training():
    """Main training function with stability measures"""
    
    print("🚀 Starting Stable Segmentation Training")
    print("=" * 50)
    
    # Clean up any existing checkpoints to save space
    save_dir = Path("checkpoints_stable")
    if save_dir.exists():
        for checkpoint in save_dir.glob("*.pth"):
            if checkpoint.name not in ['best_model.pth', 'last_model.pth']:
                checkpoint.unlink()
                print(f"🗑️ Removed old checkpoint: {checkpoint.name}")
    save_dir.mkdir(exist_ok=True)
    
    # Device selection
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets with conservative settings
    train_dataset = UnitySegmentationDataset('../data', split='train')
    val_dataset = UnitySegmentationDataset('../data', split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Small batch size to prevent memory issues
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create simple, stable model
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=12,
        activation=None
    )
    model = model.to(device)
    
    # Conservative optimizer with very low learning rate
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Stable loss function
    criterion = StableCrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                images = images.to(device)
                masks = masks.to(device)
                
                # Check for NaN in inputs
                if torch.isnan(images).any() or torch.isnan(masks).any():
                    print(f"⚠️ NaN in batch {batch_idx}, skipping")
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Check loss validity
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Invalid loss in batch {batch_idx}: {loss.item()}")
                    continue
                
                # Backward pass with gradient checking
                loss.backward()
                
                # Check gradients
                if not check_gradients(model):
                    print("⚠️ NaN gradients detected, skipping update")
                    optimizer.zero_grad()
                    continue
                
                # Clip gradients
                clip_gradients(model, max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_samples += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if train_samples == 0:
            print("❌ No valid training samples processed!")
            break
            
        avg_train_loss = train_loss / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                try:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    if torch.isnan(images).any() or torch.isnan(masks).any():
                        continue
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        val_samples += 1
                        
                except Exception as e:
                    continue
        
        if val_samples == 0:
            print("❌ No valid validation samples!")
            avg_val_loss = float('inf')
        else:
            avg_val_loss = val_loss / val_samples
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        save_dir = Path("checkpoints_stable")
        save_dir.mkdir(exist_ok=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save ONLY the best model (overwrite previous best)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, save_dir / 'best_model.pth')
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Save ONLY the last model (overwrite previous last)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss
        }, save_dir / 'last_model.pth')
            
        if patience_counter >= max_patience:
            print(f"Early stopping after {max_patience} epochs without improvement")
            break
        
        # Check disk space usage
        check_disk_space(save_dir)
    
    print("\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    stable_training()

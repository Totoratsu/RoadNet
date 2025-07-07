#!/usr/bin/env python3
"""
Simple Unity Segmentation Training
A clean, simple training script for Unity segmentation without multiprocessing issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import Unity modules
from unity_dataset import UnityDataModule, UnitySegmentationDataset
from unity_unet import create_unity_model, UNITY_CLASSES, NUM_CLASSES

def calculate_iou(pred, target, num_classes):
    """Calculate IoU for segmentation."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    # Return mean IoU, ignoring NaN values
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0

def train_unity_model():
    """Train Unity segmentation model."""
    
    print("🚗 Simple Unity Segmentation Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'epochs': 60,
        'batch_size': 8,  # Smaller batch for stability
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'save_every': 10,
        'patience': 20,
    }
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🔧 Using device: {device}")
    
    # Create dataset (no multiprocessing to avoid issues)
    train_dataset = UnitySegmentationDataset(
        data_dir="../data",
        sequence="sequence.0",
        split="train",
        image_size=(256, 512),
        augment=True
    )
    
    val_dataset = UnitySegmentationDataset(
        data_dir="../data", 
        sequence="sequence.0",
        split="val",
        image_size=(256, 512),
        augment=False
    )
    
    # Data loaders (no multiprocessing)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Create model
    model = create_unity_model(optimization="balanced")
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] * 0.01
    )
    
    # Training tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_iou = 0.0
    epochs_without_improvement = 0
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🎯 Starting training for {config['epochs']} epochs...")
    print(f"   Early stopping patience: {config['patience']}")
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n📈 Epoch {epoch}/{config['epochs']}")
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == masks).sum().item()
            train_total += masks.numel()
            
            if batch_idx % 3 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"   {progress:5.1f}% | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == masks).sum().item()
                val_total += masks.numel()
                
                # Calculate IoU for this batch
                batch_iou = calculate_iou(predictions, masks, NUM_CLASSES)
                if not np.isnan(batch_iou):
                    all_ious.append(batch_iou)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_iou = np.mean(all_ious) if all_ious else 0.0
        
        # Learning rate step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"   Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}")
        print(f"   Val IoU: {val_iou:.3f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            
            model_name = f"unity_best_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = save_dir / model_name
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_acc': val_acc,
                'config': config
            }, model_path)
            
            print(f"💾 New best model saved: {model_name} (IoU: {val_iou:.3f})")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_name = f"unity_checkpoint_epoch_{epoch:03d}.pt"
            checkpoint_path = save_dir / checkpoint_name
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_name}")
        
        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\n🛑 Early stopping: No improvement for {epochs_without_improvement} epochs")
            break
        
        # Plot progress every 10 epochs
        if epoch % 10 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.title('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(history['val_iou'])
            plt.title('Validation IoU')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(history['val_acc'])
            plt.title('Validation Accuracy')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Training completed!")
    print(f"   Total time: {total_time/3600:.1f} hours")
    print(f"   Best IoU: {best_iou:.3f}")
    print(f"   Final epoch: {epoch}")
    
    # Test inference speed
    print(f"\n⚡ Testing inference speed...")
    model.eval()
    test_image = torch.randn(1, 3, 256, 512).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_image)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(50):
            _ = model(test_image)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50
    fps = 1.0 / avg_time
    
    print(f"✅ Inference performance:")
    print(f"   Time per frame: {avg_time*1000:.1f} ms")
    print(f"   FPS: {fps:.1f}")
    
    return {
        'best_iou': best_iou,
        'total_time_hours': total_time / 3600,
        'final_epoch': epoch,
        'fps': fps,
        'history': history
    }

if __name__ == "__main__":
    results = train_unity_model()
    print(f"\n🎯 Final Results:")
    print(f"   Best IoU: {results['best_iou']:.1f}%")
    print(f"   Training time: {results['total_time_hours']:.1f} hours") 
    print(f"   Inference FPS: {results['fps']:.1f}")
    print(f"\n💡 Your Unity model is ready for deployment!")

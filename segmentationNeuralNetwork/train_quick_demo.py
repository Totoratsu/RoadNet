#!/usr/bin/env python3
"""
Quick Unity Model Training Demo
Train a basic Unity segmentation model with minimal configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path

def quick_train_unity_model():
    """Train a Unity segmentation model quickly for demo purposes."""
    
    print("🚀 Quick Unity Model Training Demo")
    print("=" * 50)
    
    # Import Unity components
    try:
        from unity_dataset import UnityDataModule
        from unity_unet import create_unity_model, NUM_CLASSES, get_unity_class_weights
    except ImportError as e:
        print(f"❌ Failed to import Unity components: {e}")
        return
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create dataset
    data_module = UnityDataModule(
        data_dir="../data",
        sequence="sequence.0",
        batch_size=8,  # Small batch for quick demo
        image_size=(256, 512),
        num_workers=2
    )
    
    train_loader, val_loader, _ = data_module.get_dataloaders()
    print(f"📊 Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model (speed optimized for quick training)
    model = create_unity_model(optimization="speed")
    model = model.to(device)
    print(f"🏗️  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Quick training loop (just a few epochs for demo)
    num_epochs = 2
    print(f"\n🎯 Training for {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress every few batches
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s, Avg Loss: {avg_loss:.4f}")
    
    # Save model
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    model_name = f"unity_quick_demo_{int(time.time())}.pt"
    model_path = save_dir / model_name
    
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Quick validation
    print(f"\n📊 Quick validation...")
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_pixels / total_pixels
    
    print(f"✅ Validation Results:")
    print(f"   Avg Loss: {avg_val_loss:.4f}")
    print(f"   Pixel Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Test inference speed
    print(f"\n⚡ Testing inference speed...")
    from optimized_inference import OptimizedSegmentationModel
    from PIL import Image
    
    # Create optimized model
    opt_model = OptimizedSegmentationModel(model_path, optimization_level='speed')
    
    # Test with dummy image
    test_image = Image.new('RGB', (512, 256), color='blue')
    
    # Warmup
    for _ in range(5):
        _ = opt_model.predict(test_image)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        result = opt_model.predict(test_image)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    fps = 1.0 / avg_time
    
    print(f"✅ Inference Performance:")
    print(f"   Avg inference time: {avg_time*1000:.1f} ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Output shape: {result.shape}")
    
    # Summary
    print(f"\n🎉 Quick training demo completed!")
    print(f"📋 Summary:")
    print(f"   • Model: Unity UNet with {NUM_CLASSES} classes")
    print(f"   • Training: {num_epochs} epochs on {len(train_loader.dataset)} samples")
    print(f"   • Performance: {fps:.1f} FPS inference speed")
    print(f"   • Saved: {model_path}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Full training: python train_unity_segmentation.py")
    print(f"   2. Real-time demo: python real_time_demo.py")
    print(f"   3. Model optimization: python model_optimizer.py")

if __name__ == "__main__":
    quick_train_unity_model()

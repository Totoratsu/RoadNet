# Training Configuration and Quick Start Guide

## 🎉 Setup Complete! Everything is Ready for Training

### ✅ What's Working:
- **182 fully labeled images** (100% complete!)
- **Balanced dataset**: 67% Front, 21% Left, 12% Right
- **Training pipeline tested** and working
- **Inference pipeline tested** and working
- **All dependencies installed**

### 🚀 Ready to Train Commands (Now with Advanced Data Augmentation!)

When you're ready to start training, use these commands:

#### Option 1: Quick Training with Light Augmentation (Recommended to start)
```bash
# Fast training with simple model and light augmentation (30-60 minutes)
python main.py train ../data/sequence.0 --model_type simple --epochs 50 --batch_size 16 --augmentation light

# Or use the training script directly:
python train_decision_model.py ../data/sequence.0 --model_type simple --epochs 50 --batch_size 16 --augmentation light
```

#### Option 2: Balanced Training with Medium Augmentation (Recommended)
```bash
# Better accuracy with ResNet model and medium augmentation (1-3 hours)
python main.py train ../data/sequence.0 --model_type resnet --epochs 100 --batch_size 8 --augmentation medium

# Or use the training script directly:
python train_decision_model.py ../data/sequence.0 --model_type resnet --epochs 100 --batch_size 8 --augmentation medium
```

#### Option 3: Maximum Quality Training with Heavy Augmentation
```bash
# Maximum quality training with heavy augmentation (3-6 hours)
python main.py train ../data/sequence.0 --model_type resnet --epochs 200 --batch_size 16 --lr 0.0005 --augmentation heavy
```

### 🎨 Data Augmentation Features Added:

#### Light Augmentation:
- Random horizontal flips (30%)
- Basic color jittering (brightness, contrast, saturation)
- Minimal transformations for fast training

#### Medium Augmentation (Default):
- Random horizontal flips (50%)
- Small rotations (-5° to +5°)
- Minor translations and scaling
- Perspective transformations
- Enhanced color jittering
- Gaussian blur variations
- Random grayscale (10%)
- Gaussian noise addition
- Random patch erasing (cutout)

#### Heavy Augmentation:
- Aggressive horizontal flips (60%)
- Larger rotations (-10° to +10°)
- More translations and scaling
- Stronger perspective transformations
- Heavy color augmentations
- More blur and grayscale
- Higher noise levels
- More aggressive patch erasing

### 🎯 Why Data Augmentation Helps:
1. **Increases effective dataset size** - Your 182 images become thousands of variations
2. **Improves generalization** - Model learns to handle different lighting, angles, noise
3. **Reduces overfitting** - Model becomes more robust to variations
4. **Better real-world performance** - Handles diverse driving conditions
5. **Simulates different weather/lighting** - Day/night, sunny/cloudy conditions

### 📊 Your Dataset Statistics:
- **Total Images**: 182 (100% labeled ✅)
- **Training Split**: ~146 images
- **Validation Split**: ~36 images
- **Classes**:
  - FRONT: 122 images (67.0%)
  - LEFT: 38 images (20.9%) 
  - RIGHT: 22 images (12.1%)

### 🎯 Expected Training Time (on your Mac M3 Pro):
- **Simple CNN**: 30-60 minutes for 50 epochs
- **ResNet**: 1-3 hours for 100 epochs
- **ResNet Extended**: 3-6 hours for 200 epochs

### 📁 Files That Will Be Created:
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/final_model.pth` - Final model after all epochs
- Training logs will be shown in terminal

### 🔍 After Training, Test Your Model:
```bash
# Test on single image
python main.py infer checkpoints/best_model.pth --image_path ../data/sequence.0/step10.camera.semantic\ segmentation.png

# Test on entire dataset
python main.py infer checkpoints/best_model.pth --data_dir ../data/sequence.0 --output_file predictions.json
```

### 🎨 Visualize Augmentation Effects:
```bash
# See what augmentations look like on your data
python visualize_augmentation.py ../data/sequence.0/step5.camera.semantic\ segmentation.png --samples 4
```

### 💡 Training Tips:
1. **Start with Simple CNN** (faster) to verify everything works
2. **Then try ResNet** for better accuracy
3. **Monitor the validation accuracy** - it should improve over epochs
4. **Training will auto-save the best model** based on validation accuracy
5. **You can stop training anytime** with Ctrl+C and resume later

### 🚨 If You Get Errors:
- **Out of Memory**: Reduce `--batch_size` to 4 or 8
- **Too Slow**: Use `--model_type simple` instead of `resnet`
- **Import Errors**: Run `python main.py setup` to reinstall dependencies

## 🎊 You're All Set with Enhanced Data Augmentation!

### ✅ New Features Added:
- **🎨 Advanced Data Augmentation** with 3 strength levels
- **🔄 Geometric Transformations** (rotation, scaling, perspective)
- **🌈 Color Augmentations** (brightness, contrast, hue variations)
- **🔊 Noise Addition** for robustness
- **✂️ Random Erasing** (cutout) for occlusion handling
- **👁️ Visualization Tool** to see augmentation effects

### 🚀 Recommended Training Approach:
1. **Start with Medium Augmentation** - Good balance of speed and quality
2. **Monitor validation accuracy** - Should improve steadily
3. **Try Heavy Augmentation** if you want maximum robustness
4. **Use Light Augmentation** for quick experiments

### 📈 Expected Improvements:
- **Better generalization** to new driving scenarios
- **Reduced overfitting** with small dataset
- **More robust predictions** in varying conditions
- **Higher validation accuracy** with longer training

Everything is tested and ready with enhanced data augmentation! Just run one of the training commands above when you return!

## 🧪 Now Test Your Trained Model!

Your model achieved **94.44% validation accuracy** - excellent performance! Here's how to test it:

### 🎯 Quick Testing Options:

#### Option 1: Simple Visual Test (Recommended)
```bash
# Test any image with visualization
python test_model.py --image ../data/sequence.0/step10.camera.semantic\ segmentation.png

# Interactive testing mode
python test_model.py --cli
```

#### Option 2: GUI Interface
```bash
# Launch user-friendly GUI
python test_model.py --gui
```

#### Option 3: Comprehensive Testing
```bash
# Test all images with analysis
python test_inference_cli.py checkpoints/best_model.pth --data_dir ../data/sequence.0 --save_viz

# Generate detailed performance report
python batch_analysis.py checkpoints/best_model.pth ../data/sequence.0
```

### 📊 What You'll See:

- **🖼️ Input Image**: Your segmentation mask
- **🚗 Prediction**: FRONT/LEFT/RIGHT decision with confidence
- **📈 Confidence Chart**: Probability for each decision
- **🎨 Color-coded Results**: 
  - 🟢 Green = FRONT
  - 🔵 Blue = LEFT
  - 🟠 Orange = RIGHT

### 💡 Testing Examples:

```bash
# Test with the GUI (most user-friendly)
python test_model.py --gui

# Quick single image test
python test_model.py --image ../data/sequence.0/step5.camera.semantic\ segmentation.png

# Test multiple images and save visualizations
python test_inference_cli.py checkpoints/best_model.pth --data_dir ../data/sequence.0 --save_viz --output results.json

# Compare with your labeled data
python test_inference_cli.py checkpoints/best_model.pth --data_dir ../data/sequence.0 --labels_file driving_labels.json
```

### 🎉 Your Model Performance:
- ✅ **Training Accuracy**: 98.63%
- ✅ **Validation Accuracy**: 94.44% (excellent!)
- ✅ **Class-specific augmentation**: Working perfectly
- ✅ **Geometric-only transforms**: Preserving segmentation integrity
- ✅ **Ready for real-world testing**!

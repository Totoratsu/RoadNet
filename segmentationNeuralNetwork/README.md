# Segmentation Neural Network

A semantic segmentation neural network designed for real-time driving applications using Unity-generated data.

## 🎯 Overview

This segmentation neural network processes Unity driving simulation data to perform semantic segmentation with 12 classes optimized for autonomous driving scenarios. The system includes stable training scripts, real-time inference capabilities, and comprehensive testing tools.

## 📦 Pre-trained Models

### Stable Model Checkpoint
The best performing stable model checkpoint is available for download:

**📁 Best Stable Model**: [Download from Google Drive](https://drive.google.com/file/d/1_Zmd_MHT9rcCJjkfZHsx-8saL2voLyut/view?usp=share_link)

- **Architecture**: MobileNetV2 UNet
- **Training**: Stable training with overfitting prevention
- **Performance**: 97.0% pixel accuracy, 0.298 mean IoU
- **Speed**: ~14.4 FPS on Apple Silicon
- **Location**: Place in `checkpoints_stable/best_model.pth`

## 🚀 Quick Start

### 1. Download Pre-trained Model
```bash
# Download the model from the link above and place it at:
segmentationNeuralNetwork/checkpoints_stable/best_model.pth
```

### 2. Test the Model
```bash
# Interactive demo with visualization
python demo_segmentation.py

# Batch performance analysis
python demo_segmentation.py --batch_analysis --num_samples 10

# Test specific sample
python demo_segmentation.py --sample 0
```

### 3. Real-time Performance Test
```bash
# Command-line performance benchmarking
python test_stable_model.py --quick        # Quick 5-sample test
python test_stable_model.py --real-time    # 30-second FPS test
```

## 📊 Model Performance

The stable model demonstrates:
- **Pixel Accuracy**: 97.0% 
- **Mean IoU**: 0.298
- **Inference Speed**: 14.4 FPS (Apple Silicon MPS)
- **Best Classes**: Void, Building, Wall detection
- **Challenging Classes**: Road, Vehicle, Pedestrian detection

## 🏗️ Architecture

- **Encoder**: MobileNetV2 (ImageNet pre-trained)
- **Decoder**: UNet architecture
- **Input**: 256×512 RGB images
- **Output**: 12-class semantic segmentation masks
- **Optimization**: Designed for real-time inference

## 📁 File Structure

```
segmentationNeuralNetwork/
├── README.md                    # This file
├── demo_segmentation.py         # Main testing/demo script
├── test_stable_model.py         # Performance benchmarking
├── train_stable.py             # Stable training script
├── unity_dataset.py            # Unity data loader
├── checkpoints_stable/         # Stable model checkpoints
│   ├── best_model.pth          # Best model (download required)
│   └── last_model.pth          # Latest training checkpoint
├── DEMO_GUIDE.md               # Testing guide
├── README_UNITY.md             # Unity-specific documentation
└── README_REALTIME.md          # Real-time optimization guide
```

## 🎮 Demo Features

The main demo script (`demo_segmentation.py`) provides:

- **6-panel Visualization**: Original, ground truth, prediction, overlay, errors, per-class IoU
- **Interactive Navigation**: Previous/Next/Random/Find Best/Worst buttons
- **Real-time Metrics**: FPS measurement and performance assessment
- **Batch Analysis**: Automated testing on multiple samples
- **Per-class Performance**: Detailed breakdown of class-wise accuracy

## 🔧 Training

For training your own model:

```bash
# Stable training (recommended)
python train_stable.py

# Other training options
python train_high_accuracy.py
python train_realtime_model.py
```

## 📋 Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install matplotlib numpy pillow
pip install opencv-python
```

## 🎯 Unity Classes

The model recognizes 12 semantic classes:
- **Void** (0): Background/undefined areas
- **Building** (1): Structures and buildings  
- **Fence** (2): Barriers and fences
- **Other** (3): Miscellaneous objects
- **Pedestrian** (4): People and pedestrians
- **Pole** (5): Posts and poles
- **RoadLine** (6): Lane markings
- **Road** (7): Driveable surfaces
- **SideWalk** (8): Pedestrian walkways
- **Vegetation** (9): Trees and plants
- **Vehicles** (10): Cars and vehicles
- **Wall** (11): Walls and barriers

## 📚 Additional Documentation

- [`DEMO_GUIDE.md`](DEMO_GUIDE.md) - Complete testing guide
- [`README_UNITY.md`](README_UNITY.md) - Unity-specific implementation details
- [`README_REALTIME.md`](README_REALTIME.md) - Real-time optimization guide
- [`ULTRA_TRAINING_GUIDE.md`](ULTRA_TRAINING_GUIDE.md) - Advanced training options

## ⚡ Performance Notes

The current model excels at structural detection (buildings, walls) but may need additional training for driving-specific classes (roads, vehicles). For production use in real-time driving applications, consider:

1. **Speed Optimization**: Model currently runs at 14.4 FPS, target 30+ FPS
2. **Class Balance**: Additional training data for road/vehicle detection  
3. **Real-time Deployment**: Use optimized inference scripts for production

## 🤝 Contributing

When training new models or making improvements, ensure compatibility with the existing demo and testing infrastructure.

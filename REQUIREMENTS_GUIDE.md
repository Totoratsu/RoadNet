# Requirements Guide

This project has multiple requirements files for different use cases. Choose the appropriate one based on your needs:

## 📁 Root Level
- **`requirements.txt`** - Complete dependencies for both segmentation and decision neural networks (recommended for development)

## 🧠 Segmentation Neural Network
- **`segmentationNeuralNetwork/requirements_training.txt`** - Full dependencies for training UNet models
- **`segmentationNeuralNetwork/requirements_inference.txt`** - Minimal dependencies for inference only

## 🚗 Decision Neural Network  
- **`decisionNeuralNetwork/requirements_training.txt`** - Full dependencies for training decision models
- **`decisionNeuralNetwork/requirements_inference.txt`** - Minimal dependencies for inference only
- **`decisionNeuralNetwork/requirements_minimal.txt`** - Legacy minimal requirements (kept for compatibility)

## 🚀 Quick Start

### For Complete Development Environment
```bash
pip install -r requirements.txt
```

### For Specific Use Cases

#### Segmentation Training
```bash
pip install -r segmentationNeuralNetwork/requirements_training.txt
```

#### Segmentation Inference Only
```bash
pip install -r segmentationNeuralNetwork/requirements_inference.txt
```

#### Decision Model Training
```bash
pip install -r decisionNeuralNetwork/requirements_training.txt
```

#### Decision Model Inference Only
```bash
pip install -r decisionNeuralNetwork/requirements_inference.txt
```

## 💡 Recommendations

1. **Development**: Use the root `requirements.txt` for full functionality
2. **Production Inference**: Use the specific inference requirements for smaller footprint
3. **Training**: Use the specific training requirements when you only need to train one type of model
4. **Docker/Containers**: Use inference-only requirements for deployment containers

## 🔧 GPU Support

All requirements files include CPU versions of PyTorch. For GPU acceleration:

```bash
# Install GPU version after installing requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📦 Virtual Environment

Recommended setup:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

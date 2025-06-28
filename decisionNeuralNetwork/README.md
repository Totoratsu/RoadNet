****# Driving Decision Neural Network

This project implements a neural network that learns to make driving decisions (go front, left, or right) based on semantic segmentation masks from driving scenes.

## Overview

The system takes semantic segmentation masks as input and predicts one of three driving actions:
- **Front**: Continue straight
- **Left**: Turn left  
- **Right**: Turn right

## Project Structure

```
decisionNeuralNetwork/
├── main.py                    # Main interface script
├── decision_dataset.py        # Dataset class for loading data
├── decision_model.py          # Neural network model definitions
├── annotate_data.py          # Tool for manually labeling data
├── train_decision_model.py   # Training script
├── inference.py              # Inference script for predictions
├── README.md                 # This file
└── checkpoints/              # Directory for saved models (created during training)
```

## Setup

### 1. Environment Setup
```bash
# Navigate to the project directory
cd decisionNeuralNetwork

# Setup environment and install dependencies
python main.py setup
```

### 2. Alternative Manual Setup
```bash
# Install required packages
pip install torch torchvision matplotlib scikit-learn pillow numpy
```

## Usage

### Step 1: Data Annotation

Before training, you need to manually label driving decisions for your segmentation masks:

```bash
# Start annotation process
python main.py annotate /path/to/data/sequence.0

# Continue annotation from where you left off
python main.py annotate /path/to/data/sequence.0 --continue

# Show annotation statistics
python main.py annotate /path/to/data/sequence.0 --stats
```

**Annotation Instructions:**
- The tool will display segmentation masks one by one
- For each image, decide what driving action should be taken
- Press: `0` for FRONT, `1` for LEFT, `2` for RIGHT
- Press: `s` to SKIP, `q` to QUIT and SAVE, `u` to UNDO last annotation

**Recommendation:** Label at least 50-100 images for decent performance, more for better results.

### Step 2: Train the Model

```bash
# Basic training with ResNet backbone
python main.py train /path/to/data/sequence.0

# Training with custom parameters
python main.py train /path/to/data/sequence.0 \
    --model_type resnet \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001

# Training with simple CNN (faster, smaller model)
python main.py train /path/to/data/sequence.0 \
    --model_type simple \
    --epochs 50
```

**Training Parameters:**
- `--model_type`: Choose between 'resnet' (better accuracy) or 'simple' (faster training)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save model checkpoints (default: 'checkpoints')

### Step 3: Run Inference

```bash
# Predict on a single image
python main.py infer checkpoints/best_model.pth \
    --image_path /path/to/segmentation_mask.png

# Predict on entire directory
python main.py infer checkpoints/best_model.pth \
    --data_dir /path/to/data/sequence.0 \
    --output_file predictions.json
```

## Model Architecture

### ResNet-based Model (default)
- Uses pre-trained ResNet18 as backbone
- Custom classifier head with dropout for regularization
- ~11M parameters
- Better accuracy but slower training

### Simple CNN Model
- Custom lightweight CNN architecture
- 4 convolutional layers with max pooling
- ~400K parameters  
- Faster training but potentially lower accuracy

## Data Format

The system expects:
1. **Segmentation masks**: PNG files named like `step{N}.camera.semantic segmentation.png`
2. **Labels file**: JSON file (`driving_labels.json`) mapping step numbers to driving decisions

Example labels file:
```json
{
  "0": 0,    // step 0 -> FRONT
  "1": 1,    // step 1 -> LEFT  
  "2": 0,    // step 2 -> FRONT
  "3": 2     // step 3 -> RIGHT
}
```

## Training Tips

1. **Data Balance**: Try to have roughly equal numbers of each class (front/left/right)
2. **Data Quality**: Ensure your manual annotations are consistent and accurate
3. **Model Selection**: Start with ResNet for better accuracy, use Simple CNN if training time is a concern
4. **Hyperparameters**: 
   - Start with default parameters
   - Increase epochs if validation accuracy is still improving
   - Reduce learning rate if loss oscillates
   - Increase batch size if you have enough GPU memory

## Troubleshooting

### Common Issues

1. **"Not enough labeled data" error**
   - You need at least 10 labeled samples to train
   - Recommended: 50+ samples for good performance

2. **Poor model performance**
   - Check data balance (equal distribution of classes)
   - Verify annotation quality
   - Try training for more epochs
   - Consider using data augmentation (already included in training)

3. **Out of memory errors**
   - Reduce batch size (`--batch_size 8` or `--batch_size 4`)
   - Use smaller image size (`--image_size 128`)
   - Use simple model instead of ResNet

4. **Slow training**
   - Use GPU if available (CUDA)
   - Reduce image size
   - Use simple model
   - Reduce number of data loader workers

## File Descriptions

- **`main.py`**: Unified interface for all operations
- **`decision_dataset.py`**: PyTorch dataset class for loading segmentation masks and labels
- **`decision_model.py`**: Neural network model definitions (ResNet and Simple CNN)
- **`annotate_data.py`**: Interactive tool for manually labeling driving decisions
- **`train_decision_model.py`**: Training script with validation and checkpointing
- **`inference.py`**: Script for making predictions with trained models

## Model Output

The model outputs:
- **Predicted class**: 0 (FRONT), 1 (LEFT), or 2 (RIGHT)
- **Confidence score**: Probability of the predicted class (0.0 to 1.0)
- **Class probabilities**: Probability distribution over all three classes

## Pre-trained Model

A pre-trained model is available for download:

**Download best_model.pth**: [Google Drive Link](https://drive.google.com/file/d/1HzyJE94FbkFGxA9s2dxgKLrx0Aj6w25J/view?usp=sharing)

This model was trained on driving decision data and can be used directly for inference without needing to train from scratch. Place the downloaded file in the `checkpoints/` directory to use with the inference scripts.

## Example Workflow

```bash
# 1. Setup environment
python main.py setup

# 2. Annotate your data (label 50-100 images)
python main.py annotate ../data/sequence.0 --batch_size 20

# 3. Train the model
python main.py train ../data/sequence.0 --epochs 100 --model_type resnet

# 4. Make predictions
python main.py infer checkpoints/best_model.pth --data_dir ../data/sequence.0 --output_file results.json
```

## Performance Monitoring

During training, the system will:
- Display training and validation loss/accuracy
- Save the best model based on validation accuracy
- Generate classification reports every 5 epochs
- Save training history for analysis

The best model is automatically saved as `checkpoints/best_model.pth`.

## Next Steps

1. **Data Collection**: Collect more diverse driving scenarios
2. **Data Augmentation**: Experiment with different augmentation techniques
3. **Model Architecture**: Try different backbone networks (ResNet50, EfficientNet, etc.)
4. **Temporal Information**: Incorporate sequential information from multiple frames
5. **Real-time Deployment**: Optimize model for real-time inference

## 🧪 Testing Your Trained Model

After training, you have several options to test your model:

### Option 1: Simple Test Interface (Recommended)
```bash
# Quick test with visualization
python test_model.py --image path/to/image.png

# Interactive command-line interface
python test_model.py --cli

# Show help with all options
python test_model.py
```

### Option 2: GUI Interface
```bash
# Launch graphical interface
python test_model.py --gui
# or
python test_inference_gui.py
```

Features:
- 🖼️ Interactive image selection
- 📊 Real-time confidence visualization
- 🎯 Easy model loading
- 📸 Sample image testing

### Option 3: Advanced Command-Line Interface
```bash
# Test single image with detailed analysis
python test_inference_cli.py checkpoints/best_model.pth --image path/to/image.png

# Test entire directory with batch analysis
python test_inference_cli.py checkpoints/best_model.pth --data_dir ../data/sequence.0 --save_viz

# Compare predictions with ground truth
python test_inference_cli.py checkpoints/best_model.pth --data_dir ../data/sequence.0 --labels_file driving_labels.json
```

### Option 4: Batch Analysis with Reports
```bash
# Generate comprehensive analysis report
python batch_analysis.py checkpoints/best_model.pth ../data/sequence.0

# Creates analysis_results/ with:
# - confusion_matrix.png
# - confidence_analysis.png  
# - detailed_results.csv
# - classification_report.json
```

### Option 5: Main Interface Testing
```bash
# Test via main interface
python main.py test --gui                    # GUI interface
python main.py test --image path/to/image    # Single image
python main.py test --data_dir path/to/dir   # Batch testing
```

### 📊 Visualization Features

All testing interfaces provide:
- **Input visualization**: Original segmentation mask
- **Prediction overlay**: Clear decision display with confidence
- **Confidence chart**: Bar chart showing all class probabilities
- **Color coding**: 
  - 🟢 Green for FRONT decisions
  - 🔵 Blue for LEFT decisions  
  - 🟠 Orange for RIGHT decisions

### 💡 Testing Tips

1. **Start with GUI**: Use `python test_model.py --gui` for interactive testing
2. **Test single images**: Verify model behavior on specific cases
3. **Batch testing**: Analyze performance across your dataset
4. **Compare with labels**: Use ground truth for validation
5. **Save visualizations**: Use `--save_viz` for documentation

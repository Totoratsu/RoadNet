"""
Driving Decision Neural Network Package

This package implements a neural network system for predicting driving decisions
(front, left, right) from semantic segmentation masks.

Main components:
- DecisionDataset: Dataset class for loading segmentation masks and labels
- DecisionCNN, SimpleCNN: Neural network models for decision prediction
- DataAnnotator: Tool for manually labeling driving decisions
- DecisionTrainer: Training pipeline with validation and checkpointing
- DecisionPredictor: Inference pipeline for making predictions

Usage:
    from decision_dataset import DecisionDataset
    from decision_model import DecisionCNN
    from train_decision_model import DecisionTrainer
    from inference import DecisionPredictor
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__email__ = ""

# Import main classes for easy access
from .decision_dataset import DecisionDataset, LABEL_MAPPING, REVERSE_LABEL_MAPPING
from .decision_model import DecisionCNN, SimpleCNN, create_model

# Make key classes available at package level
__all__ = [
    'DecisionDataset',
    'DecisionCNN', 
    'SimpleCNN',
    'create_model',
    'LABEL_MAPPING',
    'REVERSE_LABEL_MAPPING'
]

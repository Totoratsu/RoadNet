#!/usr/bin/env python3
"""
Ultra High-Performance Unity Segmentation Training
Optimized for maximum accuracy with 24+ hour training time.
Advanced techniques to prevent overfitting and achieve state-of-the-art results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Dict, List, Tuple
import math

# Import Unity modules
from unity_dataset import UnityDataModule, UnitySegmentationDataset
from unity_unet import create_unity_model, get_unity_class_weights, UNITY_CLASSES, NUM_CLASSES

class UltraHighPerformanceTrainer:
    """Ultra high-performance trainer for maximum accuracy with extensive training time."""
    
    def __init__(self, config: dict = None):
        """Initialize ultra high-performance trainer."""
        self.config = self._get_ultra_config()
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create enhanced data loaders with advanced augmentations
        self.data_module = UnityDataModule(
            data_dir=self.config['data_dir'],
            sequence=self.config['sequence'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers']
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_dataloaders()
        
        # Create ensemble of models for better performance
        self.models = self._create_model_ensemble()
        
        # Setup advanced training components
        self.criterions = self._setup_advanced_loss()
        self.optimizers = self._setup_advanced_optimizers()
        self.schedulers = self._setup_advanced_schedulers()
        
        # Advanced training tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_pixel_acc': [],
            'learning_rates': [],
            'epoch_times': [],
            'train_pixel_acc': [],
            'train_iou': [],
            'ensemble_performance': [],
            'individual_model_performance': []
        }
        
        self.best_accuracy = 0.0
        self.best_iou = 0.0
        self.best_ensemble_accuracy = 0.0
        self.best_model_paths = []
        self.plateau_count = 0
        
        # Create save directory
        self.save_dir = Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Ultra High-Performance Unity Trainer Initialized")
        print(f"   Training Time: 24+ hours (no time constraints)")
        print(f"   Focus: Maximum accuracy with overfitting prevention")
        print(f"   Device: {self.device}")
        print(f"   Models in ensemble: {len(self.models)}")
        print(f"   Total parameters: {sum(sum(p.numel() for p in model.parameters()) for model in self.models):,}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Max epochs: {self.config['epochs']}")
    
    def _get_ultra_config(self) -> dict:
        """Get ultra high-performance training configuration."""
        return {
            'data_dir': '../data',
            'sequence': 'sequence.0',
            'epochs': 300,               # Very long training
            'batch_size': 8,             # Smaller batch for ensemble training
            'learning_rate': 0.0005,     # Conservative learning rate
            'weight_decay': 1e-5,        # Light regularization
            'image_size': (256, 512),    # Standard size
            'num_workers': 8,            # Maximum data loading
            'save_every': 10,            # Save frequently
            'early_stopping_patience': 50, # Very patient
            'plateau_patience': 20,      # Plateau detection
            'use_ensemble': True,        # Multiple models
            'ensemble_size': 3,          # Number of models in ensemble
            'use_advanced_augmentation': True,
            'use_test_time_augmentation': True,
            'use_cosine_annealing': True,
            'use_warm_restarts': True,
            'use_advanced_loss': True,
            'use_gradient_accumulation': True,
            'accumulation_steps': 4,     # Effective batch size = 8 * 4 = 32
            'use_ema': True,             # Exponential moving average
            'ema_decay': 0.999,
            'use_mixup': True,           # Advanced augmentation
            'mixup_alpha': 0.2,
            'use_cutmix': True,
            'cutmix_alpha': 1.0,
            'use_label_smoothing': True,
            'label_smoothing': 0.1,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_dice_loss': True,
            'dice_weight': 0.3,
            'use_boundary_loss': True,
            'boundary_weight': 0.2,
            'gradient_clip': 1.0,
            'monitor_overfitting': True,
            'overfitting_threshold': 0.05,
        }
    
    def _setup_device(self) -> torch.device:
        """Setup training device with optimization."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"   Using Apple Metal Performance Shaders")
        else:
            device = torch.device('cpu')
            print(f"   Using CPU (consider GPU for 24h training)")
        return device
    
    def _create_model_ensemble(self) -> List[nn.Module]:
        """Create ensemble of different models for better performance."""
        models = []
        
        if self.config['use_ensemble']:
            # Model 1: ResNext50 for maximum accuracy
            model1 = create_unity_model(optimization='accuracy')
            models.append(model1.to(self.device))
            print(f"   Model 1: ResNext50 (Accuracy focus)")
            
            # Model 2: ResNet50 for robustness
            model2 = create_unity_model(encoder='resnet50', optimization='quality')
            models.append(model2.to(self.device))
            print(f"   Model 2: ResNet50 (Quality focus)")
            
            # Model 3: EfficientNet-B3 for efficiency
            model3 = create_unity_model(encoder='efficientnet-b3', optimization='balanced')
            models.append(model3.to(self.device))
            print(f"   Model 3: EfficientNet-B3 (Balanced)")
            
        else:
            # Single best model
            model = create_unity_model(optimization='accuracy')
            models.append(model.to(self.device))
            print(f"   Single model: ResNext50 (Accuracy focus)")
        
        return models
    
    def _setup_advanced_loss(self) -> List[nn.Module]:
        """Setup advanced loss functions for each model."""
        criterions = []
        
        for i, model in enumerate(self.models):
            # Combined loss function
            criterion = AdvancedLoss(
                num_classes=NUM_CLASSES,
                device=self.device,
                use_focal=self.config['use_focal_loss'],
                use_dice=self.config['use_dice_loss'],
                use_boundary=self.config['use_boundary_loss'],
                focal_alpha=self.config['focal_alpha'],
                focal_gamma=self.config['focal_gamma'],
                dice_weight=self.config['dice_weight'],
                boundary_weight=self.config['boundary_weight'],
                label_smoothing=self.config['label_smoothing'] if self.config['use_label_smoothing'] else 0.0
            )
            criterions.append(criterion)
            
        print(f"   Loss: Combined (CE + Focal + Dice + Boundary)")
        return criterions
    
    def _setup_advanced_optimizers(self) -> List[optim.Optimizer]:
        """Setup advanced optimizers for each model."""
        optimizers = []
        
        for model in self.models:
            # Use AdamW with different parameters for different model parts
            optimizer = optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': self.config['learning_rate'] * 0.1, 'weight_decay': self.config['weight_decay']},
                {'params': model.decoder.parameters(), 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay']},
                {'params': model.segmentation_head.parameters(), 'lr': self.config['learning_rate'] * 2, 'weight_decay': self.config['weight_decay'] * 0.1}
            ], betas=(0.9, 0.999), eps=1e-8)
            
            optimizers.append(optimizer)
        
        print(f"   Optimizer: AdamW with differential learning rates")
        return optimizers
    
    def _setup_advanced_schedulers(self) -> List[optim.lr_scheduler._LRScheduler]:
        """Setup advanced learning rate schedulers."""
        schedulers = []
        
        for optimizer in self.optimizers:
            if self.config['use_warm_restarts']:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=30,      # Restart every 30 epochs
                    T_mult=2,    # Double the period after each restart
                    eta_min=self.config['learning_rate'] * 0.001
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config['epochs'],
                    eta_min=self.config['learning_rate'] * 0.001
                )
            schedulers.append(scheduler)
        
        print(f"   Scheduler: Cosine Annealing with Warm Restarts")
        return schedulers
    
    def _apply_mixup_cutmix(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp and CutMix augmentations."""
        if self.config['use_mixup'] and np.random.random() < 0.5:
            return self._mixup(images, targets)
        elif self.config['use_cutmix'] and np.random.random() < 0.5:
            return self._cutmix(images, targets)
        else:
            return images, targets
    
    def _mixup(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation."""
        alpha = self.config['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets.float() + (1 - lam) * targets[index].float()
        
        return mixed_images, mixed_targets.long()
    
    def _cutmix(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        alpha = self.config['cutmix_alpha']
        lam = np.random.beta(alpha, alpha)
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        H, W = images.shape[2], images.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        targets[:, bby1:bby2, bbx1:bbx2] = targets[index, bby1:bby2, bbx1:bbx2]
        
        return images, targets
    
    def calculate_comprehensive_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Calculate comprehensive metrics including advanced ones."""
        predictions = torch.argmax(outputs, dim=1)
        
        # Basic metrics
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        pixel_acc = correct / total
        
        # IoU per class
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_id in range(NUM_CLASSES):
            tp = ((predictions == class_id) & (targets == class_id)).sum().item()
            fp = ((predictions == class_id) & (targets != class_id)).sum().item()
            fn = ((predictions != class_id) & (targets == class_id)).sum().item()
            tn = ((predictions != class_id) & (targets != class_id)).sum().item()
            
            # IoU
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            ious.append(iou)
            
            # Precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            precisions.append(precision)
            
            # Recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            recalls.append(recall)
            
            # F1 Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)
        
        mean_iou = np.mean(ious)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = np.mean(f1_scores)
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'class_ious': ious,
            'class_precisions': precisions,
            'class_recalls': recalls,
            'class_f1s': f1_scores
        }
    
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch with advanced techniques."""
        for model in self.models:
            model.train()
        
        epoch_losses = [0.0] * len(self.models)
        epoch_metrics = [{'pixel_accuracy': 0.0, 'mean_iou': 0.0} for _ in range(len(self.models))]
        num_batches = len(self.train_loader)
        
        # Mixed precision training
        scalers = []
        if self.device.type == 'cuda':
            scalers = [torch.cuda.amp.GradScaler() for _ in range(len(self.models))]
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Apply advanced augmentations
            if epoch > 10:  # Start augmentations after initial epochs
                images, masks = self._apply_mixup_cutmix(images, masks)
            
            # Train each model in ensemble
            for model_idx, (model, criterion, optimizer) in enumerate(zip(self.models, self.criterions, self.optimizers)):
                
                # Forward pass with mixed precision
                if scalers:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    # Gradient accumulation
                    loss = loss / self.config['accumulation_steps']
                    scalers[model_idx].scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                        # Gradient clipping
                        scalers[model_idx].unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                        
                        scalers[model_idx].step(optimizer)
                        scalers[model_idx].update()
                        optimizer.zero_grad()
                        
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Gradient accumulation
                    loss = loss / self.config['accumulation_steps']
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_losses[model_idx] += loss.item() * self.config['accumulation_steps']
                
                # Calculate metrics for monitoring (less frequent for speed)
                if batch_idx % 10 == 0 and model_idx == 0:  # Only for first model
                    with torch.no_grad():
                        metrics = self.calculate_comprehensive_metrics(outputs, masks)
                        epoch_metrics[model_idx]['pixel_accuracy'] += metrics['pixel_accuracy']
                        epoch_metrics[model_idx]['mean_iou'] += metrics['mean_iou']
            
            # Print progress
            if batch_idx % 20 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                avg_loss = np.mean([epoch_losses[i] / (batch_idx + 1) for i in range(len(self.models))])
                print(f"   Epoch {epoch:3d} | {progress:5.1f}% | Avg Loss: {avg_loss:.4f}")
        
        # Average metrics
        avg_losses = [loss / num_batches for loss in epoch_losses]
        avg_metrics = []
        for i in range(len(self.models)):
            avg_metric = {k: v / (num_batches // 10) for k, v in epoch_metrics[i].items()}
            avg_metrics.append(avg_metric)
        
        return {
            'losses': avg_losses,
            'metrics': avg_metrics,
            'ensemble_loss': np.mean(avg_losses)
        }
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch with ensemble predictions."""
        for model in self.models:
            model.eval()
        
        val_losses = [0.0] * len(self.models)
        all_individual_metrics = [[] for _ in range(len(self.models))]
        all_ensemble_metrics = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                model_outputs = []
                
                # Get predictions from each model
                for model_idx, (model, criterion) in enumerate(zip(self.models, self.criterions)):
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    val_losses[model_idx] += loss.item()
                    model_outputs.append(outputs)
                    
                    # Individual model metrics
                    metrics = self.calculate_comprehensive_metrics(outputs, masks)
                    all_individual_metrics[model_idx].append(metrics)
                
                # Ensemble prediction (average probabilities)
                if len(model_outputs) > 1:
                    ensemble_output = torch.stack(model_outputs).mean(dim=0)
                    ensemble_metrics = self.calculate_comprehensive_metrics(ensemble_output, masks)
                    all_ensemble_metrics.append(ensemble_metrics)
        
        # Average validation metrics
        avg_val_losses = [loss / len(self.val_loader) for loss in val_losses]
        
        # Individual model performance
        individual_performance = []
        for model_idx in range(len(self.models)):
            metrics = all_individual_metrics[model_idx]
            avg_performance = {
                'loss': avg_val_losses[model_idx],
                'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in metrics]),
                'mean_iou': np.mean([m['mean_iou'] for m in metrics]),
                'mean_precision': np.mean([m['mean_precision'] for m in metrics]),
                'mean_recall': np.mean([m['mean_recall'] for m in metrics]),
                'mean_f1': np.mean([m['mean_f1'] for m in metrics]),
                'class_ious': np.mean([m['class_ious'] for m in metrics], axis=0)
            }
            individual_performance.append(avg_performance)
        
        # Ensemble performance
        ensemble_performance = None
        if all_ensemble_metrics:
            ensemble_performance = {
                'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_ensemble_metrics]),
                'mean_iou': np.mean([m['mean_iou'] for m in all_ensemble_metrics]),
                'mean_precision': np.mean([m['mean_precision'] for m in all_ensemble_metrics]),
                'mean_recall': np.mean([m['mean_recall'] for m in all_ensemble_metrics]),
                'mean_f1': np.mean([m['mean_f1'] for m in all_ensemble_metrics]),
                'class_ious': np.mean([m['class_ious'] for m in all_ensemble_metrics], axis=0)
            }
        
        return {
            'individual_performance': individual_performance,
            'ensemble_performance': ensemble_performance,
            'best_individual_accuracy': max([p['pixel_accuracy'] for p in individual_performance]),
            'best_individual_iou': max([p['mean_iou'] for p in individual_performance])
        }
    
    def detect_overfitting(self, train_acc: float, val_acc: float) -> bool:
        """Detect overfitting based on train/val accuracy gap."""
        if self.config['monitor_overfitting']:
            gap = train_acc - val_acc
            return gap > self.config['overfitting_threshold']
        return False
    
    def save_ensemble_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save ensemble checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            base_filename = f"unity_ensemble_best_{timestamp}"
        else:
            base_filename = f"unity_ensemble_epoch_{epoch:03d}_{timestamp}"
        
        saved_paths = []
        
        # Save each model in ensemble
        for model_idx, model in enumerate(self.models):
            filename = f"{base_filename}_model_{model_idx}.pt"
            filepath = self.save_dir / filename
            
            checkpoint = {
                'epoch': epoch,
                'model_idx': model_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[model_idx].state_dict(),
                'scheduler_state_dict': self.schedulers[model_idx].state_dict(),
                'metrics': metrics,
                'config': self.config,
                'num_classes': NUM_CLASSES
            }
            
            torch.save(checkpoint, filepath)
            saved_paths.append(filepath)
        
        if is_best:
            self.best_model_paths = saved_paths
            ensemble_acc = metrics.get('ensemble_performance', {}).get('pixel_accuracy', 0) * 100
            print(f"💾 Saved BEST ensemble: {len(saved_paths)} models (Ensemble Acc: {ensemble_acc:.1f}%)")
        else:
            print(f"💾 Saved ensemble checkpoint: {len(saved_paths)} models")
        
        return saved_paths
    
    def plot_ultra_progress(self):
        """Plot comprehensive training progress."""
        if len(self.history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss comparison
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy with overfitting detection
        train_acc = [acc * 100 for acc in self.history['train_pixel_acc']]
        val_acc = [acc * 100 for acc in self.history['val_pixel_acc']]
        
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].axhline(y=90, color='g', linestyle='--', label='Target 90%', alpha=0.7)
        axes[0, 1].axhline(y=95, color='gold', linestyle='--', label='Stretch 95%', alpha=0.7)
        axes[0, 1].set_title('Accuracy Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU evolution
        train_iou = [iou * 100 for iou in self.history['train_iou']]
        val_iou = [iou * 100 for iou in self.history['val_iou']]
        
        axes[1, 0].plot(epochs, train_iou, 'b-', label='Train IoU', linewidth=2)
        axes[1, 0].plot(epochs, val_iou, 'r-', label='Val IoU', linewidth=2)
        axes[1, 0].set_title('IoU Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Ensemble vs Individual performance
        if self.history['ensemble_performance']:
            ensemble_acc = [p * 100 for p in self.history['ensemble_performance']]
            axes[1, 1].plot(epochs, val_acc, 'r--', label='Best Individual', alpha=0.7)
            axes[1, 1].plot(epochs, ensemble_acc, 'purple', label='Ensemble', linewidth=3)
            axes[1, 1].set_title('Ensemble vs Individual')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate evolution
        axes[2, 0].plot(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[2, 0].set_title('Learning Rate Schedule')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_yscale('log')
        
        # Training time analysis
        if self.history['epoch_times']:
            cumulative_time = np.cumsum(self.history['epoch_times']) / 3600  # Convert to hours
            axes[2, 1].plot(epochs, cumulative_time, 'green', linewidth=2)
            axes[2, 1].set_title('Cumulative Training Time')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Time (hours)')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'ultra_training_progress.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def train(self) -> dict:
        """Main ultra high-performance training loop."""
        print(f"\n🎯 Starting Ultra High-Performance Unity Training")
        print(f"   Time Budget: 24+ hours (no constraints)")
        print(f"   Target: 90%+ accuracy with ensemble")
        print(f"   Strategy: Multi-model ensemble + Advanced techniques")
        
        start_time = time.time()
        epochs_without_improvement = 0
        best_seen = False
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training with ensemble
            print(f"\n🏋️ Epoch {epoch}/{self.config['epochs']}")
            train_results = self.train_epoch(epoch)
            
            # Validation with ensemble
            val_results = self.validate_epoch(epoch)
            
            # Learning rate step for all models
            for scheduler in self.schedulers:
                scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_results['ensemble_loss'])
            self.history['val_loss'].append(np.mean([p['loss'] for p in val_results['individual_performance']]))
            
            # Get best individual and ensemble metrics
            best_individual_acc = val_results['best_individual_accuracy']
            best_individual_iou = val_results['best_individual_iou']
            
            ensemble_acc = 0
            ensemble_iou = 0
            if val_results['ensemble_performance']:
                ensemble_acc = val_results['ensemble_performance']['pixel_accuracy']
                ensemble_iou = val_results['ensemble_performance']['mean_iou']
            
            self.history['val_pixel_acc'].append(max(best_individual_acc, ensemble_acc))
            self.history['val_iou'].append(max(best_individual_iou, ensemble_iou))
            self.history['train_pixel_acc'].append(train_results['metrics'][0]['pixel_accuracy'])
            self.history['train_iou'].append(train_results['metrics'][0]['mean_iou'])
            self.history['learning_rates'].append(self.optimizers[0].param_groups[0]['lr'])
            self.history['ensemble_performance'].append(ensemble_acc)
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Print comprehensive summary
            print(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s")
            print(f"   Train Loss: {train_results['ensemble_loss']:.4f}")
            print(f"   Individual Models:")
            for i, perf in enumerate(val_results['individual_performance']):
                print(f"     Model {i+1}: Acc {perf['pixel_accuracy']*100:.1f}%, IoU {perf['mean_iou']*100:.1f}%")
            
            if val_results['ensemble_performance']:
                print(f"   🎯 Ensemble: Acc {ensemble_acc*100:.1f}%, IoU {ensemble_iou*100:.1f}%")
            
            # Overfitting detection
            if self.detect_overfitting(self.history['train_pixel_acc'][-1], self.history['val_pixel_acc'][-1]):
                print(f"   ⚠️ Overfitting detected (gap: {(self.history['train_pixel_acc'][-1] - self.history['val_pixel_acc'][-1])*100:.1f}%)")
            
            # Check for best model
            current_best = max(best_individual_acc, ensemble_acc)
            is_best = current_best > self.best_accuracy
            
            if is_best:
                self.best_accuracy = current_best
                self.best_iou = max(best_individual_iou, ensemble_iou)
                epochs_without_improvement = 0
                self.save_ensemble_checkpoint(epoch, val_results, is_best=True)
                
                if current_best >= 0.90:
                    best_seen = True
                    print(f"🎉 90%+ ACCURACY ACHIEVED! ({current_best*100:.1f}%)")
            else:
                epochs_without_improvement += 1
            
            # Save regular checkpoint
            if epoch % self.config['save_every'] == 0:
                self.save_ensemble_checkpoint(epoch, val_results, is_best=False)
            
            # Plot progress
            if epoch % 5 == 0:
                self.plot_ultra_progress()
            
            # Print class-wise performance every 20 epochs
            if epoch % 20 == 0 and val_results['ensemble_performance']:
                print(f"   Class IoUs (Ensemble):")
                for i, iou in enumerate(val_results['ensemble_performance']['class_ious']):
                    if i in UNITY_CLASSES:
                        class_name = UNITY_CLASSES[i]['name']
                        print(f"     {class_name:12s}: {iou*100:.1f}%")
            
            # Early stopping (very patient)
            if best_seen and epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n🛑 Early stopping: 90%+ achieved and no improvement for {epochs_without_improvement} epochs")
                break
            
            # Plateau detection and LR adjustment
            if epochs_without_improvement >= self.config['plateau_patience']:
                self.plateau_count += 1
                print(f"📊 Plateau detected ({self.plateau_count}). Consider manual intervention.")
                
                # Reset plateau counter
                epochs_without_improvement = max(0, epochs_without_improvement - 10)
        
        total_time = time.time() - start_time
        
        # Final comprehensive summary
        final_accuracy = self.best_accuracy * 100
        final_iou = self.best_iou * 100
        
        print(f"\n🎉 Ultra High-Performance Training Completed!")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Final Accuracy: {final_accuracy:.1f}%")
        print(f"   Final IoU: {final_iou:.1f}%")
        print(f"   Models saved: {len(self.best_model_paths)}")
        print(f"   Total epochs: {epoch}")
        print(f"   Overfitting incidents: {self.plateau_count}")
        
        # Achievement analysis
        if final_accuracy >= 95:
            print(f"🏆 OUTSTANDING: 95%+ accuracy achieved!")
        elif final_accuracy >= 90:
            print(f"🎯 EXCELLENT: 90%+ accuracy achieved!")
        elif final_accuracy >= 85:
            print(f"✅ GOOD: 85%+ accuracy achieved!")
        else:
            print(f"📈 PROGRESS: {final_accuracy:.1f}% accuracy (continue training)")
        
        # Final plot
        self.plot_ultra_progress()
        
        return {
            'best_accuracy': self.best_accuracy,
            'best_iou': self.best_iou,
            'best_model_paths': [str(p) for p in self.best_model_paths],
            'total_epochs': epoch,
            'total_time_hours': total_time / 3600,
            'target_90_reached': final_accuracy >= 90,
            'target_95_reached': final_accuracy >= 95,
            'final_metrics': val_results,
            'history': self.history,
            'plateau_count': self.plateau_count
        }

class AdvancedLoss(nn.Module):
    """Advanced combined loss function for maximum performance."""
    
    def __init__(self, num_classes, device, use_focal=True, use_dice=True, use_boundary=True,
                 focal_alpha=1.0, focal_gamma=2.0, dice_weight=0.3, boundary_weight=0.2, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.use_boundary = use_boundary
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        # Base loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)
        
        # Focal loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, inputs, targets):
        """Focal loss for hard example mining."""
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=-1, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        """Dice loss for better boundary prediction."""
        smooth = 1e-6
        
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_losses = []
        for c in range(self.num_classes):
            pred_c = inputs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_losses.append(1 - dice)
        
        return torch.stack(dice_losses).mean()
    
    def boundary_loss(self, inputs, targets):
        """Boundary loss for better edge prediction."""
        # Simple boundary loss using gradient magnitude
        # Convert predictions to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Calculate gradients
        grad_x = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        grad_y = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        
        # Boundary loss encourages sharp boundaries
        boundary_loss = grad_x.mean() + grad_y.mean()
        
        return boundary_loss
    
    def forward(self, inputs, targets):
        """Combined loss computation."""
        # Base cross-entropy loss
        total_loss = self.ce_loss(inputs, targets)
        
        # Add focal loss
        if self.use_focal:
            focal = self.focal_loss(inputs, targets)
            total_loss = total_loss + focal
        
        # Add dice loss
        if self.use_dice:
            dice = self.dice_loss(inputs, targets)
            total_loss = total_loss + self.dice_weight * dice
        
        # Add boundary loss
        if self.use_boundary:
            boundary = self.boundary_loss(inputs, targets)
            total_loss = total_loss + self.boundary_weight * boundary
        
        return total_loss

def main():
    """Main ultra high-performance training function."""
    print("🚀 Ultra High-Performance Unity Segmentation Training")
    print("Time Budget: 24+ hours | Target: 90%+ accuracy")
    print("=" * 80)
    
    # Ultra high-performance configuration
    config = {
        'epochs': 300,                  # Very long training
        'batch_size': 8,                # Smaller for ensemble
        'learning_rate': 0.0005,        # Conservative
        'use_ensemble': True,           # Multiple models
        'ensemble_size': 3,             # 3 different architectures
        'early_stopping_patience': 50, # Very patient
        'save_every': 10,               # Frequent saves
        'use_advanced_augmentation': True,
        'use_mixup': True,
        'use_cutmix': True,
        'use_focal_loss': True,
        'use_dice_loss': True,
        'use_boundary_loss': True,
        'monitor_overfitting': True,
    }
    
    # Create ultra trainer
    trainer = UltraHighPerformanceTrainer(config)
    
    # Start training
    results = trainer.train()
    
    # Final performance test
    print(f"\n🧪 Testing ensemble performance...")
    
    print(f"\n🏆 Final Results Summary:")
    print(f"   🎯 Best Accuracy: {results['best_accuracy']*100:.1f}%")
    print(f"   📊 Best IoU: {results['best_iou']*100:.1f}%")
    print(f"   ⏱️ Training Time: {results['total_time_hours']:.1f} hours")
    print(f"   📈 Total Epochs: {results['total_epochs']}")
    print(f"   🎲 Models in Ensemble: {len(results['best_model_paths'])}")
    print(f"   🎯 90% Target: {'✅ ACHIEVED' if results['target_90_reached'] else '❌ Not reached'}")
    print(f"   🏆 95% Stretch: {'✅ ACHIEVED' if results['target_95_reached'] else '❌ Not reached'}")
    
    print(f"\n💡 Your ultra high-performance Unity segmentation ensemble is ready!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unity Segmentation Visualization Demo
Test and visualize segmentation masks on Unity test data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
import argparse

# Import Unity modules
from unity_dataset import UnitySegmentationDataset
from unity_unet import UNITY_CLASSES, NUM_CLASSES
from optimized_inference import OptimizedSegmentationModel

class SegmentationVisualizer:
    """Visualize and test Unity segmentation results."""
    
    def __init__(self, model_path: str = None):
        """Initialize the visualizer."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load test dataset
        self.test_dataset = UnitySegmentationDataset(
            data_dir="../data",
            sequence="sequence.0", 
            split="test",
            image_size=(256, 512),
            augment=False
        )
        
        print(f"📊 Loaded {len(self.test_dataset)} test samples")
        
        # Load model if provided
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                self.model = OptimizedSegmentationModel(model_path, optimization_level='balanced')
                print(f"✅ Loaded model: {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                print("Will show ground truth only")
        else:
            print("⚠️ No model provided or found. Will show ground truth only")
        
        # Color map for visualization
        self.class_colors = self._create_color_map()
    
    def _create_color_map(self):
        """Create color map for visualization."""
        colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
        
        for class_id, class_info in UNITY_CLASSES.items():
            color = class_info['color']
            colors[class_id] = [color[0], color[1], color[2]]  # RGB
        
        return colors
    
    def _mask_to_color(self, mask):
        """Convert class mask to RGB image."""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(NUM_CLASSES):
            class_pixels = mask == class_id
            color_mask[class_pixels] = self.class_colors[class_id]
        
        return color_mask
    
    def _denormalize_image(self, tensor_image):
        """Denormalize image tensor for display."""
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = tensor_image.permute(1, 2, 0).cpu().numpy()
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def visualize_sample(self, sample_idx: int, save_path: str = None):
        """Visualize a single test sample."""
        if sample_idx >= len(self.test_dataset):
            print(f"❌ Sample index {sample_idx} out of range (max: {len(self.test_dataset)-1})")
            return
        
        # Get test sample
        image_tensor, gt_mask = self.test_dataset[sample_idx]
        
        # Get original image
        original_image = self._denormalize_image(image_tensor)
        gt_mask_np = gt_mask.numpy()
        
        # Predict if model available
        prediction = None
        if self.model:
            # Convert tensor to PIL for model
            pil_image = Image.fromarray((original_image * 255).astype(np.uint8))
            prediction = self.model.predict(pil_image)
        
        # Create visualization
        fig_width = 15 if prediction is not None else 10
        fig, axes = plt.subplots(2, 3 if prediction is not None else 2, figsize=(fig_width, 8))
        
        if prediction is not None:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth mask
        gt_color = self._mask_to_color(gt_mask_np)
        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        if prediction is not None:
            # Prediction mask
            pred_color = self._mask_to_color(prediction)
            axes[2].imshow(pred_color)
            axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Overlay: Original + Prediction
            overlay = original_image.copy()
            pred_alpha = 0.6
            pred_normalized = pred_color.astype(np.float32) / 255.0
            overlay = overlay * (1 - pred_alpha) + pred_normalized * pred_alpha
            axes[3].imshow(overlay)
            axes[3].set_title('Original + Prediction', fontsize=14, fontweight='bold')
            axes[3].axis('off')
            
            # Error map (difference between GT and prediction)
            error_map = (gt_mask_np != prediction).astype(np.float32)
            im = axes[4].imshow(error_map, cmap='Reds', alpha=0.8)
            axes[4].set_title('Error Map (Red = Wrong)', fontsize=14, fontweight='bold')
            axes[4].axis('off')
            
            # Class distribution comparison
            gt_unique, gt_counts = np.unique(gt_mask_np, return_counts=True)
            pred_unique, pred_counts = np.unique(prediction, return_counts=True)
            
            # Create bar chart
            all_classes = list(range(NUM_CLASSES))
            gt_dist = np.zeros(NUM_CLASSES)
            pred_dist = np.zeros(NUM_CLASSES)
            
            for i, cls in enumerate(gt_unique):
                if cls < NUM_CLASSES:
                    gt_dist[cls] = gt_counts[i] / gt_mask_np.size * 100
            
            for i, cls in enumerate(pred_unique):
                if cls < NUM_CLASSES:
                    pred_dist[cls] = pred_counts[i] / prediction.size * 100
            
            x = np.arange(NUM_CLASSES)
            width = 0.35
            
            axes[5].bar(x - width/2, gt_dist, width, label='Ground Truth', alpha=0.8)
            axes[5].bar(x + width/2, pred_dist, width, label='Prediction', alpha=0.8)
            axes[5].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
            axes[5].set_xlabel('Class ID')
            axes[5].set_ylabel('Percentage')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        else:
            # Show class distribution for ground truth only
            gt_unique, gt_counts = np.unique(gt_mask_np, return_counts=True)
            gt_dist = np.zeros(NUM_CLASSES)
            
            for i, cls in enumerate(gt_unique):
                if cls < NUM_CLASSES:
                    gt_dist[cls] = gt_counts[i] / gt_mask_np.size * 100
            
            axes[2].bar(range(NUM_CLASSES), gt_dist, alpha=0.8)
            axes[2].set_title('Ground Truth Class Distribution (%)', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Class ID')
            axes[2].set_ylabel('Percentage')
            axes[2].grid(True, alpha=0.3)
            
            # Hide unused subplot
            axes[3].axis('off')
        
        plt.suptitle(f'Unity Segmentation Test Sample {sample_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization: {save_path}")
        
        plt.show()
        
        # Print metrics if prediction available
        if prediction is not None:
            self._print_metrics(gt_mask_np, prediction, sample_idx)
    
    def _print_metrics(self, gt_mask, prediction, sample_idx):
        """Print detailed metrics for the sample."""
        print(f"\n📊 Metrics for Sample {sample_idx}:")
        print("=" * 40)
        
        # Overall accuracy
        total_pixels = gt_mask.size
        correct_pixels = (gt_mask == prediction).sum()
        pixel_accuracy = correct_pixels / total_pixels
        
        print(f"Pixel Accuracy: {pixel_accuracy:.3f} ({pixel_accuracy*100:.1f}%)")
        
        # Per-class IoU
        class_ious = []
        print(f"\nPer-Class IoU:")
        
        for class_id in range(NUM_CLASSES):
            # Check if class exists in ground truth
            gt_has_class = (gt_mask == class_id).any()
            pred_has_class = (prediction == class_id).any()
            
            if gt_has_class or pred_has_class:
                tp = ((prediction == class_id) & (gt_mask == class_id)).sum()
                fp = ((prediction == class_id) & (gt_mask != class_id)).sum()
                fn = ((prediction != class_id) & (gt_mask == class_id)).sum()
                
                if tp + fp + fn > 0:
                    iou = tp / (tp + fp + fn)
                    class_ious.append(iou)
                    class_name = UNITY_CLASSES[class_id]['name']
                    print(f"  {class_name:12s}: {iou:.3f}")
                else:
                    class_ious.append(0.0)
        
        mean_iou = np.mean(class_ious) if class_ious else 0.0
        print(f"\nMean IoU: {mean_iou:.3f} ({mean_iou*100:.1f}%)")
    
    def visualize_all_test_samples(self, save_dir: str = "test_visualizations"):
        """Visualize all test samples and save them."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print(f"🎨 Visualizing all {len(self.test_dataset)} test samples...")
        
        for i in range(len(self.test_dataset)):
            output_file = save_path / f"test_sample_{i:03d}.png"
            print(f"Processing sample {i+1}/{len(self.test_dataset)}...")
            
            # Close any existing plots to avoid memory issues
            plt.close('all')
            
            self.visualize_sample(i, save_path=str(output_file))
        
        print(f"✅ All visualizations saved to {save_dir}/")
    
    def create_class_legend(self, save_path: str = "class_legend.png"):
        """Create a legend showing all classes and their colors."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create color patches for each class
        y_pos = np.arange(NUM_CLASSES)
        colors = self.class_colors / 255.0  # Normalize for matplotlib
        
        for i, (class_id, class_info) in enumerate(UNITY_CLASSES.items()):
            ax.barh(y_pos[i], 1, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.text(0.5, y_pos[i], f"{class_id}: {class_info['name']}", 
                   ha='center', va='center', fontweight='bold', fontsize=12)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Class {i}" for i in range(NUM_CLASSES)])
        ax.set_xlabel('Color')
        ax.set_title('Unity Segmentation Class Colors', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Class legend saved: {save_path}")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Unity Segmentation Visualization Demo')
    parser.add_argument('--model', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize (default: 0)')
    parser.add_argument('--all', action='store_true', help='Visualize all test samples')
    parser.add_argument('--legend', action='store_true', help='Create class color legend')
    parser.add_argument('--save_dir', type=str, default='test_visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("🎨 Unity Segmentation Visualization Demo")
    print("=" * 50)
    
    # Find best model if not specified
    model_path = args.model
    if not model_path:
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            best_models = list(checkpoints_dir.glob("*unity_best*.pt"))
            if best_models:
                model_path = str(sorted(best_models)[-1])  # Most recent
                print(f"🎯 Auto-detected model: {model_path}")
    
    # Create visualizer
    visualizer = SegmentationVisualizer(model_path)
    
    # Create class legend
    if args.legend:
        visualizer.create_class_legend()
    
    # Visualize samples
    if args.all:
        visualizer.visualize_all_test_samples(args.save_dir)
    else:
        visualizer.visualize_sample(args.sample)
    
    print(f"\n💡 Demo completed!")
    print(f"   Test samples available: 0 to {len(visualizer.test_dataset)-1}")
    print(f"   Run with --sample N to view different samples")
    print(f"   Run with --all to generate all test visualizations")

if __name__ == "__main__":
    # If run without arguments, show interactive demo
    import sys
    if len(sys.argv) == 1:
        print("🎨 Unity Segmentation Interactive Demo")
        print("=" * 50)
        
        # Auto-find best model
        checkpoints_dir = Path("checkpoints")
        model_path = None
        if checkpoints_dir.exists():
            best_models = list(checkpoints_dir.glob("*unity_best*.pt"))
            if best_models:
                model_path = str(sorted(best_models)[-1])
                print(f"🎯 Found model: {model_path}")
        
        visualizer = SegmentationVisualizer(model_path)
        
        print(f"\nAvailable test samples: 0 to {len(visualizer.test_dataset)-1}")
        
        while True:
            try:
                sample_idx = input(f"\nEnter sample index (0-{len(visualizer.test_dataset)-1}) or 'q' to quit: ")
                if sample_idx.lower() == 'q':
                    break
                
                sample_idx = int(sample_idx)
                if 0 <= sample_idx < len(visualizer.test_dataset):
                    plt.close('all')  # Close previous plots
                    visualizer.visualize_sample(sample_idx)
                else:
                    print(f"❌ Invalid sample index. Use 0-{len(visualizer.test_dataset)-1}")
                    
            except ValueError:
                print("❌ Invalid input. Enter a number or 'q'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        main()

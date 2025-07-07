#!/usr/bin/env python3
"""
Universal Segmentation Model Demo
Test and visualize trained segmentation models (Stable or Fast) on Unity test data.
Automatically detects model architecture and loads appropriate configuration.
Interactive interface with performance analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import json
from pathlib import Path
import argparse
import random
import time
import segmentation_models_pytorch as smp

# Import Unity modules
from unity_dataset import UnitySegmentationDataset

class StableSegmentationDemo:
    """Interactive demo for testing segmentation models (both Stable and Fast)."""
    
    def __init__(self, model_path: str = 'checkpoints_stable/best_model.pth', data_dir: str = '../data'):
        """Initialize the demo with any segmentation model."""
        self.model_path = model_path
        self.data_dir = data_dir
        
        print(f"🚗 Loading Universal Segmentation Demo")
        print(f"   Model: {model_path}")
        print(f"   Data: {data_dir}")
        
        # Device setup
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                  'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")
        
        # Load model (automatically detects architecture)
        self.model = self._load_model()
        
        # Load test dataset
        self.test_dataset = UnitySegmentationDataset(data_dir, split='test')
        print(f"✅ Loaded {len(self.test_dataset)} test samples")
        
        # Demo state
        self.current_idx = 0
        self.fig = None
        self.axes = None
        
        # Constants for our stable model
        self.NUM_CLASSES = 12
        self.class_names = [
            'Void', 'Building', 'Fence', 'Other', 'Pedestrian', 'Pole',
            'RoadLine', 'Road', 'SideWalk', 'Vegetation', 'Vehicles', 'Wall'
        ]
        
        # Color mapping for visualization
        self.colors = self._create_color_map()
        
    def _load_model(self):
        """Load model automatically detecting architecture (MobileNetV2 or EfficientNet-B0)"""
        # Load checkpoint first to check metadata (fix for PyTorch 2.6)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Try to detect model type from path or checkpoint info
        is_fast_model = 'fast' in self.model_path or 'efficientnet' in self.model_path.lower()
        
        if is_fast_model:
            print("🚀 Detected Fast Model (EfficientNet-B0)")
            # Fast model architecture - EfficientNet-B0 with reduced decoder
            model = smp.Unet(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                in_channels=3,
                classes=12,
                activation=None,
                decoder_channels=[128, 64, 32, 16, 8]  # Reduced decoder channels
            )
        else:
            print("🏗️ Detected Stable Model (MobileNetV2)")
            # Stable model architecture - MobileNetV2 with default decoder
            model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights="imagenet", 
                in_channels=3,
                classes=12,
                activation=None
            )
        
        # Load model weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded successfully from epoch {checkpoint['epoch']}")
        except Exception as e:
            print(f"⚠️ Error loading model state: {e}")
            print("   Trying to load with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Model loaded with relaxed matching")
        
        model = model.to(self.device)
        model.eval()
        
        print(f"   Best validation loss: {checkpoint['val_loss']:.4f}")
        
        # Print performance info if available
        if 'estimated_fps' in checkpoint:
            print(f"   Estimated FPS: {checkpoint['estimated_fps']:.1f}")
        
        return model
    
    def _predict_image(self, image):
        """Make prediction with the model"""
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            output = self.model(image)
            prediction = torch.argmax(output, dim=1)
            
            return prediction.cpu().numpy()[0]
    
    def measure_inference_speed(self, num_samples=10):
        """Measure real-time inference speed for driving assessment"""
        print(f"\n🏁 Measuring Real-Time Performance ({num_samples} samples)")
        print("=" * 50)
        
        inference_times = []
        
        # Warm up the model
        print("🔥 Warming up model...")
        for _ in range(3):
            sample_idx = random.randint(0, len(self.test_dataset) - 1)
            image, _ = self.test_dataset[sample_idx]
            _ = self._predict_image(image)
        
        print("⏱️ Measuring inference times...")
        
        for i in range(num_samples):
            sample_idx = random.randint(0, len(self.test_dataset) - 1)
            image, _ = self.test_dataset[sample_idx]
            
            # Measure pure inference time
            start_time = time.time()
            _ = self._predict_image(image)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            
            if i % 5 == 0:
                current_fps = 1 / inference_time
                print(f"   Sample {i+1}: {inference_time*1000:.1f}ms ({current_fps:.1f} FPS)")
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)
        
        avg_fps = 1 / avg_time
        min_fps = 1 / max_time
        max_fps = 1 / min_time
        
        print(f"\n📊 Performance Results:")
        print(f"   Average: {avg_time*1000:.1f}ms ({avg_fps:.1f} FPS)")
        print(f"   Best:    {min_time*1000:.1f}ms ({max_fps:.1f} FPS)")
        print(f"   Worst:   {max_time*1000:.1f}ms ({min_fps:.1f} FPS)")
        print(f"   Std Dev: {std_time*1000:.1f}ms")
        
        print(f"\n🚗 Real-Time Driving Assessment:")
        if avg_fps >= 30:
            print(f"   ✅ EXCELLENT for real-time driving ({avg_fps:.1f} FPS >= 30 FPS)")
            print(f"   ✅ Smooth and responsive for autonomous navigation")
        elif avg_fps >= 20:
            print(f"   ⚠️  GOOD for driving ({avg_fps:.1f} FPS >= 20 FPS)")
            print(f"   ⚠️  Adequate but may feel slightly delayed")
        elif avg_fps >= 15:
            print(f"   ⚠️  MARGINAL for driving ({avg_fps:.1f} FPS >= 15 FPS)")
            print(f"   ⚠️  Noticeable delay, not ideal for high-speed driving")
        else:
            print(f"   ❌ TOO SLOW for real-time driving ({avg_fps:.1f} FPS < 15 FPS)")
            print(f"   ❌ Significant delay, unsafe for autonomous navigation")
        
        return avg_fps, inference_times
    
    def _create_color_map(self):
        """Create color map for visualization."""
        colors = np.zeros((self.NUM_CLASSES, 3), dtype=np.uint8)
        
        # Unity class colors
        class_colors = [
            [0, 0, 0],       # Void - Black
            [128, 64, 128],  # Building - Purple
            [244, 35, 232],  # Fence - Pink
            [70, 70, 70],    # Other - Gray
            [220, 20, 60],   # Pedestrian - Red
            [153, 153, 153], # Pole - Light Gray
            [157, 234, 50],  # RoadLine - Yellow-Green
            [128, 64, 255],  # Road - Blue-Purple
            [232, 35, 244],  # SideWalk - Magenta
            [35, 142, 107],  # Vegetation - Green
            [142, 0, 0],     # Vehicles - Dark Red
            [156, 102, 102]  # Wall - Brown
        ]
        
        for i, color in enumerate(class_colors):
            if i < self.NUM_CLASSES:
                colors[i] = color
        
        return colors
    
    def _apply_colormap(self, mask: np.ndarray) -> np.ndarray:
        """Apply color mapping to segmentation mask."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(self.NUM_CLASSES):
            mask_class = (mask == class_id)
            colored[mask_class] = self.colors[class_id]
        
        return colored
    
    def _calculate_metrics(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
        """Calculate metrics for the current prediction."""
        # Overall pixel accuracy
        correct = (pred_mask == true_mask).sum()
        total = true_mask.size
        pixel_acc = correct / total
        
        # Per-class IoU
        class_ious = []
        
        for class_id in range(self.NUM_CLASSES):
            tp = ((pred_mask == class_id) & (true_mask == class_id)).sum()
            fp = ((pred_mask == class_id) & (true_mask != class_id)).sum()
            fn = ((pred_mask != class_id) & (true_mask == class_id)).sum()
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            
            class_ious.append(iou)
        
        mean_iou = np.mean(class_ious)
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'class_ious': class_ious,
            'class_names': self.class_names
        }
    
    def _create_error_map(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
        """Create error visualization map."""
        error_map = np.zeros_like(true_mask)
        
        # Different error types
        correct = (pred_mask == true_mask)
        error_map[correct] = 0  # Correct - black
        error_map[~correct] = 1  # Error - white
        
        return error_map
    
    def show_sample(self, idx: int):
        """Show a specific test sample."""
        if idx >= len(self.test_dataset):
            idx = 0
        elif idx < 0:
            idx = len(self.test_dataset) - 1
        
        self.current_idx = idx
        
        # Get sample from our dataset
        image, true_mask = self.test_dataset[idx]
        
        # Time the inference
        start_time = time.time()
        pred_mask = self._predict_image(image)
        inference_time = time.time() - start_time
        
        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        true_mask_np = true_mask.numpy()
        
        # Calculate metrics
        metrics = self._calculate_metrics(pred_mask, true_mask_np)
        
        # Create visualizations
        pred_colored = self._apply_colormap(pred_mask)
        true_colored = self._apply_colormap(true_mask_np)
        error_map = self._create_error_map(pred_mask, true_mask_np)
        
        # Create overlay (50% image + 50% prediction)
        overlay = (image_np * 0.6 + pred_colored.astype(float) / 255 * 0.4)
        overlay = np.clip(overlay, 0, 1)
        
        # Clear and setup plot
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle(f'Stable Model Demo - Sample {idx+1}/{len(self.test_dataset)} | FPS: {1/inference_time:.1f}', fontsize=16)
        
        # Original image
        self.axes[0, 0].imshow(image_np)
        self.axes[0, 0].set_title('Original Image')
        self.axes[0, 0].axis('off')
        
        # Ground truth
        self.axes[0, 1].imshow(true_colored)
        self.axes[0, 1].set_title('Ground Truth')
        self.axes[0, 1].axis('off')
        
        # Prediction
        self.axes[0, 2].imshow(pred_colored)
        self.axes[0, 2].set_title(f'Prediction (IoU: {metrics["mean_iou"]:.3f})')
        self.axes[0, 2].axis('off')
        
        # Overlay
        self.axes[1, 0].imshow(overlay)
        self.axes[1, 0].set_title('Prediction Overlay')
        self.axes[1, 0].axis('off')
        
        # Error map
        self.axes[1, 1].imshow(error_map, cmap='Reds', alpha=0.7)
        self.axes[1, 1].imshow(image_np, alpha=0.3)
        self.axes[1, 1].set_title(f'Errors (Acc: {metrics["pixel_accuracy"]:.3f})')
        self.axes[1, 1].axis('off')
        
        # Class performance
        self.axes[1, 2].clear()
        class_names_short = [name[:8] for name in metrics['class_names']]
        bars = self.axes[1, 2].bar(range(len(metrics['class_ious'])), metrics['class_ious'])
        self.axes[1, 2].set_title('Per-Class IoU')
        self.axes[1, 2].set_xticks(range(len(metrics['class_ious'])))
        self.axes[1, 2].set_xticklabels(class_names_short, rotation=45, ha='right')
        self.axes[1, 2].set_ylim(0, 1)
        self.axes[1, 2].grid(True, alpha=0.3)
        
        # Color bars based on performance
        for bar, iou in zip(bars, metrics['class_ious']):
            if iou > 0.7:
                bar.set_color('green')
            elif iou > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        # Add navigation buttons
        self._add_navigation_buttons()
        
        # Print metrics to console
        print(f"\n📊 Sample {idx+1} Metrics:")
        print(f"   Overall IoU: {metrics['mean_iou']:.3f}")
        print(f"   Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
        print(f"   Inference Time: {inference_time*1000:.1f}ms ({1/inference_time:.1f} FPS)")
        print(f"   Best classes: {', '.join([metrics['class_names'][i] for i in np.argsort(metrics['class_ious'])[-3:]])}")
        print(f"   Worst classes: {', '.join([metrics['class_names'][i] for i in np.argsort(metrics['class_ious'])[:3]])}")
        
        return metrics
    
    def _add_navigation_buttons(self):
        """Add navigation buttons to the plot."""
        # Button axes
        ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
        ax_next = plt.axes([0.25, 0.01, 0.1, 0.05])
        ax_random = plt.axes([0.4, 0.01, 0.1, 0.05])
        ax_worst = plt.axes([0.55, 0.01, 0.15, 0.05])
        ax_best = plt.axes([0.75, 0.01, 0.15, 0.05])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_random = Button(ax_random, 'Random')
        self.btn_worst = Button(ax_worst, 'Find Worst')
        self.btn_best = Button(ax_best, 'Find Best')
        
        # Connect button events
        self.btn_prev.on_clicked(lambda x: self.show_sample(self.current_idx - 1))
        self.btn_next.on_clicked(lambda x: self.show_sample(self.current_idx + 1))
        self.btn_random.on_clicked(lambda x: self.show_sample(random.randint(0, len(self.test_samples)-1)))
        self.btn_worst.on_clicked(lambda x: self.find_worst_sample())
        self.btn_best.on_clicked(lambda x: self.find_best_sample())
    
    def find_worst_sample(self):
        """Find and show the worst performing sample."""
        print("🔍 Finding worst performing sample...")
        worst_iou = float('inf')
        worst_idx = 0
        
        for idx in range(min(20, len(self.test_dataset))):  # Check first 20 samples
            image, true_mask = self.test_dataset[idx]
            
            # Quick prediction
            pred_mask = self._predict_image(image)
            true_mask_np = true_mask.numpy()
            
            metrics = self._calculate_metrics(pred_mask, true_mask_np)
            
            if metrics['mean_iou'] < worst_iou:
                worst_iou = metrics['mean_iou']
                worst_idx = idx
        
        print(f"📉 Worst sample: {worst_idx+1} (IoU: {worst_iou:.3f})")
        self.show_sample(worst_idx)
    
    def find_best_sample(self):
        """Find and show the best performing sample."""
        print("🔍 Finding best performing sample...")
        best_iou = 0.0
        best_idx = 0
        
        for idx in range(min(20, len(self.test_dataset))):  # Check first 20 samples
            image, true_mask = self.test_dataset[idx]
            
            # Quick prediction
            pred_mask = self._predict_image(image)
            true_mask_np = true_mask.numpy()
            
            metrics = self._calculate_metrics(pred_mask, true_mask_np)
            
            if metrics['mean_iou'] > best_iou:
                best_iou = metrics['mean_iou']
                best_idx = idx
        
        print(f"📈 Best sample: {best_idx+1} (IoU: {best_iou:.3f})")
        self.show_sample(best_idx)
    
    def run_batch_analysis(self, num_samples: int = None):
        """Run analysis on multiple samples."""
        if num_samples is None:
            num_samples = len(self.test_dataset)
        
        num_samples = min(num_samples, len(self.test_dataset))
        
        print(f"📊 Running batch analysis on {num_samples} samples...")
        
        all_metrics = []
        class_ious_sum = np.zeros(self.NUM_CLASSES)
        total_time = 0
        
        for idx in range(num_samples):
            image, true_mask = self.test_dataset[idx]
            
            # Time prediction
            start_time = time.time()
            pred_mask = self._predict_image(image)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            true_mask_np = true_mask.numpy()
            
            metrics = self._calculate_metrics(pred_mask, true_mask_np)
            all_metrics.append(metrics)
            class_ious_sum += np.array(metrics['class_ious'])
            
            if (idx + 1) % 5 == 0:
                print(f"   Processed {idx + 1}/{num_samples} samples...")
        
        # Calculate summary statistics
        mean_iou = np.mean([m['mean_iou'] for m in all_metrics])
        mean_acc = np.mean([m['pixel_accuracy'] for m in all_metrics])
        class_ious_avg = class_ious_sum / num_samples
        avg_fps = num_samples / total_time
        
        print(f"\n📈 Batch Analysis Results:")
        print(f"   Samples analyzed: {num_samples}")
        print(f"   Average IoU: {mean_iou:.3f}")
        print(f"   Average Accuracy: {mean_acc:.3f}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Real-time capable: {'✅ YES' if avg_fps >= 30 else '❌ NO'}")
        print(f"\n   Per-class IoU:")
        
        for i, avg_iou in enumerate(class_ious_avg):
            class_name = self.class_names[i]
            print(f"     {class_name:15s}: {avg_iou:.3f}")
        
        return {
            'mean_iou': mean_iou,
            'mean_accuracy': mean_acc,
            'class_ious': class_ious_avg,
            'avg_fps': avg_fps,
            'all_metrics': all_metrics
        }
    
    def start_interactive_demo(self):
        """Start the interactive demo."""
        print(f"\n🎮 Starting Interactive Segmentation Demo")
        print(f"   Use the buttons to navigate:")
        print(f"   - Previous/Next: Navigate through samples")
        print(f"   - Random: Jump to random sample")
        print(f"   - Find Best/Worst: Find best/worst performing samples")
        print(f"   - Close the window to exit")
        
        # Show first sample
        self.show_sample(0)
        plt.show()

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Universal Segmentation Model Demo')
    parser.add_argument('--model', type=str, default='checkpoints_stable/best_model.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--batch_analysis', action='store_true', help='Run batch analysis')
    parser.add_argument('--num_samples', type=int, help='Number of samples for batch analysis')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to show (if not interactive)')
    parser.add_argument('--performance_test', action='store_true', help='Run performance/speed test')
    parser.add_argument('--speed_samples', type=int, default=20, help='Number of samples for speed test')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("   Available models:")
        
        # Check both stable and fast checkpoints
        for checkpoint_dir in ["checkpoints_stable", "checkpoints_fast"]:
            checkpoint_path = Path(checkpoint_dir)
            if checkpoint_path.exists():
                print(f"   {checkpoint_dir}/:")
                for model_file in checkpoint_path.glob("*.pth"):
                    print(f"     - {model_file}")
        return
    
    try:
        # Create demo
        demo = StableSegmentationDemo(args.model, args.data_dir)
        
        # Run performance test if requested
        if args.performance_test:
            avg_fps, _ = demo.measure_inference_speed(args.speed_samples)
            return avg_fps
        
        if args.batch_analysis:
            # Run batch analysis
            results = demo.run_batch_analysis(args.num_samples)
        else:
            # Start interactive demo
            demo.start_interactive_demo()
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

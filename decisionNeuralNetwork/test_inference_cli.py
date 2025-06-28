"""
Command-line interface for testing driving decision model inference with visualization.
This provides a flexible command-line tool for batch testing and single image inference.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import time

from decision_model import create_model
from decision_dataset import LABEL_MAPPING


class InferenceTester:
    """Command-line inference tester with visualization."""
    
    def __init__(self, model_path: str, model_type: str = 'resnet'):
        """
        Initialize the inference tester.
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model ('resnet' or 'simple')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        
        # Create transform (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Colors for visualization
        self.decision_colors = {
            'front': '#4CAF50',  # Green
            'left': '#2196F3',   # Blue
            'right': '#FF9800'   # Orange
        }
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_model(model_type, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Print model info
        val_acc = checkpoint.get('val_accuracy', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        print(f"Model loaded successfully!")
        print(f"Epoch: {epoch}, Validation Accuracy: {val_acc}")
        
        return model
    
    def predict_single_image(self, image_path: str, show_visualization: bool = True, 
                           save_visualization: str = None) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the segmentation mask image
            show_visualization: Whether to display the visualization
            save_visualization: Path to save the visualization image
            
        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"\n🔍 Testing image: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        start_time = time.time()
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Apply transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        inference_time = time.time() - start_time
        
        # Get prediction name
        prediction_name = LABEL_MAPPING[predicted_class]
        
        # Prepare results
        results = {
            'image_path': image_path,
            'prediction': prediction_name,
            'confidence': confidence,
            'probabilities': {
                label: float(prob) for label, prob in 
                zip(LABEL_MAPPING.values(), probabilities[0].cpu().numpy())
            },
            'inference_time': inference_time
        }
        
        # Print results
        print(f"🚗 Prediction: {prediction_name.upper()}")
        print(f"📊 Confidence: {confidence*100:.1f}%")
        print(f"⏱️  Inference time: {inference_time*1000:.1f}ms")
        print(f"📈 All probabilities:")
        for label, prob in results['probabilities'].items():
            print(f"   {label}: {prob*100:.1f}%")
        
        # Create visualization
        if show_visualization or save_visualization:
            self._create_visualization(original_image, probabilities[0].cpu().numpy(), 
                                     prediction_name, image_path, show_visualization, 
                                     save_visualization)
        
        return results
    
    def _create_visualization(self, image: np.ndarray, probabilities: np.ndarray, 
                            prediction: str, image_path: str, show: bool = True, 
                            save_path: str = None):
        """Create and display/save visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'RoadNet Inference: {os.path.basename(image_path)}', fontsize=14, fontweight='bold')
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Input Segmentation Mask')
        ax1.axis('off')
        
        # Add prediction overlay
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=self.decision_colors[prediction], alpha=0.9)
        ax1.text(0.02, 0.98, f'🚗 {prediction.upper()}', 
                transform=ax1.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', bbox=bbox_props, color='white')
        
        # Confidence badge
        confidence = probabilities[LABEL_MAPPING[prediction]]
        confidence_text = f'{confidence*100:.1f}%'
        confidence_bbox = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9)
        ax1.text(0.02, 0.02, confidence_text, 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='bottom', bbox=confidence_bbox, color='black')
        
        # Plot confidence bar chart
        labels = list(LABEL_MAPPING.values())
        colors = [self.decision_colors[label] for label in labels]
        bars = ax2.bar(labels, probabilities * 100, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
        
        # Highlight predicted class
        predicted_idx = labels.index(prediction)
        bars[predicted_idx].set_alpha(1.0)
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(3)
        
        ax2.set_title('Prediction Confidence')
        ax2.set_ylabel('Confidence (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Style improvements
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def test_directory(self, data_dir: str, output_file: str = None, 
                      save_visualizations: bool = False) -> List[Dict]:
        """
        Test all segmentation masks in a directory.
        
        Args:
            data_dir: Directory containing segmentation masks
            output_file: Optional file to save results JSON
            save_visualizations: Whether to save visualization images
            
        Returns:
            List of prediction results
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find segmentation mask files
        mask_files = [f for f in os.listdir(data_dir) 
                     if f.endswith('.camera.semantic segmentation.png')]
        
        if not mask_files:
            print(f"No segmentation mask files found in {data_dir}")
            return []
        
        print(f"\n🎯 Testing {len(mask_files)} images in {data_dir}")
        
        results = []
        viz_dir = None
        
        if save_visualizations:
            viz_dir = os.path.join(data_dir, 'inference_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            print(f"📁 Saving visualizations to: {viz_dir}")
        
        for i, filename in enumerate(sorted(mask_files), 1):
            image_path = os.path.join(data_dir, filename)
            print(f"\n[{i}/{len(mask_files)}] Processing: {filename}")
            
            try:
                # Create save path for visualization
                viz_save_path = None
                if save_visualizations:
                    viz_name = filename.replace('.png', '_inference.png')
                    viz_save_path = os.path.join(viz_dir, viz_name)
                
                # Run inference
                result = self.predict_single_image(
                    image_path, 
                    show_visualization=False,  # Don't show individual plots
                    save_visualization=viz_save_path
                )
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        
        # Calculate summary statistics
        self._print_summary(results)
        
        # Save results to JSON if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        if not results:
            return
        
        print(f"\n📊 SUMMARY STATISTICS")
        print("=" * 50)
        
        # Count predictions
        prediction_counts = {}
        confidences = []
        inference_times = []
        
        for result in results:
            pred = result['prediction']
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            confidences.append(result['confidence'])
            inference_times.append(result['inference_time'])
        
        # Print prediction distribution
        total = len(results)
        print(f"Total images tested: {total}")
        print(f"Prediction distribution:")
        for decision in ['front', 'left', 'right']:
            count = prediction_counts.get(decision, 0)
            percentage = (count / total) * 100
            print(f"  {decision.upper()}: {count} ({percentage:.1f}%)")
        
        # Print performance metrics
        avg_confidence = np.mean(confidences)
        avg_inference_time = np.mean(inference_times)
        print(f"\nPerformance metrics:")
        print(f"  Average confidence: {avg_confidence*100:.1f}%")
        print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"  Total processing time: {sum(inference_times):.1f}s")
    
    def compare_with_labels(self, data_dir: str, labels_file: str) -> Dict:
        """
        Compare predictions with ground truth labels.
        
        Args:
            data_dir: Directory containing images
            labels_file: Path to labels JSON file
            
        Returns:
            Comparison results
        """
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Load ground truth labels
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        print(f"\n🎯 Comparing predictions with ground truth...")
        
        correct = 0
        total = 0
        confusion_matrix = np.zeros((3, 3), dtype=int)
        label_names = ['front', 'left', 'right']
        
        for step_str, true_label in labels_data.items():
            # Find corresponding image
            image_filename = f"step{step_str}.camera.semantic segmentation.png"
            image_path = os.path.join(data_dir, image_filename)
            
            if not os.path.exists(image_path):
                continue
            
            # Run inference
            result = self.predict_single_image(image_path, show_visualization=False)
            predicted_label = result['prediction']
            
            # Convert to indices
            true_idx = ['front', 'left', 'right'].index(label_names[true_label])
            pred_idx = ['front', 'left', 'right'].index(predicted_label)
            
            # Update metrics
            if true_idx == pred_idx:
                correct += 1
            total += 1
            confusion_matrix[true_idx][pred_idx] += 1
            
            print(f"Step {step_str}: True={label_names[true_label]}, Pred={predicted_label}, "
                  f"{'✓' if true_idx == pred_idx else '✗'}")
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 VALIDATION RESULTS")
        print("=" * 40)
        print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"\nConfusion Matrix:")
        print("     ", "  ".join(f"{name:>6}" for name in label_names))
        for i, true_name in enumerate(label_names):
            row = "  ".join(f"{confusion_matrix[i][j]:>6}" for j in range(3))
            print(f"{true_name:>6} {row}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion_matrix': confusion_matrix.tolist()
        }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Test driving decision model inference')
    parser.add_argument('model_path', help='Path to trained model (.pth file)')
    parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                       help='Type of model')
    parser.add_argument('--image', help='Path to single image to test')
    parser.add_argument('--data_dir', help='Directory containing images to test')
    parser.add_argument('--labels_file', help='Labels file for validation (optional)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--save_viz', action='store_true', 
                       help='Save visualization images')
    parser.add_argument('--no_show', action='store_true',
                       help='Don\'t show visualizations (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = InferenceTester(args.model_path, args.model_type)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test single image
    if args.image:
        try:
            tester.predict_single_image(
                args.image, 
                show_visualization=not args.no_show,
                save_visualization='inference_result.png' if args.save_viz else None
            )
        except Exception as e:
            print(f"❌ Error testing image: {e}")
    
    # Test directory
    elif args.data_dir:
        try:
            results = tester.test_directory(
                args.data_dir,
                output_file=args.output,
                save_visualizations=args.save_viz
            )
            
            # Compare with labels if provided
            if args.labels_file:
                labels_path = args.labels_file
                if not os.path.isabs(labels_path):
                    labels_path = os.path.join(args.data_dir, labels_path)
                
                try:
                    tester.compare_with_labels(args.data_dir, labels_path)
                except Exception as e:
                    print(f"⚠️  Warning: Could not compare with labels: {e}")
        
        except Exception as e:
            print(f"❌ Error testing directory: {e}")
    
    else:
        print("❌ Please specify either --image or --data_dir")
        parser.print_help()


if __name__ == "__main__":
    main()

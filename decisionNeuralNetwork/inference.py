"""
Inference script for making predictions with the trained driving decision model.
"""

import os
import argparse
import json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from decision_model import create_model
from decision_dataset import LABEL_MAPPING


class DecisionPredictor:
    """
    Class for making driving decision predictions from segmentation masks.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'resnet',
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('resnet' or 'simple')
            device: Device to run inference on
            image_size: Input image size
        """
        self.model_type = model_type
        self.image_size = image_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"DecisionPredictor initialized with {model_type} model on {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Create model architecture
        model = create_model(self.model_type, num_classes=3)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        if 'val_accuracy' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_accuracy']:.2f}%")
        
        return model
    
    def predict_image(self, image_path: str) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction for a single image.
        
        Args:
            image_path: Path to the segmentation mask image
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            probabilities = probabilities.cpu().numpy().flatten()
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, float, np.ndarray]]:
        """
        Make predictions for a batch of images.
        
        Args:
            image_paths: List of paths to segmentation mask images
            
        Returns:
            List of tuples (predicted_class, confidence, class_probabilities)
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((0, 0.0, np.array([1.0, 0.0, 0.0])))  # Default to 'front'
        
        return results
    
    def predict_directory(
        self,
        data_dir: str,
        output_file: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Make predictions for all segmentation masks in a directory.
        
        Args:
            data_dir: Directory containing segmentation masks
            output_file: Optional file to save predictions
            confidence_threshold: Minimum confidence threshold for predictions
            
        Returns:
            Dictionary mapping step numbers to prediction results
        """
        # Find all segmentation mask files
        segmentation_files = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.camera.semantic segmentation.png'):
                segmentation_files.append(filename)
        
        segmentation_files = sorted(segmentation_files, 
                                  key=lambda x: int(x.split('.')[0].replace('step', '')))
        
        print(f"Found {len(segmentation_files)} segmentation masks")
        
        # Make predictions
        results = {}
        for filename in segmentation_files:
            # Extract step number
            step_number = int(filename.split('.')[0].replace('step', ''))
            
            # Make prediction
            image_path = os.path.join(data_dir, filename)
            predicted_class, confidence, probabilities = self.predict_image(image_path)
            
            # Store result
            results[step_number] = {
                'filename': filename,
                'predicted_class': predicted_class,
                'predicted_label': LABEL_MAPPING[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    'front': float(probabilities[0]),
                    'left': float(probabilities[1]),
                    'right': float(probabilities[2])
                },
                'high_confidence': confidence >= confidence_threshold
            }
            
            # Print progress
            if step_number % 50 == 0:
                print(f"Processed step {step_number}: {LABEL_MAPPING[predicted_class]} "
                      f"(confidence: {confidence:.3f})")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Predictions saved to {output_file}")
        
        # Print summary statistics
        self._print_prediction_summary(results, confidence_threshold)
        
        return results
    
    def _print_prediction_summary(self, results: Dict, confidence_threshold: float):
        """Print summary statistics of predictions."""
        total = len(results)
        if total == 0:
            return
        
        # Count predictions by class
        class_counts = {'front': 0, 'left': 0, 'right': 0}
        high_confidence_counts = {'front': 0, 'left': 0, 'right': 0}
        total_high_confidence = 0
        avg_confidence = 0.0
        
        for result in results.values():
            label = result['predicted_label']
            confidence = result['confidence']
            
            class_counts[label] += 1
            avg_confidence += confidence
            
            if result['high_confidence']:
                high_confidence_counts[label] += 1
                total_high_confidence += 1
        
        avg_confidence /= total
        
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total predictions: {total}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"High confidence (≥{confidence_threshold}): {total_high_confidence} ({total_high_confidence/total*100:.1f}%)")
        
        print("\nClass distribution:")
        for label, count in class_counts.items():
            percentage = count/total*100
            high_conf_count = high_confidence_counts[label]
            high_conf_percentage = high_conf_count/count*100 if count > 0 else 0
            print(f"  {label.upper()}: {count} ({percentage:.1f}%) - "
                  f"{high_conf_count} high confidence ({high_conf_percentage:.1f}%)")
        print("="*50)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained driving decision model')
    parser.add_argument('model_path', help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', help='Directory containing segmentation masks')
    parser.add_argument('--image_path', help='Path to single image for prediction')
    parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                       help='Type of model')
    parser.add_argument('--output_file', help='File to save predictions (JSON format)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for high-confidence predictions')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint {args.model_path} does not exist")
        return
    
    # Create predictor
    predictor = DecisionPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        image_size=(args.image_size, args.image_size)
    )
    
    if args.image_path:
        # Single image prediction
        if not os.path.exists(args.image_path):
            print(f"Error: Image {args.image_path} does not exist")
            return
        
        print(f"Making prediction for: {args.image_path}")
        predicted_class, confidence, probabilities = predictor.predict_image(args.image_path)
        
        print(f"\nPrediction: {LABEL_MAPPING[predicted_class].upper()}")
        print(f"Confidence: {confidence:.3f}")
        print("Class probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  {LABEL_MAPPING[i]}: {prob:.3f}")
    
    elif args.data_dir:
        # Directory prediction
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist")
            return
        
        print(f"Making predictions for all images in: {args.data_dir}")
        results = predictor.predict_directory(
            args.data_dir,
            args.output_file,
            args.confidence_threshold
        )
        
        print(f"Completed predictions for {len(results)} images")
    
    else:
        print("Error: Please specify either --image_path or --data_dir")


if __name__ == "__main__":
    main()

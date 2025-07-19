import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import io

# Add the train directory to the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from train import UNetResNet18


class DrivingSegmentationInference:
    """
    Inference class for driving segmentation model.
    
    Usage:
        predictor = DrivingSegmentationInference()
        colored_mask = predictor("path/to/image.jpg", save=True)
    """
    
    # Color mapping for visualization (same as training)
    CLASS_TO_COLOR = {
        0: (255, 255, 255),  # road - white
        1: (192, 183, 77),   # building - brown  
        2: (83, 21, 168),    # car - purple
        3: (255, 0, 0),      # traffic_light - red
        4: (255, 0, 121),    # road_block - pink
        255: (0, 0, 0),      # unknown/unlabeled - black
    }
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to model weights (.pth file). 
                       Default: '../checkpoints/driving_segmentation_model.pth'
            device: Device to run inference on. Auto-detects if None.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Set model path
        if model_path is None:
            model_path = '../checkpoints/driving_segmentation_model.pth'
        
        self.model_path = Path(model_path)
        
        # Initialize and load model
        self.model = self._load_model()
        
        print(f"Model loaded from: {self.model_path}")
    
    def _load_model(self):
        """Load the trained model"""
        # Create model (same as training)
        model = UNetResNet18(num_classes=5)

        # Load weights
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()

        return model
    
    def _preprocess_image(self, image):
        """Preprocess image for inference (same as training)"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to tensor and normalize (same as training)
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        # Normalize using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def _create_colored_mask(self, pred_mask):
        """Create colored visualization of segmentation"""
        h, w = pred_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.CLASS_TO_COLOR.items():
            mask = pred_mask == class_id
            colored[mask] = color
        
        return Image.fromarray(colored)
    
    def predict_bytes(self, image_bytes: bytes) -> bytes:
        """Run inference on image bytes and return PNG bytes of colored mask"""
        # Load image from bytes
        buf = io.BytesIO(image_bytes)
        image = Image.open(buf)
        
        # Preprocess
        input_tensor = self._preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy mask
        pred_mask = predictions.cpu().numpy()[0]
        
        # Create colored mask
        colored_mask = self._create_colored_mask(pred_mask)
        
        # Save to PNG bytes
        out_buf = io.BytesIO()
        colored_mask.save(out_buf, format='PNG')
        return out_buf.getvalue()

    def __call__(self, image_path, save=False):
        """
        Call the inference like a PyTorch module.
        
        Args:
            image_path: Path to image file
            save: Whether to save the result (default: False)
            
        Returns:
            PIL.Image: Colored segmentation mask
        """
        # Load image
        image = Image.open(image_path)
        
        # Preprocess
        input_tensor = self._preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy
        pred_mask = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Create colored visualization
        colored_mask = self._create_colored_mask(pred_mask)
        
        # Save if requested
        if save:
            image_name = Path(image_path).stem
            output_path = f"{image_name}_segmentation.png"
            colored_mask.save(output_path)
            print(f"Segmentation saved as: {output_path}")
        
        return colored_mask

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create predictor
    predictor = DrivingSegmentationInference()
    
    # Make prediction (save by default in command line)
    result = predictor(image_path, save=True)
    
    print("\n✅ Inference complete!")

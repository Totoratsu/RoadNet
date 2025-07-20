import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import io

class UNetResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()  # Initialize the parent class
        resnet = models.resnet18(pretrained=True)
        # Encoder layers
        self.input_block = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Decoder layers (upsample + skip connection)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x0 = self.input_block(x)      # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)         # (B, 64, H/4, W/4)
        x2 = self.encoder1(x1)        # (B, 64, H/4, W/4)
        x3 = self.encoder2(x2)        # (B, 128, H/8, W/8)
        x4 = self.encoder3(x3)        # (B, 256, H/16, W/16)
        x5 = self.encoder4(x4)        # (B, 512, H/32, W/32)

        # Decoder with skip connections (resize skip if needed)
        def match_size(src, target):
            if src.shape[2:] != target.shape[2:]:
                src = torch.nn.functional.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)
            return src

        d4 = self.up4(x5)             # (B, 256, H/16, W/16)
        x4m = match_size(x4, d4)
        d4 = torch.cat([d4, x4m], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)             # (B, 128, H/8, W/8)
        x3m = match_size(x3, d3)
        d3 = torch.cat([d3, x3m], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)             # (B, 64, H/4, W/4)
        x2m = match_size(x2, d2)
        d2 = torch.cat([d2, x2m], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)             # (B, 64, H/2, W/2)
        x0m = match_size(x0, d1)
        d1 = torch.cat([d1, x0m], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        # Upsample to input size
        out = torch.nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out
    
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
            model_path = '../checkpoints/driving_segmentation_fast.pth'
        
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
    
    def predict_bytes(self, image_bytes: bytes) -> tuple[bytes, bool, list]:
        """Run inference on image bytes and return PNG bytes of colored mask + red pixel detection + coordinates"""
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
        
        # Check for many red pixels
        many_red_pixels, traffic_light_coords = self._check_traffic_lights(pred_mask)
        
        # Create colored mask
        colored_mask = self._create_colored_mask(pred_mask)
        
        # Save to PNG bytes
        out_buf = io.BytesIO()
        colored_mask.save(out_buf, format='PNG')
        return out_buf.getvalue(), many_red_pixels, traffic_light_coords

    def _check_traffic_lights(self, pred_mask, threshold_percent=0.5):
        """
        Check if there are many red pixels (traffic lights - class 3)
        
        Args:
            pred_mask: Predicted segmentation mask
            threshold_percent: Percentage threshold for "many" red pixels (default: 0.5%)
            
        Returns:
            tuple: (bool, list) - (True if many traffic light pixels detected, list of coordinates)
        """
        total_pixels = pred_mask.size
        traffic_light_pixels = np.sum(pred_mask == 3)
        traffic_light_percentage = (traffic_light_pixels / total_pixels) * 100

        print(f"Traffic light pixel percentage: {traffic_light_percentage:.2f}%")
        
        # Get coordinates of traffic light pixels
        traffic_light_coords = []
        if traffic_light_percentage >= threshold_percent:
            # Find all traffic light pixel coordinates
            y_coords, x_coords = np.where(pred_mask == 3)
            
            # Return up to 2 coordinates (first two found)
            for i in range(min(2, len(y_coords))):
                traffic_light_coords.append((int(x_coords[i]), int(y_coords[i])))
        
        return traffic_light_percentage >= threshold_percent, traffic_light_coords

    def __call__(self, image_path, save=False):
        """
        Call the inference like a PyTorch module.
        
        Args:
            image_path: Path to image file
            save: Whether to save the result (default: False)
            
        Returns:
            dict: {
                'colored_mask': PIL.Image - Colored segmentation mask,
                'many_red_pixels': bool - True if many traffic light pixels detected,
                'traffic_light_coords': list - List of (x, y) coordinates of traffic light pixels
            }
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
        
        # Check for many red pixels (traffic lights)
        many_red_pixels, traffic_light_coords = self._check_traffic_lights(pred_mask)
        
        # Create colored visualization
        colored_mask = self._create_colored_mask(pred_mask)
        
        # Save if requested
        if save:
            image_name = Path(image_path).stem
            output_path = f"{image_name}_segmentation.png"
            colored_mask.save(output_path)
            print(f"Segmentation saved as: {output_path}")
            print(f"Many red pixels detected: {many_red_pixels}")
            if traffic_light_coords:
                print(f"Traffic light coordinates: {traffic_light_coords}")
        
        return {
            'colored_mask': colored_mask,
            'many_red_pixels': many_red_pixels,
            'traffic_light_coords': traffic_light_coords
        }

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
    
    print(f"\n✅ Inference complete!")
    print(f"Many red pixels detected: {result['many_red_pixels']}")
    if result['traffic_light_coords']:
        print(f"Traffic light coordinates: {result['traffic_light_coords']}")

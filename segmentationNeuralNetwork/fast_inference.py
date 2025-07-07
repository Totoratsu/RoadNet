# Lightweight Real-time Segmentation Inference
# Optimized for driving scenarios with minimal dependencies

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
from typing import Tuple, Optional

class FastSegmentationModel:
    """
    Ultra-fast segmentation model optimized for real-time driving inference.
    Minimal dependencies, maximum speed.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize fast segmentation model.
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu', 'cuda', 'mps', or 'auto'
        """
        self.device = self._select_device(device)
        self.model = self._load_optimized_model(model_path)
        self.transform = self._create_fast_transform()
        self.inference_times = []
        
        print(f"🚀 FastSegmentationModel initialized on {self.device}")
        
    def _select_device(self, device: str) -> torch.device:
        """Select best device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_optimized_model(self, model_path: str) -> nn.Module:
        """Load and optimize model for speed."""
        
        # Use lightweight MobileNetV2 encoder for speed
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=255,
            activation=None
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply TorchScript optimization
        try:
            dummy_input = torch.randn(1, 3, 256, 512).to(self.device)
            model = torch.jit.trace(model, dummy_input)
            print("✓ TorchScript optimization applied")
        except Exception as e:
            print(f"⚠ TorchScript failed: {e}")
        
        # Apply FP16 for CUDA
        if self.device.type == 'cuda':
            try:
                model = model.half()
                print("✓ FP16 optimization applied")
            except Exception as e:
                print(f"⚠ FP16 failed: {e}")
        
        return model
    
    def _create_fast_transform(self):
        """Create optimized preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Ultra-fast inference on single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Segmentation mask as numpy array
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Handle FP16
        if self.device.type == 'cuda' and hasattr(self.model, 'half'):
            input_tensor = input_tensor.half()
        
        # Fast inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=1)
            result = predictions.cpu().numpy().squeeze()
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return result
    
    def get_fps(self) -> float:
        """Get current FPS performance."""
        if not self.inference_times:
            return 0.0
        return 1.0 / np.mean(self.inference_times[-10:])  # Last 10 frames
    
    def benchmark(self, iterations: int = 100):
        """Quick benchmark test."""
        print(f"🔥 Benchmarking {iterations} iterations...")
        
        # Dummy image
        test_image = Image.new('RGB', (512, 256), color='red')
        
        # Warmup
        for _ in range(5):
            self.predict(test_image)
        
        # Clear times
        self.inference_times = []
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            self.predict(test_image)
        total_time = time.time() - start_time
        
        # Results
        avg_time = np.mean(self.inference_times) * 1000
        fps = 1.0 / np.mean(self.inference_times)
        
        print(f"📊 Results:")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Device: {self.device}")
        
        if fps >= 30:
            print("✅ Ready for real-time driving!")
        elif fps >= 15:
            print("⚠️  Suitable for assisted driving")
        else:
            print("❌ Too slow for real-time use")
        
        return fps

class DrivingSegmentationPipeline:
    """
    Complete pipeline optimized for driving scenarios.
    """
    
    def __init__(self, model_path: str):
        self.model = FastSegmentationModel(model_path)
        self.frame_count = 0
        
    def process_driving_frame(self, image: Image.Image) -> dict:
        """
        Process a single driving frame and return relevant information.
        
        Returns:
            Dictionary with segmentation and driving-relevant analysis
        """
        self.frame_count += 1
        
        # Get segmentation
        segmentation = self.model.predict(image)
        
        # Extract driving-relevant features
        analysis = self._analyze_driving_scene(segmentation)
        
        return {
            'segmentation': segmentation,
            'road_area': analysis['road_area'],
            'vehicle_count': analysis['vehicle_count'],
            'fps': self.model.get_fps(),
            'frame_number': self.frame_count
        }
    
    def _analyze_driving_scene(self, segmentation: np.ndarray) -> dict:
        """Analyze segmentation for driving-relevant information."""
        
        # Common Cityscapes class IDs for driving
        ROAD_CLASSES = [0, 1]  # road, sidewalk
        VEHICLE_CLASSES = [26, 27, 28]  # car, truck, bus
        
        # Calculate road area percentage
        road_pixels = np.isin(segmentation, ROAD_CLASSES).sum()
        total_pixels = segmentation.size
        road_area = (road_pixels / total_pixels) * 100
        
        # Count vehicles (approximate)
        vehicle_mask = np.isin(segmentation, VEHICLE_CLASSES)
        vehicle_count = len(np.unique(segmentation[vehicle_mask])) - 1  # Rough estimate
        
        return {
            'road_area': road_area,
            'vehicle_count': max(0, vehicle_count)
        }

def create_driving_ready_model(model_path: str) -> FastSegmentationModel:
    """Create a model ready for driving inference."""
    model = FastSegmentationModel(model_path)
    
    # Test performance
    fps = model.benchmark(iterations=50)
    
    if fps < 15:
        print("⚠️  Warning: Model may be too slow for safe driving use")
        print("Consider using a faster device or lighter model architecture")
    
    return model

def test_real_time_performance(model_path: str):
    """Test if model is suitable for real-time driving."""
    print("🚗 Testing real-time driving performance...")
    
    model = FastSegmentationModel(model_path)
    pipeline = DrivingSegmentationPipeline(model_path)
    
    # Create test driving scene
    test_image = Image.new('RGB', (1024, 512), color='gray')  # Road-like image
    
    print("\n📊 Performance Test Results:")
    
    # Single frame test
    start_time = time.time()
    result = pipeline.process_driving_frame(test_image)
    processing_time = time.time() - start_time
    
    print(f"  Single frame processing: {processing_time*1000:.2f} ms")
    print(f"  Road area detected: {result['road_area']:.1f}%")
    print(f"  Current FPS: {result['fps']:.1f}")
    
    # Continuous test
    print("\n🔄 Continuous processing test (30 frames)...")
    times = []
    for i in range(30):
        start = time.time()
        pipeline.process_driving_frame(test_image)
        times.append(time.time() - start)
    
    avg_fps = 1.0 / np.mean(times)
    print(f"  Average FPS: {avg_fps:.1f}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if avg_fps >= 30:
        print("  ✅ Excellent - Ready for high-speed driving")
    elif avg_fps >= 20:
        print("  ✅ Good - Suitable for city driving")
    elif avg_fps >= 15:
        print("  ⚠️  Acceptable - Use caution, consider assisted driving only")
    else:
        print("  ❌ Too slow - Not recommended for real-time driving")
        print("  Consider: Faster hardware, lighter model, or lower resolution")

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/trained_model.pt"
    
    # Quick test
    model = create_driving_ready_model(model_path)
    
    # Full performance test
    test_real_time_performance(model_path)

# Real-time Segmentation Inference Optimizations
# Multiple approaches for optimizing segmentation inference speed

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import warnings

class OptimizedSegmentationModel:
    """
    Optimized segmentation model for real-time inference while driving.
    Includes multiple optimization strategies for maximum speed.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', optimization_level: str = 'balanced'):
        """
        Initialize optimized segmentation model.
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu', 'cuda', 'mps', or 'auto' for automatic selection
            optimization_level: 'speed' (fastest), 'balanced', or 'quality' (best quality)
        """
        self.device = self._select_device(device)
        self.optimization_level = optimization_level
        
        # Load and optimize model
        self.model = self._load_and_optimize_model(model_path)
        self.preprocessing = self._setup_preprocessing()
        
        # Performance tracking
        self.inference_times = []
        
    def _select_device(self, device: str) -> torch.device:
        """Select the best available device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_and_optimize_model(self, model_path: str) -> nn.Module:
        """Load model and apply optimizations based on level."""
        
        # Import Unity model components
        try:
            from unity_unet import create_unity_model, NUM_CLASSES
            num_classes = NUM_CLASSES
            use_unity = True
            print(f"✓ Using Unity model configuration with {num_classes} classes")
        except ImportError:
            print("⚠️  Unity components not found, using Cityscapes defaults")
            num_classes = 255
            use_unity = False
        
        # Create model using Unity configuration if available
        if use_unity:
            try:
                model = create_unity_model(optimization=self.optimization_level)
                print(f"✓ Created Unity model with {self.optimization_level} optimization")
            except Exception as e:
                print(f"⚠️  Failed to create Unity model: {e}")
                use_unity = False
        
        # Fallback to standard segmentation model
        if not use_unity:
            # Choose model architecture based on optimization level
            if self.optimization_level == 'speed':
                encoder = "mobilenet_v2"
            elif self.optimization_level == 'balanced':
                encoder = "efficientnet-b0"
            else:  # quality
                encoder = "resnet34"
            
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        
        # Load weights
        try:
            # Load checkpoint to check model configuration
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Try to determine number of classes from checkpoint
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
            if hasattr(state_dict, 'items'):
                # Look for segmentation head to determine classes
                for key, value in state_dict.items():
                    if 'segmentation_head' in key and 'weight' in key and value.dim() >= 2:
                        checkpoint_classes = value.shape[0]
                        print(f"✓ Detected {checkpoint_classes} classes from checkpoint")
                        
                        # If checkpoint classes don't match our model, recreate model
                        if checkpoint_classes != num_classes:
                            print(f"⚠️  Model/checkpoint class mismatch: model={num_classes}, checkpoint={checkpoint_classes}")
                            print("🔄 Recreating model with checkpoint classes...")
                            
                            if use_unity and checkpoint_classes != NUM_CLASSES:
                                print("   Using fallback standard model...")
                                # Fallback to standard model with checkpoint classes
                                model = smp.Unet(
                                    encoder_name=encoder,
                                    encoder_weights="imagenet",
                                    in_channels=3,
                                    classes=checkpoint_classes,
                                    activation=None
                                )
                            elif not use_unity:
                                # Update standard model classes
                                model = smp.Unet(
                                    encoder_name=encoder,
                                    encoder_weights="imagenet",
                                    in_channels=3,
                                    classes=checkpoint_classes,
                                    activation=None
                                )
                        break
            
            # Load model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model weights from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                print(f"✓ Loaded model weights from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("Using randomly initialized weights for demo")
        
        model = model.to(self.device)
        model.eval()
        
        # Apply optimizations
        model = self._apply_model_optimizations(model)
        
        return model
    
    def _apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply various model optimizations for speed."""
        
        # 1. Torch JIT compilation
        if self.optimization_level in ['speed', 'balanced']:
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 3, 256, 512).to(self.device)
                model = torch.jit.trace(model, dummy_input)
                print("✓ Applied TorchScript optimization")
            except Exception as e:
                print(f"⚠ TorchScript optimization failed: {e}")
        
        # 2. Half precision (FP16) for GPU
        if self.device.type == 'cuda' and self.optimization_level == 'speed':
            try:
                model = model.half()
                print("✓ Applied FP16 optimization")
            except Exception as e:
                print(f"⚠ FP16 optimization failed: {e}")
        
        # 3. Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _setup_preprocessing(self) -> transforms.Compose:
        """Setup optimized preprocessing pipeline."""
        
        # Optimized preprocessing with minimal operations
        transform_list = []
        
        # Resize with optimized interpolation
        if self.optimization_level == 'speed':
            # Faster bilinear interpolation
            transform_list.append(transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BILINEAR))
        else:
            # Higher quality bicubic
            transform_list.append(transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def predict(self, image: Image.Image, return_confidence: bool = False) -> np.ndarray:
        """
        Perform fast inference on a single image.
        
        Args:
            image: PIL Image
            return_confidence: Whether to return confidence scores
            
        Returns:
            Segmentation mask as numpy array
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocessing(image).unsqueeze(0).to(self.device)
        
        # Handle FP16 if enabled
        if self.device.type == 'cuda' and hasattr(self.model, 'half'):
            input_tensor = input_tensor.half()
        
        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(input_tensor)
        
        # Post-process
        if return_confidence:
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            result = predictions.cpu().numpy().squeeze()
            conf_map = confidence.cpu().numpy().squeeze()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return result, conf_map
        else:
            predictions = torch.argmax(outputs, dim=1)
            result = predictions.cpu().numpy().squeeze()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return result
    
    def predict_batch(self, images: list) -> list:
        """
        Perform batch inference for multiple images.
        More efficient for processing multiple frames.
        """
        start_time = time.time()
        
        # Preprocess batch
        batch_tensor = torch.stack([
            self.preprocessing(img) for img in images
        ]).to(self.device)
        
        # Handle FP16 if enabled
        if self.device.type == 'cuda' and hasattr(self.model, 'half'):
            batch_tensor = batch_tensor.half()
        
        # Batch inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(batch_tensor)
        
        # Post-process
        predictions = torch.argmax(outputs, dim=1)
        results = predictions.cpu().numpy()
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return [results[i] for i in range(len(results))]
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {"message": "No inference performed yet"}
        
        times = np.array(self.inference_times)
        return {
            "avg_inference_time_ms": np.mean(times) * 1000,
            "min_inference_time_ms": np.min(times) * 1000,
            "max_inference_time_ms": np.max(times) * 1000,
            "fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
            "device": str(self.device),
            "optimization_level": self.optimization_level
        }
    
    def benchmark(self, image_size: Tuple[int, int] = (256, 512), num_iterations: int = 100):
        """
        Benchmark the model performance.
        
        Args:
            image_size: Size of test images (height, width)
            num_iterations: Number of benchmark iterations
        """
        print(f"🔥 Benchmarking model on {self.device} ({self.optimization_level} optimization)")
        print(f"Image size: {image_size}, Iterations: {num_iterations}")
        
        # Create dummy image
        dummy_image = Image.new('RGB', (image_size[1], image_size[0]), color='red')
        
        # Warmup
        for _ in range(5):
            _ = self.predict(dummy_image)
        
        # Clear previous times
        self.inference_times = []
        
        # Benchmark
        start_time = time.time()
        for i in range(num_iterations):
            _ = self.predict(dummy_image)
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_iterations}")
        
        total_time = time.time() - start_time
        
        # Results
        stats = self.get_performance_stats()
        print(f"\n📊 Benchmark Results:")
        print(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"FPS: {stats['fps']:.1f}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Device: {stats['device']}")
        
        # Real-time feasibility
        target_fps = 30  # Target for smooth driving
        if stats['fps'] >= target_fps:
            print(f"✅ Model is suitable for real-time driving (>{target_fps} FPS)")
        else:
            print(f"⚠️  Model may struggle with real-time driving (<{target_fps} FPS)")
            print("Consider using 'speed' optimization level or upgrading hardware")

class RealTimeSegmentationPipeline:
    """
    Complete pipeline for real-time segmentation inference.
    Includes frame buffering and asynchronous processing.
    """
    
    def __init__(self, model_path: str, buffer_size: int = 3):
        """
        Initialize real-time pipeline.
        
        Args:
            model_path: Path to trained model
            buffer_size: Number of frames to buffer for smooth processing
        """
        self.model = OptimizedSegmentationModel(model_path, optimization_level='speed')
        self.buffer_size = buffer_size
        self.frame_buffer = []
        
    def process_frame(self, frame: Image.Image) -> np.ndarray:
        """Process a single frame with buffering."""
        # Add to buffer
        self.frame_buffer.append(frame)
        
        # Keep buffer size
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Process latest frame
        return self.model.predict(frame)
    
    def process_video_stream(self, video_source, output_callback=None):
        """
        Process continuous video stream (camera/video file).
        
        Args:
            video_source: Video source (camera index or file path)
            output_callback: Function to call with each processed frame
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for video processing. Install with: pip install opencv-python")
        
        cap = cv2.VideoCapture(video_source)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and create PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Process frame
                segmentation = self.process_frame(pil_image)
                
                # Callback with results
                if output_callback:
                    output_callback(frame, segmentation)
                
                # Display performance
                stats = self.model.get_performance_stats()
                if len(self.model.inference_times) % 30 == 0:  # Every 30 frames
                    print(f"Real-time FPS: {stats['fps']:.1f}")
                
        finally:
            cap.release()

# Example usage functions
def create_speed_optimized_model(model_path: str):
    """Create a model optimized for maximum speed."""
    return OptimizedSegmentationModel(model_path, optimization_level='speed')

def create_balanced_model(model_path: str):
    """Create a model with balanced speed and quality."""
    return OptimizedSegmentationModel(model_path, optimization_level='balanced')

def benchmark_all_configurations(model_path: str):
    """Benchmark all optimization configurations."""
    configurations = ['speed', 'balanced', 'quality']
    
    print("🚀 Benchmarking all optimization configurations...\n")
    
    for config in configurations:
        print(f"{'='*50}")
        print(f"Testing {config.upper()} configuration")
        print(f"{'='*50}")
        
        model = OptimizedSegmentationModel(model_path, optimization_level=config)
        model.benchmark(num_iterations=50)
        print()

if __name__ == "__main__":
    print("🚗 Real-time Segmentation Inference Demo")
    print("=" * 50)
    
    # Try to find Unity model first, fallback to demo
    model_path = None
    
    # Look for Unity models
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        unity_models = list(checkpoints_dir.glob("*unity*.pt")) + list(checkpoints_dir.glob("*Unity*.pt"))
        if unity_models:
            model_path = str(unity_models[0])
            print(f"🎯 Found Unity model: {model_path}")
    
    # Fallback demo (no actual weights loaded)
    if not model_path:
        print("⚠️  No model weights found - running demo with random weights")
        model_path = "dummy_model.pt"  # Will fail to load, but that's OK for demo
    
    # Create optimized model
    model = create_speed_optimized_model(model_path)
    
    # Benchmark
    model.benchmark()
    
    # Test on single image
    test_image = Image.new('RGB', (512, 256), color='blue')
    result = model.predict(test_image)
    print(f"Segmentation shape: {result.shape}")

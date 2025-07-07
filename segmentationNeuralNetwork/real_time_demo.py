# Real-time Segmentation Demo for Driving
# Practical demonstration of optimized segmentation inference

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
import threading
import queue
from pathlib import Path

class RealTimeDrivingDemo:
    """
    Real-time segmentation demo optimized for driving scenarios.
    Shows FPS, processing time, and driving-relevant analysis.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize real-time demo.
        
        Args:
            model_path: Path to trained segmentation model
            device: Computing device ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._select_device(device)
        self.model = self._load_fast_model(model_path)
        self.transform = self._create_transform()
        
        # Performance tracking
        self.frame_times = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Cityscapes class colors for visualization
        self.class_colors = self._get_class_colors()
        
        print(f"🚗 Real-time driving demo initialized on {self.device}")
        self._benchmark_performance()
    
    def _select_device(self, device: str) -> torch.device:
        """Select best available device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_fast_model(self, model_path: str) -> nn.Module:
        """Load and optimize model for maximum speed."""
        
        # Use MobileNetV2 for speed (adjust if your model uses different encoder)
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=255,
            activation=None
        )
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"⚠️  Could not load model weights: {e}")
            print("Using randomly initialized model for demo purposes")
        
        model = model.to(self.device)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply optimizations
        try:
            # TorchScript optimization
            dummy_input = torch.randn(1, 3, 256, 512).to(self.device)
            model = torch.jit.trace(model, dummy_input)
            print("✓ TorchScript optimization applied")
        except:
            print("⚠️  TorchScript optimization failed, using regular model")
        
        # FP16 for CUDA
        if self.device.type == 'cuda':
            try:
                model = model.half()
                print("✓ FP16 optimization applied")
            except:
                print("⚠️  FP16 optimization failed")
        
        return model
    
    def _create_transform(self):
        """Create fast preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_class_colors(self) -> dict:
        """Get colors for different segmentation classes."""
        return {
            0: (128, 64, 128),   # road - purple
            1: (244, 35, 232),   # sidewalk - pink
            2: (70, 70, 70),     # building - dark gray
            3: (102, 102, 156),  # wall - light gray
            4: (190, 153, 153),  # fence - beige
            5: (153, 153, 153),  # pole - gray
            6: (250, 170, 30),   # traffic light - orange
            7: (220, 220, 0),    # traffic sign - yellow
            8: (107, 142, 35),   # vegetation - green
            9: (152, 251, 152),  # terrain - light green
            10: (70, 130, 180),  # sky - blue
            11: (220, 20, 60),   # person - red
            12: (255, 0, 0),     # rider - bright red
            13: (0, 0, 142),     # car - dark blue
            14: (0, 0, 70),      # truck - very dark blue
            15: (0, 60, 100),    # bus - medium blue
            16: (0, 80, 100),    # train - teal
            17: (0, 0, 230),     # motorcycle - blue
            18: (119, 11, 32),   # bicycle - dark red
        }
    
    def _benchmark_performance(self):
        """Quick performance benchmark."""
        print("🔥 Running performance benchmark...")
        
        # Create test image
        test_image = Image.new('RGB', (1024, 512), color='gray')
        
        # Warmup
        for _ in range(5):
            self.process_frame(test_image, show_analysis=False)
        
        # Benchmark
        self.frame_times = []
        start = time.time()
        for _ in range(30):
            self.process_frame(test_image, show_analysis=False)
        total_time = time.time() - start
        
        avg_fps = 30 / total_time
        avg_time = np.mean(self.frame_times) * 1000
        
        print(f"📊 Benchmark Results:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Device: {self.device}")
        
        if avg_fps >= 30:
            print("✅ Excellent performance for real-time driving")
        elif avg_fps >= 20:
            print("✅ Good performance for assisted driving")
        elif avg_fps >= 10:
            print("⚠️  Moderate performance - usable but not optimal")
        else:
            print("❌ Poor performance - not suitable for real-time use")
    
    def process_frame(self, image: Image.Image, show_analysis: bool = True) -> dict:
        """
        Process a single frame with timing and analysis.
        
        Args:
            image: Input PIL image
            show_analysis: Whether to print driving analysis
            
        Returns:
            Dictionary with results and timing info
        """
        frame_start = time.time()
        self.frame_count += 1
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Handle FP16
        if self.device.type == 'cuda' and hasattr(self.model, 'half'):
            input_tensor = input_tensor.half()
        
        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=1)
            segmentation = predictions.cpu().numpy().squeeze()
        inference_time = time.time() - inference_start
        
        # Analyze for driving
        analysis = self._analyze_driving_scene(segmentation)
        
        # Total frame time
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        
        # Keep only recent times for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        
        # Calculate current FPS
        current_fps = 1.0 / np.mean(self.frame_times[-10:]) if len(self.frame_times) >= 10 else 0
        
        results = {
            'segmentation': segmentation,
            'frame_time_ms': frame_time * 1000,
            'inference_time_ms': inference_time * 1000,
            'fps': current_fps,
            'frame_number': self.frame_count,
            'analysis': analysis
        }
        
        if show_analysis:
            self._print_frame_analysis(results)
        
        return results
    
    def _analyze_driving_scene(self, segmentation: np.ndarray) -> dict:
        """Analyze segmentation for driving-relevant information."""
        
        total_pixels = segmentation.size
        
        # Define class groups
        road_classes = [0, 1]  # road, sidewalk
        vehicle_classes = [13, 14, 15]  # car, truck, bus
        person_classes = [11, 12]  # person, rider
        obstacle_classes = [2, 3, 4, 5]  # building, wall, fence, pole
        
        # Calculate areas
        road_area = np.isin(segmentation, road_classes).sum() / total_pixels * 100
        vehicle_area = np.isin(segmentation, vehicle_classes).sum() / total_pixels * 100
        person_area = np.isin(segmentation, person_classes).sum() / total_pixels * 100
        obstacle_area = np.isin(segmentation, obstacle_classes).sum() / total_pixels * 100
        
        # Driving safety assessment
        safety_score = 100
        if vehicle_area > 15:  # Too many vehicles
            safety_score -= 30
        if person_area > 5:  # Pedestrians present
            safety_score -= 20
        if road_area < 20:  # Not enough road visible
            safety_score -= 25
        if obstacle_area > 30:  # Too many obstacles
            safety_score -= 15
        
        safety_score = max(0, safety_score)
        
        return {
            'road_area_percent': road_area,
            'vehicle_area_percent': vehicle_area,
            'person_area_percent': person_area,
            'obstacle_area_percent': obstacle_area,
            'safety_score': safety_score,
            'driving_recommendation': self._get_driving_recommendation(safety_score)
        }
    
    def _get_driving_recommendation(self, safety_score: int) -> str:
        """Get driving recommendation based on safety score."""
        if safety_score >= 80:
            return "🟢 CLEAR - Normal driving conditions"
        elif safety_score >= 60:
            return "🟡 CAUTION - Reduced speed recommended"
        elif safety_score >= 40:
            return "🟠 WARNING - High caution required"
        else:
            return "🔴 DANGER - Stop or extreme caution"
    
    def _print_frame_analysis(self, results: dict):
        """Print real-time analysis to console."""
        analysis = results['analysis']
        
        # Clear previous lines (simple version)
        print(f"\rFrame {results['frame_number']} | "
              f"FPS: {results['fps']:.1f} | "
              f"Time: {results['frame_time_ms']:.1f}ms | "
              f"Safety: {analysis['safety_score']}/100 | "
              f"{analysis['driving_recommendation']}", end="")
        
        # Detailed analysis every 30 frames
        if results['frame_number'] % 30 == 0:
            print(f"\n📊 Detailed Analysis (Frame {results['frame_number']}):")
            print(f"  Road visible: {analysis['road_area_percent']:.1f}%")
            print(f"  Vehicles: {analysis['vehicle_area_percent']:.1f}%")
            print(f"  Pedestrians: {analysis['person_area_percent']:.1f}%")
            print(f"  Obstacles: {analysis['obstacle_area_percent']:.1f}%")
            print(f"  Recommendation: {analysis['driving_recommendation']}")
    
    def create_visualization(self, image: Image.Image, segmentation: np.ndarray) -> Image.Image:
        """
        Create visualization overlay of segmentation on original image.
        
        Args:
            image: Original image
            segmentation: Segmentation mask
            
        Returns:
            Image with segmentation overlay
        """
        # Resize segmentation to match image
        seg_img = Image.fromarray(segmentation.astype(np.uint8), mode='L')
        seg_img = seg_img.resize(image.size, Image.NEAREST)
        seg_array = np.array(seg_img)
        
        # Create colored overlay
        overlay = Image.new('RGB', image.size, (0, 0, 0))
        overlay_array = np.array(overlay)
        
        for class_id, color in self.class_colors.items():
            mask = seg_array == class_id
            overlay_array[mask] = color
        
        overlay = Image.fromarray(overlay_array)
        
        # Blend with original image
        blended = Image.blend(image, overlay, alpha=0.5)
        
        return blended
    
    def run_demo_on_images(self, image_dir: str, output_dir: str = None):
        """
        Run demo on a directory of images.
        
        Args:
            image_dir: Directory containing test images
            output_dir: Directory to save visualization (optional)
        """
        image_path = Path(image_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        print(f"🎬 Running demo on images in {image_dir}")
        
        # Find images
        image_files = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))
        
        if not image_files:
            print("❌ No images found in directory")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            print(f"\n🖼️  Processing {img_file.name} ({i+1}/{len(image_files)})")
            
            # Load and process
            image = Image.open(img_file).convert('RGB')
            results = self.process_frame(image)
            
            # Create visualization if output directory specified
            if output_dir:
                vis_image = self.create_visualization(image, results['segmentation'])
                output_file = output_path / f"seg_{img_file.name}"
                vis_image.save(output_file)
                print(f"  Saved visualization: {output_file}")
        
        print(f"\n✅ Demo completed! Processed {len(image_files)} images")
        
        # Final statistics
        if self.frame_times:
            avg_fps = 1.0 / np.mean(self.frame_times)
            print(f"📊 Overall Performance:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Total frames: {len(self.frame_times)}")

def quick_driving_demo(model_path: str, test_images_dir: str = None):
    """
    Quick demonstration of real-time segmentation for driving.
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory with test images (optional)
    """
    print("🚗💨 Quick Real-time Driving Segmentation Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = RealTimeDrivingDemo(model_path)
    
    if test_images_dir and Path(test_images_dir).exists():
        # Run on test images
        demo.run_demo_on_images(test_images_dir, "demo_output")
    else:
        # Run on synthetic test images
        print("🧪 Running on synthetic test images...")
        
        test_images = [
            Image.new('RGB', (1024, 512), color='gray'),      # Road scene
            Image.new('RGB', (1024, 512), color='green'),     # Vegetation
            Image.new('RGB', (1024, 512), color='blue'),      # Sky
        ]
        
        for i, img in enumerate(test_images):
            print(f"\n🖼️  Processing synthetic image {i+1}")
            results = demo.process_frame(img)
    
    print(f"\n🏁 Demo completed!")
    return demo

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/segmentation_model.pth"
    test_dir = "../data/sequence.0"  # Directory with test images
    
    # Run quick demo
    demo = quick_driving_demo(model_path, test_dir)

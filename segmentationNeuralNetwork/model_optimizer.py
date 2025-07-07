# Model Optimization and Conversion Tools
# Convert trained models to optimized formats for real-time inference

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import time
from pathlib import Path
import warnings

class ModelOptimizer:
    """
    Convert and optimize segmentation models for maximum inference speed.
    Supports multiple optimization backends.
    """
    
    def __init__(self, model_path: str, output_dir: str = "optimized_models"):
        """
        Initialize model optimizer.
        
        Args:
            model_path: Path to original PyTorch model
            output_dir: Directory to save optimized models
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load original model
        self.model = self._load_original_model()
        
    def _load_original_model(self) -> nn.Module:
        """Load the original PyTorch model."""
        
        # Create model architecture (adjust if your model is different)
        model = smp.Unet(
            encoder_name="resnet34",  # Adjust based on your training
            encoder_weights="imagenet",
            in_channels=3,
            classes=255,
            activation=None
        )
        
        # Load weights
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def create_lightweight_model(self, encoder: str = "mobilenet_v2") -> str:
        """
        Create a lightweight version of the model with different encoder.
        
        Args:
            encoder: Lightweight encoder to use
            
        Returns:
            Path to lightweight model
        """
        print(f"🏃‍♂️ Creating lightweight model with {encoder} encoder...")
        
        # Create lightweight model
        light_model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=255,
            activation=None
        )
        
        # Transfer compatible weights (encoder may differ)
        try:
            # This is a simplified transfer - in practice, you'd retrain or use knowledge distillation
            light_model.eval()
            output_path = self.output_dir / f"lightweight_{encoder}_model.pth"
            torch.save(light_model.state_dict(), output_path)
            print(f"✅ Lightweight model saved: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"❌ Failed to create lightweight model: {e}")
            return None
    
    def optimize_with_torchscript(self, input_size: tuple = (1, 3, 256, 512)) -> str:
        """
        Optimize model using TorchScript JIT compilation.
        
        Args:
            input_size: Input tensor size for tracing
            
        Returns:
            Path to optimized model
        """
        print("⚡ Optimizing with TorchScript...")
        
        try:
            # Create dummy input for tracing
            dummy_input = torch.randn(input_size)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Optimize
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save
            output_path = self.output_dir / "torchscript_model.pt"
            traced_model.save(str(output_path))
            
            print(f"✅ TorchScript model saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ TorchScript optimization failed: {e}")
            return None
    
    def convert_to_onnx(self, input_size: tuple = (1, 3, 256, 512)) -> str:
        """
        Convert model to ONNX format for cross-platform inference.
        
        Args:
            input_size: Input tensor size
            
        Returns:
            Path to ONNX model
        """
        print("🔄 Converting to ONNX...")
        
        try:
            import onnx
            import onnxruntime
        except ImportError:
            print("❌ ONNX not installed. Install with: pip install onnx onnxruntime")
            return None
        
        try:
            # Dummy input
            dummy_input = torch.randn(input_size)
            
            # Export to ONNX
            output_path = self.output_dir / "model.onnx"
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNX model saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ ONNX conversion failed: {e}")
            return None
    
    def quantize_model(self) -> str:
        """
        Apply post-training quantization for smaller model size and faster inference.
        
        Returns:
            Path to quantized model
        """
        print("🗜️  Applying quantization...")
        
        try:
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Save quantized model
            output_path = self.output_dir / "quantized_model.pth"
            torch.save(quantized_model.state_dict(), output_path)
            
            print(f"✅ Quantized model saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return None
    
    def benchmark_optimizations(self, iterations: int = 100):
        """
        Benchmark all optimization methods to find the fastest.
        
        Args:
            iterations: Number of benchmark iterations
        """
        print("🏁 Benchmarking all optimization methods...")
        
        input_size = (1, 3, 256, 512)
        dummy_input = torch.randn(input_size)
        
        results = {}
        
        # 1. Original model
        print("\n1️⃣  Benchmarking original model...")
        times = self._benchmark_model(self.model, dummy_input, iterations)
        results['original'] = {
            'avg_time_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        # 2. TorchScript model
        torchscript_path = self.optimize_with_torchscript(input_size)
        if torchscript_path:
            print("\n2️⃣  Benchmarking TorchScript model...")
            ts_model = torch.jit.load(torchscript_path)
            times = self._benchmark_model(ts_model, dummy_input, iterations)
            results['torchscript'] = {
                'avg_time_ms': np.mean(times) * 1000,
                'fps': 1.0 / np.mean(times)
            }
        
        # 3. Quantized model
        quantized_path = self.quantize_model()
        if quantized_path:
            print("\n3️⃣  Benchmarking quantized model...")
            # Note: Quantized model benchmarking would need special handling
            print("⚠️  Quantized model benchmarking skipped (requires special setup)")
        
        # 4. ONNX model
        onnx_path = self.convert_to_onnx(input_size)
        if onnx_path:
            try:
                import onnxruntime as ort
                print("\n4️⃣  Benchmarking ONNX model...")
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                
                times = []
                for _ in range(iterations):
                    start = time.time()
                    _ = session.run(None, {input_name: dummy_input.numpy()})
                    times.append(time.time() - start)
                
                results['onnx'] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'fps': 1.0 / np.mean(times)
                }
            except ImportError:
                print("⚠️  ONNX runtime not available for benchmarking")
        
        # Print results
        print("\n📊 Benchmark Results Summary:")
        print("=" * 50)
        for method, stats in results.items():
            print(f"{method.upper():>12}: {stats['avg_time_ms']:>8.2f} ms | {stats['fps']:>6.1f} FPS")
        
        # Find best method
        best_method = min(results.keys(), key=lambda k: results[k]['avg_time_ms'])
        print(f"\n🏆 Fastest method: {best_method.upper()}")
        print(f"    Speed improvement: {results['original']['avg_time_ms'] / results[best_method]['avg_time_ms']:.2f}x")
        
        return results
    
    def _benchmark_model(self, model, dummy_input, iterations):
        """Benchmark a single model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
        return times

class OptimizedInferenceEngine:
    """
    High-performance inference engine that automatically selects the best optimization.
    """
    
    def __init__(self, model_path: str, auto_optimize: bool = True):
        """
        Initialize optimized inference engine.
        
        Args:
            model_path: Path to trained model
            auto_optimize: Whether to automatically find best optimization
        """
        self.model_path = model_path
        
        if auto_optimize:
            self._auto_optimize()
        else:
            self._load_default_model()
    
    def _auto_optimize(self):
        """Automatically find and use the best optimization."""
        print("🔍 Auto-optimizing model for best performance...")
        
        optimizer = ModelOptimizer(self.model_path)
        results = optimizer.benchmark_optimizations(iterations=50)
        
        # Use the fastest method
        best_method = min(results.keys(), key=lambda k: results[k]['avg_time_ms'])
        
        if best_method == 'torchscript':
            self.model = torch.jit.load(optimizer.output_dir / "torchscript_model.pt")
        elif best_method == 'onnx':
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(optimizer.output_dir / "model.onnx"))
                self.model = None  # Use ONNX session instead
            except ImportError:
                self._load_default_model()
        else:
            self._load_default_model()
        
        print(f"✅ Using {best_method} optimization for inference")
    
    def _load_default_model(self):
        """Load default PyTorch model."""
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=255,
            activation=None
        )
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()
        self.model = model

def optimize_for_driving(model_path: str) -> str:
    """
    One-click optimization for driving scenarios.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Path to best optimized model
    """
    print("🚗 Optimizing model for real-time driving...")
    
    optimizer = ModelOptimizer(model_path)
    
    # Test different optimizations
    results = optimizer.benchmark_optimizations(iterations=30)
    
    # Find best method for real-time (>20 FPS target)
    best_for_driving = None
    best_fps = 0
    
    for method, stats in results.items():
        if stats['fps'] > best_fps:
            best_fps = stats['fps']
            best_for_driving = method
    
    if best_fps >= 20:
        print(f"✅ Model optimized for driving: {best_fps:.1f} FPS with {best_for_driving}")
    else:
        print(f"⚠️  Model may be slow for driving: {best_fps:.1f} FPS")
        print("Consider using a lighter model architecture or faster hardware")
    
    return best_for_driving

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model.pth"
    
    # Optimize for driving
    best_method = optimize_for_driving(model_path)
    
    # Create optimized inference engine
    engine = OptimizedInferenceEngine(model_path, auto_optimize=True)

#!/usr/bin/env python3
"""
Real-time Driving Pipeline Benchmark

Tests the complete pipeline performance for real-time driving:
1. Segmentation model (semantic understanding)
2. Decision model (driving decisions)
3. Combined pipeline latency and throughput

This script provides comprehensive performance metrics to validate
that the system can run in real-time during driving.
"""

import torch
import time
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "segmentationNeuralNetwork"))
sys.path.append(str(Path(__file__).parent / "decisionNeuralNetwork"))

try:
    from optimized_inference import OptimizedSegmentationModel
    from decision_model import create_model
    from inference import DecisionPredictor
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


class RealTimePipelineBenchmark:
    """Benchmark the complete real-time driving pipeline."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize benchmark with device selection."""
        self.device = self._select_device(device)
        self.results = {}
        
    def _select_device(self, device: str) -> torch.device:
        """Select the best available device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def benchmark_segmentation_models(self) -> Dict:
        """Benchmark segmentation models across optimization levels."""
        print("🔍 Benchmarking Segmentation Models...")
        print("-" * 50)
        
        seg_results = {}
        levels = ['speed', 'balanced', 'quality']
        
        # Find model checkpoint or create dummy
        checkpoints_dir = Path("segmentationNeuralNetwork/checkpoints")
        model_files = list(checkpoints_dir.glob("*.pth")) if checkpoints_dir.exists() else []
        
        if not model_files:
            print("⚠️  No segmentation checkpoints found, using dummy model...")
            dummy_path = "dummy_seg_model.pth"
            torch.save({'model_state_dict': {}}, dummy_path)
            model_path = dummy_path
        else:
            model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
        
        for level in levels:
            try:
                print(f"\n📊 Testing {level.upper()} optimization:")
                
                # Initialize model
                seg_model = OptimizedSegmentationModel(model_path, 
                                                     device=str(self.device), 
                                                     optimization_level=level)
                
                # Benchmark with different input sizes
                input_sizes = [(256, 256), (384, 384), (512, 512)]
                level_results = {}
                
                for size in input_sizes:
                    # Create dummy input
                    dummy_input = torch.randn(1, 3, *size).to(self.device)
                    
                    # Warm up
                    for _ in range(5):
                        with torch.no_grad():
                            _ = seg_model.model(dummy_input)
                    
                    # Benchmark
                    times = []
                    for _ in range(30):
                        start = time.time()
                        with torch.no_grad():
                            output = seg_model.model(dummy_input)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        times.append(time.time() - start)
                    
                    avg_time = np.mean(times[5:])  # Skip first 5
                    fps = 1.0 / avg_time
                    
                    level_results[f"{size[0]}x{size[1]}"] = {
                        'time_ms': avg_time * 1000,
                        'fps': fps,
                        'real_time': fps >= 30
                    }
                    
                    print(f"   {size[0]}x{size[1]}: {avg_time*1000:.1f}ms ({fps:.1f} FPS) {'✓' if fps >= 30 else '✗'}")
                
                seg_results[level] = level_results
                
            except Exception as e:
                print(f"   Error testing {level}: {e}")
                seg_results[level] = {'error': str(e)}
        
        self.results['segmentation'] = seg_results
        return seg_results
    
    def benchmark_decision_models(self) -> Dict:
        """Benchmark decision models."""
        print("\n🔍 Benchmarking Decision Models...")
        print("-" * 50)
        
        decision_results = {}
        model_types = ['simple', 'resnet']
        
        for model_type in model_types:
            try:
                print(f"\n📊 Testing {model_type.upper()} model:")
                
                # Create model
                model = create_model(model_type, num_classes=3)
                model.eval()
                model.to(self.device)
                
                # Get model info
                total_params = sum(p.numel() for p in model.parameters())
                
                # Benchmark with standard input size (224x224)
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Warm up
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                # Benchmark
                times = []
                for _ in range(100):
                    start = time.time()
                    with torch.no_grad():
                        output = model(dummy_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
                
                avg_time = np.mean(times[10:])  # Skip first 10
                fps = 1.0 / avg_time
                
                decision_results[model_type] = {
                    'parameters': total_params,
                    'time_ms': avg_time * 1000,
                    'fps': fps,
                    'real_time': fps >= 30
                }
                
                print(f"   Parameters: {total_params:,}")
                print(f"   Time: {avg_time*1000:.2f}ms ({fps:.1f} FPS) {'✓' if fps >= 30 else '✗'}")
                
            except Exception as e:
                print(f"   Error testing {model_type}: {e}")
                decision_results[model_type] = {'error': str(e)}
        
        self.results['decision'] = decision_results
        return decision_results
    
    def benchmark_combined_pipeline(self) -> Dict:
        """Benchmark the complete pipeline (segmentation + decision)."""
        print("\n🔍 Benchmarking Combined Pipeline...")
        print("-" * 50)
        
        pipeline_results = {}
        
        # Test combinations of segmentation and decision models
        seg_levels = ['speed', 'balanced', 'quality']
        decision_types = ['simple', 'resnet']
        
        # Find or create dummy segmentation model
        checkpoints_dir = Path("segmentationNeuralNetwork/checkpoints")
        model_files = list(checkpoints_dir.glob("*.pth")) if checkpoints_dir.exists() else []
        
        if not model_files:
            dummy_path = "dummy_seg_model.pth"
            torch.save({'model_state_dict': {}}, dummy_path)
            seg_model_path = dummy_path
        else:
            seg_model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
        
        for seg_level in seg_levels:
            for decision_type in decision_types:
                combination = f"{seg_level}_seg + {decision_type}_decision"
                
                try:
                    print(f"\n📊 Testing {combination}:")
                    
                    # Initialize models
                    seg_model = OptimizedSegmentationModel(seg_model_path, 
                                                         device=str(self.device), 
                                                         optimization_level=seg_level)
                    
                    decision_model = create_model(decision_type, num_classes=3)
                    decision_model.eval()
                    decision_model.to(self.device)
                    
                    # Test with 256x256 input (good balance of quality/speed)
                    dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                    
                    # Warm up
                    for _ in range(5):
                        with torch.no_grad():
                            seg_output = seg_model.model(dummy_input)
                            # Resize segmentation output for decision model
                            resized = torch.nn.functional.interpolate(seg_output, size=(224, 224), mode='bilinear')
                            decision_output = decision_model(resized)
                    
                    # Benchmark pipeline
                    times = []
                    seg_times = []
                    decision_times = []
                    
                    for _ in range(50):
                        # Time complete pipeline
                        start_total = time.time()
                        
                        with torch.no_grad():
                            # Segmentation
                            start_seg = time.time()
                            seg_output = seg_model.model(dummy_input)
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize()
                            seg_time = time.time() - start_seg
                            
                            # Decision (resize segmentation output)
                            start_decision = time.time()
                            resized = torch.nn.functional.interpolate(seg_output, size=(224, 224), mode='bilinear')
                            decision_output = decision_model(resized)
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize()
                            decision_time = time.time() - start_decision
                        
                        total_time = time.time() - start_total
                        
                        times.append(total_time)
                        seg_times.append(seg_time)
                        decision_times.append(decision_time)
                    
                    # Calculate statistics
                    avg_total = np.mean(times[5:])  # Skip first 5
                    avg_seg = np.mean(seg_times[5:])
                    avg_decision = np.mean(decision_times[5:])
                    
                    fps = 1.0 / avg_total
                    
                    pipeline_results[combination] = {
                        'total_time_ms': avg_total * 1000,
                        'segmentation_time_ms': avg_seg * 1000,
                        'decision_time_ms': avg_decision * 1000,
                        'fps': fps,
                        'real_time': fps >= 30,
                        'overhead_ms': (avg_total - avg_seg - avg_decision) * 1000
                    }
                    
                    print(f"   Total: {avg_total*1000:.1f}ms ({fps:.1f} FPS) {'✓' if fps >= 30 else '✗'}")
                    print(f"   - Segmentation: {avg_seg*1000:.1f}ms")
                    print(f"   - Decision: {avg_decision*1000:.1f}ms")
                    print(f"   - Overhead: {(avg_total-avg_seg-avg_decision)*1000:.1f}ms")
                    
                except Exception as e:
                    print(f"   Error testing {combination}: {e}")
                    pipeline_results[combination] = {'error': str(e)}
        
        self.results['pipeline'] = pipeline_results
        return pipeline_results
    
    def print_summary(self):
        """Print comprehensive benchmark summary."""
        print("\n" + "="*70)
        print("🏁 REAL-TIME DRIVING PIPELINE BENCHMARK SUMMARY")
        print("="*70)
        
        print(f"\n🖥️  Device: {self.device}")
        print(f"🎯 Target: ≥30 FPS for real-time driving")
        
        # Segmentation summary
        if 'segmentation' in self.results:
            print(f"\n📊 SEGMENTATION MODELS:")
            for level, results in self.results['segmentation'].items():
                if 'error' not in results:
                    print(f"\n   {level.upper()}:")
                    for size, metrics in results.items():
                        status = "✓ REAL-TIME" if metrics['real_time'] else "✗ TOO SLOW"
                        print(f"     {size}: {metrics['time_ms']:.1f}ms ({metrics['fps']:.1f} FPS) {status}")
        
        # Decision summary
        if 'decision' in self.results:
            print(f"\n🧠 DECISION MODELS:")
            for model_type, results in self.results['decision'].items():
                if 'error' not in results:
                    status = "✓ REAL-TIME" if results['real_time'] else "✗ TOO SLOW"
                    print(f"   {model_type.upper()}: {results['time_ms']:.2f}ms ({results['fps']:.1f} FPS) {status}")
        
        # Pipeline summary
        if 'pipeline' in self.results:
            print(f"\n🚗 COMPLETE PIPELINE (256x256 input):")
            real_time_configs = []
            
            for combination, results in self.results['pipeline'].items():
                if 'error' not in results:
                    status = "✓ REAL-TIME" if results['real_time'] else "✗ TOO SLOW"
                    print(f"   {combination}: {results['total_time_ms']:.1f}ms ({results['fps']:.1f} FPS) {status}")
                    
                    if results['real_time']:
                        real_time_configs.append((combination, results['fps']))
            
            # Recommendations
            print(f"\n🎯 RECOMMENDATIONS:")
            if real_time_configs:
                # Sort by FPS (highest first)
                real_time_configs.sort(key=lambda x: x[1], reverse=True)
                
                print(f"   ✅ {len(real_time_configs)} configurations are real-time ready!")
                print(f"   🏆 Best performance: {real_time_configs[0][0]} ({real_time_configs[0][1]:.1f} FPS)")
                
                if len(real_time_configs) > 1:
                    print(f"   ⚖️  Balanced option: {real_time_configs[len(real_time_configs)//2][0]} ({real_time_configs[len(real_time_configs)//2][1]:.1f} FPS)")
                
                print(f"\n   🚀 For maximum speed: Use 'speed' segmentation + 'simple' decision")
                print(f"   🎯 For best accuracy: Use 'quality' segmentation + 'resnet' decision (if real-time)")
                print(f"   ⚖️  For balance: Use 'balanced' segmentation + 'resnet' decision")
            else:
                print(f"   ⚠️  No configurations achieve real-time performance!")
                print(f"   🔧 Consider: Smaller input sizes, model optimization, or faster hardware")
        
        print(f"\n💡 DEPLOYMENT NOTES:")
        print(f"   • Real-time = ≥30 FPS (33.3ms max latency)")
        print(f"   • Autonomous driving typically needs 10-60 FPS depending on speed")
        print(f"   • Consider GPU acceleration for better performance")
        print(f"   • Pipeline can be further optimized with TensorRT, ONNX, or quantization")


def main():
    """Run the complete benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark real-time driving pipeline')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='auto',
                        help='Device to run benchmarks on')
    parser.add_argument('--skip-seg', action='store_true', help='Skip segmentation benchmarks')
    parser.add_argument('--skip-decision', action='store_true', help='Skip decision benchmarks')
    parser.add_argument('--skip-pipeline', action='store_true', help='Skip pipeline benchmarks')
    
    args = parser.parse_args()
    
    if not IMPORTS_OK:
        print("❌ Cannot run benchmarks due to import errors")
        print("Make sure you're in the correct directory and all dependencies are installed")
        return
    
    print("🏁 Real-Time Driving Pipeline Benchmark")
    print("="*70)
    
    benchmark = RealTimePipelineBenchmark(device=args.device)
    
    # Run benchmarks
    if not args.skip_seg:
        benchmark.benchmark_segmentation_models()
    
    if not args.skip_decision:
        benchmark.benchmark_decision_models()
    
    if not args.skip_pipeline:
        benchmark.benchmark_combined_pipeline()
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ultra Performance Training Launcher
Easy-to-use launcher for 24+ hour high-accuracy training.
"""

import argparse
import sys
from pathlib import Path
from ultra_config import get_config, print_config_summary

def main():
    parser = argparse.ArgumentParser(description='Ultra Performance Unity Segmentation Training')
    parser.add_argument('--config', choices=['ultra', 'fast', 'custom'], default='ultra',
                       help='Training configuration preset')
    parser.add_argument('--target-accuracy', type=float, default=0.90,
                       help='Target accuracy (0.0-1.0)')
    parser.add_argument('--max-hours', type=float, default=24,
                       help='Maximum training hours')
    parser.add_argument('--ensemble-size', type=int, default=3,
                       help='Number of models in ensemble')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without training')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint directory')
    
    args = parser.parse_args()
    
    print("🚀 Ultra Performance Unity Segmentation Training")
    print("=" * 60)
    
    # Load base configuration
    config = get_config()
    
    # Apply preset configurations
    if args.config == 'fast':
        # Faster training preset (12-16 hours)
        config['ultra'].update({
            'epochs': 150,
            'ensemble_size': 1,
            'batch_size': 16,
            'early_stopping_patience': 25,
            'save_every': 15,
            'use_mixup': False,
            'use_cutmix': False,
        })
        print("📈 Using FAST preset (12-16 hours, 85-90% accuracy)")
        
    elif args.config == 'ultra':
        # Ultra performance preset (24+ hours)
        print("🎯 Using ULTRA preset (24+ hours, 90-95% accuracy)")
        
    elif args.config == 'custom':
        print("🛠️ Using CUSTOM configuration")
    
    # Apply command line overrides
    config['ultra']['target_accuracy'] = args.target_accuracy
    config['ultra']['ensemble_size'] = args.ensemble_size
    config['ultra']['batch_size'] = args.batch_size
    
    # Calculate max epochs based on time budget
    if args.max_hours < 24:
        estimated_epochs = int(args.max_hours * 12)  # ~12 epochs per hour
        config['ultra']['epochs'] = min(config['ultra']['epochs'], estimated_epochs)
        print(f"⏱️ Time budget: {args.max_hours}h, limiting to {config['ultra']['epochs']} epochs")
    
    print()
    print_config_summary()
    print()
    
    # Pre-flight checks
    print("🔍 Pre-flight Checks:")
    
    # Check data availability
    data_path = Path(config['ultra']['data_dir']) / config['ultra']['sequence']
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        sys.exit(1)
    else:
        print(f"✅ Data found: {data_path}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            print("⚠️ No GPU available, using CPU (will be very slow)")
    except ImportError:
        print("❌ PyTorch not available")
        sys.exit(1)
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free // 1024**3
        if free_space < 10:
            print(f"⚠️ Low disk space: {free_space} GB (recommend 10+ GB)")
        else:
            print(f"✅ Disk space: {free_space} GB available")
    except:
        print("⚠️ Could not check disk space")
    
    print()
    
    if args.dry_run:
        print("🔍 Dry run complete. Configuration looks good!")
        print("Remove --dry-run to start training.")
        return
    
    # Confirmation
    print("🚀 Ready to start ultra performance training!")
    print(f"   Target: {args.target_accuracy*100:.0f}% accuracy")
    print(f"   Models: {args.ensemble_size} in ensemble")
    print(f"   Time budget: {args.max_hours} hours")
    print(f"   Max epochs: {config['ultra']['epochs']}")
    print()
    
    try:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        return
    
    # Start training
    print("\n🎯 Starting Ultra Performance Training...")
    print("🔥 This will take 24+ hours. Make sure your system is stable!")
    print("💡 Monitor progress: tail -f checkpoints/training.log")
    print("📊 View plots: open checkpoints/ultra_training_progress.png")
    print()
    
    try:
        from train_ultra_performance import UltraHighPerformanceTrainer
        
        # Create trainer with custom configuration
        trainer = UltraHighPerformanceTrainer(config['ultra'])
        
        # Start training
        results = trainer.train()
        
        # Training completed
        print("\n🎉 Ultra Performance Training Completed!")
        print(f"   Final Accuracy: {results['best_accuracy']*100:.1f}%")
        print(f"   Training Time: {results['total_time_hours']:.1f} hours")
        print(f"   Models Saved: {len(results['best_model_paths'])}")
        
        # Test the trained models
        if results['best_accuracy'] >= args.target_accuracy:
            print(f"✅ TARGET ACHIEVED! {results['best_accuracy']*100:.1f}% >= {args.target_accuracy*100:.0f}%")
        else:
            print(f"📈 Progress made: {results['best_accuracy']*100:.1f}% (target: {args.target_accuracy*100:.0f}%)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

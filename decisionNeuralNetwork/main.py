"""
Main script for the driving decision neural network project.
This script provides a unified interface for data annotation, training, and inference.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Optional


def run_command(command: List[str], description: str) -> int:
    """
    Run a command and handle errors.
    
    Args:
        command: Command to run as list of strings
        description: Description of what the command does
        
    Returns:
        Return code of the command
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)
    
    try:
        result = subprocess.run(command, check=True)
        print(f"✅ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return 1


def check_data_directory(data_dir: str) -> bool:
    """Check if data directory exists and contains segmentation masks."""
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return False
    
    # Check for segmentation masks
    segmentation_files = [f for f in os.listdir(data_dir) 
                         if f.endswith('.camera.semantic segmentation.png')]
    
    if not segmentation_files:
        print(f"❌ No segmentation masks found in: {data_dir}")
        return False
    
    print(f"✅ Found {len(segmentation_files)} segmentation masks in {data_dir}")
    return True


def check_labels_file(data_dir: str, labels_file: str = 'driving_labels.json') -> bool:
    """Check if labels file exists and has sufficient labels."""
    labels_path = os.path.join(data_dir, labels_file)
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file does not exist: {labels_path}")
        print("   Use 'annotate' command to create labels")
        return False
    
    try:
        import json
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        if len(labels) < 10:
            print(f"⚠️  Only {len(labels)} labels found. Recommend at least 50+ for good training")
            return False
        
        print(f"✅ Found {len(labels)} labels in {labels_path}")
        return True
    except Exception as e:
        print(f"❌ Error reading labels file: {e}")
        return False


def check_model_checkpoint(model_path: str) -> bool:
    """Check if model checkpoint exists."""
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint does not exist: {model_path}")
        return False
    
    print(f"✅ Model checkpoint found: {model_path}")
    return True


def annotate_data(args):
    """Run data annotation."""
    if not check_data_directory(args.data_dir):
        return 1
    
    # Use terminal version on Mac for better compatibility
    if sys.platform == "darwin":  # macOS
        script_name = 'annotate_data_terminal.py'
        print("🍎 Using terminal-based annotator for Mac compatibility")
    else:
        script_name = 'annotate_data.py'
    
    command = [
        sys.executable, script_name,
        args.data_dir,
        '--batch_size', str(args.batch_size)
    ]
    
    if args.continue_annotation:
        command.append('--continue')
    
    if args.stats_only:
        command.append('--stats')
    
    return run_command(command, "Data annotation")


def train_model(args):
    """Run model training."""
    if not check_data_directory(args.data_dir):
        return 1
    
    if not check_labels_file(args.data_dir):
        return 1
    
    command = [
        sys.executable, 'train_decision_model.py',
        args.data_dir,
        '--model_type', args.model_type,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.learning_rate),
        '--image_size', str(args.image_size),
        '--save_dir', args.save_dir,
        '--augmentation', args.augmentation
    ]
    
    return run_command(command, "Model training")


def run_inference(args):
    """Run model inference."""
    model_path = args.model_path
    
    if not check_model_checkpoint(model_path):
        return 1
    
    command = [
        sys.executable, 'inference.py',
        model_path,
        '--model_type', args.model_type,
        '--image_size', str(args.image_size)
    ]
    
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"❌ Image file does not exist: {args.image_path}")
            return 1
        command.extend(['--image_path', args.image_path])
    
    elif args.data_dir:
        if not check_data_directory(args.data_dir):
            return 1
        command.extend(['--data_dir', args.data_dir])
        
        if args.output_file:
            command.extend(['--output_file', args.output_file])
    
    else:
        print("❌ Please specify either --image_path or --data_dir for inference")
        return 1
    
    return run_command(command, "Model inference")


def setup_environment(args):
    """Setup the Python environment and install dependencies."""
    print("Setting up environment...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment. Consider using venv or conda")
    
    # Install requirements
    requirements_path = '../requirements.txt'
    if os.path.exists(requirements_path):
        command = [sys.executable, '-m', 'pip', 'install', '-r', requirements_path]
        return run_command(command, "Installing requirements")
    else:
        print(f"❌ Requirements file not found: {requirements_path}")
        return 1


def handle_test_command(args):
    """Handle the test command."""
    print("🚗 RoadNet Model Testing")
    print("=" * 40)
    
    # Check if model exists
    model_path = args.model_path or 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Train a model first using: python main.py train <data_dir>")
        return 1
    
    if args.gui:
        print("🖼️  Launching GUI interface...")
        return run_command(['python', 'test_inference_gui.py'], 
                          "GUI Test Interface")
    
    elif args.image_path:
        print(f"🔍 Testing single image: {args.image_path}")
        return run_command(['python', 'test_model.py', '--image', args.image_path, '--model', model_path],
                          "Single Image Test")
    
    elif args.data_dir:
        print(f"📊 Testing batch of images in: {args.data_dir}")
        cmd = ['python', 'test_inference_cli.py', model_path, '--data_dir', args.data_dir]
        if args.save_viz:
            cmd.append('--save_viz')
        if args.output:
            cmd.extend(['--output', args.output])
        return run_command(cmd, "Batch Image Test")
    
    else:
        print("🎯 Available Test Options:")
        print("1. 🖼️  GUI Interface:     python main.py test --gui")
        print("2. 📸 Single Image:      python main.py test --image path/to/image.png") 
        print("3. 📊 Batch Directory:   python main.py test --data_dir path/to/directory")
        print("4. 💻 Interactive CLI:   python test_model.py --cli")
        print("\n💡 Examples:")
        print("   python main.py test --gui")
        print("   python main.py test --image ../data/sequence.0/step5.camera.semantic\\ segmentation.png")
        print("   python main.py test --data_dir ../data/sequence.0 --save_viz")
        return 0


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Driving Decision Neural Network - Main Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup environment
  python main.py setup
  
  # Annotate data
  python main.py annotate /path/to/data/sequence.0
  
  # Train model
  python main.py train /path/to/data/sequence.0 --epochs 100
  
  # Run inference on single image
  python main.py infer checkpoints/best_model.pth --image_path /path/to/image.png
  
  # Run inference on directory
  python main.py infer checkpoints/best_model.pth --data_dir /path/to/data/sequence.0
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup environment and install dependencies')
    
    # Annotate command
    annotate_parser = subparsers.add_parser('annotate', help='Annotate driving decisions')
    annotate_parser.add_argument('data_dir', help='Directory containing segmentation masks')
    annotate_parser.add_argument('--batch_size', type=int, default=10, help='Batch size for annotation')
    annotate_parser.add_argument('--continue', dest='continue_annotation', action='store_true',
                                help='Continue from where left off')
    annotate_parser.add_argument('--stats', dest='stats_only', action='store_true',
                                help='Show statistics only')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the neural network')
    train_parser.add_argument('data_dir', help='Directory containing segmentation masks')
    train_parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                             help='Type of model to train')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    train_parser.add_argument('--save_dir', default='checkpoints', help='Directory to save models')
    train_parser.add_argument('--augmentation', choices=['light', 'medium', 'heavy'], default='medium',
                             help='Data augmentation strength: light, medium, or heavy')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run model inference')
    infer_parser.add_argument('model_path', help='Path to trained model checkpoint')
    infer_parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                             help='Type of model')
    infer_parser.add_argument('--image_path', help='Path to single image for prediction')
    infer_parser.add_argument('--data_dir', help='Directory containing segmentation masks')
    infer_parser.add_argument('--output_file', help='File to save predictions (JSON format)')
    infer_parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the trained model')
    test_parser.add_argument('--model_path', help='Path to trained model checkpoint')
    test_parser.add_argument('--image_path', help='Path to single image for testing')
    test_parser.add_argument('--data_dir', help='Directory containing images for testing')
    test_parser.add_argument('--output', help='File to save test results')
    test_parser.add_argument('--save_viz', action='store_true', help='Save visualization of test results')
    test_parser.add_argument('--gui', action='store_true', help='Launch GUI for testing')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run appropriate command
    if args.command == 'setup':
        return setup_environment(args)
    elif args.command == 'annotate':
        return annotate_data(args)
    elif args.command == 'train':
        return train_model(args)
    elif args.command == 'infer':
        return run_inference(args)
    elif args.command == 'test':
        return handle_test_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Simple test interface for the driving decision model with visualization.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json

from decision_model import create_model
from decision_dataset import LABEL_MAPPING


def main():
    """Simple test interface."""
    parser = argparse.ArgumentParser(description='🚗 RoadNet - Test Your Model!')
    parser.add_argument('--model', default='checkpoints/best_model.pth',
                       help='Path to model file (default: checkpoints/best_model.pth)')
    parser.add_argument('--image', help='Path to specific image to test')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--cli', action='store_true', help='Use command-line interface')
    parser.add_argument('--batch', help='Test all images in directory')
    
    args = parser.parse_args()
    
    print("🚗 RoadNet Driving Decision Tester")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("💡 Make sure you've trained a model first!")
        return
    
    if args.gui:
        print("🖼️  Launching GUI interface...")
        try:
            import test_inference_gui
            test_inference_gui.main()
        except ImportError as e:
            print(f"❌ GUI dependencies missing: {e}")
            print("💡 Install with: pip install tkinter matplotlib")
    
    elif args.cli or args.image:
        print("💻 Using command-line interface...")
        if args.image:
            test_single_image(args.model, args.image)
        else:
            # Interactive mode
            while True:
                image_path = input("\n📁 Enter image path (or 'quit' to exit): ").strip()
                if image_path.lower() in ['quit', 'exit', 'q']:
                    break
                if os.path.exists(image_path):
                    test_single_image(args.model, image_path)
                else:
                    print(f"❌ File not found: {image_path}")
    
    elif args.batch:
        print(f"📊 Testing all images in: {args.batch}")
        try:
            import test_inference_cli
            tester = test_inference_cli.InferenceTester(args.model)
            tester.test_directory(args.batch, save_visualizations=True)
        except ImportError as e:
            print(f"❌ CLI dependencies missing: {e}")
    
    else:
        print("🎯 Quick Test Options:")
        print("1. 🖼️  GUI Interface:    python test_model.py --gui")
        print("2. 💻 CLI Interface:    python test_model.py --cli") 
        print("3. 📸 Single Image:     python test_model.py --image path/to/image.png")
        print("4. 📊 Batch Test:       python test_model.py --batch path/to/directory")
        print("\n💡 Example:")
        print("   python test_model.py --image ../data/sequence.0/step5.camera.semantic\\ segmentation.png")


def test_single_image(model_path: str, image_path: str):
    """Test a single image with simple visualization."""
    try:
        # Load model
        device = torch.device('cpu')  # Use CPU for simplicity
        checkpoint = torch.load(model_path, map_location=device)
        model = create_model('resnet', num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        prediction = LABEL_MAPPING[pred_class]
        
        # Print results
        print(f"\n🔍 Testing: {os.path.basename(image_path)}")
        print(f"🚗 PREDICTION: {prediction.upper()}")
        print(f"📊 CONFIDENCE: {confidence*100:.1f}%")
        print(f"📈 All probabilities:")
        for i, (label, prob) in enumerate(zip(LABEL_MAPPING.values(), probs[0])):
            marker = "🎯" if i == pred_class else "  "
            print(f"   {marker} {label}: {prob*100:.1f}%")
        
        # Simple visualization
        try:
            colors = {'front': '#4CAF50', 'left': '#2196F3', 'right': '#FF9800'}
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'🚗 RoadNet Prediction', fontsize=14, fontweight='bold')
            
            # Original image
            ax1.imshow(np.array(image))
            ax1.set_title(f'Input: {os.path.basename(image_path)}')
            ax1.axis('off')
            
            # Add prediction overlay
            color = colors.get(prediction, 'gray')
            ax1.text(0.02, 0.98, f'{prediction.upper()}\n{confidence*100:.1f}%', 
                    transform=ax1.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9),
                    color='white')
            
            # Confidence chart
            labels = list(LABEL_MAPPING.values())
            bar_colors = [colors[label] for label in labels]
            bars = ax2.bar(labels, probs[0].numpy() * 100, color=bar_colors, alpha=0.8)
            
            # Highlight prediction
            bars[pred_class].set_alpha(1.0)
            bars[pred_class].set_edgecolor('black')
            bars[pred_class].set_linewidth(2)
            
            ax2.set_title('Confidence Scores')
            ax2.set_ylabel('Confidence (%)')
            ax2.set_ylim(0, 100)
            
            # Add percentage labels
            for bar, prob in zip(bars, probs[0]):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as viz_error:
            print(f"⚠️  Visualization error: {viz_error}")
            print("💡 Results printed above")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return None


if __name__ == "__main__":
    main()

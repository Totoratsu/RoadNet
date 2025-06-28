"""
Batch inference script for testing the model on multiple images with comparison to ground truth.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List

from decision_model import create_model
from decision_dataset import LABEL_MAPPING


def load_model(model_path: str, model_type: str = 'resnet'):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(model_type, num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def batch_inference(model, device, data_dir: str, labels_file: str = None):
    """Run batch inference and compare with ground truth."""
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load labels if provided
    labels_data = {}
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
    
    # Find segmentation mask files
    mask_files = [f for f in os.listdir(data_dir) 
                 if f.endswith('.camera.semantic segmentation.png')]
    
    results = []
    label_names = ['front', 'left', 'right']
    
    print(f"Testing {len(mask_files)} images...")
    
    for filename in sorted(mask_files):
        # Extract step number
        step_num = filename.split('.')[0].replace('step', '')
        
        # Load image
        image_path = os.path.join(data_dir, filename)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        prediction_name = LABEL_MAPPING[predicted_class]
        
        # Get ground truth if available
        true_label = None
        true_label_name = None
        if step_num in labels_data:
            true_label = labels_data[step_num]
            true_label_name = label_names[true_label]
        
        result = {
            'step': int(step_num),
            'filename': filename,
            'prediction': prediction_name,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'true_label': true_label,
            'true_label_name': true_label_name,
            'correct': predicted_class == true_label if true_label is not None else None,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        results.append(result)
        
        # Print progress
        status = ""
        if true_label is not None:
            status = "✓" if result['correct'] else "✗"
        print(f"Step {step_num}: {prediction_name} ({confidence*100:.1f}%) {status}")
    
    return results


def create_analysis_report(results: List[Dict], output_dir: str):
    """Create comprehensive analysis report with visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter results with ground truth
    labeled_results = [r for r in results if r['true_label'] is not None]
    
    if not labeled_results:
        print("No ground truth labels available for analysis")
        return
    
    # Calculate metrics
    y_true = [r['true_label'] for r in labeled_results]
    y_pred = [r['predicted_class'] for r in labeled_results]
    
    accuracy = np.mean([r['correct'] for r in labeled_results])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    label_names = ['FRONT', 'LEFT', 'RIGHT']
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    plt.colorbar(im)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}',
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(range(len(label_names)), label_names)
    plt.yticks(range(len(label_names)), label_names)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Plot confidence distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confidence by correctness
    correct_conf = [r['confidence'] for r in labeled_results if r['correct']]
    incorrect_conf = [r['confidence'] for r in labeled_results if not r['correct']]
    
    ax1.hist([correct_conf, incorrect_conf], bins=20, alpha=0.7, 
             label=['Correct', 'Incorrect'], color=['green', 'red'])
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    
    # Confidence by class
    for i, class_name in enumerate(label_names):
        class_conf = [r['confidence'] for r in labeled_results if r['predicted_class'] == i]
        ax2.hist(class_conf, bins=15, alpha=0.7, label=class_name)
    
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence by Predicted Class')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=150)
    plt.close()
    
    # Print detailed report
    print(f"\n📊 DETAILED ANALYSIS REPORT")
    print("=" * 50)
    print(f"Total images: {len(results)}")
    print(f"Labeled images: {len(labeled_results)}")
    print(f"Overall accuracy: {accuracy*100:.1f}%")
    
    # Per-class analysis
    print(f"\nPer-class performance:")
    for i, class_name in enumerate(label_names):
        class_true = [r for r in labeled_results if r['true_label'] == i]
        class_correct = [r for r in class_true if r['correct']]
        
        if class_true:
            precision = len(class_correct) / len(class_true)
            avg_conf = np.mean([r['confidence'] for r in class_true])
            print(f"  {class_name}: {len(class_correct)}/{len(class_true)} "
                  f"({precision*100:.1f}%, avg conf: {avg_conf*100:.1f}%)")
    
    # Misclassification analysis
    misclassified = [r for r in labeled_results if not r['correct']]
    if misclassified:
        print(f"\nMisclassified samples ({len(misclassified)}):")
        for r in misclassified:
            print(f"  Step {r['step']}: {label_names[r['true_label']]} → "
                  f"{r['prediction']} (conf: {r['confidence']*100:.1f}%)")
    
    # Save detailed results as CSV manually
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    with open(csv_path, 'w') as f:
        # Write header
        f.write('step,filename,prediction,predicted_class,confidence,true_label,true_label_name,correct\n')
        # Write data
        for r in labeled_results:
            f.write(f"{r['step']},{r['filename']},{r['prediction']},{r['predicted_class']},"
                   f"{r['confidence']},{r['true_label']},{r['true_label_name']},{r['correct']}\n")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_dir}")
    print("  - confusion_matrix.png")
    print("  - confidence_analysis.png") 
    print("  - detailed_results.csv")
    print("  - classification_report.json")


def main():
    """Main function for batch analysis."""
    parser = argparse.ArgumentParser(description='Batch inference and analysis')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('data_dir', help='Directory containing test images')
    parser.add_argument('--labels_file', default='driving_labels.json',
                       help='Labels file name (default: driving_labels.json)')
    parser.add_argument('--model_type', choices=['resnet', 'simple'], default='resnet',
                       help='Model type')
    parser.add_argument('--output_dir', default='analysis_results',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.model_path, args.model_type)
    
    # Run batch inference
    labels_path = os.path.join(args.data_dir, args.labels_file)
    results = batch_inference(model, device, args.data_dir, labels_path)
    
    # Create analysis report
    create_analysis_report(results, args.output_dir)
    
    # Save all results
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

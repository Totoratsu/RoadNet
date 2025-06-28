"""
Visualization script to show the effects of different data augmentation levels.
This helps you understand what transformations are being applied to your training data.
"""

import os
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms
from train_decision_model import create_advanced_augmentation_transform
import argparse


def visualize_augmentations(image_path: str, num_samples: int = 6, save_path: str = None):
    """
    Visualize the effects of different augmentation levels on a sample image.
    
    Args:
        image_path: Path to a segmentation mask image
        num_samples: Number of augmented samples to show per level
        save_path: Optional path to save the visualization
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create transforms for different augmentation levels
    transforms_dict = {
        'Original': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        'Light': create_advanced_augmentation_transform((224, 224), 'light'),
        'Medium': create_advanced_augmentation_transform((224, 224), 'medium'),
        'Heavy': create_advanced_augmentation_transform((224, 224), 'heavy')
    }
    
    # Create visualization
    fig, axes = plt.subplots(4, num_samples + 1, figsize=(16, 12))
    fig.suptitle(f'Data Augmentation Effects\nImage: {os.path.basename(image_path)}', fontsize=16)
    
    for row, (aug_level, transform) in enumerate(transforms_dict.items()):
        # Show original in first column
        if aug_level == 'Original':
            for col in range(num_samples + 1):
                tensor = transform(original_image)
                # Denormalize if needed
                if tensor.max() <= 1.0:
                    img_array = tensor.permute(1, 2, 0).numpy()
                else:
                    img_array = tensor.permute(1, 2, 0).numpy()
                axes[row, col].imshow(img_array)
                axes[row, col].set_title(f'{aug_level}')
                axes[row, col].axis('off')
        else:
            # Show label
            axes[row, 0].text(0.5, 0.5, f'{aug_level}\nAugmentation', 
                            ha='center', va='center', fontsize=12, fontweight='bold')
            axes[row, 0].set_xlim(0, 1)
            axes[row, 0].set_ylim(0, 1)
            axes[row, 0].axis('off')
            
            # Show augmented samples
            for col in range(1, num_samples + 1):
                try:
                    tensor = transform(original_image)
                    # Handle normalized tensors
                    if hasattr(transform.transforms[-1], 'mean'):  # Has normalization
                        # Denormalize
                        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                        tensor = tensor * std + mean
                        tensor = torch.clamp(tensor, 0, 1)
                    
                    img_array = tensor.permute(1, 2, 0).numpy()
                    axes[row, col].imshow(img_array)
                    axes[row, col].set_title(f'Sample {col}')
                    axes[row, col].axis('off')
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', 
                                      ha='center', va='center', fontsize=8)
                    axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description='Visualize data augmentation effects')
    parser.add_argument('image_path', help='Path to segmentation mask image')
    parser.add_argument('--samples', type=int, default=6, help='Number of samples per augmentation level')
    parser.add_argument('--save', help='Path to save the visualization')
    
    args = parser.parse_args()
    
    print("🎨 Generating augmentation visualization...")
    print(f"📸 Image: {args.image_path}")
    print(f"📊 Samples per level: {args.samples}")
    
    visualize_augmentations(args.image_path, args.samples, args.save)


if __name__ == "__main__":
    main()

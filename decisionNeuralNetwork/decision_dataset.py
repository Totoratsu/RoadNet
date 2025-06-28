"""
Dataset class for loading segmentation masks and driving decision labels.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms


class DecisionDataset(Dataset):
    """
    Dataset for loading segmentation masks and corresponding driving decisions.
    
    Expected structure:
    - Each sample has a segmentation mask (.png) and a label (0: front, 1: left, 2: right)
    - Labels are stored in a separate JSON file or CSV file
    """
    
    def __init__(
        self, 
        data_dir: str, 
        labels_file: str, 
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        class_specific_augmentation: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing segmentation mask images
            labels_file: Path to JSON file containing labels
            transform: Optional transforms to apply to images
            image_size: Target image size (height, width)
            class_specific_augmentation: Whether to apply different augmentations based on class
        """
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.image_size = image_size
        self.class_specific_augmentation = class_specific_augmentation
        
        # Load labels
        self.labels_data = self._load_labels()
        
        # Get list of available segmentation masks
        self.image_files = self._get_segmentation_files()
        
        # Filter to only include files that have labels
        self.image_files = [f for f in self.image_files if self._extract_step_number(f) is not None]
        
        # Store the base transform
        self.base_transform = transform
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_labels(self) -> Dict:
        """Load labels from JSON file."""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _get_segmentation_files(self) -> List[str]:
        """Get list of segmentation mask files."""
        files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.camera.semantic segmentation.png'):
                files.append(filename)
        return sorted(files)
    
    def _extract_step_number(self, filename: str) -> Optional[int]:
        """Extract step number from filename."""
        try:
            # Extract step number from filename like "step4.camera.semantic segmentation.png"
            step_part = filename.split('.')[0]  # "step4"
            step_number = int(step_part.replace('step', ''))  # 4
            return step_number
        except:
            return None
    
    def _get_class_specific_transform(self, label: int):
        """Get transform based on class label."""
        if not self.class_specific_augmentation or self.base_transform is None:
            return self.transform
        
        # Only augment LEFT (1) and RIGHT (2) classes, not FRONT (0)
        if label == 0:  # FRONT - minimal/no augmentation
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # LEFT (1) or RIGHT (2) - apply augmentation
            return self.base_transform
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
            where label is 0: front, 1: left, 2: right
        """
        # Get image filename
        img_filename = self.image_files[idx]
        step_number = self._extract_step_number(img_filename)
        
        # Load segmentation mask
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        # Get label (default to 0 - front if not labeled)
        label = self.labels_data.get(str(step_number), 0)
        
        # Apply class-specific transform if enabled
        if self.class_specific_augmentation:
            transform = self._get_class_specific_transform(label)
        else:
            transform = self.transform
        
        # Apply transforms
        if transform:
            image = transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples for each class."""
        counts = {'front': 0, 'left': 0, 'right': 0}
        class_names = ['front', 'left', 'right']
        
        for i in range(len(self)):
            _, label = self[i]
            counts[class_names[label]] += 1
            
        return counts
    
    def get_unlabeled_samples(self) -> List[Tuple[int, str]]:
        """Get list of unlabeled samples (step_number, filename)."""
        unlabeled = []
        for filename in self.image_files:
            step_number = self._extract_step_number(filename)
            if step_number is not None and str(step_number) not in self.labels_data:
                unlabeled.append((step_number, filename))
        return unlabeled


# Label mapping
LABEL_MAPPING = {
    0: 'front',
    1: 'left', 
    2: 'right'
}

REVERSE_LABEL_MAPPING = {
    'front': 0,
    'left': 1,
    'right': 2
}

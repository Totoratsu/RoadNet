# Custom Segmentation Dataset for Unity-generated Data
# Adapted for the data in the RoadNet/data folder

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class UnitySegmentationDataset(Dataset):
    """
    Dataset for Unity-generated semantic segmentation data.
    
    Data structure:
    - Images: step{N}.camera.png
    - Masks: step{N}.camera.semantic segmentation.png  
    - Metadata: step{N}.frame_data.json
    """
    
    def __init__(self, 
                 data_dir: str,
                 sequence: str = "sequence.0",
                 image_size: Tuple[int, int] = (256, 512),
                 split: str = "train",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.15,
                 augment: bool = True):
        """
        Initialize Unity segmentation dataset.
        
        Args:
            data_dir: Path to data directory (e.g., "data")
            sequence: Sequence folder name (e.g., "sequence.0") 
            image_size: Target image size (height, width)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation (remaining goes to test)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.sequence_dir = self.data_dir / sequence
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Define class mapping from Unity colors to class IDs
        self.class_mapping = self._create_class_mapping()
        self.num_classes = len(self.class_mapping)
        
        # Get all available files
        self.image_files, self.mask_files, self.metadata_files = self._get_file_lists()
        
        # Split data
        self.indices = self._create_data_split(train_ratio, val_ratio)
        
        # Setup transforms
        self.image_transform = self._get_image_transform()
        self.mask_transform = self._get_mask_transform()
        
        print(f"Unity Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.indices)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Image size: {image_size}")
        print(f"  Augmentation: {self.augment}")
    
    def _create_class_mapping(self) -> Dict:
        """
        Create mapping from Unity pixel colors to class IDs.
        Based on common Unity semantic segmentation classes.
        """
        # Standard Unity semantic segmentation classes
        # These match common driving scenarios
        class_mapping = {
            # Color tuple (R, G, B): (class_id, class_name)
            (255, 255, 255): (0, 'road'),           # White - Road surface
            (192, 183, 77): (1, 'building'),       # Brown - Buildings  
            (83, 21, 168): (2, 'car'),             # Purple - Vehicles
            (255, 0, 0): (3, 'traffic_light'),     # Red - Traffic lights
            (255, 0, 121): (4, 'road_block'),      # Pink - Road blocks/barriers
            (0, 255, 0): (5, 'vegetation'),        # Green - Trees/plants
            (0, 0, 255): (6, 'sky'),               # Blue - Sky
            (255, 255, 0): (7, 'traffic_sign'),    # Yellow - Traffic signs
            (0, 255, 255): (8, 'sidewalk'),        # Cyan - Sidewalks
            (255, 0, 255): (9, 'person'),          # Magenta - Pedestrians
            (128, 128, 128): (10, 'pole'),         # Gray - Poles
            (0, 0, 0): (11, 'background'),         # Black - Background/unknown
        }
        
        return class_mapping
    
    def _get_file_lists(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """Get sorted lists of all data files."""
        
        # Find all step files
        image_files = sorted(self.sequence_dir.glob("step*.camera.png"))
        mask_files = sorted(self.sequence_dir.glob("step*.camera.semantic segmentation.png"))
        metadata_files = sorted(self.sequence_dir.glob("step*.frame_data.json"))
        
        # Ensure we have matching files
        valid_steps = []
        for img_file in image_files:
            step_num = img_file.name.split('.')[0]  # Extract "stepN"
            mask_file = self.sequence_dir / f"{step_num}.camera.semantic segmentation.png"
            meta_file = self.sequence_dir / f"{step_num}.frame_data.json"
            
            if mask_file.exists() and meta_file.exists():
                valid_steps.append(step_num)
        
        # Filter to only valid steps
        image_files = [self.sequence_dir / f"{step}.camera.png" for step in valid_steps]
        mask_files = [self.sequence_dir / f"{step}.camera.semantic segmentation.png" for step in valid_steps]
        metadata_files = [self.sequence_dir / f"{step}.frame_data.json" for step in valid_steps]
        
        print(f"Found {len(valid_steps)} complete data samples")
        return image_files, mask_files, metadata_files
    
    def _create_data_split(self, train_ratio: float, val_ratio: float) -> List[int]:
        """Create data split indices for train/val/test."""
        
        total_samples = len(self.image_files)
        indices = list(range(total_samples))
        
        # Calculate split points
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        if self.split == 'train':
            return indices[:train_end]
        elif self.split == 'val':
            return indices[train_end:val_end]
        else:  # test
            return indices[val_end:]
    
    def _get_image_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        
        transform_list = [
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Add augmentation for training
        if self.augment:
            augment_transforms = [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            # Insert augmentations before normalization
            transform_list = transform_list[:2] + augment_transforms + transform_list[2:]
        
        return transforms.Compose(transform_list)
    
    def _get_mask_transform(self) -> transforms.Compose:
        """Get mask preprocessing transforms."""
        
        transform_list = [
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ]
        
        # Add augmentation for training (only geometric, no color changes)
        if self.augment:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        return transforms.Compose(transform_list)
    
    def _color_to_class_id(self, color_mask: np.ndarray) -> np.ndarray:
        """
        Convert colored mask to class ID mask.
        
        Args:
            color_mask: (H, W, 3) RGB mask
            
        Returns:
            (H, W) class ID mask
        """
        h, w = color_mask.shape[:2]
        class_mask = np.full((h, w), 11, dtype=np.uint8)  # Default to background class
        
        # Convert each color to class ID
        for color, (class_id, _) in self.class_mapping.items():
            # Create mask for this color
            color_match = np.all(color_mask == color, axis=2)
            class_mask[color_match] = class_id
        
        return class_mask
    
    def _load_metadata(self, metadata_file: Path) -> Dict:
        """Load metadata from frame_data.json file."""
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def get_class_info(self) -> Dict:
        """Get information about segmentation classes."""
        class_info = {}
        for color, (class_id, class_name) in self.class_mapping.items():
            class_info[class_id] = {
                'name': class_name,
                'color': color,
                'rgb_hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            }
        return class_info
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Get actual file index
        file_idx = self.indices[idx]
        
        # Load image
        image_path = self.image_files[file_idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_files[file_idx]
        mask_img = Image.open(mask_path).convert('RGB')  # Convert to RGB to remove alpha
        mask_array = np.array(mask_img)
        
        # Convert colored mask to class IDs
        class_mask = self._color_to_class_id(mask_array)
        mask = Image.fromarray(class_mask, mode='L')
        
        # Apply transforms
        # Note: We need to apply the same random transforms to both image and mask
        if self.augment:
            # Create a random state for consistent transforms
            seed = np.random.randint(2147483647)
            
            # Apply to image
            torch.manual_seed(seed)
            image = self.image_transform(image)
            
            # Apply to mask with same random state
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

class UnityDataModule:
    """
    Data module for easy dataset management.
    """
    
    def __init__(self, 
                 data_dir: str = "../data",
                 sequence: str = "sequence.0",
                 batch_size: int = 16,
                 image_size: Tuple[int, int] = (256, 512),
                 num_workers: int = 4):
        """
        Initialize data module.
        
        Args:
            data_dir: Path to data directory
            sequence: Sequence folder name
            batch_size: Training batch size
            image_size: Target image size
            num_workers: Number of data loading workers
        """
        self.data_dir = data_dir
        self.sequence = sequence
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = UnitySegmentationDataset(
            data_dir, sequence, image_size, split='train', augment=True
        )
        self.val_dataset = UnitySegmentationDataset(
            data_dir, sequence, image_size, split='val', augment=False
        )
        self.test_dataset = UnitySegmentationDataset(
            data_dir, sequence, image_size, split='test', augment=False
        )
        
        self.num_classes = self.train_dataset.num_classes
        
    def get_dataloaders(self):
        """Get PyTorch data loaders."""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_info(self):
        """Get class information."""
        return self.train_dataset.get_class_info()

def analyze_dataset(data_dir: str = "../data", sequence: str = "sequence.0"):
    """
    Analyze the Unity dataset to understand class distribution.
    """
    print("🔍 Analyzing Unity segmentation dataset...")
    
    dataset = UnitySegmentationDataset(data_dir, sequence, split='train')
    
    # Analyze a few samples
    class_counts = {}
    sample_count = min(10, len(dataset))
    
    if sample_count == 0:
        print("❌ No samples found! Check your data path.")
        print(f"   Looking in: {Path(data_dir) / sequence}")
        return None
    
    for i in range(sample_count):
        image, mask = dataset[i]
        
        # Count classes in this mask
        unique_classes, counts = torch.unique(mask, return_counts=True)
        
        for class_id, count in zip(unique_classes.numpy(), counts.numpy()):
            if class_id.item() not in class_counts:
                class_counts[class_id.item()] = 0
            class_counts[class_id.item()] += count.item()
    
    # Print analysis
    print(f"\n📊 Dataset Analysis (first {sample_count} samples):")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Image size: {dataset.image_size}")
    print(f"  Number of classes: {dataset.num_classes}")
    
    print(f"\n🎨 Class Distribution:")
    class_info = dataset.get_class_info()
    total_pixels = sum(class_counts.values())
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total_pixels * 100
        class_name = class_info[class_id]['name']
        color = class_info[class_id]['rgb_hex']
        print(f"  {class_id:2d} | {class_name:12s} | {color} | {percentage:6.2f}% ({count:,} pixels)")
    
    return dataset

if __name__ == "__main__":
    # Analyze the dataset
    dataset = analyze_dataset()
    
    if dataset is None or len(dataset) == 0:
        print("❌ Dataset is empty. Please check:")
        print("  1. Data path is correct (../data/sequence.0)")
        print("  2. Files exist: step*.camera.png, step*.camera.semantic segmentation.png")
        print("  3. File naming matches expected pattern")
        exit(1)
    
    # Test data loading
    print(f"\n🧪 Testing data loading...")
    
    data_module = UnityDataModule()
    
    if len(data_module.train_dataset) == 0:
        print("❌ No training data found. Exiting.")
        exit(1)
    
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch
    if len(train_loader) > 0:
        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"  Batch {batch_idx}: images {images.shape}, masks {masks.shape}")
            break
    
    print("✅ Dataset setup successful!")

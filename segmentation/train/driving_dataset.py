import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import random
import torchvision.transforms.functional as TF


class DrivingDataset(Dataset):
    # Color mapping from Unity segmentation
    COLOR_TO_CLASS = {
        (255, 255, 255): 0,  # road 
        (192, 183, 77): 1,   # building
        (83, 21, 168): 2,    # car
        (255, 0, 0): 3,      # traffic_light
        (255, 0, 121): 4,    # road_block
    }
    
    def __init__(self, data_dir, augment=True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        
        # Get all step numbers
        camera_files = list(self.data_dir.glob("step*.camera.png"))
        self.steps = sorted([int(f.stem.split('.')[0][4:]) for f in camera_files])
        
        print(f"Found {len(self.steps)} samples")
        if self.augment:
            print("Data augmentation enabled (rotation + stretch)")
    
    def _apply_augmentation(self, image, mask):
        """Apply same augmentation to both image and mask"""
        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Random horizontal stretch (0.8x to 1.2x)
        if random.random() > 0.5:
            original_size = image.size
            stretch_factor = random.uniform(0.8, 1.2)
            new_width = int(original_size[0] * stretch_factor)

            # Resize with stretch
            image = image.resize((new_width, original_size[1]), Image.BILINEAR)
            mask = mask.resize((new_width, original_size[1]), Image.NEAREST)

            # Crop or pad back to original size
            if new_width > original_size[0]:
                # Crop center
                left = (new_width - original_size[0]) // 2
                image = image.crop((left, 0, left + original_size[0], original_size[1]))
                mask = mask.crop((left, 0, left + original_size[0], original_size[1]))
            elif new_width < original_size[0]:
                # Pad with black
                pad_width = original_size[0] - new_width
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad

                new_image = Image.new('RGB', original_size, (0, 0, 0))
                new_image.paste(image, (left_pad, 0))
                image = new_image

                new_mask = Image.new('L', original_size, 0)
                new_mask.paste(mask, (left_pad, 0))
                mask = new_mask

        # Random crop and resize back to original size
        if random.random() > 0.5:
            crop_scale = random.uniform(0.7, 1.0)  # Crop between 70% and 100% of original size
            orig_w, orig_h = image.size
            crop_w, crop_h = int(orig_w * crop_scale), int(orig_h * crop_scale)
            if crop_w < orig_w and crop_h < orig_h:
                left = random.randint(0, orig_w - crop_w)
                top = random.randint(0, orig_h - crop_h)
                image = image.crop((left, top, left + crop_w, top + crop_h))
                mask = mask.crop((left, top, left + crop_w, top + crop_h))
                # Resize back to original size
                image = image.resize((orig_w, orig_h), Image.BILINEAR)
                mask = mask.resize((orig_w, orig_h), Image.NEAREST)

        return image, mask
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, idx):
        step = self.steps[idx]
        
        # Load image
        img_path = self.data_dir / f"step{step}.camera.png"
        image = Image.open(img_path).convert('RGB')
        
        # Load segmentation mask
        mask_path = self.data_dir / f"step{step}.camera.semantic segmentation.png"
        seg_image = Image.open(mask_path).convert('RGB')
        
        # Convert colors to class indices first
        seg_array = np.array(seg_image)
        mask_array = np.zeros(seg_array.shape[:2], dtype=np.uint8)
        
        for color, class_id in self.COLOR_TO_CLASS.items():
            color_mask = np.all(seg_array == color, axis=2)
            mask_array[color_mask] = class_id
        
        # Convert mask to PIL Image for augmentation
        mask = Image.fromarray(mask_array, mode='L')
        
        # Apply augmentation to both image and mask
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # Normalize using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


if __name__ == "__main__":
    # Test the dataset
    dataset = DrivingDataset("../../data/sequence.0", augment=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique classes in mask: {torch.unique(mask)}")
    
    print("✅ Dataset works!")

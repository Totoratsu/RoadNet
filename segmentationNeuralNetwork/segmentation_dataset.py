import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_dir, mask_dir, image_label, mask_label, split='train'):
        self.root_dir = root_dir
        self.split = split

        # These lists are going to be filled with the paths of the files
        self.images = []
        self.masks = []

        # Explore image_dir and mask_dir from root_dir to get files paths
        img_dir = os.path.join(root_dir, image_dir, split)
        mask_dir = os.path.join(root_dir, mask_dir, split)


        for folder in os.listdir(img_dir):
            folder_img_dir = os.path.join(img_dir, folder)
            folder_mask_dir = os.path.join(mask_dir, folder)
            
            for img_name in os.listdir(folder_img_dir):
                if img_name.endswith('.png'):
                    # Get the corresponding mask for the normal image
                    img_path = os.path.join(folder_img_dir, img_name)
                    mask_name = img_name.replace(image_label, mask_label) 
                    mask_path = os.path.join(folder_mask_dir, mask_name)
                    
                    # Ensure the mask for the current image exists 
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
                    else:
                        print(f"Warning: Mask not found for {img_path}. Skipping.")

        print(f"Found {len(self.images)} images for {split} split.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Open images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) 
        
        # Transform Images
        image = TRANSFORM_IMG(image)
        mask = TRANSFORM_IMG(mask)
        mask = mask.squeeze(0)

        return image, mask.long()
    

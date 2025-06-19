from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
from dotenv import load_dotenv
from unet import model
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os

load_dotenv()

# DATA
CITYSCAPES_ROOT = os.getenv("CITYSCAPES_ROOT")
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
IMAGE_SIZE = (256, 512)

train_dataset = SegmentationDataset(
    root_dir=CITYSCAPES_ROOT,
    image_dir="leftImg8bit",
    mask_dir="gtCoarse",
    image_label="leftImg8bit",
    mask_label="gtCoarse_labelIds",
    split="train"
)

val_dataset = SegmentationDataset(
    root_dir=CITYSCAPES_ROOT,
    image_dir="leftImg8bit",
    mask_dir="gtCoarse",
    image_label="leftImg8bit",
    mask_label="gtCoarse_labelIds",
    split="val"
)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

# LOSS
l = nn.CrossEntropyLoss()

# OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

# TRAIN
model.train(True)
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}")

    epoch_loss = 0
    for images, masks in train_loader:
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # Backward
        loss = l(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() # Metric
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Average Train Loss : {avg_loss}")

# SAVE
now = datetime.now()
suffix = now.strftime("%Y_%m_%d_%H_%M_%S")
torch.save(model.state_dict(), f"model_cityscapes_{suffix}.pt")
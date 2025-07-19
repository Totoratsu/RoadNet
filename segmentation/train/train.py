import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.models as models
from driving_dataset import DrivingDataset


# --- U-Net with ResNet18 encoder ---
class UNetResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()  # Initialize the parent class
        resnet = models.resnet18(pretrained=True)
        # Encoder layers
        self.input_block = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Decoder layers (upsample + skip connection)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x0 = self.input_block(x)      # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)         # (B, 64, H/4, W/4)
        x2 = self.encoder1(x1)        # (B, 64, H/4, W/4)
        x3 = self.encoder2(x2)        # (B, 128, H/8, W/8)
        x4 = self.encoder3(x3)        # (B, 256, H/16, W/16)
        x5 = self.encoder4(x4)        # (B, 512, H/32, W/32)

        # Decoder with skip connections (resize skip if needed)
        def match_size(src, target):
            if src.shape[2:] != target.shape[2:]:
                src = torch.nn.functional.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)
            return src

        d4 = self.up4(x5)             # (B, 256, H/16, W/16)
        x4m = match_size(x4, d4)
        d4 = torch.cat([d4, x4m], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)             # (B, 128, H/8, W/8)
        x3m = match_size(x3, d3)
        d3 = torch.cat([d3, x3m], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)             # (B, 64, H/4, W/4)
        x2m = match_size(x2, d2)
        d2 = torch.cat([d2, x2m], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)             # (B, 64, H/2, W/2)
        x0m = match_size(x0, d1)
        d1 = torch.cat([d1, x0m], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        # Upsample to input size
        out = torch.nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

def train_model():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader (resolve data path relative to this script)
    # Determine project root (two levels up from segmentation/train)
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / 'data' / 'sequence.0'
    train_dataset_full = DrivingDataset(data_dir, augment=True)
    val_dataset_full = DrivingDataset(data_dir, augment=False)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    
    # Use the same indices for both datasets to ensure same split
    torch.manual_seed(42)  # For reproducible splits
    indices = torch.randperm(len(train_dataset_full))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = UNetResNet18(num_classes=6).to(device)  # Update to 6 classes

    # Loss and optimizer
    # Ignore unknown/unlabeled class (index 5) during loss computation
    criterion = nn.CrossEntropyLoss(ignore_index=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(outputs, dim=1)
                correct_pixels += (predicted == masks).sum().item()
                total_pixels += masks.numel()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_pixels / total_pixels

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("-" * 50)

    # Save model
    torch.save(model.state_dict(), '../checkpoints/driving_segmentation_model.pth')
    print("Model saved as '../checkpoints/driving_segmentation_model.pth'")


if __name__ == "__main__":
    train_model()

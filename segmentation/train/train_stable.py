import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
from driving_dataset import DrivingDataset


def train_stable_model():
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve data directory (same as train.py)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data' / 'sequence.0'

    # Create datasets and dataloaders
    train_dataset = DrivingDataset(data_dir, augment=True)
    val_dataset = DrivingDataset(data_dir, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model, loss, optimizer
    # use a simple stable Unet with MobileNetV2 encoder pretrained on ImageNet
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
        activation=None
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                running_val += criterion(outputs, masks).item()
        avg_val_loss = running_val / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save final model
    save_dir = project_root / 'checkpoints_stable'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / 'driving_segmentation_stable.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_stable_model()

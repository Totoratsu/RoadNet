import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from driving_dataset import DrivingDataset

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Use ResNet18 as backbone
        backbone = models.resnet18(pretrained=True)
        
        # Remove the final classification layers
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.decoder(features)
        
        # Resize output to match input size
        output = torch.nn.functional.interpolate(
            output, 
            size=(x.shape[2], x.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )
        
        return output

def train_model():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    train_dataset_full = DrivingDataset("../../data/sequence.0", augment=True)
    val_dataset_full = DrivingDataset("../../data/sequence.0", augment=False)
    
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
    model = SegmentationModel(num_classes=5).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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

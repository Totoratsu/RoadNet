# Unity-optimized UNet Model
# Adapted for Unity-generated semantic segmentation data

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

class UnityUNet:
    """
    UNet model factory for Unity semantic segmentation data.
    Supports different encoders optimized for real-time inference.
    """
    
    def __init__(self, num_classes: int = 12, encoder: str = "resnet34"):
        """
        Initialize Unity UNet model.
        
        Args:
            num_classes: Number of segmentation classes (default 12 for Unity data)
            encoder: Encoder backbone architecture
        """
        self.num_classes = num_classes
        self.encoder = encoder
        
        # Define available encoders with their characteristics
        self.encoder_specs = {
            # Speed-optimized encoders
            'mobilenet_v2': {
                'speed': 'fastest',
                'memory': 'low',
                'accuracy': 'medium',
                'params': '~3M'
            },
            'efficientnet-b0': {
                'speed': 'fast', 
                'memory': 'low',
                'accuracy': 'good',
                'params': '~5M'
            },
            'efficientnet-b3': {
                'speed': 'medium',
                'memory': 'medium',
                'accuracy': 'very good',
                'params': '~12M'
            },
            # Balanced encoders
            'resnet18': {
                'speed': 'medium',
                'memory': 'medium', 
                'accuracy': 'good',
                'params': '~11M'
            },
            'resnet34': {
                'speed': 'medium',
                'memory': 'medium',
                'accuracy': 'high',
                'params': '~21M'
            },
            # Quality-optimized encoders
            'resnet50': {
                'speed': 'slow',
                'memory': 'high',
                'accuracy': 'highest', 
                'params': '~23M'
            },
            'resnext50_32x4d': {
                'speed': 'slow',
                'memory': 'high',
                'accuracy': 'highest',
                'params': '~25M'
            }
        }
    
    def create_model(self, pretrained: bool = True) -> nn.Module:
        """
        Create UNet model for Unity data.
        
        Args:
            pretrained: Whether to use ImageNet pretrained encoder
            
        Returns:
            UNet model
        """
        encoder_weights = "imagenet" if pretrained else None
        
        model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self.num_classes,
            activation=None  # We'll use CrossEntropyLoss
        )
        
        return model
    
    def create_realtime_model(self) -> nn.Module:
        """Create model optimized for real-time inference."""
        return UnityUNet(self.num_classes, 'mobilenet_v2').create_model()
    
    def create_balanced_model(self) -> nn.Module:
        """Create model with balanced speed/accuracy."""
        return UnityUNet(self.num_classes, 'efficientnet-b0').create_model()
    
    def create_quality_model(self) -> nn.Module:
        """Create model optimized for best accuracy."""
        return UnityUNet(self.num_classes, 'resnet50').create_model()
    
    def get_encoder_info(self) -> dict:
        """Get information about the current encoder."""
        return self.encoder_specs.get(self.encoder, {})
    
    def list_available_encoders(self):
        """Print available encoders and their characteristics."""
        print("🏗️  Available UNet Encoders for Unity Data:")
        print("=" * 60)
        
        for encoder, specs in self.encoder_specs.items():
            print(f"{encoder:20s} | {specs['speed']:8s} | {specs['accuracy']:8s} | {specs['params']}")
        
        print("=" * 60)
        print("Speed: fastest -> slowest")
        print("Accuracy: medium -> highest")

# Unity-specific class definitions
UNITY_CLASSES = {
    0: {'name': 'road', 'color': (255, 255, 255), 'description': 'Driveable road surface'},
    1: {'name': 'building', 'color': (192, 183, 77), 'description': 'Buildings and structures'},
    2: {'name': 'car', 'color': (83, 21, 168), 'description': 'Vehicles and cars'},
    3: {'name': 'traffic_light', 'color': (255, 0, 0), 'description': 'Traffic lights'},
    4: {'name': 'road_block', 'color': (255, 0, 121), 'description': 'Road blocks and barriers'},
    5: {'name': 'vegetation', 'color': (0, 255, 0), 'description': 'Trees and plants'},
    6: {'name': 'sky', 'color': (0, 0, 255), 'description': 'Sky area'},
    7: {'name': 'traffic_sign', 'color': (255, 255, 0), 'description': 'Traffic signs'},
    8: {'name': 'sidewalk', 'color': (0, 255, 255), 'description': 'Sidewalks and walkways'},
    9: {'name': 'person', 'color': (255, 0, 255), 'description': 'Pedestrians'},
    10: {'name': 'pole', 'color': (128, 128, 128), 'description': 'Poles and posts'},
    11: {'name': 'background', 'color': (0, 0, 0), 'description': 'Background/unknown'}
}

NUM_CLASSES = len(UNITY_CLASSES)

def create_unity_model(encoder: str = "resnet34", optimization: str = "balanced") -> nn.Module:
    """
    Factory function to create Unity-optimized segmentation models.
    
    Args:
        encoder: Encoder architecture
        optimization: 'speed', 'balanced', or 'quality'
        
    Returns:
        Configured UNet model
    """
    
    # Select encoder based on optimization preference
    if optimization == "speed":
        recommended_encoder = "mobilenet_v2"
    elif optimization == "balanced": 
        recommended_encoder = "efficientnet-b0"
    elif optimization == "quality":
        recommended_encoder = "resnet50"
    elif optimization == "accuracy":
        recommended_encoder = "resnext50_32x4d"  # Best accuracy model
    else:
        recommended_encoder = encoder
    
    # Use provided encoder or recommendation
    final_encoder = encoder if encoder != "resnet34" else recommended_encoder
    
    print(f"🏗️  Creating Unity UNet model:")
    print(f"   Encoder: {final_encoder}")
    print(f"   Classes: {NUM_CLASSES}")
    print(f"   Optimization: {optimization}")
    
    unity_unet = UnityUNet(NUM_CLASSES, final_encoder)
    model = unity_unet.create_model()
    
    # Print model info
    encoder_info = unity_unet.get_encoder_info()
    if encoder_info:
        print(f"   Speed: {encoder_info['speed']}")
        print(f"   Accuracy: {encoder_info['accuracy']}")
        print(f"   Parameters: {encoder_info['params']}")
    
    return model

def get_unity_class_weights(dataset_path: str = "data/sequence.0") -> torch.Tensor:
    """
    Calculate class weights for Unity dataset to handle class imbalance.
    
    Args:
        dataset_path: Path to Unity dataset
        
    Returns:
        Class weights tensor
    """
    try:
        from unity_dataset import UnitySegmentationDataset
        import numpy as np
        
        # Create dataset to analyze
        dataset = UnitySegmentationDataset(
            data_dir="data",
            sequence="sequence.0",
            split="train"
        )
        
        # Count class frequencies
        class_counts = np.zeros(NUM_CLASSES)
        sample_size = min(50, len(dataset))  # Sample for efficiency
        
        for i in range(sample_size):
            _, mask = dataset[i]
            unique, counts = torch.unique(mask, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                if 0 <= class_id < NUM_CLASSES:
                    class_counts[class_id] += count.item()
        
        # Calculate inverse frequency weights
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (NUM_CLASSES * class_counts + 1e-6)  # Avoid division by zero
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        
        return torch.FloatTensor(class_weights)
        
    except Exception as e:
        print(f"⚠️  Could not calculate class weights: {e}")
        print("Using uniform weights")
        return torch.ones(NUM_CLASSES)

# Legacy compatibility - create model with original interface
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet" 
# NUM_CLASSES defined above

def create_legacy_model():
    """Create model with original interface for backward compatibility."""
    return create_unity_model(ENCODER, "balanced")

# Create the model (for backward compatibility)
model = create_legacy_model()

if __name__ == "__main__":
    # Demonstrate different model configurations
    print("🚀 Unity UNet Model Configurations\n")
    
    # Show available encoders
    unity_unet = UnityUNet()
    unity_unet.list_available_encoders()
    
    print(f"\n🎯 Unity Class Definitions ({NUM_CLASSES} classes):")
    for class_id, info in UNITY_CLASSES.items():
        color_hex = f"#{info['color'][0]:02x}{info['color'][1]:02x}{info['color'][2]:02x}"
        print(f"  {class_id:2d} | {info['name']:12s} | {color_hex} | {info['description']}")
    
    print(f"\n🏗️  Creating different model configurations:")
    
    # Speed-optimized model
    speed_model = create_unity_model(optimization="speed")
    speed_params = sum(p.numel() for p in speed_model.parameters())
    print(f"   Speed model parameters: {speed_params:,}")
    
    # Balanced model  
    balanced_model = create_unity_model(optimization="balanced")
    balanced_params = sum(p.numel() for p in balanced_model.parameters())
    print(f"   Balanced model parameters: {balanced_params:,}")
    
    # Quality model
    quality_model = create_unity_model(optimization="quality") 
    quality_params = sum(p.numel() for p in quality_model.parameters())
    print(f"   Quality model parameters: {quality_params:,}")
    
    print(f"\n💡 Recommendations:")
    print(f"   🏃‍♂️ Real-time driving: Use speed model (MobileNetV2)")
    print(f"   ⚖️  Balanced performance: Use balanced model (EfficientNet-B0)")
    print(f"   🎯 Best accuracy: Use quality model (ResNet50)")
    
    # Calculate class weights
    print(f"\n⚖️  Calculating class weights...")
    class_weights = get_unity_class_weights()
    print(f"   Class weights shape: {class_weights.shape}")
    print(f"   Min weight: {class_weights.min():.3f}")
    print(f"   Max weight: {class_weights.max():.3f}")

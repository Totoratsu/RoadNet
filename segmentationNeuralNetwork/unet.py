import segmentation_models_pytorch

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 255

model = segmentation_models_pytorch.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None # CrossEntropyLoss
)
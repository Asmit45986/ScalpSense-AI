import torch
import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def get_scalp_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Loads MobileNetV2 and updates the final classifier head for our scalp disorder classes.
    Optimized for low VRAM usage on the RTX 3050.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Customize classifier head with dropout to prevent overfitting
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model
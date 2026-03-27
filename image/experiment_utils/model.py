import torch.nn as nn
import timm
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# ResNet-50
# ============================================================
def get_resnet50(num_classes: int, pretrained: bool = True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)
    return model


# ============================================================
# EfficientNet-B0
# ============================================================
def get_efficientnet_b0(num_classes: int, pretrained: bool = True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_dim = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_dim, num_classes)
    return model

# ============================================================
# RepVGG-B0
# ============================================================
def get_repvgg_b0(num_classes, pretrained=True):
    model = timm.create_model(
        "repvgg_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model
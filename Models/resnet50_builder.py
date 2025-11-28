# Models/resnet50_builder.py
import torch.nn as nn
import torchvision.models as models

def get_activation(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()
    return nn.ReLU(inplace=True)

def build_resnet50(num_classes=3, dense_layers=1, activation="ReLU", pretrained=True, freeze_backbone=True, dropout=0.5):
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=pretrained)

    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    act = get_activation(activation)

    if dense_layers == 1:
        model.fc = nn.Linear(in_features, num_classes)
    elif dense_layers == 2:
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    elif dense_layers == 3:
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            act,
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model

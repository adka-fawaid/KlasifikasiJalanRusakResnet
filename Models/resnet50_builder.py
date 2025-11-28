# models/resnet50_builder.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        try:
            # swish approximation using SiLU
            return nn.SiLU()
        except:
            return nn.ReLU(inplace=True)
    return nn.ReLU(inplace=True)

def build_resnet50(num_classes=3, dense_layers=1, activation="ReLU", pretrained=True, freeze_backbone=True, dropout=0.5):
    # load model
    if pretrained:
        # newest torchvision uses weights enum; to keep compatibility fallback:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)

    # optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = False

    # original fc in resnet50: in_features = 2048
    in_features = model.fc.in_features

    act = get_activation(activation)
    layers = []
    if dense_layers == 1:
        layers = [nn.Linear(in_features, num_classes)]
    elif dense_layers == 2:
        layers = [
            nn.Linear(in_features, 512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        ]
    elif dense_layers == 3:
        layers = [
            nn.Linear(in_features, 512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            act,
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        ]
    else:
        # fallback to single fc
        layers = [nn.Linear(in_features, num_classes)]

    model.fc = nn.Sequential(*layers)
    return model

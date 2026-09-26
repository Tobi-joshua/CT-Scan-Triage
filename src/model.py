from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class TinyCXRNet(nn.Module):
    """Small CNN used for the PneumoniaMNIST proof-of-concept deployment."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def build_model(num_classes: int = 2, dropout: float = 0.25, pretrained: bool = True, architecture: str = "mobilenet_v3_small") -> nn.Module:
    if architecture == "tiny_cxr":
        return TinyCXRNet(num_classes=num_classes, dropout=dropout)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256), nn.Hardswish(), nn.Dropout(dropout), nn.Linear(256, num_classes),
    )
    return model

def gradcam_target_layer(model: nn.Module, architecture: str):
    if architecture == "tiny_cxr":
        return model.features[6]
    return model.features[-1]

def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.inference_mode()
def mc_predict(model: nn.Module, x: torch.Tensor, passes: int = 20):
    model.eval(); enable_mc_dropout(model)
    samples = torch.stack([torch.softmax(model(x), dim=1) for _ in range(passes)])
    mean = samples.mean(0)
    variance = samples.var(0, unbiased=False)
    entropy = -(mean.clamp_min(1e-8) * mean.clamp_min(1e-8).log()).sum(1)
    sample_entropy = -(samples.clamp_min(1e-8) * samples.clamp_min(1e-8).log()).sum(2).mean(0)
    mutual_information = entropy - sample_entropy
    return mean, variance, entropy, mutual_information

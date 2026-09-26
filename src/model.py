from __future__ import annotations
import torch
from torch import nn
from torchvision import models

def build_model(num_classes: int = 2, dropout: float = 0.25, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model

def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.inference_mode()
def mc_predict(model: nn.Module, x: torch.Tensor, passes: int = 20):
    """Multi-class Monte Carlo Dropout using softmax probabilities."""
    model.eval()
    enable_mc_dropout(model)
    probs = []
    for _ in range(passes):
        probs.append(torch.softmax(model(x), dim=1))
    samples = torch.stack(probs)
    mean = samples.mean(0)
    variance = samples.var(0, unbiased=False)
    entropy = -(mean.clamp_min(1e-8) * mean.clamp_min(1e-8).log()).sum(1)
    sample_entropy = -(samples.clamp_min(1e-8) * samples.clamp_min(1e-8).log()).sum(2).mean(0)
    mutual_information = entropy - sample_entropy
    return mean, variance, entropy, mutual_information

@torch.inference_mode()
def mc_predict_multilabel(model: nn.Module, x: torch.Tensor, passes: int = 20):
    """Multi-label Monte Carlo Dropout using independent sigmoid outputs."""
    model.eval()
    enable_mc_dropout(model)
    probs = []
    for _ in range(passes):
        probs.append(torch.sigmoid(model(x)))
    samples = torch.stack(probs)
    mean = samples.mean(0)
    variance = samples.var(0, unbiased=False)
    p = mean.clamp(1e-8, 1 - 1e-8)
    entropy = -(p * p.log() + (1-p) * (1-p).log())
    sp = samples.clamp(1e-8, 1 - 1e-8)
    sample_entropy = -(sp * sp.log() + (1-sp) * (1-sp).log()).mean(0)
    mutual_information = entropy - sample_entropy
    return mean, variance, entropy, mutual_information

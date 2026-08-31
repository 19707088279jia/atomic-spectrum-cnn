from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mixture_model(num_classes: int = 6, pretrained: bool = False) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def initialize_model_for_training(model: nn.Module, device: str | None = None) -> nn.Module:
    if device is None or device == "cpu":
        return model
    return model


__all__ = ["build_mixture_model"]

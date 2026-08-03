"""CNN model definitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleSpectrumCNN(nn.Module):
    """A compact CNN suitable for quick CPU smoke tests."""

    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 16)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def create_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    """Create a configured classifier."""
    model_config = config["model"]
    name = str(model_config["name"]).lower()
    dropout = float(model_config.get("dropout", 0.25))

    if name == "simple_cnn":
        return SimpleSpectrumCNN(num_classes=num_classes, dropout=dropout)

    if name == "resnet18":
        use_pretrained = bool(model_config.get("pretrained", False))
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        try:
            model = resnet18(weights=weights)
        except Exception as exc:
            if not use_pretrained:
                raise
            raise RuntimeError(
                "Could not download/load ResNet18 pretrained weights. "
                "Set model.pretrained to false for offline training."
            ) from exc
        input_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, num_classes))
        return model

    raise ValueError(f"Unsupported model name: {name}. Use simple_cnn or resnet18.")

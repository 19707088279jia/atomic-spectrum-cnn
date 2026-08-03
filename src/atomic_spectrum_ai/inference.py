"""Checkpoint loading and image prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .dataset import build_transforms
from .model import create_model
from .utils import resolve_device


def load_model_checkpoint(
    checkpoint_path: str | Path, requested_device: str = "auto"
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    """Load a model and checkpoint metadata."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = resolve_device(requested_device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    required = {"model_state_dict", "model_name", "class_names", "image_size"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

    config = checkpoint.get("config") or {
        "data": {
            "image_size": int(checkpoint["image_size"]),
            "classes": checkpoint["class_names"],
        },
        "model": {
            "name": checkpoint["model_name"],
            "pretrained": False,
            "dropout": checkpoint.get("dropout", 0.25),
        },
    }
    # Never request network downloads when reconstructing a saved model.
    config["model"]["pretrained"] = False
    model = create_model(config, num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    """Predict top-k classes for a PIL image."""
    class_names = list(checkpoint["class_names"])
    top_k = max(1, min(top_k, len(class_names)))
    config = checkpoint.get("config") or {
        "data": {"image_size": checkpoint["image_size"]},
        "augmentation": {},
    }
    transform = build_transforms(config, training=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = probabilities.topk(top_k)

    return [
        {
            "element": class_names[int(index)],
            "probability": float(value),
        }
        for value, index in zip(values.cpu(), indices.cpu(), strict=True)
    ]

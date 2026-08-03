"""Standalone checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from .dataset import build_dataloaders
from .inference import load_model_checkpoint
from .metrics import calculate_metrics, save_confusion_matrix
from .training import run_epoch
from .utils import save_json


def evaluate_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    """Evaluate a saved checkpoint against the configured test split."""
    loaders, class_names = build_dataloaders(config)
    requested_device = str(config["training"].get("device", "auto"))
    model, checkpoint, device = load_model_checkpoint(checkpoint_path, requested_device)

    if list(checkpoint["class_names"]) != class_names:
        raise ValueError(
            f"Checkpoint classes {checkpoint['class_names']} do not match dataset {class_names}"
        )

    loss, targets, predictions = run_epoch(
        model, loaders["test"], nn.CrossEntropyLoss(), device
    )
    metrics = calculate_metrics(targets, predictions, class_names)
    metrics["test_loss"] = loss
    metrics["device"] = str(device)

    output_root = Path(config["output"]["root"])
    save_json(metrics, output_root / "evaluation_metrics.json")
    save_confusion_matrix(
        targets,
        predictions,
        class_names,
        output_root / "evaluation_confusion_matrix.png",
    )
    return metrics

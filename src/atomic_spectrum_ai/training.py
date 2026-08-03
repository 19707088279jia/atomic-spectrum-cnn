"""Model training and evaluation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_dataloaders
from .metrics import calculate_metrics, save_confusion_matrix
from .model import create_model
from .utils import resolve_device, save_json, set_seed


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    """Run one train or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        predictions = logits.argmax(dim=1)
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    return average_loss, targets_all, predictions_all


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    """Save loss and accuracy curves."""
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["validation_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["validation_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="train")
    axes[1].plot(epochs, val_accuracy, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Train a model from configuration and save all outputs."""
    seed = int(config["project"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    loaders, class_names = build_dataloaders(config)

    model = create_model(config, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "best_model.pt"

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("early_stopping_patience", epochs))
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_targets, validation_predictions = run_epoch(
            model, loaders["validation"], criterion, device
        )

        train_metrics = calculate_metrics(train_targets, train_predictions, class_names)
        validation_metrics = calculate_metrics(
            validation_targets, validation_predictions, class_names
        )
        entry = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(entry)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config["model"]["name"],
                    "class_names": class_names,
                    "image_size": int(config["data"]["image_size"]),
                    "dropout": float(config["model"].get("dropout", 0.25)),
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    save_json(history, output_root / "history.json")
    save_json(class_names, output_root / "class_names.json")
    save_training_curves(history, output_root / "training_curves.png")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_targets, test_predictions = run_epoch(
        model, loaders["test"], criterion, device
    )
    test_metrics = calculate_metrics(test_targets, test_predictions, class_names)
    test_metrics["test_loss"] = test_loss
    test_metrics["device"] = str(device)
    test_metrics["checkpoint"] = str(checkpoint_path)
    save_json(test_metrics, output_root / "metrics.json")
    save_confusion_matrix(
        test_targets,
        test_predictions,
        class_names,
        output_root / "confusion_matrix.png",
    )
    return test_metrics

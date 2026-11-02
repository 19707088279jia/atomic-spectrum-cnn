#!/usr/bin/env python
"""Train a full-spectrum Pb presence/absence CNN on fixed Z-903 train/val splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from atomic_spectrum_ai.models.z903_cnn import Z903CNN
from atomic_spectrum_ai.z903 import TARGETS, Z903Dataset

ROOT = Path(__file__).resolve().parents[1]
PB_INDEX = TARGETS.index("Pb")
PB_THRESHOLD_PPM = 10.0
CLASSIFICATION_THRESHOLD = 0.5
WAVELENGTH_RANGE_NM = (180.0, 960.0)
EXPECTED_POINTS = 23401


class PbDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Select the existing Pb label and mask while retaining full spectra."""

    def __init__(self, manifest_path: Path) -> None:
        self.dataset = Z903Dataset(manifest_path, preprocess="robust")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        spectrum, raw_labels, masks = self.dataset[index]
        label = (raw_labels[PB_INDEX] > PB_THRESHOLD_PPM).to(dtype=torch.float32).view(1)
        mask = masks[PB_INDEX].view(1)
        return spectrum, label, mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def label_counts(dataset: PbDataset) -> dict[str, int]:
    frame = dataset.dataset.manifest
    raw = pd.to_numeric(frame["Pb_raw"], errors="coerce")
    known = pd.to_numeric(frame["Pb_mask"], errors="coerce").fillna(0) > 0
    positive = int((known & (raw > PB_THRESHOLD_PPM)).sum())
    negative = int((known & (raw <= PB_THRESHOLD_PPM)).sum())
    return {"positive": positive, "negative": negative, "missing": int((~known).sum())}


def masked_bce(logits: Tensor, labels: Tensor, masks: Tensor, pos_weight: Tensor | None = None) -> Tensor:
    known = masks > 0
    if not torch.any(known):
        raise ValueError("The batch contains no known Pb labels")
    return F.binary_cross_entropy_with_logits(
        logits[known], labels[known], pos_weight=pos_weight
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_total = 0.0
    correct = 0
    known_total = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for spectra, labels, masks in loader:
            spectra, labels, masks = spectra.to(device), labels.to(device), masks.to(device)
            logits = model(spectra)
            known = masks > 0
            if not torch.any(known):
                continue
            batch_count = int(known.sum().item())
            batch_loss = F.binary_cross_entropy_with_logits(logits[known], labels[known])
            probabilities = torch.sigmoid(logits[known])
            loss_total += float(batch_loss.item()) * batch_count
            correct += int(((probabilities >= CLASSIFICATION_THRESHOLD) == (labels[known] > 0.5)).sum().item())
            known_total += batch_count
            all_labels.append(labels[known].cpu().numpy().astype(int))
            all_probabilities.append(probabilities.cpu().numpy())
    if known_total == 0:
        raise ValueError("The evaluation split contains no known Pb labels")
    return (
        loss_total / known_total,
        correct / known_total,
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def create_run_directory(base_dir: Path) -> Path:
    """Create a unique run folder and never overwrite earlier results."""
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def assert_split_isolation(datasets: dict[str, PbDataset]) -> None:
    """Reject any reused target or group identity across the supplied splits."""
    for column in ("target_id", "group_id"):
        if not all(column in dataset.dataset.manifest.columns for dataset in datasets.values()):
            continue
        identities = {
            split: set(dataset.dataset.manifest[column].dropna().astype(str))
            for split, dataset in datasets.items()
        }
        split_names = list(identities)
        for index, first in enumerate(split_names):
            for second in split_names[index + 1 :]:
                overlap = identities[first] & identities[second]
                if overlap:
                    examples = sorted(overlap)[:5]
                    raise ValueError(
                        f"{column} leakage between {first} and {second}: {examples}"
                    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    actual_positive = labels == 1
    actual_negative = labels == 0
    tp = int(np.sum(actual_positive & predictions))
    tn = int(np.sum(actual_negative & ~predictions))
    fp = int(np.sum(actual_negative & predictions))
    fn = int(np.sum(actual_positive & ~predictions))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    roc_auc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "accuracy_formula": "(TP + TN) / (TP + TN + FP + FN)",
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "known_samples": total,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    train_dataset = PbDataset(ROOT / "data/processed/z903_train.csv")
    val_dataset = PbDataset(ROOT / "data/processed/z903_val.csv")
    test_dataset = PbDataset(ROOT / "data/processed/z903_test.csv")
    datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    assert_split_isolation(datasets)
    counts = {"train": label_counts(train_dataset), "validation": label_counts(val_dataset)}
    for split_name, split_counts in counts.items():
        print(
            f"{split_name}: Pb-positive={split_counts['positive']}, "
            f"Pb-negative={split_counts['negative']}, missing={split_counts['missing']}"
        )

    output_dir = create_run_directory(ROOT / "reports" / "pb_presence_absence")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = Z903CNN(output_size=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        threshold=1e-4,
        min_lr=1e-6,
    )
    train_counts = counts["train"]
    positive_weight = torch.tensor(
        [train_counts["negative"] / max(1, train_counts["positive"])],
        dtype=torch.float32,
        device=device,
    )
    checkpoint_path = output_dir / "pb_full_spectrum_best.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_known_total = 0
        for spectra, labels, masks in train_loader:
            spectra, labels, masks = spectra.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(spectra)
            known = masks > 0
            if not torch.any(known):
                continue
            loss = masked_bce(logits, labels, masks, pos_weight=positive_weight)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                metric_logits = model(spectra)
                metric_probabilities = torch.sigmoid(metric_logits[known])
                batch_count = int(known.sum().item())
                train_loss_total += float(
                    F.binary_cross_entropy_with_logits(
                        metric_logits[known], labels[known]
                    ).item()
                ) * batch_count
                train_correct += int(
                    ((metric_probabilities >= CLASSIFICATION_THRESHOLD) == (labels[known] > 0.5))
                    .sum()
                    .item()
                )
                train_known_total += batch_count
            model.train()
        if train_known_total == 0:
            raise ValueError("The training split contains no known Pb labels")
        train_loss = train_loss_total / train_known_total
        train_accuracy = train_correct / train_known_total
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch:04d} train_loss={train_loss:.6f} "
            f"validation_loss={val_loss:.6f} train_accuracy={train_accuracy:.4f} "
            f"validation_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "Z903CNN",
                    "target": "Pb",
                    "label_definition": "Pb_raw > 10 ppm; missing Pb labels excluded by mask",
                    "input_shape": [1, 1, EXPECTED_POINTS],
                    "wavelength_range_nm": list(WAVELENGTH_RANGE_NM),
                    "preprocess": "robust",
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "classification_threshold": CLASSIFICATION_THRESHOLD,
                    "seed": args.seed,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_accuracy, val_labels, val_probabilities = evaluate(model, val_loader, device)
    test_loss, test_accuracy, test_labels, test_probabilities = evaluate(model, test_loader, device)
    metrics = {
        "validation": classification_metrics(val_labels, val_probabilities),
        "test": classification_metrics(test_labels, test_probabilities),
    }
    metrics["training_seconds"] = time.perf_counter() - started
    metrics.update(
        {
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "dataset_counts": counts,
            "test_counts": label_counts(test_dataset),
            "threshold": CLASSIFICATION_THRESHOLD,
            "label_definition": "Pb_raw > 10 ppm; missing Pb labels excluded by mask",
            "input_wavelength_range_nm": list(WAVELENGTH_RANGE_NM),
            "input_points": EXPECTED_POINTS,
            "preprocessing": "existing robust median/IQR scaling",
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "report_directory": str(output_dir.relative_to(ROOT)),
            "early_stopping_patience": args.patience,
            "learning_rate_patience": args.lr_patience,
            "seed": args.seed,
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame([metrics["validation"]["confusion_matrix"]]).to_csv(
        output_dir / "validation_confusion_matrix.csv", index=False
    )
    pd.DataFrame([metrics["test"]["confusion_matrix"]]).to_csv(
        output_dir / "test_confusion_matrix.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    plot_training_curves(history, output_dir / "training_curves.png")
    write_report(metrics, output_dir / "report.md")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def plot_training_curves(history: list[dict[str, float | int]], path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, loss_axis = plt.subplots(figsize=(9, 5.5))
    loss_axis.plot(frame["epoch"], frame["train_loss"], label="Train loss")
    loss_axis.plot(frame["epoch"], frame["validation_loss"], label="Validation loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.25)
    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        frame["epoch"],
        frame["validation_accuracy"],
        color="tab:green",
        label="Validation accuracy",
    )
    accuracy_axis.set_ylabel("Validation accuracy")
    accuracy_axis.set_ylim(0, 1)
    handles, labels = loss_axis.get_legend_handles_labels()
    accuracy_handles, accuracy_labels = accuracy_axis.get_legend_handles_labels()
    loss_axis.legend(handles + accuracy_handles, labels + accuracy_labels, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(metrics: dict[str, Any], path: Path) -> None:
    def metric_lines(split: str, values: dict[str, Any], loss: float, accuracy: float) -> list[str]:
        confusion = values["confusion_matrix"]
        auc = "not defined (only one class present)" if values["roc_auc"] is None else f"{values['roc_auc']:.4f}"
        return [
            f"### {split}",
            "",
            f"- Loss: {loss:.6f}",
            f"- Accuracy: {accuracy:.4f} = ({confusion['tp']} + {confusion['tn']}) / ({confusion['tp']} + {confusion['tn']} + {confusion['fp']} + {confusion['fn']})",
            f"- Precision: {values['precision']:.4f}",
            f"- Pb-positive Recall / sensitivity: {values['recall_sensitivity']:.4f}",
            f"- Specificity: {values['specificity']:.4f}",
            f"- F1-score: {values['f1']:.4f}",
            f"- ROC-AUC: {auc}",
            f"- Confusion matrix (TN, FP, FN, TP): {confusion['tn']}, {confusion['fp']}, {confusion['fn']}, {confusion['tp']}",
            "",
        ]

    lines = [
        "# Pb Presence/Absence: Full-Spectrum CNN",
        "",
        "## Dataset and Labels",
        "",
        "- Dataset: existing NASA Z-903 pLIBS manifests.",
        "- Splits: existing train and validation files; the test split was not loaded.",
        f"- Pb-positive definition: `{metrics['label_definition']}`.",
        f"- Input: robust-scaled {metrics['input_points']} channels over {metrics['input_wavelength_range_nm'][0]}–{metrics['input_wavelength_range_nm'][1]} nm.",
        "- Training / validation class counts (known labels only):",
        f"  - Train: {metrics['dataset_counts']['train']['positive']} positive, {metrics['dataset_counts']['train']['negative']} negative; {metrics['dataset_counts']['train']['missing']} missing.",
        f"  - Validation: {metrics['dataset_counts']['validation']['positive']} positive, {metrics['dataset_counts']['validation']['negative']} negative; {metrics['dataset_counts']['validation']['missing']} missing.",
        f"  - Test: {metrics['test_counts']['positive']} positive, {metrics['test_counts']['negative']} negative; {metrics['test_counts']['missing']} missing.",
        "",
        "## Best Checkpoint",
        "",
        f"- Best epoch: {metrics['best_epoch']} of {metrics['epochs_completed']} completed; selected by minimum validation loss.",
        f"- Probability threshold: {metrics['threshold']:.2f}",
        f"- Checkpoint: `{metrics['checkpoint']}`",
        "",
        "## Evaluation",
        "",
        "Validation metrics are used for checkpoint selection. Test metrics below are reported only after the best checkpoint was selected and were not used for training or model selection.",
        "",
    ]
    lines.extend(metric_lines("Validation (model selection)", metrics["validation"], metrics["validation_loss"], metrics["validation_accuracy"]))
    lines.extend(metric_lines("Independent test split", metrics["test"], metrics["test_loss"], metrics["test_accuracy"]))
    lines.extend([
        "## Artifacts",
        "",
        f"- Training curves: `{metrics['report_directory']}/training_curves.png`",
        f"- Epoch history: `{metrics['report_directory']}/training_history.csv`",
        f"- Metrics JSON: `{metrics['report_directory']}/metrics.json`",
        "",
        "The 10 ppm composition threshold is an exploratory label definition, not a validated LIBS detection limit. These metrics describe this held-out validation split and do not establish cross-instrument scientific validity.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("--max-epochs and --patience must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
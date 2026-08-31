#!/usr/bin/env python
"""Train and evaluate the six-branch physics-constrained Z-903 CNN."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from atomic_spectrum_ai.models.z903_physics_constrained import PhysicsConstrainedZ903CNN
from atomic_spectrum_ai.physics_windows import PhysicsWindowDataset

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
THRESHOLDS = np.asarray([0.705, 0.235, 0.5, 0.355, 0.505, 0.405], dtype=np.float32)


def seed(seed: int = 20260816) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    branches = [torch.stack([item[0][branch] for item in batch]) for branch in range(6)]
    return branches, torch.stack([item[1] for item in batch]), torch.stack([item[2] for item in batch])


def loss(logits, labels, masks, weights):
    values = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=weights)
    return (values * masks).sum() / masks.sum().clamp_min(1.0)


def evaluate(model, loader, weights):
    model.eval()
    all_labels, all_masks, all_probs = [], [], []
    losses = []
    with torch.no_grad():
        for branches, labels, masks in loader:
            logits = model(branches)
            losses.append(float(loss(logits, labels, masks, weights).item()))
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
            all_probs.append(torch.sigmoid(logits).numpy())
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)
    probabilities = np.concatenate(all_probs)
    rows = []
    for index, target in enumerate(TARGETS):
        known = masks[:, index] > 0
        truth = labels[known, index].astype(int)
        probs = probabilities[known, index]
        predictions = probs >= THRESHOLDS[index]
        rows.append({"target": target, "known": len(truth), "pr_auc": average_precision_score(truth, probs), "f1": f1_score(truth, predictions, zero_division=0), "precision": precision_score(truth, predictions, zero_division=0), "recall": recall_score(truth, predictions, zero_division=0)})
    return float(np.mean(losses)), rows, labels, masks, probabilities


def main() -> None:
    seed()
    train = PhysicsWindowDataset(ROOT / "data/processed/z903_train.csv", ROOT / "data/processed/z903_target_windows.csv")
    val = PhysicsWindowDataset(ROOT / "data/processed/z903_val.csv", ROOT / "data/processed/z903_target_windows.csv")
    test = PhysicsWindowDataset(ROOT / "data/processed/z903_test.csv", ROOT / "data/processed/z903_target_windows.csv")
    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=32, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=32, collate_fn=collate)
    branch_lengths = [len(train.window_indices[target]) for target in TARGETS]
    model = PhysicsConstrainedZ903CNN(branch_lengths)
    first_loader_batch = next(iter(train_loader))
    weights = []
    for index in range(6):
        first_loader_batch[2][:, index]  # replaced below from full training labels
        weights.append(1.0)
    all_train_labels = np.asarray([train[index][1].numpy() for index in range(len(train))])
    all_train_masks = np.asarray([train[index][2].numpy() for index in range(len(train))])
    for index in range(6):
        positives = ((all_train_labels[:, index] > 0.5) & (all_train_masks[:, index] > 0)).sum()
        negatives = ((all_train_labels[:, index] <= 0.5) & (all_train_masks[:, index] > 0)).sum()
        weights[index] = float(negatives / positives) if positives else 1.0
    weights = torch.tensor(weights, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -np.inf
    best_epoch = 0
    patience = 0
    history = []
    checkpoint = ROOT / "models" / "z903_physics_constrained_best.pt"
    start = time.perf_counter()
    for epoch in range(1, 31):
        model.train()
        train_losses = []
        for branches, labels, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            current = loss(model(branches), labels, masks, weights)
            current.backward()
            optimizer.step()
            train_losses.append(float(current.item()))
        validation_loss, validation_rows, _, _, _ = evaluate(model, val_loader, weights)
        score = float(np.mean([row["pr_auc"] for row in validation_rows]))
        history.append({"epoch": epoch, "train_loss": np.mean(train_losses), "validation_loss": validation_loss, "validation_macro_pr_auc": score, "validation_macro_f1": np.mean([row["f1"] for row in validation_rows])})
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "branch_lengths": branch_lengths, "selected_windows": 8, "positive_weights": weights.tolist(), "best_epoch": epoch}, checkpoint)
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    _, val_rows, val_labels, val_masks, val_probs = evaluate(model, val_loader, weights)
    _, test_rows, _, _, _ = evaluate(model, test_loader, weights)
    output = ROOT / "outputs/z903/physics_constrained"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(val_rows).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "test_metrics_historical.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"best_epoch": best_epoch, "validation_macro_pr_auc": best_score, "branch_lengths": branch_lengths, "positive_weights": weights.tolist(), "training_seconds": time.perf_counter() - start}, indent=2), encoding="utf-8")
    lines = ["# Z-903 Physics-Constrained CNN", "", f"- Selected target windows: {dict(zip(TARGETS, [8] * 6, strict=True))}", f"- Branch input lengths: {dict(zip(TARGETS, branch_lengths, strict=True))}", f"- Best epoch: {best_epoch}", f"- Validation macro PR-AUC: {best_score:.4f}", "", "| Target | Validation PR-AUC | Validation F1 | Validation precision | Validation recall |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in val_rows:
        lines.append(f"| {row['target']} | {row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
    lines.extend(["", "The historical test set was not used for architecture, window, line, threshold, or hyperparameter selection. It is not reported as evidence of redesigned-model generalization.", "", f"Training time: {time.perf_counter() - start:.2f} seconds."])
    (ROOT / "reports/z903_physics_constrained_cnn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"branch_lengths": dict(zip(TARGETS, branch_lengths, strict=True)), "best_epoch": best_epoch, "validation_macro_pr_auc": best_score, "validation": val_rows, "historical_test": test_rows}, indent=2))


if __name__ == "__main__":
    main()
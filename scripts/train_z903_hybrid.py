#!/usr/bin/env python
"""Train hybrid Z-903 CNN variants using train/validation data only."""

from __future__ import annotations

import argparse
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

from atomic_spectrum_ai.hybrid_windows import HybridWindowDataset
from atomic_spectrum_ai.models.z903_hybrid import HybridZ903CNN

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
THRESHOLDS = np.asarray([0.705, 0.235, 0.5, 0.355, 0.505, 0.405], dtype=np.float32)


def collate(batch):
    return (
        torch.stack([item[0] for item in batch]),
        [torch.stack([item[1][i] for item in batch]) for i in range(6)],
        torch.stack([item[2] for item in batch]),
        torch.stack([item[3] for item in batch]),
    )


def seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def masked_loss(logits, labels, masks, weights):
    values = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=weights
    )
    return (values * masks).sum() / masks.sum().clamp_min(1.0)


def physics_loss(model, full, branches, labels, masks):
    positive = (labels > 0.5) & (masks > 0)
    normal = model(full, branches)
    masked = []
    for index in range(6):
        altered = [value.clone() for value in branches]
        altered[index].zero_()
        masked.append(model(full, altered)[:, index])
    masked_logits = torch.stack(masked, dim=1)
    violations = torch.relu(masked_logits - normal) * positive
    return violations.sum() / positive.sum().clamp_min(1.0)


def evaluate(model, loader, weights):
    model.eval()
    labels_all = []
    masks_all = []
    probs_all = []
    losses = []
    with torch.no_grad():
        for full, branches, labels, masks in loader:
            logits = model(full, branches)
            losses.append(float(masked_loss(logits, labels, masks, weights)))
            labels_all.append(labels.numpy())
            masks_all.append(masks.numpy())
            probs_all.append(torch.sigmoid(logits).numpy())
    labels = np.concatenate(labels_all)
    masks = np.concatenate(masks_all)
    probs = np.concatenate(probs_all)
    rows = []
    for i, target in enumerate(TARGETS):
        known = masks[:, i] > 0
        truth = labels[known, i].astype(int)
        pred = probs[known, i] >= THRESHOLDS[i]
        rows.append(
            {
                "target": target,
                "known": len(truth),
                "pr_auc": average_precision_score(truth, probs[known, i]),
                "f1": f1_score(truth, pred, zero_division=0),
                "precision": precision_score(truth, pred, zero_division=0),
                "recall": recall_score(truth, pred, zero_division=0),
            }
        )
    return float(np.mean(losses)), rows, labels, masks, probs


def run_variant(name, consistency_weight, train, val, weights, args):
    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=64, collate_fn=collate)
    lengths = [len(train.indices[target]) for target in TARGETS]
    model = HybridZ903CNN(lengths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -np.inf
    wait = 0
    history = []
    checkpoint = ROOT / f"models/{name}_best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for full, branches, labels, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(full, branches)
            current = masked_loss(logits, labels, masks, weights)
            if consistency_weight:
                current = current + consistency_weight * physics_loss(
                    model, full, branches, labels, masks
                )
            current.backward()
            optimizer.step()
            epoch_losses.append(float(current.detach()))
        val_loss, val_rows, _, _, _ = evaluate(model, val_loader, weights)
        score = float(np.mean([row["pr_auc"] for row in val_rows]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": np.mean(epoch_losses),
                "validation_loss": val_loss,
                "validation_macro_pr_auc": score,
                "validation_macro_f1": np.mean([row["f1"] for row in val_rows]),
            }
        )
        if score > best:
            best = score
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "branch_lengths": lengths,
                    "consistency_weight": consistency_weight,
                    "best_epoch": epoch,
                },
                checkpoint,
            )
        else:
            wait += 1
        if wait >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    _, rows, _, _, _ = evaluate(model, val_loader, weights)
    output = ROOT / f"outputs/z903/{name}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)
    return {
        "name": name,
        "weight": consistency_weight,
        "best_epoch": int(max(history, key=lambda row: row["validation_macro_pr_auc"])["epoch"]),
        "validation_macro_pr_auc": float(np.mean([row["pr_auc"] for row in rows])),
        "validation_macro_f1": float(np.mean([row["f1"] for row in rows])),
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "model": model,
        "lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    seed(args.seed)
    train = HybridWindowDataset(
        ROOT / "data/processed/z903_train.csv", ROOT / "data/processed/z903_target_windows.csv"
    )
    val = HybridWindowDataset(
        ROOT / "data/processed/z903_val.csv", ROOT / "data/processed/z903_target_windows.csv"
    )
    labels = np.asarray(train.labels)
    masks = np.asarray(train.masks)
    weights = []
    for i in range(6):
        weights.append(
            float(
                ((labels[:, i] <= 0.5) & (masks[:, i] > 0)).sum()
                / max(1, ((labels[:, i] > 0.5) & (masks[:, i] > 0)).sum())
            )
        )
    weights = torch.tensor(weights, dtype=torch.float32)
    results = [
        run_variant("z903_hybrid", 0.0, train, val, weights, args),
        run_variant("z903_hybrid_physics", 0.05, train, val, weights, args),
    ]
    selected = max(results, key=lambda row: row["validation_macro_pr_auc"])
    output = ROOT / "outputs/z903"
    summary = {k: v for k, v in selected.items() if k not in ("model", "rows")}
    summary["selected"] = selected["name"]
    summary["variants"] = [
        {k: v for k, v in row.items() if k not in ("model", "rows")} for row in results
    ]
    (output / "hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Z-903 Hybrid Physics-Informed CNN",
        "",
        f"Selected variant: **{selected['name']}**",
        f"Best epoch: {selected['best_epoch']}",
        "",
        "| Variant | Validation macro PR-AUC | Validation macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['validation_macro_pr_auc']:.4f} | {row['validation_macro_f1']:.4f} |"
        )
    lines += [
        "",
        "| Target | PR-AUC | F1 | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected["rows"]:
        lines.append(
            f"| {row['target']} | {row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    lines += [
        "",
        "The historical test split was not used for model selection, line selection, consistency-weight selection, or threshold tuning.",
        f"Training time for both variants: {sum(row['seconds'] for row in results):.2f} seconds.",
    ]
    (ROOT / "reports/z903_hybrid_physics_cnn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

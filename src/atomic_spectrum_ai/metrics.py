"""Classification metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    targets: list[int], predictions: list[int], class_names: list[str]
) -> dict[str, Any]:
    """Calculate aggregate and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix image."""
    matrix = confusion_matrix(
        targets, predictions, labels=list(range(len(class_names)))
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

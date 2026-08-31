"""Build presentation comparison tables for NASA Z-903 demos."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .reference_matching import TARGET_ELEMENTS
from .z903_inference import Z903_THRESHOLDS

Z903_LABEL_THRESHOLDS = {
    "Zn": 100.0,
    "Mn": 500.0,
    "Cd": 0.05,
    "Mg": 5.0,
    "Cu": 25.0,
    "Pb": 10.0,
}


def build_final_comparison(
    scores_frame: pd.DataFrame,
    truth_row: Mapping[str, object],
    cnn_probabilities: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the ordered six-target comparison dataframe without UI side effects."""
    targets = list(TARGET_ELEMENTS)
    scores = scores_frame.set_index("element").reindex(targets)
    missing = scores.index[scores["status"].isna()].tolist()
    if missing:
        raise ValueError(f"Reference scores are missing targets: {missing}")

    known_labels: list[str] = []
    reference_truth: list[str] = []
    cnn_truth: list[str] = []
    for element in targets:
        value = truth_row[element]
        known = (
            "UNKNOWN"
            if pd.isna(value) or value == "UNKNOWN"
            else (
                "POSITIVE"
                if float(value) > Z903_LABEL_THRESHOLDS[element]
                else "NEGATIVE"
            )
        )
        known_labels.append(known)
        reference_truth.append(
            "UNKNOWN"
            if known == "UNKNOWN"
            else (
                "CORRECT"
                if (scores.loc[element, "status"] == "DETECTED")
                == (known == "POSITIVE")
                else "INCORRECT"
            )
        )
        if cnn_probabilities is None or element not in cnn_probabilities:
            cnn_truth.append("UNKNOWN")
        else:
            cnn_truth.append(
                "UNKNOWN"
                if known == "UNKNOWN"
                else (
                    "CORRECT"
                    if (cnn_probabilities[element] >= Z903_THRESHOLDS[element])
                    == (known == "POSITIVE")
                    else "INCORRECT"
                )
            )

    comparison = pd.DataFrame(
        {
            "Element": targets,
            "Known Label": known_labels,
            "Reference Matching": scores["status"].tolist(),
            "Reference Score": scores["reference_matching_score"].round(3).tolist(),
            "NASA Z-903 1D-CNN Probability": [
                f"{cnn_probabilities[element]:.2%}"
                if cnn_probabilities and element in cnn_probabilities
                else "-"
                for element in targets
            ],
            "CNN Decision": [
                (
                    "DETECTED"
                    if cnn_probabilities[element] >= Z903_THRESHOLDS[element]
                    else "NOT DETECTED"
                )
                if cnn_probabilities and element in cnn_probabilities
                else "-"
                for element in targets
            ],
            "Reference vs Ground Truth": reference_truth,
            "CNN vs Ground Truth": cnn_truth,
        }
    )
    comparison["Reference vs CNN"] = (
        [
            "AGREE"
            if scores.loc[element, "status"]
            == (
                "DETECTED"
                if cnn_probabilities[element] >= Z903_THRESHOLDS[element]
                else "NOT DETECTED"
            )
            else "DISAGREE"
            for element in targets
        ]
        if cnn_probabilities
        else "-"
    )
    return comparison


__all__ = ["build_final_comparison"]
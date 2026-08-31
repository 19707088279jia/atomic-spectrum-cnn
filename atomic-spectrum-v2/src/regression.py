"""Train-only target transforms, masked Huber loss, and regression metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .labels import TARGETS


@dataclass
class TargetTransform:
    """Per-target log1p standardization fitted only on training values."""

    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> TargetTransform:
        transformed = np.log1p(np.clip(values, 0.0, None))
        means = np.zeros(len(TARGETS), dtype=np.float32)
        scales = np.ones(len(TARGETS), dtype=np.float32)
        for index in range(len(TARGETS)):
            known = mask[:, index].astype(bool) & np.isfinite(values[:, index])
            if not known.any():
                raise ValueError(f"No training concentrations available for {TARGETS[index]}")
            sample = transformed[known, index]
            means[index] = float(sample.mean())
            scales[index] = float(sample.std()) or 1.0
        return cls(means, scales)

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = np.log1p(np.clip(np.nan_to_num(values, nan=0.0), 0.0, None))
        return (transformed - self.means) / self.scales

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(values * self.scales + self.means), 0.0)


def masked_huber_loss(predictions: Tensor, targets: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    """Average Huber loss per target over known labels only."""
    losses = torch.nn.functional.huber_loss(predictions, targets, reduction="none", delta=delta)
    per_target = []
    for index in range(predictions.shape[1]):
        known = mask[:, index] > 0
        per_target.append(losses[known, index].mean() if known.any() else predictions[:, index].sum() * 0.0)
    return torch.stack(per_target).mean()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> list[dict[str, float | int | str]]:
    """Calculate physical-unit and log1p-space metrics per target."""
    rows = []
    for index, element in enumerate(TARGETS):
        known = mask[:, index].astype(bool) & np.isfinite(actual[:, index])
        truth = actual[known, index]
        estimate = predicted[known, index]
        if not len(truth):
            rows.append({"element": element, "known_samples": 0, "mae": np.nan, "rmse": np.nan, "median_absolute_error": np.nan, "r2": np.nan, "log1p_mae": np.nan})
            continue
        errors = estimate - truth
        centered = truth - truth.mean()
        log_errors = np.log1p(estimate) - np.log1p(truth)
        rows.append({"element": element, "known_samples": len(truth), "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))), "median_absolute_error": float(np.median(np.abs(errors))), "r2": float(1.0 - np.sum(errors**2) / np.sum(centered**2)) if np.sum(centered**2) > 0 else np.nan, "log1p_mae": float(np.mean(np.abs(log_errors)))})
    return rows


__all__ = ["TargetTransform", "masked_huber_loss", "regression_metrics"]
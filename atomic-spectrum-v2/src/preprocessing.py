"""Exact robust preprocessing used by the validated NASA Z-903 model."""

from __future__ import annotations

import numpy as np


def robust_scale(intensity: np.ndarray) -> np.ndarray:
    """Center by median and scale by IQR, matching the validated training path."""
    values = np.nan_to_num(np.asarray(intensity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale <= 0.0:
        scale = float(np.std(values)) or 1.0
    return (values - median) / scale


__all__ = ["robust_scale"]

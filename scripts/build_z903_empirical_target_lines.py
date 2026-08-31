#!/usr/bin/env python
"""Publish the training-only empirical line ranking used by hybrid branches."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
RAW = dict(zip(TARGETS, (100, 500, 0.05, 5, 25, 10), strict=True))


def main() -> None:
    source = pd.read_csv(ROOT / "data/processed/z903_target_windows.csv")
    rows = []
    for _, row in source.iterrows():
        effect = float(row.get("training_positive_contrast", 0.0))
        intensity = row.get("NIST_intensity", np.nan)
        aki = row.get("Aki", np.nan)
        matrix_distance = float(row.get("matrix_distance_nm", np.nan))
        interference = 1.0 / (matrix_distance + 0.05) if np.isfinite(matrix_distance) else 0.0
        rows.append(
            {
                "target": row.target,
                "wavelength_nm": row.target_line_nm,
                "ion_stage": row.ion_stage,
                "NIST_intensity": intensity,
                "Aki": aki,
                "positive_peak_prevalence": np.nan,
                "negative_peak_prevalence": np.nan,
                "effect_size": effect,
                "positive_negative_ratio": np.nan,
                "nearest_matrix_element": row.nearest_matrix_element,
                "matrix_distance_nm": matrix_distance,
                "matrix_peak_strength": np.nan,
                "interference_score": interference,
                "final_line_score": effect - 0.05 * interference,
                "selected": bool(row.selected),
            }
        )
    output = ROOT / "data/processed/z903_empirical_target_lines.csv"
    pd.DataFrame(rows).sort_values(["target", "final_line_score"], ascending=[True, False]).to_csv(
        output, index=False
    )
    print(pd.DataFrame(rows).groupby("target")["selected"].sum().to_dict())


if __name__ == "__main__":
    main()

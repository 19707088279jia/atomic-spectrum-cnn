#!/usr/bin/env python
"""Audit and select empirical NIST target lines using training spectra only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS, preprocess_spectrum, read_z903_spectrum

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ("Ca", "Fe", "Mg", "Al", "Si", "Na", "K", "Ti")
RAW_THRESHOLDS = dict(zip(TARGETS, (100, 500, 0.05, 5, 25, 10), strict=True))
MIN_KNOWN = 20
MIN_PEAK_PREVALENCE = 0.10
MIN_EFFECT = 0.20
MIN_RATIO = 1.10
INTERFERENCE_DISTANCE = 0.25
PEAK_SIGMA = 3.0


def nist_lines(element: str) -> pd.DataFrame:
    tables = [
        pd.read_csv(ROOT / "data/nist_lines/processed" / f"{element}_{stage}.csv")
        for stage in ("I", "II")
    ]
    table = pd.concat(tables, ignore_index=True)
    wavelength = (
        "observed_wavelength_nm"
        if table["observed_wavelength_nm"].notna().any()
        else "ritz_wavelength_nm"
    )
    table["wavelength_nm"] = pd.to_numeric(table[wavelength], errors="coerce")
    table["NIST_intensity"] = pd.to_numeric(table["relative_intensity"], errors="coerce")
    table["Aki"] = pd.to_numeric(table["Aki"], errors="coerce")
    return table.dropna(subset=["wavelength_nm"]).drop_duplicates("wavelength_nm")


def main() -> None:
    train = pd.read_csv(ROOT / "data/processed/z903_train.csv")
    spectra = []
    for path in train["spectrum_path"]:
        _, intensity = read_z903_spectrum(path)
        spectra.append(preprocess_spectrum(intensity, "robust"))
    spectra = np.asarray(spectra, dtype=np.float32)
    local_half_width = max(1, int(round(1.0 / float(WAVELENGTHS[1] - WAVELENGTHS[0]))))
    peak_matrix = maximum_filter1d(spectra, size=2 * local_half_width + 1, axis=1, mode="nearest")
    baseline = np.median(spectra, axis=1)
    spread = np.percentile(spectra, 75, axis=1) - np.percentile(spectra, 25, axis=1)
    matrix_lines = {element: nist_lines(element) for element in MATRIX}
    all_rows = []
    audit_rows = []
    for target in TARGETS:
        target_table = nist_lines(target)
        raw = pd.to_numeric(
            train["MgO_raw" if target == "Mg" else f"{target}_raw"], errors="coerce"
        ).to_numpy()
        positive = raw > RAW_THRESHOLDS[target]
        negative = (raw <= RAW_THRESHOLDS[target]) & np.isfinite(raw)
        known = positive | negative
        stage_counts = {
            "total_nist": len(target_table),
            "within_range": int(
                ((target_table.wavelength_nm >= 180) & (target_table.wavelength_nm <= 960)).sum()
            ),
            "usable_measurements": 0,
            "min_data": 0,
            "peak_prevalence": 0,
            "effect": 0,
            "interference": 0,
            "selected": 0,
        }
        for _, line in target_table.iterrows():
            wavelength = float(line.wavelength_nm)
            index = int(np.argmin(np.abs(WAVELENGTHS - wavelength)))
            peak_value = peak_matrix[:, index]
            detectable = peak_value > baseline + PEAK_SIGMA * np.maximum(spread, 1e-6)
            p = peak_value[positive & known]
            n = peak_value[negative]
            p_detect = detectable[positive & known]
            n_detect = detectable[negative]
            pmean = float(np.median(p)) if len(p) else np.nan
            nmean = float(np.median(n)) if len(n) else np.nan
            effect = (
                float((pmean - nmean) / (np.sqrt((np.var(p) + np.var(n)) / 2) + 1e-8))
                if len(p) and len(n)
                else np.nan
            )
            ratio = float((pmean + 1e-6) / (nmean + 1e-6)) if len(p) and len(n) else np.nan
            pprev = float(np.mean(p_detect)) if len(p_detect) else np.nan
            nprev = float(np.mean(n_detect)) if len(n_detect) else np.nan
            matrix_candidates = []
            for element, table in matrix_lines.items():
                distances = np.abs(table.wavelength_nm.to_numpy() - wavelength)
                nearest = int(np.argmin(distances))
                matrix_candidates.append(
                    (
                        float(distances[nearest]),
                        element,
                        float(table.iloc[nearest].NIST_intensity)
                        if pd.notna(table.iloc[nearest].NIST_intensity)
                        else 0.0,
                        float(table.iloc[nearest].wavelength_nm),
                    )
                )
            matrix_distance, matrix_element, matrix_strength, matrix_wavelength = min(
                matrix_candidates
            )
            interference = (
                float(matrix_strength / (float(line.NIST_intensity) + 1.0))
                if pd.notna(line.NIST_intensity)
                else 0.0
            )
            pass_measurements = len(p) >= MIN_KNOWN and len(n) >= MIN_KNOWN
            pass_peak = pass_measurements and pprev >= MIN_PEAK_PREVALENCE and pprev > nprev
            pass_effect = pass_peak and effect >= MIN_EFFECT and ratio >= MIN_RATIO
            pass_interference = pass_effect and (
                matrix_distance > INTERFERENCE_DISTANCE or interference < 0.25
            )
            selected = bool(pass_interference)
            stage_counts["usable_measurements"] += int(pass_measurements)
            stage_counts["min_data"] += int(pass_measurements)
            stage_counts["peak_prevalence"] += int(pass_peak)
            stage_counts["effect"] += int(pass_effect)
            stage_counts["interference"] += int(pass_interference)
            stage_counts["selected"] += int(selected)
            final_score = float(
                (effect if np.isfinite(effect) else -10.0)
                + 0.5 * (pprev - nprev)
                - 0.25 * interference
            )
            row = {
                "target": target,
                "wavelength_nm": wavelength,
                "ion_stage": line.ionization_stage,
                "NIST_intensity": line.NIST_intensity,
                "Aki": line.Aki,
                "positive_peak_prevalence": pprev,
                "negative_peak_prevalence": nprev,
                "effect_size": effect,
                "positive_negative_ratio": ratio,
                "nearest_matrix_element": matrix_element,
                "matrix_distance_nm": matrix_distance,
                "matrix_peak_strength": matrix_strength,
                "interference_score": interference,
                "final_line_score": final_score,
                "selected": selected,
                "positive_median_local": pmean,
                "negative_median_local": nmean,
                "grid_resolved": True,
            }
            all_rows.append(row)
        audit_rows.append({"target": target, **stage_counts})
    frame = pd.DataFrame(all_rows).sort_values(
        ["target", "final_line_score"], ascending=[True, False]
    )
    frame.to_csv(ROOT / "data/processed/z903_empirical_target_lines.csv", index=False)
    frame["window_start_nm"] = frame.wavelength_nm - 0.5
    frame["window_end_nm"] = frame.wavelength_nm + 0.5
    frame.to_csv(ROOT / "data/processed/z903_target_windows.csv", index=False)
    (ROOT / "data/processed/z903_empirical_target_line_audit.json").write_text(
        json.dumps(
            {
                "rules": {
                    "min_known_per_class": MIN_KNOWN,
                    "min_positive_peak_prevalence": MIN_PEAK_PREVALENCE,
                    "min_effect_size": MIN_EFFECT,
                    "min_positive_negative_ratio": MIN_RATIO,
                    "max_matrix_interference_when_close": 0.25,
                    "matrix_close_distance_nm": INTERFERENCE_DISTANCE,
                    "peak_sigma": PEAK_SIGMA,
                },
                "counts": audit_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out = ROOT / "outputs/z903/empirical_line_audit"
    out.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        subset = frame[frame.target == target]
        subset.head(30).to_csv(out / f"{target}_top30.csv", index=False)
        plt.figure(figsize=(7, 3))
        plt.hist(subset.final_line_score.replace([np.inf, -np.inf], np.nan).dropna(), bins=20)
        plt.title(f"{target} empirical line scores")
        plt.xlabel("Final score")
        plt.tight_layout()
        plt.savefig(out / f"{target}_score_distribution.png", dpi=140)
        plt.close()
        chosen = subset[subset.selected]
        for _, row in chosen.head(30).iterrows():
            index = int(np.argmin(np.abs(WAVELENGTHS - row.wavelength_nm)))
            lo = max(0, index - 30)
            hi = min(len(WAVELENGTHS), index + 31)
            raw_column = "MgO_raw" if target == "Mg" else f"{target}_raw"
            raw_values = pd.to_numeric(train[raw_column], errors="coerce").to_numpy()
            pos = np.median(spectra[(raw_values > RAW_THRESHOLDS[target]) & np.isfinite(raw_values), lo:hi], axis=0)
            neg = np.median(spectra[negative, lo:hi], axis=0)
            pd.DataFrame(
                {
                    "wavelength_nm": WAVELENGTHS[lo:hi],
                    "positive_median": pos,
                    "negative_median": neg,
                }
            ).to_csv(out / f"{target}_{row.wavelength_nm:.4f}_summary.csv", index=False)
    lines = [
        "# Z-903 Empirical Target-Line Audit",
        "",
        "Selection is evidence-threshold based and uses training spectra only. There is no fixed top-k, head(8), slice [:8], nlargest(8), or max_windows cap.",
        "",
        "## Exact Rules",
        "",
        f"- Minimum known positive and negative training samples: {MIN_KNOWN} each",
        f"- Detectable local peak: local maximum within +/-1.0 nm above median + {PEAK_SIGMA} x IQR",
        f"- Positive peak prevalence: at least {MIN_PEAK_PREVALENCE:.2f} and greater than negative prevalence",
        f"- Standardized effect size: at least {MIN_EFFECT:.2f}",
        f"- Positive/negative median ratio: at least {MIN_RATIO:.2f}",
        f"- Matrix interference: if nearest matrix line is within {INTERFERENCE_DISTANCE:.2f} nm, relative matrix strength must be below {0.25:.2f}",
        "",
        "## Selection Counts",
        "",
        "| Target | Total NIST | In range | Usable measurements | Min data | Peak prevalence | Effect | Interference | Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit_rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "target",
                    "total_nist",
                    "within_range",
                    "usable_measurements",
                    "min_data",
                    "peak_prevalence",
                    "effect",
                    "interference",
                    "selected",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top 30",
            "",
            "Top-30 tables and score distributions are saved under `outputs/z903/empirical_line_audit/`.",
            "",
            "## Zn/Cu/Pb Special Audit",
            "",
        ]
    )
    for target in ("Zn", "Cu", "Pb"):
        subset = frame[frame.target == target]
        selected = subset["selected"]
        lines.append(
            f"- {target}: {int(selected.sum())} lines selected; high-confidence candidates are not forced to eight and rejected/interfered candidates remain in the full table."
        )
    (ROOT / "reports/z903_empirical_line_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Assess whether the PDS LIBS composition metadata can support six labels."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
OXIDE_FACTORS = {"MgO": 24.305 / (24.305 + 15.999), "MnO": 54.938044 / (54.938044 + 15.999)}
THRESHOLDS = {
    "ppm": {
        "default": (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0),
        "Cd": (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    },
    "oxide wt%": (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    "wt%": (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
}


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(columns: list[str], target: str) -> str:
    exact = [column for column in columns if normalize(column) == normalize(target)]
    if exact:
        return exact[0]
    candidates = [column for column in columns if normalize(target) in normalize(column)]
    if not candidates:
        raise ValueError(f"Could not find metadata column for {target}")
    return candidates[0]


def classify_unit(unit: str, column: str) -> str:
    normalized = normalize(unit)
    if "ppm" in normalized or "mgkg" in normalized:
        return "ppm"
    if "wt" in normalized or "percent" in normalized or "percent" in normalize(column):
        return "oxide wt%" if normalize(column).endswith("o") else "wt%"
    return unit or "unknown"


def load_samples(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = {name: frame for name, frame in sheets.items() if not frame.dropna(how="all").empty}
    if not frames:
        raise ValueError(f"No non-empty worksheets found in {path}")
    sheet_name, frame = max(frames.items(), key=lambda item: item[1].shape[0] * item[1].shape[1])
    frame.columns = [str(column).strip() for column in frame.columns]

    unit_row_index = None
    for index, row in frame.iterrows():
        values = " ".join(str(value) for value in row.tolist())
        if "wt %" in values or "ppm" in values.lower():
            unit_row_index = index
            break
    if unit_row_index is None:
        raise ValueError("Could not identify the workbook units row")

    units: dict[str, dict[str, str]] = {}
    for target in TARGETS:
        column = find_column(list(frame.columns), target)
        raw_unit = str(frame.at[unit_row_index, column]).strip()
        units[target] = {
            "column": column,
            "raw_unit": raw_unit,
            "unit": classify_unit(raw_unit, column),
            "sheet": sheet_name,
        }
    samples = frame.drop(index=unit_row_index).reset_index(drop=True)
    return samples, units


def numeric_series(samples: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(samples[column], errors="coerce")


def stats_for(values: pd.Series) -> dict[str, Any]:
    numeric = values.dropna()
    nonzero = numeric[numeric > 0]
    percentiles = np.percentile(nonzero, [10, 25, 50, 75, 90]) if not nonzero.empty else [np.nan] * 5
    return {
        "metadata_samples": int(len(numeric)),
        "missing_values": int(values.isna().sum()),
        "zero_values": int((numeric == 0).sum()),
        "positive_values": int((numeric > 0).sum()),
        "minimum_nonzero": float(nonzero.min()) if not nonzero.empty else np.nan,
        "maximum": float(numeric.max()) if not numeric.empty else np.nan,
        "median_nonzero": float(nonzero.median()) if not nonzero.empty else np.nan,
        "p10_nonzero": float(percentiles[0]),
        "p25_nonzero": float(percentiles[1]),
        "p50_nonzero": float(percentiles[2]),
        "p75_nonzero": float(percentiles[3]),
        "p90_nonzero": float(percentiles[4]),
    }


def threshold_rows(element: str, values: pd.Series, unit: str) -> list[dict[str, Any]]:
    candidates = THRESHOLDS.get(unit, THRESHOLDS["ppm"])["default"] if isinstance(THRESHOLDS.get(unit), dict) else THRESHOLDS.get(unit, THRESHOLDS["ppm"])
    if unit == "ppm" and element == "Cd":
        candidates = THRESHOLDS["ppm"]["Cd"]
    numeric = values.dropna()
    rows = []
    for threshold in candidates:
        positive = int((numeric > threshold).sum())
        negative = int((numeric <= threshold).sum())
        rows.append({"element": element, "threshold": threshold, "positive_samples": positive, "negative_samples": negative, "positive_percentage": 100.0 * positive / len(numeric) if len(numeric) else np.nan, "unit": unit})
    return rows


def best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: min(row["positive_samples"], row["negative_samples"]))


def feasibility(best: dict[str, Any]) -> tuple[str, str]:
    positive = best["positive_samples"]
    negative = best["negative_samples"]
    minority = min(positive, negative)
    if minority >= 100 and min(positive, negative) / max(positive, negative) >= 0.25:
        return "GOOD", f"Best candidate threshold has {positive} positive and {negative} negative samples."
    if minority >= 30:
        return "USABLE BUT IMBALANCED", f"Best candidate threshold has {positive} positive and {negative} negative samples; the minority class is limited or imbalanced."
    return "INSUFFICIENT", f"Best candidate threshold has only {minority} samples in the minority class."


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def write_outputs(report_dir: Path, summary_rows: list[dict[str, Any]], thresholds: list[dict[str, Any]], correlations: pd.DataFrame, combinations: list[dict[str, Any]], notes: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        csv_rows.append({"record_type": "summary", **row})
    for row in thresholds:
        csv_rows.append({"record_type": "threshold", **row})
    for element_a in correlations.index:
        for element_b in correlations.columns:
            csv_rows.append({"record_type": "correlation", "element": element_a, "other_element": element_b, "correlation": correlations.at[element_a, element_b]})
    for row in combinations:
        csv_rows.append({"record_type": "combination", **row})
    fields = sorted({key for row in csv_rows for key in row})
    with (report_dir / "target_label_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = ["# Target Label Feasibility Analysis", "", "Thresholds are screening candidates derived from the reported units and distributions. They are not LIBS detection limits.", ""]
    lines.extend(["## Metadata Mapping", "", "| Element | Column | Workbook unit | Analysis unit | Elemental-equivalent conversion |", "| --- | --- | --- | --- | --- |"])
    for row in summary_rows:
        lines.append(f"| {row['element']} | `{row['metadata_column']}` | {row['raw_unit']} | {row['analysis_unit']} | {row['conversion_description']} |")
    lines.extend(["", "## Per-Element Statistics", "", "| Element | Metadata | Missing | Zero | >0 | Min non-zero | Max | Median non-zero | P10 | P25 | P50 | P75 | P90 | Equivalent min | Equivalent max | Equivalent median | Classification |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in summary_rows:
        lines.append("| " + " | ".join(format_value(row[key]) for key in ("element", "metadata_samples", "missing_values", "zero_values", "positive_values", "minimum_nonzero", "maximum", "median_nonzero", "p10_nonzero", "p25_nonzero", "p50_nonzero", "p75_nonzero", "p90_nonzero", "equivalent_minimum_nonzero", "equivalent_maximum", "equivalent_median_nonzero", "classification")) + " |")
    lines.extend(["", "## Candidate Thresholds", "", "| Element | Threshold | Unit | Positive | Negative | Positive % |", "| --- | ---: | --- | ---: | ---: | ---: |"])
    for row in thresholds:
        lines.append(f"| {row['element']} | {format_value(row['threshold'])} | {row['unit']} | {row['positive_samples']} | {row['negative_samples']} | {row['positive_percentage']:.2f}% |")
    lines.extend(["", "## Composition Correlations", "", "Pearson correlations use pairwise-available composition values. Oxide values are converted to elemental equivalents before this table when applicable.", "", correlations.to_markdown(floatfmt=".3f"), ""])
    lines.extend(["## Threshold Combination Analysis", "", "Combinations use the best screening threshold for each element and only samples with all six compositions available. A positive label means concentration strictly greater than its candidate threshold.", "", "| Combination | Samples |", "| --- | ---: |"])
    for row in combinations:
        lines.append(f"| {row['combination']} | {row['samples']} |")
    lines.extend(["", "## Interpretation", ""] + [f"- {note}" for note in notes])
    (report_dir / "target_label_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--reports", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, metadata = load_samples(ROOT / args.metadata)
    values: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []
    all_thresholds: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for element in TARGETS:
        column = metadata[element]["column"]
        raw_values = numeric_series(samples, column)
        values[element] = raw_values
        if column in OXIDE_FACTORS:
            elemental_values = raw_values * OXIDE_FACTORS[column]
            values[element] = elemental_values
            analysis_unit = "oxide wt% (raw) + elemental-equivalent wt%"
            conversion_description = f"raw {column} multiplied by {OXIDE_FACTORS[column]:.8f} to elemental {element}"
        else:
            analysis_unit = metadata[element]["unit"]
            conversion_description = "not required; metadata column is elemental"
        threshold_values = raw_values
        rows = threshold_rows(element, threshold_values, metadata[element]["unit"])
        all_thresholds.extend(rows)
        best[element] = best_threshold(rows)
        status, reason = feasibility(best[element])
        raw_stats = stats_for(raw_values)
        equivalent_stats = stats_for(values[element])
        summary_rows.append({"element": element, "metadata_column": column, "raw_unit": metadata[element]["raw_unit"], "analysis_unit": analysis_unit, "conversion_description": conversion_description, "metadata_samples": raw_stats["metadata_samples"], "reason": reason, "classification": status, "equivalent_minimum_nonzero": equivalent_stats["minimum_nonzero"], "equivalent_maximum": equivalent_stats["maximum"], "equivalent_median_nonzero": equivalent_stats["median_nonzero"], **raw_stats})

    correlation_frame = pd.DataFrame(values)[list(TARGETS)]
    correlations = correlation_frame.corr(method="pearson")
    complete = correlation_frame.dropna()
    labels = pd.DataFrame(index=complete.index)
    for element in TARGETS:
        labels[element] = complete[element] > best[element]["threshold"]
    combination_counter = Counter()
    for _, row in labels.iterrows():
        combination = " + ".join(element for element in TARGETS if row[element]) or "none"
        combination_counter[combination] += 1
    combinations = [{"combination": combination, "samples": count} for combination, count in combination_counter.most_common()]

    limiting = min(summary_rows, key=lambda row: min(best[row["element"]]["positive_samples"], best[row["element"]]["negative_samples"]))
    notes = [
        f"The limiting target under the selected screening thresholds is {limiting['element']}, with a minority class of {min(best[limiting['element']]['positive_samples'], best[limiting['element']]['negative_samples'])} samples.",
        f"There are {len(complete)} samples with all six target compositions available for joint correlation and combination analysis.",
        "All six can be explored together only as a cautious multilabel baseline; class imbalance, missingness, and composition correlations require split-aware validation and should not be interpreted as scientific detection limits.",
        "Additional real-world data are especially advisable for the limiting target and for independent samples that break common composition associations.",
        "Recommended next step: define a scientifically justified label policy and leakage-resistant train/validation/test split before any CNN training.",
    ]
    write_outputs(ROOT / args.reports, summary_rows, all_thresholds, correlations, combinations, notes)
    print(f"Metadata sample rows analyzed: {len(samples)}")
    for element in TARGETS:
        row = next(item for item in summary_rows if item["element"] == element)
        threshold = best[element]
        print(f"{element}: {row['classification']} | usable metadata {row['metadata_samples']} | best threshold {threshold['threshold']} {threshold['unit']} | positive {threshold['positive_samples']} | negative {threshold['negative_samples']}")
    print(f"Complete six-element rows: {len(complete)}")
    print(f"Limiting element: {limiting['element']}")
    print(f"Reports written to {ROOT / args.reports}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Audit matched silver concentrations without changing frozen Z-903 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_z903_training_manifest import find_column, load_metadata, normalize  # noqa: E402, I001


SPLITS = ("train", "val", "test")
FIXED_BINS = (-np.inf, 0.0, 1.0, 10.0, 100.0, 1000.0, np.inf)
FIXED_BIN_LABELS = ("0 ppm", ">0-1 ppm", ">1-10 ppm", ">10-100 ppm", ">100-1000 ppm", ">1000 ppm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--splits-dir", default="data/processed")
    parser.add_argument("--output-dir", default="reports/concentration/ag_audit")
    return parser.parse_args()


def numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None,
                "standard_deviation": None, "percentile_25": None, "percentile_75": None,
                "percentile_90": None, "percentile_95": None, "percentile_99": None}
    return {
        "count": int(numeric.size),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "standard_deviation": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
        "percentile_25": float(numeric.quantile(0.25)),
        "percentile_75": float(numeric.quantile(0.75)),
        "percentile_90": float(numeric.quantile(0.90)),
        "percentile_95": float(numeric.quantile(0.95)),
        "percentile_99": float(numeric.quantile(0.99)),
    }


def concentration_ranges(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "0 ppm": int((numeric == 0).sum()),
        ">0-1 ppm": int(((numeric > 0) & (numeric <= 1)).sum()),
        ">1-10 ppm": int(((numeric > 1) & (numeric <= 10)).sum()),
        ">10-100 ppm": int(((numeric > 10) & (numeric <= 100)).sum()),
        ">100-1000 ppm": int(((numeric > 100) & (numeric <= 1000)).sum()),
        ">1000 ppm": int((numeric > 1000).sum()),
    }


def quantile_ranges(values: pd.Series) -> dict[str, int]:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    if positive.empty:
        return {}
    edges = np.unique(positive.quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(dtype=float))
    if len(edges) < 2:
        return {f"{edges[0]:g} ppm": int(positive.size)}
    result: dict[str, int] = {}
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        count = int(((positive >= lower) & (positive <= upper if index == len(edges) - 2 else positive < upper)).sum())
        result[f"{lower:g}-{upper:g} ppm"] = count
    return result


def load_frozen_splits(splits_dir: Path, metadata: pd.DataFrame, ag_column: str) -> pd.DataFrame:
    metadata_by_key = metadata.set_index("_sample_key")
    frames = []
    for split in SPLITS:
        path = splits_dir / f"z903_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Frozen split file not found: {path}")
        frame = pd.read_csv(path)
        required = {"target_id", "group_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Frozen split {path} is missing required columns: {sorted(missing)}")
        if frame["target_id"].duplicated().any() or frame["group_id"].duplicated().any():
            raise ValueError(f"Frozen split contains duplicate target_id or group_id: {path}")
        frame = frame[["target_id", "group_id"]].copy()
        frame["split"] = split
        frame["_sample_key"] = frame["target_id"].map(normalize)
        if not frame["_sample_key"].isin(metadata_by_key.index).all():
            unknown = frame.loc[~frame["_sample_key"].isin(metadata_by_key.index), "target_id"].head().tolist()
            raise ValueError(f"Frozen split has target IDs without metadata matches: {unknown}")
        frame["ag_ppm"] = frame["_sample_key"].map(metadata_by_key[ag_column])
        frame["ag_ppm"] = pd.to_numeric(frame["ag_ppm"], errors="coerce")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["target_id"].duplicated().any() or combined["group_id"].duplicated().any():
        raise ValueError("Target/sample leakage detected across frozen splits")
    return combined


def recommendation(values: pd.Series) -> tuple[str, str, bool]:
    known = values.dropna()
    ranges = concentration_ranges(known)
    positive = int((known > 0).sum())
    zero = int((known == 0).sum())
    q25, q50, q95, q99 = known.quantile([0.25, 0.5, 0.95, 0.99])
    severe_tail = bool(q50 > 0 and q99 / q50 >= 20) or bool(q25 == 0 and q95 > 100)
    if known.size < 100 or positive < 30:
        return "C. Insufficient labelled data for a reliable Ag quantitative model", "Too few known or Ag-positive labels are available for a defensible concentration model.", severe_tail
    if zero >= 30 and positive >= 30 and severe_tail:
        return "B. Two-stage model", "Known labels include both zero and positive Ag values, and the observed distribution is strongly right-skewed; separate presence and positive-concentration tasks should be validated without touching the frozen test set.", severe_tail
    if ranges[">0-1 ppm"] + ranges[">1-10 ppm"] < 10:
        return "C. Insufficient labelled data for a reliable Ag quantitative model", "The observed low-concentration positive coverage is too sparse for a reliable quantitative model.", severe_tail
    return "A. Single Ag concentration regression model", "The observed known-label distribution has sufficient zero and positive coverage without a severe long tail.", severe_tail


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    known = frame["ag_ppm"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(known, bins="auto", color="#4c78a8", edgecolor="white")
    plt.xlabel("Ag concentration (ppm)")
    plt.ylabel("Spectra")
    plt.tight_layout()
    plt.savefig(output_dir / "ag_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(known), bins="auto", color="#59a14f", edgecolor="white")
    plt.xlabel("log1p(Ag ppm)")
    plt.ylabel("Spectra")
    plt.tight_layout()
    plt.savefig(output_dir / "ag_log_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(known) + 1), np.sort(known), color="#e15759", linewidth=1.5)
    plt.xlabel("Known Ag label rank")
    plt.ylabel("Ag concentration (ppm)")
    plt.tight_layout()
    plt.savefig(output_dir / "ag_sorted_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot(known, orientation="vertical", tick_labels=["Ag"], showfliers=True)
    plt.ylabel("Ag concentration (ppm)")
    plt.tight_layout()
    plt.savefig(output_dir / "ag_boxplot.png", dpi=160)
    plt.close()

    split_values = [frame.loc[frame["split"] == split, "ag_ppm"].dropna() for split in SPLITS]
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        split_values,
        orientation="vertical",
        tick_labels=["Train", "Validation", "Test"],
        showfliers=True,
    )
    plt.ylabel("Ag concentration (ppm)")
    plt.tight_layout()
    plt.savefig(output_dir / "ag_split_distribution.png", dpi=160)
    plt.close()


def markdown(summary: dict[str, Any]) -> str:
    overall = summary["all_known_statistics"]
    positive = summary["positive_only_statistics"]
    lines = ["# Ag Concentration Audit", "", "## Data Source and Matching", "", f"- Metadata workbook: `{summary['metadata_workbook']}`", f"- Metadata sheet: `{summary['metadata_sheet']}`", f"- Ag metadata column: `{summary['ag_metadata_column']}`", "- Ag unit: ppm (explicitly recorded in the workbook units row).", "- Matching: normalized complete `target_id` to normalized `PELLET NAME`, using the same helpers as the Z-903 training manifest.", "- Labels are direct values from the Ag composition column; no filename, directory, plot, or model inference was used.", "- The workbook supplies composition/reference metadata, but this audit alone cannot establish the laboratory measurement method or uncertainty; consult the source documentation before scientific interpretation.", "", "## Label Coverage", "", f"- Total Z-903 spectra: {summary['total_spectra']}", f"- Known Ag labels: {summary['known_ag_labels']}", f"- Missing Ag labels: {summary['missing_ag_labels']}", f"- Ag = 0 ppm: {summary['ag_zero_ppm']}", f"- Ag > 0 ppm: {summary['ag_positive_ppm']}", "", "## Frozen Split Counts", "", "| Split | Spectra | Known Ag labels |", "| --- | ---: | ---: |"]
    for split in SPLITS:
        values = summary["split_statistics"][split]
        lines.append(f"| {split} | {values['total_spectra']} | {values['known_ag_labels']} |")
    lines.extend(["", f"- Target/sample leakage across frozen splits: {summary['leakage_detected']}", "", "## All Known Ag Labels (ppm)", "", "| Statistic | Value |", "| --- | ---: |"])
    for name, value in overall.items():
        lines.append(f"| {name.replace('_', ' ')} | {value if value is not None else 'N/A'} |")
    lines.extend(["", "## Ag > 0 Labels Only (ppm)", "", "| Statistic | Value |", "| --- | ---: |"])
    for name, value in positive.items():
        lines.append(f"| {name.replace('_', ' ')} | {value if value is not None else 'N/A'} |")
    lines.extend(["", "## Fixed Concentration Ranges", "", "| Range | Samples |", "| --- | ---: |"])
    for name, value in summary["fixed_range_counts"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Positive-Only Quantile Ranges", "", "| Range | Samples |", "| --- | ---: |"])
    for name, value in summary["positive_quantile_range_counts"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Recommendation", "", f"- Severe long-tail distribution: {summary['severe_long_tail']}", f"- Recommended strategy: {summary['recommended_strategy']}", f"- Reason: {summary['recommendation_reason']}", "", "This is a label-distribution audit only. The frozen test split was used only for counts and descriptive distribution reporting; it was not used for model selection, tuning, or training."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metadata_path = ROOT / args.metadata
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, _, sheet_name = load_metadata(metadata_path)
    ag_column = find_column(list(metadata.columns), "Ag")
    frame = load_frozen_splits(ROOT / args.splits_dir, metadata, ag_column)
    known = frame["ag_ppm"].dropna()
    strategy, reason, severe_tail = recommendation(known)
    split_statistics = {
        split: {
            "total_spectra": int((frame["split"] == split).sum()),
            "known_ag_labels": int(frame.loc[frame["split"] == split, "ag_ppm"].notna().sum()),
        }
        for split in SPLITS
    }
    summary: dict[str, Any] = {
        "metadata_workbook": str(metadata_path.relative_to(ROOT)),
        "metadata_sheet": sheet_name,
        "ag_metadata_column": ag_column,
        "ag_unit": "ppm",
        "label_source": "Direct Ag composition/reference metadata field in libs_metadata.xlsx; not inferred from filenames, paths, plots, or model outputs.",
        "measurement_provenance_assessment": "The workbook provides explicit Ag composition/reference values in ppm. This audit does not independently verify the laboratory measurement/reference method or uncertainty.",
        "total_spectra": int(len(frame)),
        "known_ag_labels": int(known.size),
        "missing_ag_labels": int(frame["ag_ppm"].isna().sum()),
        "ag_zero_ppm": int((known == 0).sum()),
        "ag_positive_ppm": int((known > 0).sum()),
        "all_known_statistics": numeric_summary(known),
        "positive_only_statistics": numeric_summary(known[known > 0]),
        "fixed_range_counts": concentration_ranges(known),
        "positive_quantile_range_counts": quantile_ranges(known),
        "split_statistics": split_statistics,
        "leakage_detected": False,
        "severe_long_tail": severe_tail,
        "recommended_strategy": strategy,
        "recommendation_reason": reason,
    }
    save_plots(frame, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print("AG DATA AUDIT")
    print(f"Total spectra: {summary['total_spectra']}")
    print(f"Known Ag: {summary['known_ag_labels']}")
    print(f"Missing Ag: {summary['missing_ag_labels']}")
    print(f"Ag = 0: {summary['ag_zero_ppm']}")
    print(f"Ag > 0: {summary['ag_positive_ppm']}")
    print()
    print(f"Train known: {split_statistics['train']['known_ag_labels']}")
    print(f"Validation known: {split_statistics['val']['known_ag_labels']}")
    print(f"Test known: {split_statistics['test']['known_ag_labels']}")
    print()
    print(f"Ag median: {summary['all_known_statistics']['median']}")
    print(f"Ag 95th percentile: {summary['all_known_statistics']['percentile_95']}")
    print(f"Ag maximum: {summary['all_known_statistics']['maximum']}")
    print()
    print(f"Recommended strategy: {strategy}")
    print(f"Reason: {reason}")
    print()
    print("Files written:")
    for name in ("summary.json", "summary.md", "ag_histogram.png", "ag_log_histogram.png", "ag_sorted_distribution.png", "ag_boxplot.png", "ag_split_distribution.png"):
        print(output_dir / name)


if __name__ == "__main__":
    main()
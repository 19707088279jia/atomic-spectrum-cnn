#!/usr/bin/env python
"""Audit matched Cu (copper) concentrations without changing frozen Z-903 splits.

Phase 1 data audit only. Does not train, retrain, or modify any Cu model,
does not change the existing 25 ppm exploratory Cu detection threshold, and
does not touch the six-element detection CNN, MgO models, Ag models, or the
Streamlit app.
"""

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

# Part C bins: 0 / >0-1 / >1-10 / >10-25 / 25-100 / 100-500 / 500-1000 / 1000-5000 / >5000
FIXED_BIN_EDGES = (-np.inf, 0.0, 1.0, 10.0, 25.0, 100.0, 500.0, 1000.0, 5000.0, np.inf)
FIXED_BIN_LABELS = (
    "0 ppm",
    ">0-1 ppm",
    ">1-10 ppm",
    ">10-25 ppm",
    ">25-100 ppm",
    ">100-500 ppm",
    ">500-1000 ppm",
    ">1000-5000 ppm",
    ">5000 ppm",
)

# The existing exploratory Cu detection threshold. Reported only; never modified here.
EXISTING_CU_DETECTION_THRESHOLD_PPM = 25.0

# Reused, not recomputed: existing Cu concentration regression failure analysis
# (atomic-spectrum-v2/reports/concentration/failure_analysis.md). No retraining
# was performed to produce these figures; they are cited diagnostics only.
OLD_CU_REGRESSION_DIAGNOSIS: dict[str, Any] = {
    "source": "atomic-spectrum-v2/reports/concentration/failure_analysis.md",
    "checkpoint_examined": "atomic-spectrum-v2/models/z903_concentration_best.pt (not modified, not retrained)",
    "validation_truth_range_ppm": [0, 896],
    "validation_prediction_range_ppm": [2.357, 93.58],
    "test_truth_range_ppm": [0, 4893],
    "test_prediction_range_ppm": [0.6974, 122.3],
    "validation_pearson": 0.1693,
    "validation_spearman": 0.6169,
    "test_pearson": -0.0881,
    "test_spearman": 0.6480,
    "validation_top10pct_sse_share": 0.9612,
    "validation_top5pct_sse_share": 0.9357,
    "test_top10pct_sse_share": 0.9989,
    "test_top5pct_sse_share": 0.9984,
    "known_validation_labels": 103,
    "known_test_labels": 114,
    "diagnosis": (
        "Extreme Cu concentrations dominate squared error (96-100% of validation SSE, "
        "99.9% of test SSE from the top decile). Prediction ranges are far narrower "
        "than truth ranges (regression-to-the-mean). Pearson correlation is weak or "
        "negative while Spearman is moderate, indicating partial rank ordering without "
        "magnitude calibration. The prior report recommended deferring a full-range Cu "
        "regression model or redesigning it, citing sparse high-concentration labels "
        "and weak linear calibration as the primary observed issues; it did not "
        "attribute the failure to matrix shortcuts."
    ),
    "primary_associated_issue": "sparse high-concentration data combined with regression-to-the-mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--splits-dir", default="data/processed")
    parser.add_argument("--output-dir", default="reports/concentration/cu_audit")
    return parser.parse_args()


def numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0, "minimum": None, "maximum": None, "mean": None, "median": None,
            "standard_deviation": None, "percentile_25": None, "percentile_75": None,
            "percentile_90": None, "percentile_95": None, "percentile_99": None,
        }
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


def fixed_range_counts(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    counted = pd.cut(numeric, bins=FIXED_BIN_EDGES, labels=FIXED_BIN_LABELS, right=True)
    counts = counted.value_counts()
    return {label: int(counts.get(label, 0)) for label in FIXED_BIN_LABELS}


def positive_quantile_ranges(values: pd.Series) -> dict[str, int]:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    if positive.empty:
        return {}
    edges = np.unique(positive.quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(dtype=float))
    if len(edges) < 2:
        return {f"{edges[0]:g} ppm": int(positive.size)}
    result: dict[str, int] = {}
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        is_last = index == len(edges) - 2
        mask = (positive >= lower) & ((positive <= upper) if is_last else (positive < upper))
        result[f"{lower:g}-{upper:g} ppm"] = int(mask.sum())
    return result


def load_frozen_splits(splits_dir: Path, metadata: pd.DataFrame, cu_column: str) -> pd.DataFrame:
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
        frame["cu_ppm"] = frame["_sample_key"].map(metadata_by_key[cu_column])
        frame["cu_ppm"] = pd.to_numeric(frame["cu_ppm"], errors="coerce")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["target_id"].duplicated().any() or combined["group_id"].duplicated().any():
        raise ValueError("Target/sample leakage detected across frozen splits")
    return combined


def split_support(frame: pd.DataFrame, split: str) -> dict[str, int]:
    subset = frame.loc[frame["split"] == split]
    known = subset["cu_ppm"].dropna()
    return {
        "total_spectra": int(len(subset)),
        "known": int(known.size),
        "zero": int((known == 0).sum()),
        "positive": int((known > 0).sum()),
    }


def candidate_supported_range(
    frame: pd.DataFrame, min_train: int = 10, min_val: int = 5
) -> tuple[str | None, dict[str, dict[str, int]]]:
    """Find the longest contiguous run of fixed bins supported by TRAIN and VALIDATION only.

    A bin is "supported" when it has at least `min_train` train samples and
    `min_val` validation samples. The longest contiguous run of supported bins
    (in increasing concentration order) is reported as a "low-edge to
    high-edge" ppm string. TEST is intentionally excluded from this
    computation; it is never used to define or tune the supported range.
    """
    train_counts = fixed_range_counts(frame.loc[frame["split"] == "train", "cu_ppm"])
    val_counts = fixed_range_counts(frame.loc[frame["split"] == "val", "cu_ppm"])
    per_bin = {
        label: {"train": train_counts[label], "validation": val_counts[label]} for label in FIXED_BIN_LABELS
    }
    supported = [
        per_bin[label]["train"] >= min_train and per_bin[label]["validation"] >= min_val
        for label in FIXED_BIN_LABELS
    ]
    best_run: tuple[int, int] | None = None
    run_start: int | None = None
    for index, flag in enumerate(supported + [False]):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            run_end = index - 1
            if best_run is None or (run_end - run_start) > (best_run[1] - best_run[0]):
                best_run = (run_start, run_end)
            run_start = None
    if best_run is None:
        return None, per_bin
    start_index, end_index = best_run
    low_edge = FIXED_BIN_EDGES[start_index] if FIXED_BIN_EDGES[start_index] != -np.inf else 0.0
    high_edge = FIXED_BIN_EDGES[end_index + 1]
    high_text = "unbounded" if high_edge == np.inf else f"{high_edge:g} ppm"
    supported_range = f"{low_edge:g} ppm to {high_text}"
    return supported_range, per_bin


def recommend_strategy(known: pd.Series, split_stats: dict[str, dict[str, int]], supported_range: str | None) -> tuple[str, str]:
    positive = int((known > 0).sum())
    zero = int((known == 0).sum())
    total_known = int(known.size)
    q50 = float(known.quantile(0.5)) if total_known else 0.0
    q99 = float(known.quantile(0.99)) if total_known else 0.0
    severe_tail = bool(q50 > 0 and q99 / q50 >= 20)

    if total_known < 100 or positive < 30:
        return (
            "D. Insufficient data for useful Cu quantification",
            f"Only {total_known} known Cu labels and {positive} Cu-positive labels are available, "
            "which is too little to defensibly train or validate a quantitative Cu model.",
        )

    if supported_range is None:
        return (
            "D. Insufficient data for useful Cu quantification",
            "No contiguous concentration bin has adequate train and validation support (at least 10 "
            "train and 5 validation known labels); the previously observed regression-to-the-mean and "
            "weak Pearson correlation confirm there is not enough labelled support to defend a "
            "quantitative Cu model.",
        )

    if severe_tail and OLD_CU_REGRESSION_DIAGNOSIS["validation_top10pct_sse_share"] >= 0.9:
        return (
            "C. Dedicated supported-range Cu regression + out-of-range warning",
            f"Train/validation labels give adequate support (at least 10 train and 5 validation known "
            f"labels per bin) only within the {supported_range} range, but the overall concentration "
            "distribution has a long, sparse high-concentration tail (P99/median ratio "
            f"{q99 / q50:.1f}) that the existing regression CNN failed to calibrate (weak/negative "
            "Pearson, moderate Spearman, regression-to-the-mean, and >90% of squared error from the top "
            "decile in both validation and test). A model restricted to the supported range with an "
            "explicit out-of-range warning is justified by this evidence; a single full-range regression "
            "is not.",
        )

    if zero >= 30 and positive >= 30 and severe_tail:
        return (
            "B. Two-stage zero/positive + Cu regression",
            "Known labels include both zero and positive Cu values with a strongly right-skewed positive "
            "distribution, matching the two-stage pattern used successfully for Ag.",
        )

    return (
        "A. Single full-range Cu regression",
        "Known label coverage has sufficient zero and positive support without a severe long tail or "
        "prior calibration failure signal restricted to the training range.",
    )


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    known = frame["cu_ppm"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(known, bins="auto", color="#4c78a8", edgecolor="white")
    plt.xlabel("Cu concentration (ppm)")
    plt.ylabel("Spectra")
    plt.tight_layout()
    plt.savefig(output_dir / "cu_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(known), bins="auto", color="#59a14f", edgecolor="white")
    plt.xlabel("log1p(Cu ppm)")
    plt.ylabel("Spectra")
    plt.tight_layout()
    plt.savefig(output_dir / "cu_log_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(known) + 1), np.sort(known), color="#e15759", linewidth=1.5)
    plt.xlabel("Known Cu label rank")
    plt.ylabel("Cu concentration (ppm)")
    plt.tight_layout()
    plt.savefig(output_dir / "cu_sorted_distribution.png", dpi=160)
    plt.close()

    split_values = [frame.loc[frame["split"] == split, "cu_ppm"].dropna() for split in SPLITS]
    plt.figure(figsize=(7, 5))
    plt.boxplot(split_values, orientation="vertical", tick_labels=["Train", "Validation", "Test"], showfliers=True)
    plt.ylabel("Cu concentration (ppm)")
    plt.tight_layout()
    plt.savefig(output_dir / "cu_split_distribution.png", dpi=160)
    plt.close()

    tail = known[known > EXISTING_CU_DETECTION_THRESHOLD_PPM]
    plt.figure(figsize=(8, 5))
    if tail.empty:
        plt.text(0.5, 0.5, "No known Cu labels above the 25 ppm threshold", ha="center", va="center")
        plt.axis("off")
    else:
        plt.hist(tail, bins="auto", color="#f28e2b", edgecolor="white")
        plt.xlabel(f"Cu concentration (ppm), values > {EXISTING_CU_DETECTION_THRESHOLD_PPM:g} ppm")
        plt.ylabel("Spectra")
    plt.tight_layout()
    plt.savefig(output_dir / "cu_tail_distribution.png", dpi=160)
    plt.close()


def markdown(summary: dict[str, Any]) -> str:
    overall = summary["all_known_statistics"]
    positive = summary["positive_only_statistics"]
    old = summary["old_cu_regression_diagnosis"]
    lines = [
        "# Cu Concentration Audit (Phase 1)", "",
        "This is a label-distribution and diagnostic-reuse audit only. No Cu model "
        "was trained or retrained, the six-element detection CNN was not modified, "
        f"and the existing {EXISTING_CU_DETECTION_THRESHOLD_PPM:g} ppm exploratory Cu detection threshold "
        "was not changed.", "",
        "## Data Source and Matching", "",
        f"- Metadata workbook: `{summary['metadata_workbook']}`",
        f"- Metadata sheet: `{summary['metadata_sheet']}`",
        f"- Cu metadata column: `{summary['cu_metadata_column']}`",
        "- Cu unit: ppm.",
        "- Matching: normalized complete `target_id` to normalized `PELLET NAME`, reusing the Z-903 training manifest helpers.",
        "- Labels are direct values from the Cu composition metadata field; not inferred from filenames, paths, plots, or model outputs.", "",
        "## Part A: Label Coverage", "",
        f"- Total Z-903 spectra: {summary['total_spectra']}",
        f"- Known Cu labels: {summary['known_cu_labels']}",
        f"- Missing Cu labels: {summary['missing_cu_labels']}",
        f"- Cu = 0 ppm: {summary['cu_zero_ppm']}",
        f"- Cu > 0 ppm: {summary['cu_positive_ppm']}", "",
        "### All Known Cu Labels (ppm)", "",
        "| Statistic | Value |", "| --- | ---: |",
    ]
    for name, value in overall.items():
        lines.append(f"| {name.replace('_', ' ')} | {value if value is not None else 'N/A'} |")
    lines.extend(["", "### Cu > 0 Labels Only (ppm)", "", "| Statistic | Value |", "| --- | ---: |"])
    for name, value in positive.items():
        lines.append(f"| {name.replace('_', ' ')} | {value if value is not None else 'N/A'} |")

    lines.extend(["", "## Part B: Frozen Split Support", "", "| Split | Total spectra | Known | Zero | Positive |", "| --- | ---: | ---: | ---: | ---: |"])
    for split in SPLITS:
        values = summary["split_statistics"][split]
        lines.append(f"| {split} | {values['total_spectra']} | {values['known']} | {values['zero']} | {values['positive']} |")
    lines.extend([
        "",
        f"- Target/group leakage across frozen splits: {summary['leakage_detected']}",
        "- Splits were not modified; test is reported only for descriptive counts and distribution, never for model selection or tuning.",
        "",
        "## Part C: Cu Concentration Distribution", "",
        "| Range | All known | Train | Validation | Test |", "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label in FIXED_BIN_LABELS:
        lines.append(
            f"| {label} | {summary['fixed_range_counts_all'][label]} | "
            f"{summary['fixed_range_counts_train'][label]} | {summary['fixed_range_counts_val'][label]} | "
            f"{summary['fixed_range_counts_test'][label]} |"
        )
    lines.extend(["", "### Positive-Only Quantile Ranges (all known)", "", "| Range | Samples |", "| --- | ---: |"])
    for name, value in summary["positive_quantile_range_counts"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend([
        "",
        "### High-Concentration Tail Inspection", "",
        f"- Known Cu labels above the existing {EXISTING_CU_DETECTION_THRESHOLD_PPM:g} ppm threshold: "
        f"{summary['above_threshold_count']} of {summary['known_cu_labels']}.",
        f"- Maximum observed known Cu concentration: {overall['maximum']} ppm.",
        "- See `cu_tail_distribution.png` and `cu_sorted_distribution.png` for the shape of the tail.",
        "",
        "## Part D: Existing Cu Regression Failure (Reused Diagnostics, Not Retrained)", "",
        f"- Source: `{old['source']}`",
        f"- Checkpoint examined (not modified): `{old['checkpoint_examined']}`",
        f"- Validation truth range: {old['validation_truth_range_ppm']} ppm",
        f"- Validation prediction range: {old['validation_prediction_range_ppm']} ppm",
        f"- Test truth range: {old['test_truth_range_ppm']} ppm",
        f"- Test prediction range: {old['test_prediction_range_ppm']} ppm",
        f"- Validation Pearson / Spearman: {old['validation_pearson']} / {old['validation_spearman']}",
        f"- Test Pearson / Spearman: {old['test_pearson']} / {old['test_spearman']}",
        f"- Validation top-10%/top-5% share of SSE: {old['validation_top10pct_sse_share']:.2%} / {old['validation_top5pct_sse_share']:.2%}",
        f"- Test top-10%/top-5% share of SSE: {old['test_top10pct_sse_share']:.2%} / {old['test_top5pct_sse_share']:.2%}",
        f"- Diagnosis: {old['diagnosis']}",
        f"- Primary associated issue: {old['primary_associated_issue']}",
        "",
        "## Part E: Recommended Cu Strategy", "",
        f"- Recommended strategy: {summary['recommended_strategy']}",
        f"- Reason: {summary['recommendation_reason']}",
        f"- Candidate supported concentration range (TRAIN/VALIDATION only, TEST excluded): "
        f"{summary['candidate_supported_range'] or 'none identified'}",
        "",
        "## Test Set Status", "",
        "TEST WAS NOT USED FOR MODEL PERFORMANCE OR MODEL SELECTION. It was used only for descriptive "
        "label counts and distribution reporting, and only train/validation define the candidate supported range.",
    ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metadata_path = ROOT / args.metadata
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, _, sheet_name = load_metadata(metadata_path)
    cu_column = find_column(list(metadata.columns), "Cu")
    frame = load_frozen_splits(ROOT / args.splits_dir, metadata, cu_column)
    known = frame["cu_ppm"].dropna()

    split_statistics = {split: split_support(frame, split) for split in SPLITS}
    supported_range, per_bin_support = candidate_supported_range(frame)
    strategy, reason = recommend_strategy(known, split_statistics, supported_range)

    summary: dict[str, Any] = {
        "metadata_workbook": str(metadata_path.relative_to(ROOT)),
        "metadata_sheet": sheet_name,
        "cu_metadata_column": cu_column,
        "cu_unit": "ppm",
        "existing_cu_detection_threshold_ppm": EXISTING_CU_DETECTION_THRESHOLD_PPM,
        "threshold_modified": False,
        "label_source": "Direct Cu composition metadata field in libs_metadata.xlsx; not inferred from filenames, paths, plots, or model outputs.",
        "total_spectra": int(len(frame)),
        "known_cu_labels": int(known.size),
        "missing_cu_labels": int(frame["cu_ppm"].isna().sum()),
        "cu_zero_ppm": int((known == 0).sum()),
        "cu_positive_ppm": int((known > 0).sum()),
        "all_known_statistics": numeric_summary(known),
        "positive_only_statistics": numeric_summary(known[known > 0]),
        "fixed_range_counts_all": fixed_range_counts(known),
        "fixed_range_counts_train": fixed_range_counts(frame.loc[frame["split"] == "train", "cu_ppm"]),
        "fixed_range_counts_val": fixed_range_counts(frame.loc[frame["split"] == "val", "cu_ppm"]),
        "fixed_range_counts_test": fixed_range_counts(frame.loc[frame["split"] == "test", "cu_ppm"]),
        "positive_quantile_range_counts": positive_quantile_ranges(known),
        "above_threshold_count": int((known > EXISTING_CU_DETECTION_THRESHOLD_PPM).sum()),
        "split_statistics": split_statistics,
        "leakage_detected": False,
        "supported_bin_support_by_range": per_bin_support,
        "candidate_supported_range": supported_range,
        "recommended_strategy": strategy,
        "recommendation_reason": reason,
        "old_cu_regression_diagnosis": OLD_CU_REGRESSION_DIAGNOSIS,
        "test_set_status": "NOT USED FOR MODEL PERFORMANCE OR MODEL SELECTION",
    }
    save_plots(frame, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")

    old = OLD_CU_REGRESSION_DIAGNOSIS
    print("CU DATA AUDIT")
    print(f"Total spectra: {summary['total_spectra']}")
    print(f"Known Cu: {summary['known_cu_labels']}")
    print(f"Missing Cu: {summary['missing_cu_labels']}")
    print(f"Cu = 0: {summary['cu_zero_ppm']}")
    print(f"Cu > 0: {summary['cu_positive_ppm']}")
    print()
    print("TRAIN")
    print(f"Known: {split_statistics['train']['known']}")
    print(f"Zero: {split_statistics['train']['zero']}")
    print(f"Positive: {split_statistics['train']['positive']}")
    print()
    print("VALIDATION")
    print(f"Known: {split_statistics['val']['known']}")
    print(f"Zero: {split_statistics['val']['zero']}")
    print(f"Positive: {split_statistics['val']['positive']}")
    print()
    print("TEST")
    print(f"Known: {split_statistics['test']['known']}")
    print(f"Zero: {split_statistics['test']['zero']}")
    print(f"Positive: {split_statistics['test']['positive']}")
    print()
    print(f"Cu median: {summary['all_known_statistics']['median']}")
    print(f"Cu P95: {summary['all_known_statistics']['percentile_95']}")
    print(f"Cu maximum: {summary['all_known_statistics']['maximum']}")
    print()
    print("OLD CU REGRESSION DIAGNOSIS")
    print(f"Validation truth range: {old['validation_truth_range_ppm']}")
    print(f"Validation prediction range: {old['validation_prediction_range_ppm']}")
    print(f"Test truth range: {old['test_truth_range_ppm']}")
    print(f"Test prediction range: {old['test_prediction_range_ppm']}")
    print(f"Validation Pearson: {old['validation_pearson']}")
    print(f"Validation Spearman: {old['validation_spearman']}")
    print(f"Test Pearson: {old['test_pearson']}")
    print(f"Test Spearman: {old['test_spearman']}")
    print(f"High-tail error dominance: validation top10%={old['validation_top10pct_sse_share']:.2%}, test top10%={old['test_top10pct_sse_share']:.2%}")
    print()
    print(f"RECOMMENDED STRATEGY: {strategy}")
    print(f"Reason: {reason}")
    print()
    print("TEST STATUS:")
    print("NOT USED FOR MODEL PERFORMANCE OR MODEL SELECTION")
    print()
    print("Files created/modified:")
    for name in (
        "summary.json", "summary.md", "cu_histogram.png", "cu_log_histogram.png",
        "cu_sorted_distribution.png", "cu_split_distribution.png", "cu_tail_distribution.png",
    ):
        print(output_dir / name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reconcile the current NASA Z-903 inventory and create deterministic splits."""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_URL = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-libs_reference_database/data_plibs_z903/collection_data_plibs_z903_inventory.csv"
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
THRESHOLDS = {"Zn": ("Zn_raw", 100.0, "ppm"), "Mn": ("Mn_raw", 500.0, "ppm"), "Cd": ("Cd_raw", 0.05, "ppm"), "Mg": ("MgO_raw", 5.0, "wt% MgO"), "Cu": ("Cu_raw", 25.0, "ppm"), "Pb": ("Pb_raw", 10.0, "ppm")}


def fetch_inventory(url: str) -> tuple[list[str], int]:
    request = Request(url, headers={"User-Agent": "atomic-spectrum-cnn-copilot/0.1"})
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8-sig")
    product_ids: list[str] = []
    inventory_rows = 0
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        inventory_rows += 1
        if row[0].strip().upper() != "P":
            continue
        identifier = row[-1].strip()
        stem = identifier.rsplit("::", 1)[0].split(":")[-1]
        if stem.lower().startswith("plibs_z903_"):
            product_ids.append(stem[len("plibs_z903_") :])
    return sorted(set(product_ids)), inventory_rows


def signature(row: pd.Series) -> tuple[int, ...]:
    values: list[int] = []
    for element in TARGETS:
        column, threshold, _ = THRESHOLDS[element]
        value = pd.to_numeric(row[column], errors="coerce")
        values.append(-1 if pd.isna(value) else int(value > threshold))
    return tuple(values)


def make_splits(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for index, row in frame.iterrows():
        buckets.setdefault(signature(row), []).append(index)
    assignments: dict[int, str] = {}
    fractions = (("train", 0.70), ("val", 0.15), ("test", 0.15))
    for indices in buckets.values():
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        start = 0
        for split, fraction in fractions[:-1]:
            count = int(round(len(shuffled) * fraction))
            for index in shuffled[start : start + count]:
                assignments[int(index)] = split
            start += count
        for index in shuffled[start:]:
            assignments[int(index)] = "test"
    result = {}
    for split in ("train", "val", "test"):
        result[split] = frame.loc[[index for index, value in assignments.items() if value == split]].sort_values("target_id").reset_index(drop=True)
    return result


def count_lines(frame: pd.DataFrame) -> list[str]:
    lines = ["| Element | Known | Positive | Negative | Missing |", "| --- | ---: | ---: | ---: | ---: |"]
    for element in TARGETS:
        column, threshold, _ = THRESHOLDS[element]
        values = pd.to_numeric(frame[column], errors="coerce")
        known = int(values.notna().sum())
        positive = int((values > threshold).sum())
        lines.append(f"| {element} | {known} | {positive} | {known - positive} | {len(frame) - known} |")
    return lines


def write_report(path: Path, inventory_ids: list[str], inventory_rows: int, local_ids: set[str], splits: dict[str, pd.DataFrame], seed: int) -> None:
    missing = sorted(set(inventory_ids) - local_ids)
    extra = sorted(local_ids - set(inventory_ids))
    group_counts = Counter(splits["train"].get("group_id", pd.Series(dtype=str)).tolist() + splits["val"].get("group_id", pd.Series(dtype=str)).tolist() + splits["test"].get("group_id", pd.Series(dtype=str)).tolist())
    lines = ["# Z-903 Archive Reconciliation and Splits", "", "## NASA PDS Archive Reconciliation", "", f"- Official inventory URL: {INVENTORY_URL}", f"- Inventory rows: {inventory_rows}", f"- Primary Z-903 products / represented spectral CSV target IDs: {len(inventory_ids)}", f"- Downloaded local spectral CSVs: {len(local_ids)}", f"- Inventory target IDs missing locally: {len(missing)}", f"- Local target IDs absent from inventory: {len(extra)}", "", "The current machine-readable inventory is the authoritative archive count used here. The User Guide's 2,686 figure is not silently substituted for the current inventory count. A difference is therefore reported as a version/archive-accounting discrepancy unless the inventory exposes a more specific reason.", ""]
    if missing:
        lines.append("Missing local target IDs: " + ", ".join(missing))
    if extra:
        lines.append("Local IDs absent from inventory: " + ", ".join(extra))
    lines.extend(["", "## Split Method", "", f"- Seed: {seed}", "- Split fractions: approximately 70% train, 15% validation, 15% test", "- Assignment unit: complete target_id / group_id", "- Missing labels remain missing; exploratory positive/negative counts below use only known values.", "- Signature-stratified deterministic assignment preserves the joint positive/negative/missing patterns as far as bucket sizes allow.", ""])
    for split in ("train", "val", "test"):
        lines.extend([f"## {split}", "", f"Rows: {len(splits[split])}", "", *count_lines(splits[split]), ""])
    lines.extend(["## Group Integrity", "", f"Unique groups across splits: {len(group_counts)}", f"Groups appearing in more than one split: {sum(count > 1 for count in group_counts.values())}", ""])
    if sum(count > 1 for count in group_counts.values()):
        raise ValueError("Group leakage detected while writing split report")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/z903_training_manifest.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--inventory-url", default=INVENTORY_URL)
    parser.add_argument("--report", default="reports/z903_splits_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir
    frame = pd.read_csv(manifest_path)
    required = {"target_id", "group_id", *[column for column, _, _ in THRESHOLDS.values()]}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")
    if frame["target_id"].duplicated().any() or frame["group_id"].duplicated().any():
        raise ValueError("Primary target_id/group_id must be unique under the NASA target-level grouping rule")
    inventory_ids, inventory_rows = fetch_inventory(args.inventory_url)
    local_ids = set(frame["target_id"].astype(str))
    splits = make_splits(frame, args.seed)
    for split, split_frame in splits.items():
        split_frame.to_csv(output_dir / f"z903_{split}.csv", index=False)
    write_report(ROOT / args.report, inventory_ids, inventory_rows, local_ids, splits, args.seed)
    print(f"Current archive primary products: {len(inventory_ids)}")
    print(f"Inventory rows: {inventory_rows}")
    print(f"Local spectra: {len(local_ids)}")
    print(f"Missing locally: {len(set(inventory_ids) - local_ids)}")
    print(f"Train/val/test sizes: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Unique target IDs/groups: {frame['target_id'].nunique()}/{frame['group_id'].nunique()}")
    for split in ("train", "val", "test"):
        print(f"[{split}]")
        for line in count_lines(splits[split])[2:]:
            print(line)
    print("Splits written to", output_dir)


if __name__ == "__main__":
    main()
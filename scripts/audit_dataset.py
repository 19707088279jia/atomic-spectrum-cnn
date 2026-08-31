#!/usr/bin/env python
"""Audit downloaded NASA PDS pLIBS Z-903 spectra and metadata."""

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
ELEMENTS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
THRESHOLDS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def read_spectrum(path: Path) -> tuple[int, float, float, int]:
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        header = next(handle, "").strip().split(",")
    fields = {normalize(field): index for index, field in enumerate(header)}
    wavelength_index = next((fields[key] for key in fields if key in {"wavelength", "wavelengthnm"}), None)
    intensity_index = next((fields[key] for key in fields if key in {"intensity", "sum"}), None)
    if wavelength_index is None or intensity_index is None:
        raise ValueError(f"missing wavelength/intensity columns: {header}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(wavelength_index, intensity_index), ndmin=2)
    if data.size == 0:
        raise ValueError("no data rows")
    if not np.isfinite(data).all():
        raise ValueError("non-finite wavelength or intensity")
    wavelengths = data[:, 0]
    return int(data.shape[0]), float(wavelengths.min()), float(wavelengths.max()), int(np.unique(wavelengths).size)


def load_metadata(path: Path) -> tuple[pd.DataFrame, str, dict[str, str]]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = {name: frame.dropna(how="all").reset_index(drop=True) for name, frame in sheets.items()}
    frames = {name: frame for name, frame in frames.items() if not frame.empty}
    if not frames:
        raise ValueError("workbook contains no non-empty sheets")
    sheet_name, frame = max(frames.items(), key=lambda item: item[1].shape[0] * item[1].shape[1])
    frame.columns = [str(column).strip() for column in frame.columns]
    column_map: dict[str, str] = {}
    for element in ELEMENTS:
        candidates = [column for column in frame.columns if re.search(rf"(^|[^A-Za-z]){element}(?=$|[^A-Za-z])", column, re.I)]
        if not candidates:
            candidates = [column for column in frame.columns if normalize(element) in normalize(column)]
        if candidates:
            column_map[element] = candidates[0]
    return frame, sheet_name, column_map


def find_metadata_matches(frame: pd.DataFrame, files: list[Path]) -> tuple[dict[str, str], list[str]]:
    identifier_columns = [
        column
        for column in frame.columns
        if any(marker in normalize(column) for marker in ("pellet", "sample", "specimen", "file", "name", "id"))
    ]
    value_rows: dict[str, set[int]] = {}
    for column in identifier_columns:
        for index, value in frame[column].items():
            key = normalize(value)
            if key:
                value_rows.setdefault(key, set()).add(index)

    token_rows: dict[str, set[int]] = {}
    for index, row in frame.astype(str).fillna("").iterrows():
        row_tokens = set(re.findall(r"[a-z0-9]+", " ".join(row.tolist()).lower()))
        for token in row_tokens:
            if len(token) >= 4:
                token_rows.setdefault(token, set()).add(index)

    rows: dict[str, str] = {}
    for file_path in files:
        stem = normalize(file_path.stem)
        sample_key = re.sub(r"^plibsz903", "", stem)
        exact_matches = value_rows.get(sample_key, set())
        if not exact_matches:
            exact_matches = value_rows.get(stem, set())
        if exact_matches:
            rows[file_path.name] = str(min(exact_matches))
            continue
        candidates: set[int] = set()
        for token in re.findall(r"[a-z0-9]+", stem):
            candidates.update(token_rows.get(token, set()))
        if candidates:
            rows[file_path.name] = str(min(candidates))
    matched_rows = set(rows.values())
    unmatched_metadata = [str(index) for index in frame.index if str(index) not in matched_rows]
    return rows, unmatched_metadata


def write_reports(report_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "dataset_audit.csv", index=False)
    lines = ["# NASA PDS Z-903 Dataset Audit", "", f"Generated: {pd.Timestamp.now().isoformat()}", ""]
    for heading, value in summary.items():
        lines.extend([f"## {heading}", "", str(value), ""])
    (report_dir / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", default="data/raw/pds_z903")
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--reports", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spectra_dir = ROOT / args.spectra
    metadata_path = ROOT / args.metadata
    report_dir = ROOT / args.reports
    files = sorted(spectra_dir.glob("*.csv"))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_path}")

    good_rows: list[dict[str, Any]] = []
    corrupted: list[str] = []
    for path in files:
        try:
            channels, minimum, maximum, unique_channels = read_spectrum(path)
            good_rows.append({"file": path.name, "channels": channels, "unique_channels": unique_channels, "min_wavelength_nm": minimum, "max_wavelength_nm": maximum, "status": "readable"})
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            corrupted.append(f"{path.name}: {exc}")

    frame, sheet_name, column_map = load_metadata(metadata_path)
    matches, unmatched_metadata = find_metadata_matches(frame, [spectra_dir / row["file"] for row in good_rows])
    distributions: dict[str, str] = {}
    available_counts: dict[str, int] = {}
    threshold_lines: list[str] = []
    for element in ELEMENTS:
        column = column_map.get(element)
        if column is None:
            available_counts[element] = 0
            distributions[element] = "metadata column not found"
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        available_counts[element] = int(values.size)
        distributions[element] = values.describe().to_string()
        for threshold in THRESHOLDS:
            positive = int((values > threshold).sum())
            negative = int((values <= threshold).sum())
            threshold_lines.append(f"{element} | > {threshold:g} | {positive} | {negative}")

    channel_counts = Counter(row["channels"] for row in good_rows)
    wavelength_min = min((row["min_wavelength_nm"] for row in good_rows), default=None)
    wavelength_max = max((row["max_wavelength_nm"] for row in good_rows), default=None)
    summary = {
        "Sample and Spectrum Counts": f"Metadata rows: {len(frame)}\nSpectral files discovered locally: {len(files)}\nReadable spectra: {len(good_rows)}\nCorrupted/unreadable spectra: {len(corrupted)}",
        "Wavelength and Channels": f"Range: {wavelength_min} to {wavelength_max} nm\nChannel counts: {dict(channel_counts)}",
        "Metadata Element Columns": f"Workbook sheet: {sheet_name}\n" + "\n".join(f"{element}: {column_map.get(element, 'NOT FOUND')}" for element in ELEMENTS),
        "Composition Availability": "\n".join(f"{element}: {available_counts[element]}" for element in ELEMENTS),
        "Concentration Distributions": "\n\n".join(f"{element}:\n{distributions[element]}" for element in ELEMENTS),
        "Threshold Counts": "Element | threshold | positive (> threshold) | negative (<= threshold)\n--- | --- | --- | ---\n" + "\n".join(line.replace(" | ", " | ") for line in threshold_lines),
        "Missing Metadata and Matching": f"Spectra matched to metadata rows: {len(matches)}\nSpectrum files without a metadata match: {len(good_rows) - len(matches)}\nMetadata rows without a corresponding spectrum match: {len(unmatched_metadata)}",
        "Corrupted or Unreadable Files": "\n".join(corrupted) if corrupted else "None detected",
    }
    write_reports(report_dir, summary, good_rows + [{"file": item, "status": "corrupted"} for item in corrupted])
    print(f"Metadata samples: {len(frame)}")
    print(f"Spectra discovered: {len(files)}")
    print(f"Readable spectra: {len(good_rows)}")
    print(f"Corrupted/unreadable spectra: {len(corrupted)}")
    for element in ELEMENTS:
        print(f"{element} samples with composition data: {available_counts[element]}")
    print(f"Reports written to {report_dir}")


if __name__ == "__main__":
    main()
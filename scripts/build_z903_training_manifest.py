#!/usr/bin/env python
"""Build a matched, masked-label manifest for the downloaded pLIBS Z-903 data."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
THRESHOLDS = {"Zn": ("Zn_raw", 100.0, "ppm"), "Mn": ("Mn_raw", 500.0, "ppm"), "Cd": ("Cd_raw", 0.05, "ppm"), "Mg": ("MgO_raw", 5.0, "wt% MgO"), "Cu": ("Cu_raw", 25.0, "ppm"), "Pb": ("Pb_raw", 10.0, "ppm")}
OXIDE_FACTOR_MG = 24.305 / (24.305 + 15.999)


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(columns: list[str], target: str) -> str:
    exact = [column for column in columns if normalize(column) == normalize(target)]
    if exact:
        return exact[0]
    candidates = [column for column in columns if normalize(target) in normalize(column)]
    if not candidates:
        raise ValueError(f"Metadata column not found for {target}")
    return candidates[0]


def load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, str], str]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = {name: frame for name, frame in sheets.items() if not frame.dropna(how="all").empty}
    if not frames:
        raise ValueError(f"No non-empty worksheets found in {path}")
    sheet_name, frame = max(frames.items(), key=lambda item: item[1].shape[0] * item[1].shape[1])
    frame.columns = [str(column).strip() for column in frame.columns]
    units_row = next(
        (index for index, row in frame.iterrows() if "ppm" in " ".join(str(value) for value in row).lower() or "wt %" in " ".join(str(value) for value in row)),
        None,
    )
    if units_row is None:
        raise ValueError("Could not identify the metadata units row")
    columns = {target: find_column(list(frame.columns), target) for target in TARGETS}
    frame = frame.drop(index=units_row).reset_index(drop=True)
    frame["_sample_key"] = frame["PELLET NAME"].map(normalize)
    frame = frame[frame["_sample_key"].ne("")].copy()
    if frame["_sample_key"].duplicated().any():
        duplicates = frame.loc[frame["_sample_key"].duplicated(), "PELLET NAME"].tolist()
        raise ValueError(f"Duplicate metadata sample IDs prevent a safe join: {duplicates[:5]}")
    return frame, columns, sheet_name


def sample_id_from_filename(path: Path) -> str:
    prefix = "plibs_z903_"
    if not path.stem.lower().startswith(prefix):
        raise ValueError(f"Unexpected Z-903 filename: {path.name}")
    return path.stem[len(prefix) :]


def group_candidate(sample_id: str) -> str:
    """Retain the optional filename-family diagnostic without using it for grouping."""
    candidate = re.sub(r"[A-Za-z]+$", "", sample_id).upper()
    if not candidate or not any(character.isdigit() for character in candidate):
        return sample_id.upper()
    return candidate


def spectrum_structure(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["wavelength", "intensity"]:
        raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
    wavelengths = pd.to_numeric(frame["wavelength"], errors="raise").to_numpy()
    spacing = np.diff(wavelengths)
    return {"columns": ", ".join(frame.columns), "rows": len(frame), "wavelength_min": wavelengths.min(), "wavelength_max": wavelengths.max(), "spacing_min": spacing.min(), "spacing_max": spacing.max(), "spacing_median": np.median(spacing), "intensity_columns": "intensity", "multiple_replicates": False, "averaged_status": "not determinable from the two-column CSV"}


def concentration_bins(element: str, values: pd.Series) -> list[tuple[str, int]]:
    if element == "Mg":
        edges = [-1e-9, 1e-9, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, np.inf]
        labels = ["0", ">0-0.01", "0.01-0.1", "0.1-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">=50"]
    else:
        edges = [-1e-9, 1e-9, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, np.inf]
        labels = ["0", ">0-0.01", "0.01-0.1", "0.1-1", "1-10", "10-100", "100-1000", "1000-10000", "10000-100000", ">=100000"]
    numeric = pd.to_numeric(values, errors="coerce")
    counts = pd.cut(numeric, bins=edges, labels=labels, include_lowest=True, right=False).value_counts(sort=False)
    return [(str(label), int(counts.get(label, 0))) for label in labels]


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(path: Path, rows: list[dict[str, Any]], structures: list[dict[str, Any]], group_counts: Counter[str], metadata_columns: dict[str, str], sheet_name: str, matched_metadata_rows: int) -> None:
    total = len(rows)
    lines = ["# Z-903 Training Manifest Report", "", "NASA User Guide grouping rule: each exported Z-903 CSV is one averaged 4x3-raster target-level measurement. The complete filename target ID is the primary sample and leakage group identity. Filename-family information is diagnostic only and is not used as the group ID.", "", "## Manifest Integrity", "", f"- Total spectrum rows: {total}", f"- Unique spectrum files: {len({row['spectrum_filename'] for row in rows})}", f"- Matched metadata rows: {matched_metadata_rows}", f"- Unique target IDs: {len({row['target_id'] for row in rows})}", f"- Unique group IDs: {len(group_counts)}", f"- Possible filename families (diagnostic only): {len({row['possible_family_id'] for row in rows})}", "", "## Label Counts Restricted to Matched Z-903 Spectra", "", "| Element | Known labels | Positive | Negative | Missing | Positive % of known |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for element, (column, threshold, _unit) in THRESHOLDS.items():
        mask = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce").notna()
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce")
        positive = int((values > threshold).sum())
        known = int(mask.sum())
        negative = known - positive
        lines.append(f"| {element} | {known} | {positive} | {negative} | {total - known} | {100 * positive / known:.2f}% |")
    lines.extend(["", "Exploratory thresholds only; they are not validated LIBS detection limits.", "", "## Exploratory Threshold Partitions", "", "| Element | Threshold | Unit | Known | Positive | Negative | Missing |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for element, (column, threshold, unit) in THRESHOLDS.items():
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce")
        known = int(values.notna().sum())
        positive = int((values > threshold).sum())
        lines.append(f"| {element} | {threshold:g} | {unit} | {known} | {positive} | {known - positive} | {total - known} |")
    lines.extend(["", "## Concentration Ranges", "", "Bins are exploratory distribution bins. ppm elements use logarithmic-style bins; Mg uses MgO wt% bins.", ""])
    for element, (column, _, unit) in THRESHOLDS.items():
        lines.extend([f"### {element} ({unit})", "", "| Range | Samples |", "| --- | ---: |"])
        for label, count in concentration_bins(element, pd.Series([row[column] for row in rows])):
            lines.append(f"| {label} | {count} |")
        missing = sum(pd.isna(row[column]) for row in rows)
        lines.append(f"| missing | {missing} |")
        lines.append("")
    lines.extend(["## Actual Spectrum Structure", "", "| Representative file | Columns | Rows | Wavelength range | Spacing median | Intensity columns | Replicates in file | Averaged status |", "| --- | --- | ---: | --- | ---: | --- | --- | --- |"])
    for structure in structures:
        lines.append(f"| {structure['filename']} | {structure['columns']} | {structure['rows']} | {structure['wavelength_min']:.6g}-{structure['wavelength_max']:.6g} nm | {structure['spacing_median']:.8f} nm | {structure['intensity_columns']} | no | {structure['averaged_status']} |")
    lines.extend(["", "All inspected CSVs contain one two-column wavelength/intensity vector with 23,401 channels and approximately 0.03333333 nm spacing. The NASA User Guide states that 12 datapoints from a 4x3 raster were averaged before export, so one CSV is treated as one averaged target-level spectrum.", "", "## Leakage Groups", "", f"Metadata sheet: `{sheet_name}`", "", "| Metadata column | Role |", "| --- | --- |", "| PELLET NAME | metadata join key for target_id |"])
    for element, column in metadata_columns.items():
        lines.append(f"| {column} | {element} composition source |")
    lines.extend(["", "No separate material, aliquot, replicate, or group column is available in the workbook. NASA states that reference target names are unique, so trailing-letter filename families are not treated as physical replicates. `group_id` equals the complete target ID; `possible_family_id` is retained only for later investigation.", "", "| Group size | Number of groups | Spectra |", "| ---: | ---: | ---: |"])
    for size in sorted(set(group_counts.values())):
        lines.append(f"| {size} | {sum(count == size for count in group_counts.values())} | {sum(count for count in group_counts.values() if count == size)} |")
    lines.extend(["", "## Manifest Columns", "", "The six `_mask` columns are 1 when the corresponding composition is known and 0 when it is missing. Missing labels are not converted to negative labels; future masked binary cross entropy should ignore rows with mask 0 for that target.", "", "## Verification", "", f"- Every manifest spectrum path exists: {all(Path(row['spectrum_path']).exists() for row in rows)}", f"- Positive + negative + missing equals {total} for every target: {all((pd.to_numeric(pd.Series([row[column] for row in rows]), errors='coerce') > threshold).sum() + (pd.to_numeric(pd.Series([row[column] for row in rows]), errors='coerce') <= threshold).sum() + pd.to_numeric(pd.Series([row[column] for row in rows]), errors='coerce').isna().sum() == total for column, threshold, _ in THRESHOLDS.values())}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", default="data/raw/pds_z903")
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--manifest", default="data/processed/z903_training_manifest.csv")
    parser.add_argument("--report", default="reports/z903_training_manifest_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spectra_dir = ROOT / args.spectra
    metadata_path = ROOT / args.metadata
    manifest_path = ROOT / args.manifest
    report_path = ROOT / args.report
    metadata, columns, sheet_name = load_metadata(metadata_path)
    metadata_by_key = metadata.set_index("_sample_key")
    files = sorted(spectra_dir.glob("plibs_z903_*.csv"))
    rows: list[dict[str, Any]] = []
    candidates: Counter[str] = Counter(group_candidate(sample_id_from_filename(path)) for path in files)
    for path in files:
        sample_id = sample_id_from_filename(path)
        sample_key = normalize(sample_id)
        if sample_key not in metadata_by_key.index:
            raise ValueError(f"Spectrum has no metadata match: {path.name}")
        metadata_row = metadata_by_key.loc[sample_key]
        family = group_candidate(sample_id)
        row: dict[str, Any] = {"spectrum_path": str(path), "spectrum_filename": path.name, "target_id": sample_id, "sample_id": str(metadata_row["PELLET NAME"]), "material_id": "", "group_id": sample_id, "possible_family_id": family if candidates[family] > 1 else ""}
        for element, (_, _, _) in THRESHOLDS.items():
            output_column = "MgO_raw" if element == "Mg" else f"{element}_raw"
            value = pd.to_numeric(pd.Series([metadata_row[columns[element]]]), errors="coerce").iloc[0]
            row[output_column] = value
            row[f"{element}_mask"] = int(pd.notna(value))
        row["Mg_elemental_equivalent"] = row["MgO_raw"] * OXIDE_FACTOR_MG if pd.notna(row["MgO_raw"]) else np.nan
        rows.append(row)
    write_manifest(manifest_path, rows)
    structures = []
    for path in (files[0], files[len(files) // 2], files[-1]):
        structure = spectrum_structure(path)
        structure["filename"] = path.name
        structures.append(structure)
    group_counts = Counter(row["group_id"] for row in rows)
    write_report(report_path, rows, structures, group_counts, columns, sheet_name, len(rows))
    print(f"Total spectra in manifest: {len(rows)}")
    print(f"Unique target IDs: {len({row['target_id'] for row in rows})}")
    print(f"Unique group IDs: {len(group_counts)}")
    print(f"Potential replicate groups: {sum(count > 1 for count in group_counts.values())}")
    for element, (column, threshold, unit) in THRESHOLDS.items():
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce")
        known = int(values.notna().sum())
        positive = int((values > threshold).sum())
        print(f"{element}: known={known} positive={positive} negative={known - positive} missing={len(rows) - known} threshold={threshold:g} {unit}")
    print(f"Complete six-element labels: {sum(all(row[f'{element}_mask'] for element in TARGETS) for row in rows)}")
    print(f"Manifest written to {manifest_path}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
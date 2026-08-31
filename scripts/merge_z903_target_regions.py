#!/usr/bin/env python
"""Merge selected empirical target lines into non-overlapping Z-903 regions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")


def main() -> None:
    source = pd.read_csv(ROOT / "data/processed/z903_empirical_target_lines.csv")
    selected = source[source["selected"]].copy()
    rows = []
    counts = {}
    for target in TARGETS:
        lines = selected[selected["target"] == target].sort_values("wavelength_nm")
        groups: list[list[dict]] = []
        for record in lines.to_dict("records"):
            record["window_start_nm"] = float(record["wavelength_nm"]) - 0.5
            record["window_end_nm"] = float(record["wavelength_nm"]) + 0.5
            if not groups or record["window_start_nm"] > max(item["window_end_nm"] for item in groups[-1]):
                groups.append([record])
            else:
                groups[-1].append(record)
        counts[target] = {"original_selected_lines": len(lines), "merged_regions": len(groups)}
        for region_id, group in enumerate(groups, start=1):
            rows.append({
                "target": target,
                "region_id": f"{target}_{region_id:03d}",
                "window_start_nm": min(item["window_start_nm"] for item in group),
                "window_end_nm": max(item["window_end_nm"] for item in group),
                "center_nm": sum(item["wavelength_nm"] for item in group) / len(group),
                "transition_count": len(group),
                "transitions": "; ".join(f"{item['ion_stage']}:{item['wavelength_nm']:.6f}" for item in group),
                "aggregate_score": max(item["final_line_score"] for item in group),
                "mean_effect_size": sum(item["effect_size"] for item in group) / len(group),
                "mean_positive_peak_prevalence": sum(item["positive_peak_prevalence"] for item in group) / len(group),
                "nearest_matrix_elements": "; ".join(sorted(set(str(item["nearest_matrix_element"]) for item in group))),
            })
    regions = pd.DataFrame(rows)
    regions.to_csv(ROOT / "data/processed/z903_target_regions.csv", index=False)
    windows = regions.assign(target_line_nm=regions["center_nm"], selected=True)
    windows.to_csv(ROOT / "data/processed/z903_target_windows.csv", index=False)
    (ROOT / "data/processed/z903_target_regions_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
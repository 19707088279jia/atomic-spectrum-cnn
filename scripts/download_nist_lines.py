#!/usr/bin/env python
"""Retrieve authoritative NIST ASD line tables using the live Lines Form fields."""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
ELEMENTS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb", "Ca", "Fe", "Al", "Si", "Na", "K", "Ti")
STAGES = ("I", "II")
FORM_PARAMS = {
    "output_type": "0",
    "unit": "1",
    "submit": "Retrieve Data",
    "de": "0",
    "plot_out": "0",
    "format": "2",
    "line_out": "0",
    "remove_js": "1",
    "no_spaces": "1",
    "en_unit": "0",
    "output": "0",
    "bibrefs": "1",
    "page_size": "15",
    "show_obs_wl": "1",
    "show_calc_wl": "1",
    "unc_out": "1",
    "order_out": "0",
    "show_av": "2",
    "tsb_value": "0",
    "A_out": "0",
    "intens_out": "1",
    "allowed_out": "1",
    "forbid_out": "1",
    "conf_out": "1",
    "term_out": "1",
    "enrg_out": "1",
    "J_out": "1",
}
REQUIRED_COLUMNS = {
    "observed_wavelength_nm": "obs_wl_air(nm)",
    "ritz_wavelength_nm": "ritz_wl_air(nm)",
    "relative_intensity": "intens",
    "Aki": "Aki(s^-1)",
    "lower_energy": "Ei(cm-1)",
    "upper_energy": "Ek(cm-1)",
}


def query(element: str, stage: str, low: float, high: float) -> tuple[str, dict[str, str], dict[str, str]]:
    params = {**FORM_PARAMS, "spectra": f"{element} {stage}", "low_w": str(low), "upp_w": str(high)}
    query_string = urlencode(params)
    request = Request(
        f"{NIST_URL}?{query_string}",
        headers={"User-Agent": "Mozilla/5.0 atomic-spectrum-cnn-copilot"},
    )
    with urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8", errors="replace")
        headers = {key: value for key, value in response.headers.items()}
        status = str(response.status)
    if status != "200" or "text/plain" not in headers.get("Content-Type", ""):
        raise RuntimeError(f"NIST request failed for {element} {stage}: HTTP {status}, headers={headers}")
    lines = body.splitlines()
    if not lines or not lines[0].startswith("obs_wl") or len(lines) < 2:
        raise RuntimeError(f"NIST response for {element} {stage} was not a line table: {lines[:3]}")
    return body, params, headers


def clean_cell(value: str) -> str:
    value = value.strip()
    if value.startswith('="') and value.endswith('"'):
        return value[2:-1]
    return value.strip('"')


def normalize_table(body: str, element: str, stage: str) -> pd.DataFrame:
    reader = csv.DictReader(io.StringIO(body))
    fieldnames = reader.fieldnames or []
    observed_column = next((name for name in fieldnames if name.startswith("obs_wl_") and name.endswith("(nm)")), None)
    ritz_column = next((name for name in fieldnames if name.startswith("ritz_wl_") and name.endswith("(nm)")), None)
    if observed_column is None and ritz_column is None:
        raise RuntimeError(f"NIST response for {element} {stage} has no wavelength columns: {fieldnames[:8]}")
    rows = []
    for raw_row in reader:
        row = {key: clean_cell(value or "") for key, value in raw_row.items() if key is not None}
        normalized = {"element": element, "ionization_stage": stage}
        sources = {**REQUIRED_COLUMNS, "observed_wavelength_nm": observed_column, "ritz_wavelength_nm": ritz_column}
        for output, source in sources.items():
            if source is None:
                normalized[output] = None
                continue
            value = row.get(source, "")
            try:
                normalized[output] = float(value) if value else None
            except ValueError:
                normalized[output] = None
        rows.append(normalized)
    table = pd.DataFrame(rows, columns=["element", "ionization_stage", *REQUIRED_COLUMNS])
    table = table[(table["observed_wavelength_nm"].notna()) | (table["ritz_wavelength_nm"].notna())]
    if table.empty:
        raise RuntimeError(f"NIST response for {element} {stage} contained no parseable wavelengths")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=float, default=180.0)
    parser.add_argument("--high", type=float, default=960.0)
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "nist_lines" / "raw"
    processed_dir = ROOT / "data" / "nist_lines" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    test_body, test_params, test_headers = query("Cu", "I", 300.0, 400.0)
    test_table = normalize_table(test_body, "Cu", "I")
    (raw_dir / "test_Cu_I_300_400.csv").write_text(test_body, encoding="utf-8")
    (raw_dir / "test_Cu_I_300_400.json").write_text(json.dumps({"url": NIST_URL, "parameters": test_params, "response_headers": test_headers, "retrieved": date.today().isoformat(), "records": len(test_table)}, indent=2), encoding="utf-8")
    print(f"Verified Cu I 300-400 nm query: {len(test_table)} line records")

    counts: dict[str, int] = {}
    for element in ELEMENTS:
        for stage in STAGES:
            body, params, headers = query(element, stage, args.low, args.high)
            table = normalize_table(body, element, stage)
            stem = f"{element}_{stage}_{int(args.low)}_{int(args.high)}"
            (raw_dir / f"{stem}.csv").write_text(body, encoding="utf-8")
            table.to_csv(processed_dir / f"{element}_{stage}.csv", index=False)
            (raw_dir / f"{stem}.json").write_text(json.dumps({"url": NIST_URL, "parameters": params, "response_headers": headers, "retrieved": date.today().isoformat(), "records": len(table)}, indent=2), encoding="utf-8")
            counts[f"{element} {stage}"] = len(table)
    (processed_dir / "provenance.json").write_text(json.dumps({"source": NIST_URL, "form": "https://physics.nist.gov/PhysRefData/ASD/lines_form.html", "parameters_template": FORM_PARAMS, "range_nm": [args.low, args.high], "retrieved": date.today().isoformat(), "counts": counts}, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
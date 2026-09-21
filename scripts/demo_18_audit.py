"""Forensic audit script (diagnostic only, read-only w.r.t. models/splits/weights).

Runs the existing frozen inference pipeline (unchanged) against the 18 demo_real_120
spectra that have all three (MgO, Ag, Cu) reference values known, and against the
suspicious kn88956 sample individually. Writes:
  reports/demo_18_inference_audit.csv

Does NOT retrain, tune thresholds, modify weights, or modify splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "atomic-spectrum-v2"
sys.path.insert(0, str(V2))

from src.ag_inference import classify_ag, load_ag_classifier, load_ag_concentration, predict_ag_concentration  # noqa: E402
from src.cu_inference import EXISTING_DETECTION_THRESHOLD_PPM, load_cu_concentration, predict_cu_concentration  # noqa: E402
from src.data import load_spectrum  # noqa: E402
from src.inference import load_checkpoint, predict  # noqa: E402
from src.labels import CNN_THRESHOLDS, RAW_THRESHOLDS  # noqa: E402
from src.mgo_ridge_inference import load_pipeline, predict_mgo  # noqa: E402

DEMO_DIR = ROOT / "demo_real_120"
REFERENCE_CSV = DEMO_DIR / "reference_values_120.csv"
SPLIT_DIR = V2 / "data" / "splits"
REPORTS = ROOT / "reports"


def resolve_split(target_id: str) -> str:
    for split in ("train", "val", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"z903_{split}.csv", usecols=["target_id"])
        if target_id.lower() in frame["target_id"].str.lower().to_numpy():
            return split
    return "unknown"


def main() -> None:
    reference = pd.read_csv(REFERENCE_CSV)
    fully_known = reference.dropna(subset=["MgO_reference_wt_pct", "Ag_reference_ppm", "Cu_reference_ppm"]).copy()
    print(f"Fully-labelled (MgO+Ag+Cu known) rows found: {len(fully_known)}")

    six_model, six_checkpoint = load_checkpoint()
    mgo_pipeline = load_pipeline()
    ag_classifier = load_ag_classifier()
    ag_model, ag_checkpoint = load_ag_concentration()
    cu_model, cu_checkpoint = load_cu_concentration()

    print("Resolved model paths:")
    print(" six-element CNN:", (V2 / "models" / "z903_cnn_robust_best.pt").resolve())
    print(" mgo ridge:", (V2 / "models" / "mgo_ridge_final.joblib").resolve())
    print(" ag classifier:", (ROOT / "models" / "ag_classifier_final.joblib").resolve())
    print(" ag concentration:", (ROOT / "models" / "ag_concentration_final.pt").resolve())
    print(" cu concentration:", (ROOT / "models" / "cu_supported_range_final.pt").resolve())

    rows = []
    for _, ref_row in fully_known.iterrows():
        filename = str(ref_row["spectrum_filename"])
        target_id = str(ref_row["target_id"])
        path = DEMO_DIR / filename
        wavelengths, intensity = load_spectrum(path)

        detections = predict(intensity, six_model)
        mg_prob = detections["Mg"]["probability"]
        mg_result = detections["Mg"]["result"]
        cu_prob = detections["Cu"]["probability"]
        cu_result = detections["Cu"]["result"]

        mgo_pred = predict_mgo(intensity, mgo_pipeline)

        ag_class = classify_ag(intensity, ag_classifier)
        ag_positive = ag_class["positive"]
        ag_pred = predict_ag_concentration(intensity, ag_model, ag_checkpoint) if ag_positive else np.nan

        cu_pred = predict_cu_concentration(intensity, cu_model, cu_checkpoint)

        mgo_ref = float(ref_row["MgO_reference_wt_pct"])
        ag_ref = float(ref_row["Ag_reference_ppm"])
        cu_ref = float(ref_row["Cu_reference_ppm"])

        mg_ref_class = "POSITIVE" if mgo_ref > RAW_THRESHOLDS["Mg"] else "NEGATIVE"
        mg_pred_class = "POSITIVE" if mg_result == "DETECTED" else "NEGATIVE"

        ag_ref_class = "POSITIVE" if ag_ref > 0 else "ZERO"
        ag_pred_class = "POSITIVE" if ag_positive else "ZERO"

        cu_ref_class = "POSITIVE" if cu_ref > EXISTING_DETECTION_THRESHOLD_PPM else "NEGATIVE"
        cu_pred_class = "POSITIVE" if cu_result == "DETECTED" else "NEGATIVE"

        split = resolve_split(target_id)

        rows.append({
            "spectrum_filename": filename,
            "target_id": target_id,
            "split": split,
            "MgO_reference": mgo_ref,
            "MgO_prediction": mgo_pred,
            "MgO_absolute_error": abs(mgo_pred - mgo_ref),
            "Mg_detect_probability": mg_prob,
            "Mg_detect_threshold": CNN_THRESHOLDS["Mg"],
            "Mg_reference_class": mg_ref_class,
            "Mg_predicted_class": mg_pred_class,
            "Mg_class_correct": mg_ref_class == mg_pred_class,
            "Ag_reference": ag_ref,
            "Ag_prediction": ag_pred,
            "Ag_absolute_error": abs(ag_pred - ag_ref) if ag_positive else np.nan,
            "Ag_classifier_probability": ag_class["probability"],
            "Ag_classifier_threshold": ag_class["threshold"],
            "Ag_reference_class": ag_ref_class,
            "Ag_predicted_class": ag_pred_class,
            "Ag_class_correct": ag_ref_class == ag_pred_class,
            "Cu_reference": cu_ref,
            "Cu_prediction": cu_pred,
            "Cu_absolute_error": abs(cu_pred - cu_ref),
            "Cu_detect_probability": cu_prob,
            "Cu_detect_threshold": CNN_THRESHOLDS["Cu"],
            "Cu_reference_class": cu_ref_class,
            "Cu_predicted_class": cu_pred_class,
            "Cu_class_correct": cu_ref_class == cu_pred_class,
        })

    audit = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "demo_18_inference_audit.csv"
    audit.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(audit)} rows)")

    def metric_block(name: str, ref_col: str, pred_col: str) -> dict[str, float]:
        sub = audit.dropna(subset=[pred_col])
        y = sub[ref_col].to_numpy(dtype=float)
        p = sub[pred_col].to_numpy(dtype=float)
        errors = p - y
        return {
            "n": int(len(y)),
            "mae": float(np.mean(np.abs(errors))) if len(y) else float("nan"),
            "median_ae": float(np.median(np.abs(errors))) if len(y) else float("nan"),
            "rmse": float(np.sqrt(np.mean(errors**2))) if len(y) else float("nan"),
            "bias": float(np.mean(errors)) if len(y) else float("nan"),
            "min_error": float(np.min(np.abs(errors))) if len(y) else float("nan"),
            "max_error": float(np.max(np.abs(errors))) if len(y) else float("nan"),
        }

    print("\n== MgO diagnostics (n=18) ==")
    print(metric_block("MgO", "MgO_reference", "MgO_prediction"))
    print("\n== Ag diagnostics (only positive-classified subset has a prediction) ==")
    print(metric_block("Ag", "Ag_reference", "Ag_prediction"))
    print("\n== Cu diagnostics (n=18) ==")
    print(metric_block("Cu", "Cu_reference", "Cu_prediction"))

    for element in ("Mg", "Ag", "Cu"):
        correct_col = f"{element}_class_correct"
        ref_col = f"{element}_reference_class"
        pred_col = f"{element}_predicted_class"
        total = len(audit)
        correct = int(audit[correct_col].sum())
        pos_label = "POSITIVE"
        fp = int(((audit[ref_col] != pos_label) & (audit[pred_col] == pos_label)).sum())
        fn = int(((audit[ref_col] == pos_label) & (audit[pred_col] != pos_label)).sum())
        print(f"\n{element} classification: {correct}/{total} correct, FP={fp}, FN={fn}")

    print("\nSplit membership counts:")
    print(audit["split"].value_counts())

    print("\n=== kn88956 direct Python inference (bypassing Streamlit) ===")
    kn_row = audit[audit["spectrum_filename"] == "plibs_z903_kn88956.csv"].iloc[0]
    for col in [
        "MgO_prediction", "Mg_detect_probability", "Mg_detect_threshold", "Mg_predicted_class",
        "Ag_classifier_probability", "Ag_classifier_threshold", "Ag_predicted_class", "Ag_prediction",
        "Cu_detect_probability", "Cu_detect_threshold", "Cu_predicted_class", "Cu_prediction",
    ]:
        print(f"  {col}: {kn_row[col]}")


if __name__ == "__main__":
    main()

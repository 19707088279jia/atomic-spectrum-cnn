"""Minimal Atomic Spectrum Identification V2 application."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEMO_DIR, demo_labels, load_demo, load_spectrum  # noqa: E402
from src.inference import load_checkpoint, predict  # noqa: E402
from src.labels import TARGETS  # noqa: E402
from src.mgo_ridge_inference import build_display, load_pipeline, predict_mgo  # noqa: E402
from src.reference_matching import match  # noqa: E402

st.set_page_config(page_title="Atomic Spectrum Identification", layout="wide")
st.title("Atomic Spectrum Identification")


def spectrum_plot(wavelengths, intensity) -> None:
    figure, axis = plt.subplots(figsize=(10, 4), dpi=120)
    axis.plot(wavelengths, intensity, color="black", linewidth=0.8)
    axis.set_xlim(180.0, 960.0)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Intensity")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def main() -> None:
    demo_metadata = pd.read_csv(DEMO_DIR / "demo_ground_truth.csv")
    demo_names = ["No demo sample", *demo_metadata["sample_name"].tolist()]
    selected = st.selectbox("Built-in NASA demo sample", demo_names)
    uploaded = st.file_uploader("Upload numerical CSV", type=["csv"])
    if selected == "No demo sample" and uploaded is None:
        st.info("Select a built-in NASA demo or upload a CSV with wavelength,intensity columns.")
        return
    try:
        if selected != "No demo sample":
            wavelengths, intensity, truth_row = load_demo(selected)
        else:
            wavelengths, intensity = load_spectrum(uploaded)
            truth_row = None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return

    st.subheader("Spectrum")
    spectrum_plot(wavelengths, intensity)

    model, _ = load_checkpoint()
    cnn = predict(intensity, model)
    st.subheader("CNN Detection")
    st.dataframe(
        pd.DataFrame(
            [{"Element": element, **cnn[element]} for element in TARGETS]
        ),
        use_container_width=True,
        hide_index=True,
    )

    scores, _ = match(wavelengths, intensity)
    st.subheader("Reference / Peak Matching")
    st.dataframe(
        scores.rename(columns={"element": "Element", "score": "Score", "result": "Result", "matched_peaks": "Matched Peaks"}),
        use_container_width=True,
        hide_index=True,
    )

    if truth_row is not None:
        st.subheader("Demo composition ground truth")
        st.dataframe(
            pd.DataFrame({"Element": list(TARGETS), "Known Label": [demo_labels(truth_row)[element] for element in TARGETS]}),
            use_container_width=True,
            hide_index=True,
        )

    try:
        ridge_pipeline = load_pipeline()
        predicted_mgo = predict_mgo(intensity, ridge_pipeline)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return

    st.subheader("MgO Quantitative Estimate")
    if truth_row is not None:
        ground_truth = pd.to_numeric(truth_row["Mg"], errors="coerce")
        display = build_display(predicted_mgo, float(ground_truth) if pd.notna(ground_truth) else None)
    else:
        display = build_display(predicted_mgo)
    st.write(f"**Predicted MgO:** {display['Predicted MgO']:.2f} wt%")
    st.write(f"**Model:** {display['Model']}")
    st.write(f"**Status:** {display['Status']}")
    if "Ground Truth MgO" in display:
        st.write(f"**Ground Truth MgO:** {display['Ground Truth MgO']:.2f} wt%")
        st.write(f"**Absolute Error:** {display['Absolute Error']:.2f} wt%")

    with st.expander("MgO Model Performance"):
        st.write("Known test samples: 386")
        st.write("MAE: 1.6989 wt%")
        st.write("RMSE: 2.4681 wt%")
        st.write("R²: 0.8957")
        st.write("Pearson: 0.9467")
        st.write("Spearman: 0.9188")
        st.warning(
            "This model provides an experimental quantitative estimate. Some high-MgO "
            "samples, particularly around 20–35 wt%, remain systematically underestimated. "
            "The prediction is not a certified laboratory measurement."
        )


if __name__ == "__main__":
    main()

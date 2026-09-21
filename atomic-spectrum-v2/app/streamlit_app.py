"""LIBS Multi-Element Analysis Demo: integrates frozen MgO, Ag, and Cu models."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ag_inference import (  # noqa: E402
    LOW_CONCENTRATION_NOTE,
    ZERO_CLASS_NOTE,
    classify_ag,
    load_ag_classifier,
    load_ag_concentration,
    predict_ag_concentration,
)
from src.cu_inference import (  # noqa: E402
    EXISTING_DETECTION_THRESHOLD_PPM,
    HIGH_CONCENTRATION_NOTE,
    SUPPORTED_RANGE_PPM,
    load_cu_concentration,
    predict_cu_concentration,
)
from src.data import DEMO_DIR, load_demo, load_spectrum  # noqa: E402
from src.inference import load_checkpoint, predict  # noqa: E402
from src.mgo_ridge_inference import load_pipeline, predict_mgo  # noqa: E402

st.set_page_config(page_title="LIBS Multi-Element Analysis Demo", layout="wide")
st.title("LIBS Multi-Element Analysis Demo")
st.caption("Experimental MgO, Ag, and Cu identification and quantitative estimation")

MODEL_ERRORS = (OSError, ValueError, RuntimeError, KeyError)


def spectrum_plot(wavelengths, intensity) -> None:
    figure, axis = plt.subplots(figsize=(10, 4), dpi=120)
    axis.plot(wavelengths, intensity, color="black", linewidth=0.8)
    axis.set_xlim(180.0, 960.0)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Intensity")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def load_six_element_detections(intensity) -> tuple[dict[str, dict[str, object]] | None, str | None]:
    """Run the existing frozen six-element CNN once; both Mg and Cu detection reuse it."""
    try:
        model, _ = load_checkpoint()
        return predict(intensity, model), None
    except MODEL_ERRORS as exc:
        return None, str(exc)


def render_mg_section(intensity, detections, detection_error) -> None:
    with st.container(border=True):
        st.subheader("Mg / MgO")
        if detections is not None:
            result = detections["Mg"]["result"]
            st.metric("Detection", "Detected" if result == "DETECTED" else "Not Detected")
        else:
            st.error(f"Mg detection unavailable: {detection_error}")
        try:
            pipeline = load_pipeline()
            estimate = predict_mgo(intensity, pipeline)
        except MODEL_ERRORS as exc:
            st.error(f"MgO quantitative estimate unavailable: {exc}")
        else:
            st.metric("Experimental MgO quantitative estimate", f"{estimate:.2f} wt%")


def render_ag_section(intensity) -> None:
    with st.container(border=True):
        st.subheader("Ag")
        try:
            classifier = load_ag_classifier()
            classification = classify_ag(intensity, classifier)
        except MODEL_ERRORS as exc:
            st.error(f"Ag classification unavailable: {exc}")
            return
        positive = classification["positive"]
        st.metric("Classification", "Positive" if positive else "Zero-class")
        if not positive:
            st.info(ZERO_CLASS_NOTE)
            return
        try:
            model, checkpoint = load_ag_concentration()
            estimate = predict_ag_concentration(intensity, model, checkpoint)
        except MODEL_ERRORS as exc:
            st.error(f"Ag quantitative estimate unavailable: {exc}")
            return
        st.metric("Experimental Ag estimate", f"{estimate:.3f} ppm")
        st.caption(LOW_CONCENTRATION_NOTE)


def render_cu_section(intensity, detections, detection_error) -> None:
    with st.container(border=True):
        st.subheader("Cu")
        if detections is not None:
            result = detections["Cu"]["result"]
            st.metric("Detection", "Detected" if result == "DETECTED" else "Not Detected")
        else:
            st.error(f"Cu detection unavailable: {detection_error}")
        try:
            model, checkpoint = load_cu_concentration()
            estimate = predict_cu_concentration(intensity, model, checkpoint)
        except MODEL_ERRORS as exc:
            st.error(f"Cu quantitative estimate unavailable: {exc}")
            return
        st.metric("Experimental Cu quantitative estimate", f"{estimate:.1f} ppm")
        st.metric("Data-supported quantitative range", f"{SUPPORTED_RANGE_PPM[0]:.0f}\u2013{SUPPORTED_RANGE_PPM[1]:.0f} ppm")
        st.caption(
            f"The {EXISTING_DETECTION_THRESHOLD_PPM:.0f} ppm classification threshold is not a physical "
            "detection limit and must not be used alone to decide whether a sample is out of range."
        )
        st.warning(HIGH_CONCENTRATION_NOTE)


def main() -> None:
    st.header("Spectrum Input")
    demo_metadata = pd.read_csv(DEMO_DIR / "demo_ground_truth.csv")
    demo_names = ["No demo sample", *demo_metadata["sample_name"].tolist()]
    selected = st.selectbox("Built-in demo sample", demo_names)
    uploaded = st.file_uploader("Upload numerical LIBS CSV", type=["csv"])
    if selected == "No demo sample" and uploaded is None:
        st.info("Select a built-in demo sample or upload a CSV with wavelength,intensity columns.")
        return
    try:
        if selected != "No demo sample":
            wavelengths, intensity, _ = load_demo(selected)
        else:
            wavelengths, intensity = load_spectrum(uploaded)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return

    st.header("Spectrum Plot")
    spectrum_plot(wavelengths, intensity)

    detections, detection_error = load_six_element_detections(intensity)

    st.header("Results")
    mg_column, ag_column, cu_column = st.columns(3)
    with mg_column:
        render_mg_section(intensity, detections, detection_error)
    with ag_column:
        render_ag_section(intensity)
    with cu_column:
        render_cu_section(intensity, detections, detection_error)

    st.header("Model Notes")
    st.markdown(
        "- MgO quantitative result is reported in wt%.\n"
        "- Ag concentration estimation is experimental and mainly supported at low concentrations.\n"
        "- Cu quantitative model was developed for a data-supported 1-500 ppm range.\n"
        "- These models are research prototypes and are not certified analytical measurements."
    )


if __name__ == "__main__":
    main()

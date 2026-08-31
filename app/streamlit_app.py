"""Streamlit UI for independent reference matching and CNN classification."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.demo_comparison import build_final_comparison  # noqa: E402
from atomic_spectrum_ai.inference import load_model_checkpoint, predict_image  # noqa: E402
from atomic_spectrum_ai.reference_matching import (  # noqa: E402
    TARGET_ELEMENTS,
    analyze_spectrum,
    load_reference_database,
    load_spectrum_csv,
    plot_matching_result,
)
from atomic_spectrum_ai.z903_inference import (  # noqa: E402
    Z903_CNN_CHECKPOINT,
    load_z903_cnn,
    predict_z903_csv,
)

st.set_page_config(page_title="Atomic Spectrum Identification", layout="wide")
st.title("Atomic Spectrum Identification")
mode = st.radio(
    "Identification method",
    ["Reference / Peak Matching (No AI)", "CNN Detection"],
    horizontal=True,
)

DEMO_DIR = ROOT / "demo_samples"
DEMO_OPTIONS = {
    "No demo sample": None,
    "Simple Demo": "simple_barite.csv",
    "Mixed Demo": "mixed_mix801.csv",
    "Clear Demo": "clear_mix349.csv",
    "Challenging / Near-threshold Demo": "challenging_near_threshold_agv1a.csv",
}
demo_choice = st.sidebar.selectbox("Demo Samples", list(DEMO_OPTIONS))


def selected_demo_path() -> Path | None:
    filename = DEMO_OPTIONS[demo_choice]
    if filename is None:
        return None
    path = DEMO_DIR / filename
    if not path.is_file():
        st.error(f"Demo sample is missing: {path}")
        return None
    return path


def load_demo_spectrum(path: Path):
    return load_spectrum_csv(path)


def run_reference_analysis(spectrum):
    tolerance = st.session_state.get("demo_tolerance", 0.25)
    prominence = st.session_state.get("demo_prominence", 0.05)
    normalized_intensity = st.session_state.get("demo_normalized_intensity", 0.05)
    distance = st.session_state.get("demo_distance", 0.25)
    references = load_reference_database(
        (float(spectrum.wavelength_nm.min()), float(spectrum.wavelength_nm.max()))
    )
    return analyze_spectrum(
        spectrum,
        references,
        tolerance_nm=float(tolerance),
        minimum_prominence=float(prominence),
        minimum_normalized_intensity=float(normalized_intensity),
        minimum_peak_distance_nm=float(distance),
    )


def render_final_comparison(result, cnn_probabilities: dict[str, float] | None = None) -> None:
    truth_path = DEMO_DIR / "demo_ground_truth.csv"
    truth = pd.read_csv(truth_path).set_index("sample_name")
    sample_name = demo_choice.replace(" Demo", "")
    row = truth.loc[sample_name]
    comparison = build_final_comparison(result.scores, row, cnn_probabilities)
    st.subheader("Final comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_cnn_mode() -> dict[str, str]:
    st.caption("NASA demo samples use the frozen numerical Z-903 1D-CNN. Manual images use the legacy image CNN separately.")
    checkpoint_text = st.text_input("Checkpoint path", value="outputs/quickstart/best_model.pt")
    demo_path = selected_demo_path()
    upload = None if demo_path else st.file_uploader("Upload a spectrum image", type=["png", "jpg", "jpeg"], key="cnn_image")
    labels: dict[str, str] = {}
    if demo_path is None and upload is None:
        return labels
    if demo_path:
        st.subheader("NASA Z-903 1D-CNN")
        st.caption(f"Numerical input, robust median/IQR preprocessing, checkpoint: {Z903_CNN_CHECKPOINT.name}")
        try:
            model, _, device = load_z903_cnn(Z903_CNN_CHECKPOINT)
            predictions = predict_z903_csv(demo_path, model, device)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
            return labels
        for element, prediction in predictions.items():
            labels[element] = f"{prediction['probability']:.2%}"
            st.write(f"**{element}** - {prediction['probability']:.2%} ({'DETECTED' if prediction['decision'] else 'NOT DETECTED'})")
        reference_result = run_reference_analysis(load_demo_spectrum(demo_path))
        render_final_comparison(reference_result, {element: float(prediction["probability"]) for element, prediction in predictions.items()})
    else:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Uploaded spectrum plot", use_container_width=True)
        try:
            model, checkpoint, device = load_model_checkpoint(checkpoint_text)
            results = predict_image(image, model, checkpoint, device, top_k=3)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
            return labels
        st.subheader("Legacy image CNN")
        for result in results:
            labels[result["element"]] = f"{result['probability']:.2%}"
            st.write(f"**{result['element']}** - {result['probability']:.2%}")
    return labels


def render_reference_mode() -> dict[str, str]:
    st.caption("Deterministic numerical matching against cached NIST I/II line tables.")
    st.info("Upload wavelength/intensity data first. No neural network or checkpoint is used in this mode.")
    demo_path = selected_demo_path()
    upload = None if demo_path else st.file_uploader("Upload spectrum CSV", type=["csv"], key="reference_csv")
    if demo_path is None and upload is None:
        return {}
    try:
        spectrum = load_demo_spectrum(demo_path) if demo_path else load_spectrum_csv(BytesIO(upload.getvalue()))
    except (pd.errors.ParserError, ValueError) as exc:
        st.error(str(exc))
        return {}
    st.subheader("Spectrum preview")
    st.line_chart(pd.DataFrame({"wavelength_nm": spectrum.wavelength_nm, "intensity": spectrum.intensity}).set_index("wavelength_nm"))
    with st.sidebar:
        st.header("Reference matching")
        tolerance = st.select_slider("Wavelength tolerance (nm)", options=[0.1, 0.25, 0.5], value=0.25, key="demo_tolerance")
        prominence = st.number_input("Minimum prominence", min_value=0.0, value=0.05, step=0.01, key="demo_prominence")
        normalized_intensity = st.number_input("Minimum normalized intensity", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="demo_normalized_intensity")
        distance = st.number_input("Minimum peak distance (nm)", min_value=0.01, value=0.25, step=0.05, key="demo_distance")
    references = load_reference_database((float(spectrum.wavelength_nm.min()), float(spectrum.wavelength_nm.max())))
    result = analyze_spectrum(
        spectrum,
        references,
        tolerance_nm=float(tolerance),
        minimum_prominence=float(prominence),
        minimum_normalized_intensity=float(normalized_intensity),
        minimum_peak_distance_nm=float(distance),
    )
    st.subheader("Reference matching results")
    score_view = result.scores[["element", "status", "reference_matching_score", "matched_lines", "strong_matched_lines", "low_interference_matches", "mean_wavelength_error_nm"]].copy()
    st.dataframe(score_view, use_container_width=True, hide_index=True)
    st.download_button("Download extracted peaks", result.peaks.to_csv(index=False), "extracted_peaks.csv", "text/csv")
    st.download_button("Download match table", result.matches.to_csv(index=False), "reference_matches.csv", "text/csv")
    with TemporaryDirectory() as temporary_directory:
        plot_path = Path(temporary_directory) / "reference_matching.png"
        plot_matching_result(spectrum, result, plot_path)
        st.image(str(plot_path), caption="Numerical spectrum, detected peaks, and matched reference lines", use_container_width=True)
    for element in TARGET_ELEMENTS:
        score = result.scores[result.scores.element == element].iloc[0]
        with st.expander(f"{element}: {score.status}"):
            st.write({"score": round(float(score.reference_matching_score), 3), "reference lines evaluated": int(score.reference_lines_evaluated), "strong lines evaluated": int(score.strong_lines_evaluated), "matched lines": int(score.matched_lines), "strong matched lines": int(score.strong_matched_lines), "low-interference matches": int(score.low_interference_matches), "mean wavelength error (nm)": score.mean_wavelength_error_nm})
            st.dataframe(result.matches[result.matches.element == element], use_container_width=True, hide_index=True)
    if demo_path:
        render_final_comparison(result)
    reference_labels = {row.element: row.status for row in result.scores.itertuples()}
    st.subheader("Optional CNN comparison")
    st.caption("CNN output is shown separately and never changes reference-matching results.")
    cnn_upload = st.file_uploader("Upload a spectrum plot image for CNN comparison", type=["png", "jpg", "jpeg"], key="comparison_image")
    if cnn_upload is not None:
        checkpoint_text = st.text_input("CNN checkpoint path", value="outputs/quickstart/best_model.pt", key="comparison_checkpoint")
        try:
            model, checkpoint, device = load_model_checkpoint(checkpoint_text)
            cnn_results = predict_image(Image.open(cnn_upload).convert("RGB"), model, checkpoint, device, top_k=6)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            cnn_labels = {item["element"]: item["probability"] for item in cnn_results}
            comparison = pd.DataFrame(
                {
                    "Element": TARGET_ELEMENTS,
                    "Reference Matching": [reference_labels[element] for element in TARGET_ELEMENTS],
                    "CNN": [f"{cnn_labels[element]:.2%}" if element in cnn_labels else "-" for element in TARGET_ELEMENTS],
                    "Agreement": ["AGREE" if reference_labels[element] == "DETECTED" and cnn_labels.get(element, 0.0) >= 0.5 else "DISAGREE" for element in TARGET_ELEMENTS],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    return reference_labels


if mode == "CNN Detection":
    render_cnn_mode()
else:
    render_reference_mode()

"""Streamlit user interface for spectrum image classification."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.inference import load_model_checkpoint, predict_image

st.set_page_config(page_title="Atomic Spectrum CNN", layout="centered")
st.title("Atomic Spectrum CNN")
st.caption("Single-element image classifier: Fe, Cu, Na, Ca, Mg")
st.warning(
    "This interface reports image-classification predictions. Synthetic-data results "
    "are not scientific validation for real samples."
)

checkpoint_text = st.text_input(
    "Checkpoint path", value="outputs/quickstart/best_model.pt"
)
upload = st.file_uploader("Upload a spectrum image", type=["png", "jpg", "jpeg"])

if upload is not None:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    try:
        model, checkpoint, device = load_model_checkpoint(checkpoint_text)
        results = predict_image(image, model, checkpoint, device, top_k=3)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.subheader("Top predictions")
        for result in results:
            st.write(f"**{result['element']}** — {result['probability']:.2%}")

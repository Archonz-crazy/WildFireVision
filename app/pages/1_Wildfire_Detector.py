"""
Wildfire Detector page — upload an image, get a prediction and GradCAM heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

# Make src/ importable when running from repo root or as a Spaces app
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    from wildfirevision.predict import load_model_for_inference

    return load_model_for_inference(
        local_path="models/model.h5",
        hf_repo_id="Archonz-crazy/wildfirevision",
        hf_filename="model.h5",
    )


st.title("🔥 Wildfire Detector")
st.markdown(
    "Upload a satellite or aerial image to classify it as **fire** or **no fire**.  \n"
    "The GradCAM heatmap shows which regions the model focused on."
)

model, source = load_model()

if model is None:
    st.error(
        "**Model not loaded.** The trained model weights are required for predictions.\n\n"
        "**To get the model:**\n"
        "1. Download the data: `python scripts/download_data.py`\n"
        "2. Prepare the data: `python scripts/prepare_data.py`\n"
        "3. Train: `python -m wildfirevision.train`\n"
        "4. Push to HuggingFace Hub: `python -m wildfirevision.train --push-to-hub`\n\n"
        "Or download a pre-trained checkpoint from "
        "[HuggingFace Hub](https://huggingface.co/Archonz-crazy/wildfirevision) "
        "and place it at `models/model.h5`."
    )
    st.stop()

source_label = {"local": "local `models/model.h5`", "huggingface": "HuggingFace Hub"}.get(
    source, source
)
st.success(f"Model loaded from: {source_label}")

uploaded = st.file_uploader(
    "Upload a wildfire image", type=["jpg", "jpeg", "png"], help="JPG or PNG, any resolution."
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Running inference…"):
        from wildfirevision.gradcam import generate_gradcam_for_app
        from wildfirevision.predict import predict_single

        result = predict_single(model, image)
        heatmap_img = generate_gradcam_for_app(model, image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("GradCAM Heatmap")
        st.image(heatmap_img, use_container_width=True)
        st.caption("Red = highest model attention. Blue = lowest.")

    with col3:
        st.subheader("Prediction")
        label = result["class"].upper()
        confidence = result["confidence"]
        color = "red" if result["class"] == "fire" else "green"

        st.markdown(f"## :{color}[{label}]")
        st.metric("Confidence", f"{confidence:.1%}")

        df = pd.DataFrame(
            [
                {"Class": "Fire", "Probability": result["probabilities"]["fire"]},
                {"Class": "No Fire", "Probability": result["probabilities"]["nofire"]},
            ]
        )
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Class", sort="-x"),
                color=alt.Color(
                    "Class",
                    scale=alt.Scale(
                        domain=["Fire", "No Fire"],
                        range=["#e74c3c", "#27ae60"],
                    ),
                    legend=None,
                ),
                tooltip=["Class", alt.Tooltip("Probability", format=".1%")],
            )
            .properties(height=120)
        )
        st.altair_chart(chart, use_container_width=True)

        if result["class"] == "fire":
            st.warning("⚠️ Fire detected. Please notify relevant authorities if this is a live image.")
        else:
            st.info("✅ No fire detected in this image.")
else:
    st.markdown(
        "**Sample images to try:**  \n"
        "You can download test images from the "
        "[Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) "
        "or [FlameVision](https://www.kaggle.com/datasets/anamibnjafar0/flamevision)."
    )

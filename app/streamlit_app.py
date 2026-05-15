"""
WildFireVision — Streamlit entry point.
Used by HuggingFace Spaces (app_file: app/streamlit_app.py) and Docker.
Streamlit's multi-page feature auto-discovers app/pages/*.py.
"""

import streamlit as st

st.set_page_config(
    page_title="WildFireVision",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔥 WildFireVision")
st.sidebar.markdown(
    "Deep learning wildfire detection using **VGG16 transfer learning**  \n"
    "~90% accuracy on binary fire / no-fire classification."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[GitHub](https://github.com/Archonz-crazy/WildFireVision) · "
    "[HuggingFace](https://huggingface.co/Archonz-crazy/wildfirevision)"
)

st.title("WildFireVision")
st.markdown(
    """
    Welcome to **WildFireVision** — an end-to-end deep learning system for detecting
    wildfires in aerial and satellite imagery.

    Use the sidebar to navigate between pages:

    | Page | Description |
    |------|-------------|
    | **Wildfire Detector** | Upload an image for instant prediction + GradCAM explanation |
    | **Model Performance** | VGG16 vs ResNet50 accuracy/loss comparison |
    | **Dataset** | Training data overview and preprocessing pipeline |
    | **About** | Tech stack, architecture details, and team |
    """
)

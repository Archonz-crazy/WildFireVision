"""
About page — project overview, architecture, tech stack, and team.
"""

import streamlit as st

st.title("ℹ️ About WildFireVision")

st.markdown(
    """
    **WildFireVision** is an end-to-end deep learning system for binary wildfire classification
    (fire / no fire) from aerial and satellite imagery. It was built as a capstone project for
    George Washington University's Deep Learning course, then extended into a full MLOps pipeline.
    """
)

st.markdown("---")
st.subheader("Architecture")

st.markdown(
    """
    ```
    Input (224×224 RGB)
         ↓
    VGG16 Base (ImageNet, frozen — 14.7M params)
         ↓
    BatchNormalization
         ↓
    MaxPool (2×2)
         ↓
    Flatten
         ↓
    Dense(512, relu) → Dropout(0.3) → Dense(256, relu) → Dense(128, relu)
         ↓
    Dense(2, softmax)  →  [P(fire), P(no fire)]
         ↓
    GradCAM heatmap on block5_conv3 (last conv layer)
    ```
    """
)

st.markdown("---")
st.subheader("Tech Stack")

import pandas as pd

st.table(
    pd.DataFrame(
        [
            {"Layer": "Deep Learning", "Technology": "TensorFlow 2.14 / Keras", "Purpose": "Model training and inference"},
            {"Layer": "Explainability", "Technology": "GradCAM (tf.GradientTape)", "Purpose": "Heatmap visualisation on block5_conv3"},
            {"Layer": "Experiment Tracking", "Technology": "MLflow", "Purpose": "Hyperparameters, metrics, artifact logging"},
            {"Layer": "Data Pipeline", "Technology": "DVC", "Purpose": "Reproducible download → prepare → train → evaluate"},
            {"Layer": "Web App", "Technology": "Streamlit", "Purpose": "Interactive prediction and model analysis UI"},
            {"Layer": "Model Distribution", "Technology": "HuggingFace Hub", "Purpose": "Hosted model weights for Spaces deployment"},
            {"Layer": "CI/CD", "Technology": "GitHub Actions", "Purpose": "Linting (ruff), formatting (black), tests (pytest)"},
            {"Layer": "Containerisation", "Technology": "Docker", "Purpose": "Reproducible local deployment"},
        ]
    )
)

st.markdown("---")
st.subheader("Results")

st.markdown(
    """
    | Model | Test Accuracy | Val Loss | Notes |
    |-------|-------------|----------|-------|
    | **VGG16** | ~90% | ~0.224 | Selected — best generalization |
    | ResNet50 | ~85% | ~1.6 | Overfit despite heavy regularization |
    | Baseline CNN | ~75% | — | Simple 3-conv custom network |
    """
)

st.markdown("---")
st.subheader("Team")

st.markdown(
    """
    - **Mahikshit Kurapati** — ML pipeline, Streamlit app, MLOps
    - **Pooja Chandrashekara** — Model training and evaluation
    - **Mohammad Kanu** — Data collection and EDA

    *Advised by Prof. Amir Jafari, Dept. of Data Science, George Washington University.*
    """
)

st.markdown("---")
st.subheader("Links")

col1, col2, col3 = st.columns(3)
col1.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Archonz-crazy/WildFireVision)")
col2.markdown("[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/Archonz-crazy/wildfirevision)")
col3.markdown("[![Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Archonz-crazy/wildfirevision)")

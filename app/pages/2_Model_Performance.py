"""
Model Performance page — VGG16 vs ResNet50 training results and comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("📊 Model Performance")

ASSETS = Path(__file__).parents[2] / "assets" / "results"
EVAL_DIR = Path("models/eval")

tab1, tab2, tab3 = st.tabs(["VGG16 (Best)", "ResNet50", "Comparison"])

with tab1:
    st.subheader("VGG16 — ~90% Test Accuracy")
    st.markdown(
        "VGG16 achieved the best results with a validation loss of **~0.224**, "
        "indicating strong generalization on unseen wildfire imagery."
    )
    col1, col2, col3 = st.columns(3)
    col1.image(str(ASSETS / "vgg1.jpg"), caption="Confusion Matrix", use_container_width=True)
    col2.image(str(ASSETS / "vgg2.jpg"), caption="Accuracy Curves", use_container_width=True)
    col3.image(str(ASSETS / "vgg3.jpg"), caption="Loss Curves (val ~0.224)", use_container_width=True)

    if (EVAL_DIR / "confusion_matrix.png").exists():
        st.markdown("---")
        st.subheader("Latest Training Run")
        c1, c2 = st.columns(2)
        c1.image(str(EVAL_DIR / "confusion_matrix.png"), caption="Confusion Matrix (current run)")
        if (EVAL_DIR / "training_curves.png").exists():
            c2.image(str(EVAL_DIR / "training_curves.png"), caption="Training Curves (current run)")

        if (EVAL_DIR / "classification_report.json").exists():
            with open(EVAL_DIR / "classification_report.json") as f:
                report = json.load(f)
            rows = []
            for cls in ["fire", "nofire"]:
                if cls in report:
                    rows.append(
                        {
                            "Class": cls.capitalize(),
                            "Precision": f"{report[cls]['precision']:.3f}",
                            "Recall": f"{report[cls]['recall']:.3f}",
                            "F1-Score": f"{report[cls]['f1-score']:.3f}",
                            "Support": int(report[cls]["support"]),
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("ResNet50 — Higher Overfitting")
    st.markdown(
        "ResNet50 showed high training accuracy but a validation loss of **~1.6**, "
        "indicating significant overfitting despite heavy regularization (L2 + Dropout 0.6–0.7)."
    )
    col1, col2, col3 = st.columns(3)
    col1.image(
        str(ASSETS / "resnet1.jpeg"), caption="Confusion Matrix", use_container_width=True
    )
    col2.image(
        str(ASSETS / "resnet2.jpeg"), caption="Accuracy Curves", use_container_width=True
    )
    col3.image(
        str(ASSETS / "resnet3.jpeg"),
        caption="Loss Curves (val ~1.6 — overfitting)",
        use_container_width=True,
    )

with tab3:
    st.subheader("Model Comparison")

    comparison = pd.DataFrame(
        [
            {
                "Model": "VGG16 ✅",
                "Test Accuracy": "~90%",
                "Val Loss": "~0.224",
                "Optimizer": "Adam",
                "Dropout": "0.3",
                "Regularization": "None",
                "Verdict": "Selected",
            },
            {
                "Model": "ResNet50",
                "Test Accuracy": "~85%",
                "Val Loss": "~1.6",
                "Optimizer": "SGD",
                "Dropout": "0.6–0.7",
                "Regularization": "L2 (0.001)",
                "Verdict": "Overfit",
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.info(
        "**Why VGG16 won:** Despite ResNet50 being a deeper and newer architecture, "
        "VGG16's simpler classification head (Adam + lighter regularization) "
        "generalized significantly better on this dataset size (~11,000 images). "
        "ResNet50's higher capacity led to overfitting that even heavy dropout couldn't prevent."
    )

    st.subheader("Architecture Details")
    arch_df = pd.DataFrame(
        [
            {
                "Component": "Base",
                "VGG16": "VGG16 (ImageNet, frozen)",
                "ResNet50": "ResNet50 (ImageNet, frozen)",
            },
            {
                "Component": "Head",
                "VGG16": "BN → MaxPool → Flatten → Dense(512) → Dropout(0.3) → Dense(256) → Dense(128) → Dense(2)",
                "ResNet50": "BN → MaxPool → Flatten → Dense(512,L2) → Dropout(0.7) → … → Dense(2)",
            },
            {
                "Component": "Input",
                "VGG16": "224 × 224 × 3",
                "ResNet50": "224 × 224 × 3",
            },
            {
                "Component": "Preprocessing",
                "VGG16": "vgg16.preprocess_input (ImageNet channel means)",
                "ResNet50": "resnet.preprocess_input (scale to [-1, 1])",
            },
            {
                "Component": "Loss",
                "VGG16": "categorical_crossentropy",
                "ResNet50": "categorical_crossentropy",
            },
        ]
    )
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

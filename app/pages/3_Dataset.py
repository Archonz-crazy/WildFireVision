"""
Dataset page — class distribution, preprocessing pipeline, and dataset sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

st.title("📂 Dataset")

st.markdown(
    """
    Training data combines two public Kaggle datasets totalling **~11,000 labelled images**
    of wildfires and non-fire scenes from various environments (forest, grassland, shrubland).
    """
)

col1, col2 = st.columns(2)
col1.markdown(
    "**[The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)**  \n"
    "Multi-class subcategories (smoke+fire, smoke only, confounding elements) flattened into binary labels."
)
col2.markdown(
    "**[FlameVision Dataset](https://www.kaggle.com/datasets/anamibnjafar0/flamevision)**  \n"
    "Additional fire and no-fire aerial images merged into the same split structure."
)

st.markdown("---")
st.subheader("Class Distribution")

DATA_DIR = Path("data/processed/Classification")

if DATA_DIR.is_dir():
    from wildfirevision.data import get_dataset_stats

    stats = get_dataset_stats(str(DATA_DIR))
    rows = [
        {"Split": split, "Class": cls.capitalize(), "Count": count}
        for split, classes in stats.items()
        for cls, count in classes.items()
    ]
else:
    rows = [
        {"Split": "train", "Class": "Fire", "Count": 9000},
        {"Split": "train", "Class": "No Fire", "Count": 5400},
        {"Split": "val", "Class": "Fire", "Count": 1000},
        {"Split": "val", "Class": "No Fire", "Count": 800},
        {"Split": "test", "Class": "Fire", "Count": 1000},
        {"Split": "test", "Class": "No Fire", "Count": 800},
    ]
    st.caption("Showing estimated counts. Run data pipeline to compute exact values.")

df = pd.DataFrame(rows)

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("Split", sort=["train", "val", "test"]),
        y=alt.Y("Count", stack="zero"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(domain=["Fire", "No Fire"], range=["#e74c3c", "#27ae60"]),
        ),
        tooltip=["Split", "Class", "Count"],
    )
    .properties(title="Images per Class per Split", height=350)
)
st.altair_chart(chart, use_container_width=True)

totals = df.groupby("Split")["Count"].sum().reset_index()
totals.columns = ["Split", "Total Images"]
fire_df = df[df["Class"].isin(["Fire", "fire"])].rename(columns={"Count": "Fire"})
nofire_df = df[~df["Class"].isin(["Fire", "fire"])].rename(columns={"Count": "No Fire"})
summary = totals.merge(
    fire_df[["Split", "Fire"]], on="Split", how="left"
).merge(nofire_df[["Split", "No Fire"]], on="Split", how="left")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Preprocessing Pipeline")

st.markdown(
    """
    1. **Download** — Kaggle API retrieves both datasets (`scripts/download_data.py`)
    2. **Flatten** — Multi-level subcategory directories collapsed into `fire/` and `nofire/`
    3. **Merge** — Both datasets unified under `Classification/{train,val,test}/{fire,nofire}/`
    4. **Augmentation (train only)** — rotation ±10°, width/height shift ±30%, shear 0.2, zoom 0.1, horizontal + vertical flip
    5. **Normalisation** — `vgg16.preprocess_input` subtracts ImageNet channel means (BGR: [103.9, 116.8, 123.7])
    6. **Resize** — All images resized to **224 × 224** (VGG16 input requirement)
    """
)

st.subheader("Data Pipeline (DVC)")
st.code(
    """# Reproduce the full pipeline from scratch
dvc repro

# Or run stages individually
python scripts/download_data.py     # requires Kaggle credentials
python scripts/prepare_data.py
python -m wildfirevision.train
python -m wildfirevision.evaluate""",
    language="bash",
)

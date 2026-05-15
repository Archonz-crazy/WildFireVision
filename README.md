---
title: WildFireVision
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.28.0
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# WildFireVision

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.8-blue?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-3.30-purple)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/Archonz-crazy/WildFireVision/actions/workflows/ci.yml/badge.svg)](https://github.com/Archonz-crazy/WildFireVision/actions/workflows/ci.yml)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-WildFireVision-yellow)](https://huggingface.co/spaces/Archonz-crazy/wildfirevision)

End-to-end deep learning system for binary wildfire classification (fire / no fire) from aerial and satellite imagery, with **GradCAM explainability**, **MLflow experiment tracking**, **DVC data pipelines**, and a live **Streamlit** demo.

**Live demo:** [huggingface.co/spaces/Archonz-crazy/wildfirevision](https://huggingface.co/spaces/Archonz-crazy/wildfirevision)

---

## Pipeline

```
Kaggle Data
    ↓  scripts/download_data.py
data/raw/
    ↓  scripts/prepare_data.py
data/processed/Classification/{train,val,test}/{fire,nofire}/
    ↓  python -m wildfirevision.train  (MLflow tracked)
models/model.h5  +  mlruns/
    ↓  python -m wildfirevision.evaluate
models/eval/{confusion_matrix.png, training_curves.png, classification_report.json}
    ↓  python -m wildfirevision.train --push-to-hub
HuggingFace Hub  →  Streamlit App (HF Spaces)
```

---

## Results

| Model | Test Accuracy | Val Loss | Optimizer | Notes |
|-------|-------------|----------|-----------|-------|
| **VGG16** ✅ | **~90%** | **~0.224** | Adam | Selected — best generalization |
| ResNet50 | ~85% | ~1.6 | SGD | Overfit despite L2 + Dropout(0.6–0.7) |
| Baseline CNN | ~75% | — | Adam | Custom 3-conv network |

VGG16 achieves significantly lower validation loss (0.224 vs 1.6), indicating strong generalization on unseen wildfire imagery.

---

## Quick Start

### Run the App Locally

```bash
git clone https://github.com/Archonz-crazy/WildFireVision
cd WildFireVision
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

The Wildfire Detector page will auto-download the model from HuggingFace Hub on first use (~500MB). Place a local model at `models/model.h5` to skip the download.

### Docker

```bash
docker build -t wildfirevision .
docker run -p 8501:8501 wildfirevision
# Open http://localhost:8501
```

---

## Train the Model

### Option A: Local (GPU recommended)

```bash
# 1. Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# 2. Download datasets (~3GB)
python scripts/download_data.py

# 3. Flatten + merge into unified Classification/ tree
python scripts/prepare_data.py

# 4. Train (logs to mlruns/)
python -m wildfirevision.train

# 5. View experiment tracking
mlflow ui

# 6. Push model to HuggingFace Hub for Spaces deployment
python -m wildfirevision.train --push-to-hub
```

### Option B: DVC (fully reproducible)

```bash
# Runs all stages: download → prepare → train → evaluate
dvc repro
```

### Option C: Google Colab (GPU, free tier)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open `notebooks/eda.ipynb` and follow the training instructions there.

---

## Architecture

**VGG16 Transfer Learning** — base frozen (ImageNet weights), custom classification head fine-tuned on wildfire imagery:

```
VGG16(ImageNet, frozen — 14.7M params)
    → BatchNormalization
    → MaxPool(2×2)
    → Flatten
    → Dense(512, relu) → Dropout(0.3)
    → Dense(256, relu)
    → Dense(128, relu)
    → Dense(2, softmax)
```

**GradCAM** explainability targets `block5_conv3` (VGG16's last convolutional layer), producing spatial heatmaps that highlight which image regions most influenced the prediction.

---

## Project Structure

```
WildFireVision/
├── app/                    # Streamlit multi-page app (HF Spaces entry point)
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Wildfire_Detector.py   # Upload → prediction + GradCAM
│       ├── 2_Model_Performance.py   # Training results comparison
│       ├── 3_Dataset.py             # Class distribution + pipeline
│       └── 4_About.py              # Tech stack and team
├── src/wildfirevision/     # Core Python package
│   ├── data.py             # ImageDataGenerator + VGG16 preprocessing fix
│   ├── model.py            # build_vgg16_model(), build_resnet50_model()
│   ├── train.py            # Training loop with MLflow + HF Hub push
│   ├── evaluate.py         # Confusion matrix, curves, classification report
│   ├── predict.py          # Inference + HF Hub fallback model loading
│   └── gradcam.py          # GradCAM heatmap (handles nested Sequential+VGG16)
├── scripts/
│   ├── download_data.py    # Kaggle API dataset download
│   └── prepare_data.py     # Dataset flatten + merge
├── configs/                # Hyperparameter YAML files
├── assets/results/         # Training result images (committed)
├── tests/                  # pytest test suite
├── dvc.yaml                # Reproducible 4-stage ML pipeline
├── pyproject.toml          # Package definition + ruff/black config
└── Dockerfile
```

---

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check src/ app/ scripts/ tests/
black src/ app/ scripts/ tests/

# Tests (fast — no model weights required)
pytest tests/ -m "not slow"

# Full test suite (downloads VGG16 ImageNet weights ~500MB)
pytest tests/
```

---

## Dataset

| Source | Images | Classes |
|--------|--------|---------|
| [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) | ~7,200 | fire, nofire (multi-subcategory, flattened) |
| [FlameVision](https://www.kaggle.com/datasets/anamibnjafar0/flamevision) | ~3,800 | fire, nofire |
| **Total** | **~11,000** | train / val / test split |

---

## Team

- **Mahikshit Kurapati** — ML pipeline, MLOps, Streamlit app
- **Pooja Chandrashekara** — Model training and evaluation
- **Mohammad Kanu** — Data collection and EDA

*Capstone project for GWU Data Science — Advisor: Prof. Amir Jafari*

---

## References

- [VGGNet (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [GradCAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [MLflow Documentation](https://mlflow.org/)
- [DVC Documentation](https://dvc.org/)

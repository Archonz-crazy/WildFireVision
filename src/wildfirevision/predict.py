"""Single-image inference utilities with HuggingFace Hub fallback model loading."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["fire", "nofire"]

_DEFAULT_HF_REPO = "Archonz-crazy/wildfirevision"
_DEFAULT_HF_FILENAME = "model.h5"
_DEFAULT_LOCAL_PATH = "models/model.h5"


def get_preprocessing_fn(architecture: str):
    """Lazy import to avoid loading TF at module level."""
    if architecture == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input
    from tensorflow.keras.applications.resnet import preprocess_input

    return preprocess_input


def preprocess_image(
    img: Image.Image,
    architecture: str = "vgg16",
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resizes to target_size, converts to RGB, applies architecture-specific
    preprocessing, and adds batch dimension.

    Returns ndarray of shape (1, H, W, 3).
    """
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img, dtype="float32")
    img_array = get_preprocessing_fn(architecture)(img_array)
    return np.expand_dims(img_array, axis=0)


def predict_single(
    model,
    img: Image.Image,
    architecture: str = "vgg16",
) -> dict:
    """
    Runs inference on a single PIL image.

    Returns:
        {
            "class": "fire",
            "confidence": 0.94,
            "probabilities": {"fire": 0.94, "nofire": 0.06}
        }
    """
    processed = preprocess_image(img, architecture=architecture)
    preds = model.predict(processed, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    return {
        "class": CLASS_NAMES[class_idx],
        "confidence": float(preds[class_idx]),
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)},
    }


def load_model_for_inference(
    local_path: str = _DEFAULT_LOCAL_PATH,
    hf_repo_id: str = _DEFAULT_HF_REPO,
    hf_filename: str = _DEFAULT_HF_FILENAME,
) -> tuple:
    """
    Model loading with fallback chain:
      1. local_path  → load if file exists
      2. HuggingFace Hub → hf_hub_download, then load
      3. Return (None, "not_loaded") if both fail

    Returns (model, source) where source ∈ {"local", "huggingface", "not_loaded"}.
    """
    import tensorflow as tf

    if os.path.isfile(local_path):
        try:
            model = tf.keras.models.load_model(local_path)
            return model, "local"
        except Exception:
            pass

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
        model = tf.keras.models.load_model(cached)
        return model, "huggingface"
    except Exception:
        pass

    return None, "not_loaded"

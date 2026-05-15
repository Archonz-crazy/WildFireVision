"""Data loading and preprocessing utilities."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_preprocessing_fn(architecture: str):
    """Returns the correct preprocess_input function for the given architecture."""
    if architecture == "vgg16":
        return vgg16_preprocess
    elif architecture == "resnet50":
        return resnet_preprocess
    raise ValueError(f"Unknown architecture: {architecture}. Choose 'vgg16' or 'resnet50'.")


def get_data_generators(
    data_dir: str,
    config: dict[str, Any],
    architecture: str = "vgg16",
    target_size: tuple[int, int] = (224, 224),
):
    """
    Returns (train_data, val_data, test_data) Keras directory iterators.

    Augmentation is applied only to the training split.
    Preprocessing uses the architecture-specific function (fixes the original
    bug where resnet.preprocess_input was used for VGG16).
    """
    preprocess_fn = get_preprocessing_fn(architecture)
    aug = config.get("augmentation", {})

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=aug.get("rotation_range", 10),
        width_shift_range=aug.get("width_shift_range", 0.3),
        height_shift_range=aug.get("height_shift_range", 0.3),
        shear_range=aug.get("shear_range", 0.2),
        zoom_range=aug.get("zoom_range", 0.1),
        horizontal_flip=aug.get("horizontal_flip", True),
        vertical_flip=aug.get("vertical_flip", True),
        dtype="float32",
    )
    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_fn, dtype="float32")

    batch_size = config.get("training", {}).get("batch_size", 16)

    train_data = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_data = eval_gen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    test_data = eval_gen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, val_data, test_data


def get_dataset_stats(data_dir: str) -> dict[str, dict[str, int]]:
    """Counts images per class per split. Used by the Streamlit Dataset page."""
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        stats[split] = {}
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                stats[split][cls] = len(
                    [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
                )
    return stats

"""Model architecture definitions for VGG16 and ResNet50 transfer learning."""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def build_vgg16_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    VGG16 transfer learning model (best-performing architecture, ~90% accuracy).

    Architecture:
        VGG16(ImageNet, frozen) → BatchNorm → MaxPool(2×2) → Flatten
        → Dense(512, relu) → Dropout(0.3) → Dense(256, relu)
        → Dense(128, relu) → Dense(num_classes, softmax)
    """
    base = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = not freeze_base

    model = Sequential(
        [
            base,
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_resnet50_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    ResNet50 transfer learning model (higher overfitting; val loss ~1.6 vs VGG16's ~0.224).

    Uses categorical output and crossentropy (consistent with VGG16 — original used
    sigmoid + binary_crossentropy which caused evaluation mismatches).
    """
    base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = not freeze_base

    model = Sequential(
        [
            base,
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.7),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.6),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.6),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.6),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model_from_path(model_path: str) -> tf.keras.Model | None:
    """Loads a saved .h5 model. Returns None if the file does not exist (never raises)."""
    if not os.path.isfile(model_path):
        return None
    return tf.keras.models.load_model(model_path)

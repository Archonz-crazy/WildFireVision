"""Tests for predict.py — no TF model required, runs fast in CI."""

import numpy as np
import pytest
from PIL import Image


def _make_dummy_image(width=300, height=400):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def test_preprocess_image_shape_vgg16():
    from wildfirevision.predict import preprocess_image

    img = _make_dummy_image()
    out = preprocess_image(img, architecture="vgg16")
    assert out.shape == (1, 224, 224, 3)


def test_preprocess_image_shape_resnet50():
    from wildfirevision.predict import preprocess_image

    img = _make_dummy_image()
    out = preprocess_image(img, architecture="resnet50")
    assert out.shape == (1, 224, 224, 3)


def test_preprocess_image_converts_to_rgb():
    from wildfirevision.predict import preprocess_image

    gray = Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L")
    out = preprocess_image(gray, architecture="vgg16")
    assert out.shape == (1, 224, 224, 3)


def test_load_model_returns_none_for_missing_file(tmp_path):
    from wildfirevision.predict import load_model_for_inference

    model, source = load_model_for_inference(
        local_path=str(tmp_path / "nonexistent.h5"),
        hf_repo_id="__invalid_repo_that_does_not_exist__/model",
        hf_filename="model.h5",
    )
    assert model is None
    assert source == "not_loaded"

"""Tests for model.py — marked slow because they download VGG16 ImageNet weights."""

import pytest


@pytest.mark.slow
def test_build_vgg16_output_shape():
    from wildfirevision.model import build_vgg16_model

    model = build_vgg16_model()
    assert model.output_shape == (None, 2)


@pytest.mark.slow
def test_vgg16_base_is_first_layer():
    from wildfirevision.model import build_vgg16_model

    model = build_vgg16_model()
    # model.layers[0] is the VGG16 submodel; it must expose get_layer
    assert hasattr(model.layers[0], "get_layer")


@pytest.mark.slow
def test_vgg16_base_layers_frozen():
    from wildfirevision.model import build_vgg16_model

    model = build_vgg16_model(freeze_base=True)
    vgg_base = model.layers[0]
    assert all(not layer.trainable for layer in vgg_base.layers)


@pytest.mark.slow
def test_vgg16_base_layers_unfrozen():
    from wildfirevision.model import build_vgg16_model

    model = build_vgg16_model(freeze_base=False)
    vgg_base = model.layers[0]
    assert any(layer.trainable for layer in vgg_base.layers)


@pytest.mark.slow
def test_build_resnet50_output_shape():
    from wildfirevision.model import build_resnet50_model

    model = build_resnet50_model()
    assert model.output_shape == (None, 2)


def test_load_model_from_path_missing_returns_none(tmp_path):
    from wildfirevision.model import load_model_from_path

    result = load_model_from_path(str(tmp_path / "nonexistent.h5"))
    assert result is None

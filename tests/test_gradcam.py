"""Tests for gradcam.py — overlay test runs fast; heatmap test is slow (needs VGG16)."""

import numpy as np
import pytest
from PIL import Image


def _make_pil(h=224, w=224):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def test_overlay_gradcam_returns_pil():
    from wildfirevision.gradcam import overlay_gradcam

    heatmap = np.random.rand(7, 7).astype("float32")
    img = _make_pil()
    result = overlay_gradcam(img, heatmap)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_overlay_gradcam_custom_alpha():
    from wildfirevision.gradcam import overlay_gradcam

    heatmap = np.ones((7, 7), dtype="float32")
    img = _make_pil()
    result = overlay_gradcam(img, heatmap, alpha=0.8)
    assert isinstance(result, Image.Image)


@pytest.mark.slow
def test_get_gradcam_heatmap_shape():
    """block5_conv3 in VGG16 produces (7,7) spatial output for 224x224 input."""
    from wildfirevision.gradcam import get_gradcam_heatmap
    from wildfirevision.model import build_vgg16_model
    from wildfirevision.predict import preprocess_image

    model = build_vgg16_model()
    img = _make_pil()
    img_array = preprocess_image(img, architecture="vgg16")
    heatmap = get_gradcam_heatmap(model, img_array, architecture="vgg16")
    assert heatmap.shape == (7, 7)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

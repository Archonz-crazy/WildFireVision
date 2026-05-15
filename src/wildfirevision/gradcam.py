"""GradCAM heatmap generation for model explainability."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

LAST_CONV_LAYER = {"vgg16": "block5_conv3", "resnet50": "conv5_block3_out"}


def _get_last_conv_layer(model: tf.keras.Model, architecture: str) -> tf.keras.layers.Layer:
    """
    Handles the nested Sequential(VGG16/ResNet50) structure.
    model.get_layer("block5_conv3") fails because the conv layer lives inside
    the base submodel at model.layers[0]. We must traverse into it.
    """
    layer_name = LAST_CONV_LAYER[architecture]
    base_model = model.layers[0]
    return base_model.get_layer(layer_name)


def get_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    architecture: str = "vgg16",
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Computes a GradCAM heatmap via tf.GradientTape.

    Args:
        model:       Keras model (Sequential wrapping VGG16 or ResNet50 base).
        img_array:   Preprocessed image, shape (1, H, W, 3).
        architecture: "vgg16" or "resnet50".
        pred_index:  Class index to explain. None → argmax of model prediction.

    Returns:
        Normalized heatmap as float32 ndarray of shape (conv_h, conv_w),
        e.g. (7, 7) for VGG16 block5_conv3.
    """
    last_conv = _get_last_conv_layer(model, architecture)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap.astype("float32")


def overlay_gradcam(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlays a GradCAM heatmap on the original image using JET colormap.

    Args:
        original_img: Original PIL Image (any size).
        heatmap:      Float32 heatmap array normalized to [0, 1].
        alpha:        Heatmap blend weight (0 = invisible, 1 = fully opaque).

    Returns:
        PIL Image with the heatmap blended onto the original.
    """
    w, h = original_img.size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(np.array(original_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def generate_gradcam_for_app(
    model: tf.keras.Model,
    pil_image: Image.Image,
    architecture: str = "vgg16",
) -> Image.Image:
    """
    Convenience wrapper: takes a raw PIL image and returns a GradCAM-overlaid PIL image.
    Used directly by the Streamlit Wildfire Detector page.
    """
    from wildfirevision.predict import preprocess_image

    img_array = preprocess_image(pil_image, architecture=architecture)
    heatmap = get_gradcam_heatmap(model, img_array, architecture=architecture)
    return overlay_gradcam(pil_image, heatmap)

"""
Grad-CAM for the deployed MobileNetV2 malaria classifier.

Works on the nested-model structure produced by the training notebook:

    Sequential([
        Input,
        Lambda(preprocess_input),
        MobileNetV2(...),          <- name: 'mobilenetv2_1.00_224'
        GlobalAveragePooling2D,
        Dropout,
        Dense(2, softmax),
    ])

Implementation note: accessing a nested Functional submodel's internal
layer output (base_model.get_layer(name).output) directly on a model that
was reloaded from disk is unreliable in Keras 3 - it can raise
"layer X has never been called and thus has no defined output" even
though the model predicts fine. To sidestep this entirely, we build a
fresh, standalone MobileNetV2 with a clean node graph and copy the
trained weights into it. This avoids depending on the reloaded model's
internal graph state at all.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

BASE_MODEL_NAME = "mobilenetv2_1.00_224"
LAST_CONV_LAYER_NAME = "out_relu"  # last activated feature map before pooling, (7,7,1280)

_clean_base_cache = {}  # keyed by id(model) so we only rebuild once per loaded model


def _get_clean_conv_model(model: tf.keras.Model):
    """Build (once) a standalone MobileNetV2 with the trained weights, and
    return a Model from its input to the last conv layer's output."""
    key = id(model)
    if key in _clean_base_cache:
        return _clean_base_cache[key]

    base_model = model.get_layer(BASE_MODEL_NAME)

    clean_base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    clean_base.set_weights(base_model.get_weights())

    conv_model = tf.keras.models.Model(
        clean_base.input, clean_base.get_layer(LAST_CONV_LAYER_NAME).output
    )
    _clean_base_cache[key] = conv_model
    return conv_model


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, pred_index: int = None):
    """
    img_array: preprocessed input, shape (1, 224, 224, 3) - same preprocessing as prediction
    model: the full loaded Sequential model
    pred_index: which class to explain; defaults to the predicted class

    Returns: heatmap as a (7,7) numpy array, values in [0, 1]
    """
    base_model = model.get_layer(BASE_MODEL_NAME)
    conv_model = _get_clean_conv_model(model)

    base_idx = model.layers.index(base_model)
    pre_layers = model.layers[:base_idx]        # e.g. the Lambda preprocessing layer
    post_layers = model.layers[base_idx + 1:]   # e.g. GAP, Dropout, Dense

    with tf.GradientTape() as tape:
        x = img_array
        for layer in pre_layers:
            x = layer(x)
        conv_output = conv_model(x)
        tape.watch(conv_output)

        y = conv_output
        for layer in post_layers:
            y = layer(y)
        predictions = y

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    heatmap: (h,w) array in [0,1] from make_gradcam_heatmap
    original_rgb: original image as RGB uint8 array, any size
    Returns: RGB uint8 array, same size as original_rgb, heatmap overlaid
    """
    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = heatmap_colored * alpha + original_rgb * (1 - alpha)
    return np.uint8(overlay)

"""
Malaria Cell Detection - Deployment App (Member 5 / MLOps part)

Loads the trained MobileNetV2 model, runs predictions on uploaded cell
images, and logs every prediction for monitoring.

Grad-CAM visualization (Part 1) and full UI polish (Part 2) are owned by
other teammates - this file is a self-contained, runnable baseline that
Part 1/2 code can be merged into.
"""

import csv
import datetime
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    from gradcam import make_gradcam_heatmap, overlay_heatmap
except ModuleNotFoundError:
    from app.gradcam import make_gradcam_heatmap, overlay_heatmap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL_PATH = os.path.join(MODEL_DIR, f"malaria_mobilenetv2_{MODEL_VERSION}.keras")

LOG_PATH = "logs/predictions_log.csv"
IMG_SIZE = (224, 224)

# Matches tf.keras.utils.image_dataset_from_directory's alphabetical class
# order for folders 'Parasitized' / 'Uninfected' used in the training notebook.
CLASS_NAMES = ["Parasitized", "Uninfected"]


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at '{MODEL_PATH}'. "
            f"Export it from the training notebook and place it in '{MODEL_DIR}/'."
        )
        st.stop()
    # The Lambda layer inside the model calls preprocess_input by name;
    # Keras needs it passed explicitly here to resolve it on load.
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )


# ---------------------------------------------------------------------------
# Preprocessing (must match training preprocessing exactly)
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)  # scales to [-1, 1], MobileNetV2 convention
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------
def log_prediction(filename: str, label: str, confidence: float) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "prediction", "confidence", "model_version"])
        writer.writerow(
            [datetime.datetime.now().isoformat(), filename, label, f"{confidence:.4f}", MODEL_VERSION]
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Malaria Cell Detection", page_icon="🔬")
    st.title("🔬 Malaria Cell Detection")
    st.caption(f"Model version: {MODEL_VERSION}")

    model = load_model()

    uploaded = st.file_uploader("Upload a cell image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", width="stretch")

        x = preprocess_image(image)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.progress(confidence)

        if st.checkbox("Show Grad-CAM heatmap", value=True):
            heatmap = make_gradcam_heatmap(x, model, pred_index=idx)
            original_rgb = np.array(image.convert("RGB").resize(IMG_SIZE))
            overlay = overlay_heatmap(heatmap, original_rgb)
            st.image(overlay, caption="Grad-CAM: model focus area", width="stretch")

        log_prediction(uploaded.name, label, confidence)


if __name__ == "__main__":
    main()

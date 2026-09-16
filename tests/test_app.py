"""
Basic tests for the deployment app (Member 5 / MLOps part).

Run with:  pytest tests/
"""

import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.app import CLASS_NAMES, IMG_SIZE, log_prediction, preprocess_image
from app.gradcam import overlay_heatmap


def test_preprocess_image_shape():
    img = Image.new("RGB", (100, 150), color=(120, 40, 40))
    out = preprocess_image(img)
    assert out.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)


def test_preprocess_image_value_range():
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    out = preprocess_image(img)
    # mobilenet_v2 preprocess_input maps [0,255] -> [-1,1]
    assert out.max() <= 1.0
    assert out.min() >= -1.0


def test_class_names_defined():
    assert len(CLASS_NAMES) == 2
    assert "Parasitized" in CLASS_NAMES
    assert "Uninfected" in CLASS_NAMES


def test_log_prediction_writes_row(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions_log.csv"
    monkeypatch.setattr("app.app.LOG_PATH", str(log_file))

    log_prediction("sample.png", "Parasitized", 0.9421)

    assert log_file.exists()
    with open(log_file) as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["timestamp", "filename", "prediction", "confidence", "model_version"]
    assert rows[1][1] == "sample.png"
    assert rows[1][2] == "Parasitized"
    assert rows[1][3] == "0.9421"


def test_log_prediction_appends(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions_log.csv"
    monkeypatch.setattr("app.app.LOG_PATH", str(log_file))

    log_prediction("a.png", "Uninfected", 0.80)
    log_prediction("b.png", "Parasitized", 0.91)

    with open(log_file) as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3  # header + 2 predictions


def test_overlay_heatmap_shape_and_type():
    heatmap = np.random.rand(7, 7)  # dummy Grad-CAM output
    original = np.zeros((224, 224, 3), dtype=np.uint8)
    overlay = overlay_heatmap(heatmap, original)
    assert overlay.shape == original.shape
    assert overlay.dtype == np.uint8

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import P3SDER


def test_p3sder_shape():
    """Test that P3SDER preserves the input shape and initializes residual scaling to zero."""
    x = torch.randn(1, 256, 80, 80)
    m = P3SDER(256, 256)
    y = m(x)

    assert y.shape == x.shape
    assert float(m.gamma.detach()) == 0.0


def test_p3sder_yaml_build():
    """Test that YOLOv8s/YOLO11s P3SDER YAMLs build with one inserted module."""
    for cfg in ("ultralytics/cfg/models/v8/yolov8s_p3sder.yaml", "ultralytics/cfg/models/11/yolo11s_p3sder.yaml"):
        model = YOLO(cfg).model.eval()
        x = torch.randn(1, 3, 64, 64)
        _ = model(x)
        assert sum(type(m).__name__ == "P3SDER" for m in model.modules()) == 1

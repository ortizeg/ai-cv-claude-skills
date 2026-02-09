"""Tests for the model module."""

from __future__ import annotations

import torch


def test_model_forward_shape() -> None:
    """Test that the model produces correct output shape."""
    from ${package_name}.model import Classifier, ModelConfig

    config = ModelConfig(num_classes=10, pretrained=False)
    model = Classifier(config=config)
    model.eval()

    batch = torch.randn(2, 3, 224, 224)
    output = model(batch)
    assert output.shape == (2, 10)

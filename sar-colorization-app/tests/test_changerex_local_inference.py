import numpy as np
import pytest
import torch
from PIL import Image

import changerex_local.inference as inference_module
from changerex_local.inference import ChangerExInferenceError, predict_change


class _Model(torch.nn.Module):
    def forward(self, pair):
        logits = torch.zeros(pair.shape[0], 2, pair.shape[2], pair.shape[3], device=pair.device)
        logits[:, 1, : pair.shape[2] // 2] = 10
        return logits


class _Lifecycle:
    selected_device = "cpu"
    load_time = 0.01
    checkpoint_report = None
    warnings = []

    def __init__(self):
        self.count = 0

    def ensure_loaded(self):
        reused = self.count > 0
        self.count += 1
        return _Model().eval(), reused

    def record_inference(self, elapsed):
        pass

    def status(self):
        return {
            "load_count": 1, "reuse_count": max(0, self.count - 1),
            "inference_count": self.count, "state": "ready", "warnings": [],
        }


def test_probability_extraction_source_restoration_and_reuse(monkeypatch):
    lifecycle = _Lifecycle()
    monkeypatch.setattr(inference_module, "get_lifecycle", lambda **kwargs: lifecycle)
    earlier = Image.new("RGB", (7, 5), (0, 0, 0))
    later = Image.new("RGB", (7, 5), (255, 255, 255))
    first = predict_change(earlier, later, device="cpu", maximum_dimension=10)
    second = predict_change(earlier, later, device="cpu", maximum_dimension=10)
    assert first.probability_map.shape == (5, 7)
    assert first.binary_mask.shape == (5, 7)
    assert np.isfinite(first.probability_map).all()
    assert 0 <= first.probability_map.min() <= first.probability_map.max() <= 1
    assert not first.load_reuse_status["was_reused"]
    assert second.load_reuse_status["was_reused"]


def test_safe_inference_failure(monkeypatch):
    monkeypatch.setattr(
        inference_module, "preprocess_pair", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad\ninput"))
    )
    with pytest.raises(ChangerExInferenceError, match="bad input") as captured:
        predict_change(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2)), device="cpu")
    assert "\n" not in str(captured.value)

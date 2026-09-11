from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ttp_service.inference import changed_probability_map


def test_two_class_logits_become_changed_probability_map() -> None:
    logits = np.array([
        [[4.0, -4.0], [0.0, 0.0]],
        [[-4.0, 4.0], [0.0, 2.0]],
    ], dtype=np.float32)
    sample = SimpleNamespace(seg_logits=SimpleNamespace(data=logits))
    probability, source = changed_probability_map([sample])
    assert source == "seg_logits_softmax"
    assert probability.shape == (2, 2)
    assert probability[0, 0] < 0.001
    assert probability[0, 1] > 0.999
    assert probability[1, 0] == 0.5
    assert probability[1, 1] > 0.8
    assert np.isfinite(probability).all()


def test_old_binary_prediction_remains_compatible() -> None:
    binary = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    probability, source = changed_probability_map({"predictions": binary})
    assert source == "binary_prediction_compatibility"
    assert np.array_equal(probability, binary.astype(np.float32))

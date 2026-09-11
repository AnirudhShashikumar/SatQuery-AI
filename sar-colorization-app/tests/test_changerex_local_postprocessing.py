import numpy as np
import pytest
from PIL import Image

from changerex_local.postprocessing import (
    ProbabilityValidationError,
    connected_components,
    make_overlay,
    threshold_probability,
    validate_probability_map,
)


def test_probability_validation_and_thresholding():
    probability = np.array([[0.0, 0.499], [0.5, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        threshold_probability(probability, 0.5), np.array([[0, 0], [1, 1]], dtype=np.uint8)
    )
    with pytest.raises(ProbabilityValidationError, match="non-finite"):
        validate_probability_map(np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ProbabilityValidationError, match=r"\[0, 1\]"):
        validate_probability_map(np.array([[1.1]], dtype=np.float32))


def test_connected_components_regions_and_overlay():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[3:5, 4:6] = 1
    regions = connected_components(mask)
    assert [region.area for region in regions] == [4, 4]
    assert {region.bbox_xyxy for region in regions} == {(0, 0, 2, 2), (4, 3, 6, 5)}
    later = Image.new("RGB", (6, 5), (20, 30, 40))
    overlay = np.asarray(make_overlay(later, mask))
    assert overlay.shape == (5, 6, 3)
    assert overlay[0, 0, 0] > 20
    np.testing.assert_array_equal(overlay[2, 2], [20, 30, 40])

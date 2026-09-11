import numpy as np
import pytest
from PIL import Image

from changerex_local.config import RGB_MEAN, RGB_STD
from changerex_local.preprocessing import ImageValidationError, preprocess_pair


def test_exact_preprocessing_and_temporal_order():
    earlier = np.zeros((1, 2, 3), dtype=np.uint8)
    earlier[..., 0] = 255
    later = np.zeros((1, 2, 3), dtype=np.uint8)
    later[..., 2] = 255
    tensor, details, _ = preprocess_pair(earlier, later, maximum_dimension=2)
    assert tensor.shape == (1, 6, 32, 32)
    assert details.source_size == (2, 1)
    assert details.resized_size == (2, 1)
    assert details.pad == (0, 0, 30, 31)
    observed = tensor[0, :, 0, 0].numpy()
    raw = np.array([255, 0, 0, 0, 0, 255], dtype=np.float32)
    expected = (raw - np.array(RGB_MEAN * 2)) / np.array(RGB_STD * 2)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-6)
    # Official normalized-space padding is exactly zero.
    assert float(tensor[0, :, -1, -1].abs().max()) == 0.0


def test_dimension_validation_and_aspect_resize():
    first = Image.new("RGB", (7, 5))
    second = Image.new("RGB", (8, 5))
    with pytest.raises(ImageValidationError, match="identical dimensions"):
        preprocess_pair(first, second, maximum_dimension=32)
    tensor, details, _ = preprocess_pair(first, first, maximum_dimension=10)
    assert details.resized_size == (10, 7)
    assert details.padded_size == (32, 32)
    assert tensor.dtype.is_floating_point

"""Official ChangerEx test preprocessing without Open-CD runtime dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as functional

from .config import DEFAULT_MAXIMUM_DIMENSION, RGB_MEAN, RGB_STD, SIZE_DIVISOR
from .schemas import PreprocessingDetails


ImageInput = Union[str, Path, Image.Image, np.ndarray]


class ImageValidationError(ValueError):
    pass


def load_rgb_image(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise ImageValidationError(f"Image file does not exist: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB").copy()
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ImageValidationError("NumPy images must have shape HxWx3 or HxWx4")
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ImageValidationError("Image array must contain finite numeric values")
            if array.min() < 0 or array.max() > 255:
                raise ImageValidationError("Image array values must be within [0, 255]")
            array = array.astype(np.uint8)
        return Image.fromarray(array[..., :3])
    raise ImageValidationError(f"Unsupported image type: {type(image).__name__}")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32)


def _official_resize_shape(width: int, height: int, maximum_dimension: int) -> tuple[int, int, float]:
    if maximum_dimension <= 0:
        raise ImageValidationError("maximum_dimension must be a positive integer")
    scale = maximum_dimension / max(width, height)
    resized_width = max(1, int(width * scale + 0.5))
    resized_height = max(1, int(height * scale + 0.5))
    return resized_width, resized_height, scale


def preprocess_pair(
    earlier_image: ImageInput,
    later_image: ImageInput,
    *,
    maximum_dimension: int = DEFAULT_MAXIMUM_DIMENSION,
) -> tuple[torch.Tensor, PreprocessingDetails, Image.Image]:
    """Return normalized NCHW pair, details, and the source-sized later image."""
    earlier = load_rgb_image(earlier_image)
    later = load_rgb_image(later_image)
    if earlier.size != later.size:
        raise ImageValidationError(
            f"Paired images must have identical dimensions; got {earlier.size} and {later.size}"
        )
    width, height = earlier.size
    if width < 1 or height < 1:
        raise ImageValidationError("Images must have non-zero dimensions")

    resized_width, resized_height, scale = _official_resize_shape(
        width, height, maximum_dimension
    )
    pair = torch.cat((_image_to_tensor(earlier), _image_to_tensor(later)), dim=0).unsqueeze(0)
    if (resized_height, resized_width) != (height, width):
        pair = functional.interpolate(
            pair, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

    mean = pair.new_tensor(RGB_MEAN * 2).view(1, 6, 1, 1)
    std = pair.new_tensor(RGB_STD * 2).view(1, 6, 1, 1)
    pair = (pair - mean) / std

    padded_width = ((resized_width + SIZE_DIVISOR - 1) // SIZE_DIVISOR) * SIZE_DIVISOR
    padded_height = ((resized_height + SIZE_DIVISOR - 1) // SIZE_DIVISOR) * SIZE_DIVISOR
    pad_right = padded_width - resized_width
    pad_bottom = padded_height - resized_height
    if pad_right or pad_bottom:
        pair = functional.pad(pair, (0, pad_right, 0, pad_bottom), value=0.0)

    details = PreprocessingDetails(
        source_size=(width, height),
        resized_size=(resized_width, resized_height),
        padded_size=(padded_width, padded_height),
        scale=scale,
        channel_order="RGB",
        pair_order="earlier RGB, then later RGB",
        input_range="float32 source values in [0, 255] before normalization",
        mean=RGB_MEAN,
        std=RGB_STD,
        resize_interpolation="bilinear, align_corners=False",
        pad=(0, 0, pad_right, pad_bottom),
        size_divisor=SIZE_DIVISOR,
    )
    return pair.contiguous(), details, later


def preprocessing_tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().cpu().contiguous()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sum": float(data.sum()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }

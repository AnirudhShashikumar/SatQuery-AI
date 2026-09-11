"""Dependency-free ChangerEx probability and mask postprocessing."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from .schemas import Region


class ProbabilityValidationError(ValueError):
    pass


def validate_probability_map(probability_map: np.ndarray) -> np.ndarray:
    array = np.asarray(probability_map, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ProbabilityValidationError("Probability map must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ProbabilityValidationError("Probability map contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ProbabilityValidationError(
            f"Probability values must be within [0, 1]; received [{minimum}, {maximum}]"
        )
    return np.clip(array, 0.0, 1.0)


def threshold_probability(probability_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    return (validate_probability_map(probability_map) >= threshold).astype(np.uint8)


def connected_components(mask: np.ndarray, *, minimum_area: int = 1) -> list[Region]:
    """Compute 8-connected regions; bboxes use exclusive x2/y2 coordinates."""
    if minimum_area < 1:
        raise ValueError("minimum_area must be at least one")
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("Binary mask must be 2D")
    binary = binary.astype(bool, copy=False)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    regions: list[Region] = []
    label = 0
    for start_y, start_x in np.argwhere(binary):
        y0 = int(start_y)
        x0 = int(start_x)
        if visited[y0, x0]:
            continue
        label += 1
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = True
        area = 0
        min_x = max_x = x0
        min_y = max_y = y0
        while queue:
            y, x = queue.popleft()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if binary[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
        if area >= minimum_area:
            regions.append(Region(label=label, area=area, bbox_xyxy=(min_x, min_y, max_x + 1, max_y + 1)))
    regions.sort(key=lambda region: region.area, reverse=True)
    return regions


def make_display_mask(mask: np.ndarray) -> Image.Image:
    binary = (np.asarray(mask).astype(bool) * 255).astype(np.uint8)
    red = np.zeros((*binary.shape, 3), dtype=np.uint8)
    red[..., 0] = binary
    return Image.fromarray(red)


def make_overlay(later_image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    base = np.asarray(later_image.convert("RGB"), dtype=np.float32).copy()
    binary = np.asarray(mask).astype(bool)
    if binary.shape != base.shape[:2]:
        raise ValueError("Overlay mask and later image dimensions must match")
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    base[binary] = (1.0 - alpha) * base[binary] + alpha * red[binary]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

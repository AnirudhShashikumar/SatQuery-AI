"""Dataclasses returned by the standalone ChangerEx package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessingDetails:
    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    padded_size: tuple[int, int]
    scale: float
    channel_order: str
    pair_order: str
    input_range: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_interpolation: str
    pad: tuple[int, int, int, int]
    size_divisor: int


@dataclass(frozen=True)
class Region:
    label: int
    area: int
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class RuntimeStats:
    load_seconds: float
    inference_seconds: float
    preprocessing_seconds: float
    model_seconds: float
    postprocessing_seconds: float
    peak_memory_mb: float | None


@dataclass(frozen=True)
class CheckpointVerificationReport:
    path: str
    file_size: int
    sha256: str
    top_level_keys: tuple[str, ...]
    tensor_count: int
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    key_transformations: tuple[str, ...]
    checkpoint_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangerExResult:
    source_width: int
    source_height: int
    model_input_width: int
    model_input_height: int
    selected_device: str
    probability_map: np.ndarray = field(repr=False)
    binary_mask: np.ndarray = field(repr=False)
    changed_pixel_count: int = 0
    changed_percentage: float = 0.0
    connected_component_count: int = 0
    largest_component_size: int = 0
    component_bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    preprocessing: PreprocessingDetails | None = None
    threshold: float = 0.5
    checkpoint_provenance: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeStats | None = None
    load_reuse_status: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    stage_durations_ms: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary without embedding full raster arrays."""
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "model_input_width": self.model_input_width,
            "model_input_height": self.model_input_height,
            "selected_device": self.selected_device,
            "probability_shape": list(self.probability_map.shape),
            "probability_min": float(self.probability_map.min()),
            "probability_max": float(self.probability_map.max()),
            "binary_mask_shape": list(self.binary_mask.shape),
            "changed_pixel_count": self.changed_pixel_count,
            "changed_percentage": self.changed_percentage,
            "connected_component_count": self.connected_component_count,
            "largest_component_size": self.largest_component_size,
            "component_bounding_boxes": [list(box) for box in self.component_bounding_boxes],
            "preprocessing": asdict(self.preprocessing) if self.preprocessing else None,
            "threshold": self.threshold,
            "checkpoint_provenance": self.checkpoint_provenance,
            "runtime": asdict(self.runtime) if self.runtime else None,
            "load_reuse_status": self.load_reuse_status,
            "stage_durations_ms": self.stage_durations_ms,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }

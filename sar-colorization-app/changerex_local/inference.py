"""Public inference API for standalone ChangerEx."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as functional

from .config import DEFAULT_MAXIMUM_DIMENSION, PROVENANCE
from .lifecycle import get_lifecycle, peak_memory_mb
from .postprocessing import connected_components, threshold_probability, validate_probability_map
from .preprocessing import ImageInput, preprocess_pair
from .schemas import ChangerExResult, RuntimeStats


class ChangerExInferenceError(RuntimeError):
    pass


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def predict_change(
    earlier_image: ImageInput,
    later_image: ImageInput,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    maximum_dimension: int = DEFAULT_MAXIMUM_DIMENSION,
    allow_device_fallback: bool = False,
) -> ChangerExResult:
    """Predict changed pixels for an ordered, co-registered RGB image pair."""
    total_started = time.perf_counter()
    try:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        preprocessing_started = time.perf_counter()
        pair, preprocessing, _ = preprocess_pair(
            earlier_image, later_image, maximum_dimension=maximum_dimension
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        lifecycle = get_lifecycle(
            device=device, allow_device_fallback=allow_device_fallback
        )
        model, was_reused = lifecycle.ensure_loaded()
        selected = lifecycle.selected_device
        assert selected is not None
        pair = pair.to(device=selected, dtype=torch.float32)

        _synchronize(selected)
        model_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(pair)
        _synchronize(selected)
        model_seconds = time.perf_counter() - model_started

        probability_started = time.perf_counter()
        with torch.inference_mode():
            changed_probability = torch.softmax(logits, dim=1)[:, 1:2]
        _synchronize(selected)
        probability_seconds = time.perf_counter() - probability_started

        postprocessing_started = time.perf_counter()
        resized_width, resized_height = preprocessing.resized_size
        source_width, source_height = preprocessing.source_size
        changed_probability = changed_probability[..., :resized_height, :resized_width]
        threshold_started = time.perf_counter()
        resized_mask = (changed_probability >= threshold).to(dtype=torch.float32)
        _synchronize(selected)
        threshold_seconds = time.perf_counter() - threshold_started
        restoration_started = time.perf_counter()
        if (resized_height, resized_width) != (source_height, source_width):
            restored_probability = functional.interpolate(
                changed_probability,
                size=(source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
            restored_mask = functional.interpolate(
                resized_mask, size=(source_height, source_width), mode="nearest"
            )
        else:
            restored_probability = changed_probability
            restored_mask = resized_mask
        probability_map = restored_probability[0, 0].detach().cpu().numpy().astype(np.float32)
        probability_map = validate_probability_map(probability_map)
        binary_mask = restored_mask[0, 0].detach().cpu().numpy().astype(np.uint8)
        # Ensure threshold semantics are identical when no restoration occurred.
        if (resized_height, resized_width) == (source_height, source_width):
            binary_mask = threshold_probability(probability_map, threshold)
        restoration_seconds = time.perf_counter() - restoration_started
        components_started = time.perf_counter()
        regions = connected_components(binary_mask)
        components_seconds = time.perf_counter() - components_started
        changed_pixel_count = int(binary_mask.sum())
        changed_percentage = 100.0 * changed_pixel_count / binary_mask.size
        postprocessing_seconds = time.perf_counter() - postprocessing_started
        total_seconds = time.perf_counter() - total_started
        lifecycle.record_inference(total_seconds)
        status = lifecycle.status()
        report = lifecycle.checkpoint_report
        provenance = asdict(PROVENANCE)
        if report:
            provenance.update(
                {
                    "verified_path": report.path,
                    "verified_sha256": report.sha256,
                    "strict_missing_keys": list(report.missing_keys),
                    "strict_unexpected_keys": list(report.unexpected_keys),
                    "key_transformations": list(report.key_transformations),
                    "verification_report": report.to_dict(),
                }
            )
        return ChangerExResult(
            source_width=source_width,
            source_height=source_height,
            model_input_width=preprocessing.padded_size[0],
            model_input_height=preprocessing.padded_size[1],
            selected_device=selected,
            probability_map=probability_map,
            binary_mask=binary_mask,
            changed_pixel_count=changed_pixel_count,
            changed_percentage=changed_percentage,
            connected_component_count=len(regions),
            largest_component_size=regions[0].area if regions else 0,
            component_bounding_boxes=[region.bbox_xyxy for region in regions],
            regions=regions,
            preprocessing=preprocessing,
            threshold=threshold,
            checkpoint_provenance=provenance,
            runtime=RuntimeStats(
                load_seconds=lifecycle.load_time,
                inference_seconds=total_seconds,
                preprocessing_seconds=preprocessing_seconds,
                model_seconds=model_seconds,
                postprocessing_seconds=postprocessing_seconds,
                peak_memory_mb=peak_memory_mb(selected),
            ),
            load_reuse_status={
                "was_reused": was_reused,
                "load_count": status["load_count"],
                "reuse_count": status["reuse_count"],
                "inference_count": status["inference_count"],
                "state": status["state"],
            },
            stage_durations_ms={
                "preprocessing": max(0, round(preprocessing_seconds * 1000)),
                "inference": max(0, round(model_seconds * 1000)),
                "probability_extraction": max(0, round(probability_seconds * 1000)),
                "thresholding": max(0, round(threshold_seconds * 1000)),
                "source_restoration": max(0, round(restoration_seconds * 1000)),
                "connected_components": max(0, round(components_seconds * 1000)),
            },
            warnings=list(status["warnings"]),
            limitations=[
                "The output is a model prediction, not ground truth.",
                "LEVIR-CD primarily represents building change and may not generalize to all land-cover changes.",
                "Inputs must be geometrically co-registered; misregistration can appear as change.",
            ],
        )
    except Exception as error:
        if isinstance(error, ChangerExInferenceError):
            raise
        message = f"{type(error).__name__}: {error}".replace("\n", " ")[:500]
        raise ChangerExInferenceError(message) from error

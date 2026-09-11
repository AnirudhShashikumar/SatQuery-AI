"""Artifact writing and environment diagnostics for standalone runs."""

from __future__ import annotations

import json
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .config import OPENCD_COMMIT, OPENCD_VERSION
from .postprocessing import make_display_mask, make_overlay
from .schemas import ChangerExResult


def environment_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_required": False,
        "opencd_runtime_required": False,
        "opencd_reference_version": OPENCD_VERSION,
        "opencd_reference_commit": OPENCD_COMMIT,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark_summary(
    inference_seconds: Sequence[float], *, repeat_max_abs_difference: float | None = None
) -> dict[str, Any]:
    values = [float(value) for value in inference_seconds]
    return {
        "runs": len(values),
        "seconds": values,
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": _percentile(values, 95),
        "repeat_max_abs_probability_difference": repeat_max_abs_difference,
        "deterministic_repeatability": (
            repeat_max_abs_difference == 0.0 if repeat_max_abs_difference is not None else None
        ),
    }


def save_artifacts(
    result: ChangerExResult,
    later_image: Image.Image,
    output_dir: str | Path,
    *,
    benchmark: dict[str, Any] | None = None,
    save_probability_npy: bool = True,
    smoke_source: dict[str, Any] | None = None,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "probability_map": str(output / "probability_map.npy"),
        "probability_png": str(output / "probability_map.png"),
        "binary_mask": str(output / "binary_mask.png"),
        "display_mask": str(output / "display_mask.png"),
        "overlay": str(output / "overlay.png"),
        "regions": str(output / "regions.json"),
        "result": str(output / "result.json"),
        "checkpoint_verification": str(output / "checkpoint_verification.json"),
        "environment": str(output / "environment.json"),
        "benchmark_report": str(output / "benchmark_report.md"),
    }
    if save_probability_npy:
        np.save(paths["probability_map"], result.probability_map, allow_pickle=False)
    probability_u16 = np.rint(np.clip(result.probability_map, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(probability_u16).save(paths["probability_png"])
    Image.fromarray((result.binary_mask * 255).astype(np.uint8)).save(paths["binary_mask"])
    make_display_mask(result.binary_mask).save(paths["display_mask"])
    make_overlay(later_image, result.binary_mask).save(paths["overlay"])

    regions = [asdict(region) for region in result.regions]
    Path(paths["regions"]).write_text(json.dumps(regions, indent=2), encoding="utf-8")
    summary = result.summary()
    summary["benchmark"] = benchmark
    summary["smoke_source"] = smoke_source
    summary["artifacts"] = paths
    Path(paths["result"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    verification = result.checkpoint_provenance.get("verification_report") or {
        key: result.checkpoint_provenance.get(key)
        for key in (
            "verified_path",
            "verified_sha256",
            "strict_missing_keys",
            "strict_unexpected_keys",
            "key_transformations",
        )
    }
    verification["provenance"] = {
        key: result.checkpoint_provenance.get(key)
        for key in ("checkpoint_filename", "checkpoint_url", "opencd_version", "opencd_commit")
    }
    Path(paths["checkpoint_verification"]).write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    Path(paths["environment"]).write_text(
        json.dumps(environment_report(), indent=2), encoding="utf-8"
    )

    benchmark = benchmark or {}
    source_lines = []
    if smoke_source:
        source_lines = [
            f"- Imagery source: {smoke_source.get('source', 'unspecified')}",
            f"- Source URL: {smoke_source.get('url', 'unspecified')}",
            f"- License/use basis: {smoke_source.get('license', 'unspecified')}",
        ]
    report = "\n".join(
        [
            "# ChangerEx standalone benchmark report",
            "",
            "> The binary mask and overlay are model predictions, not ground truth. This operational smoke run does not measure accuracy.",
            "",
            "## Input and output",
            "",
            f"- Source dimensions: {result.source_width} × {result.source_height}",
            f"- Model input dimensions: {result.model_input_width} × {result.model_input_height}",
            f"- Device: {result.selected_device}",
            f"- Threshold: {result.threshold}",
            f"- Probability range: {float(result.probability_map.min()):.8f}–{float(result.probability_map.max()):.8f}",
            f"- Changed percentage: {result.changed_percentage:.6f}%",
            f"- Connected regions: {result.connected_component_count}",
            *source_lines,
            "",
            "## Runtime",
            "",
            f"- Model load: {result.runtime.load_seconds if result.runtime else None} s",
            f"- First inference: {benchmark.get('first_total_seconds', result.runtime.inference_seconds if result.runtime else None)} s",
            f"- First model forward: {benchmark.get('first_model_seconds')} s",
            f"- Benchmark runs: {benchmark.get('runs')}",
            f"- Mean: {benchmark.get('mean_seconds')} s",
            f"- Median: {benchmark.get('median_seconds')} s",
            f"- P95: {benchmark.get('p95_seconds')} s",
            f"- Peak memory: {result.runtime.peak_memory_mb if result.runtime else None} MB",
            f"- Reuse count: {result.load_reuse_status.get('reuse_count')}",
            f"- Deterministic repeatability: {benchmark.get('deterministic_repeatability')}",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in result.limitations],
            "",
        ]
    )
    Path(paths["benchmark_report"]).write_text(report, encoding="utf-8")
    return paths

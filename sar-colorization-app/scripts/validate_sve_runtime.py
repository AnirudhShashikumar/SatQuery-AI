#!/usr/bin/env python3
"""Run local SVE smoke validation without exporting images, paths, or embeddings."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import resource
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satquery_agent.services.sve_service import SVEManager
from satquery_agent.models import (
    AgentResponse,
    ExecutionStep,
    ExecutionSummary,
    ImageFormat,
    ImageMetadata,
    ImageModality,
    InputMode,
    Modality,
    ReportFormat,
    RepresentationType,
    ResponseStatus,
    TaskType,
    ToolStatus,
    ValidationStatus,
)
from satquery_agent.reporting import MISSION_STORE, generate_report, mark_fresh
from satquery_agent.specialists.single_image_evidence import extract_single_image_evidence
from satquery_agent.specialists.vqa import get_vqa
from satquery_agent.sve_artifacts import verify_sve_artifacts


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(usage / divisor, 1)


def png_sha256(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def run() -> dict:
    os.environ.setdefault("SVE_ENABLED", "true")
    os.environ.setdefault("SVE_DEVICE", "auto")
    service = SVEManager()
    verified_started = time.perf_counter()
    artifacts = verify_sve_artifacts()
    verification_ms = round((time.perf_counter() - verified_started) * 1000)

    with Image.open(ROOT / "satquery_agent/demo_samples/single-optical.png") as source:
        png = source.convert("RGB")
    with Image.open(ROOT / "satquery_agent/demo_samples/cross-optical.tif") as source:
        geotiff = source.convert("RGB")
    with Image.open(ROOT / "satquery_agent/demo_samples/change-before.png") as source:
        before = source.convert("RGB")
    with Image.open(ROOT / "satquery_agent/demo_samples/change-after.png") as source:
        after = source.convert("RGB")
    hashes = {
        name: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for name, relative in {
            "png": "satquery_agent/demo_samples/single-optical.png",
            "geotiff": "satquery_agent/demo_samples/cross-optical.tif",
            "before": "satquery_agent/demo_samples/change-before.png",
            "after": "satquery_agent/demo_samples/change-after.png",
        }.items()
    }
    warm_image = png.copy()
    original_pixel = warm_image.getpixel((0, 0))
    warm_image.putpixel((0, 0), tuple((value + 1) % 256 for value in original_pixel))

    metadata = ImageMetadata(
        file_id="runtime-validation", original_name="approved-optical-sample.png", safe_name="approved-optical-sample.png",
        format=ImageFormat.PNG, mime_type="image/png", size_bytes=0, width=png.width, height=png.height,
        band_count=3, dtype="uint8", color_interpretation=["red", "green", "blue"], warnings=[],
        representation=RepresentationType.DISPLAY_PREVIEW, auto_detected_modality=ImageModality.OPTICAL_RGB,
        effective_modality=ImageModality.OPTICAL_RGB,
    )
    evidence = extract_single_image_evidence(np.asarray(png), metadata)
    vqa = get_vqa()
    intent = vqa.classify_question("Is vegetation visible?")
    controlled = vqa.answer("Is vegetation visible?", evidence.result, metadata, intent)

    cold = service.analyze(
        png,
        hashes["png"],
        captions=["a satellite scene containing vegetation", "an urban satellite scene"],
        vqa_category=controlled.details.question_category.value,
        vqa_answer=controlled.answer,
        grounding_target="road",
    )
    warm_cached = service.analyze(png, hashes["png"], captions=["a satellite scene containing vegetation"])
    warm_uncached = service.analyze(warm_image, png_sha256(warm_image))
    tiff = service.analyze(geotiff, hashes["geotiff"])
    temporal = service.compare(
        before,
        hashes["before"],
        after,
        hashes["after"],
        label="Before/after scene-level semantic similarity",
        disclaimer="Embedding differences are scene-level semantic evidence, not spatial change localization.",
    )
    raw_sar = service.unsupported_raw_sar_result()
    trace_steps = [ExecutionStep(
        tool=item["tool"], status=ToolStatus(item["status"]), duration_ms=item["duration_ms"], parameters=item.get("parameters") or {}
    ) for item in cold.trace]
    report_response = mark_fresh(AgentResponse(
        request_id=str(uuid.uuid4()), task=TaskType.VQA, answer=controlled.answer,
        confidence=controlled.details.confidence, evidence=[],
        execution=ExecutionSummary(
            input_mode=InputMode.SINGLE, selected_tools=["input_validator", "rs_vqa"], steps=trace_steps,
            duration_ms=cold.result.runtime_ms or 0, permitted_parameters={"validation": "local_runtime"},
            validation=ValidationStatus(valid=True, errors=[]), selection_reason="Runtime validation of controlled VQA with optional SVE evidence.",
        ), warnings=[], status=ResponseStatus.SUCCESS, primary_image_metadata=metadata,
        vqa_details=controlled.details, sve_result=cold.result,
    ), "runtime-validation-cache-key")
    MISSION_STORE.put(report_response, "runtime-validation-cache-key", report_response.cache.original_generation_timestamp if report_response.cache else None)
    report = generate_report(report_response.request_id, [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.CSV, ReportFormat.ZIP])
    metrics = service.metrics()
    try:
        import torch
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_available = False
    return {
        "schema_version": "sve-runtime-validation-1.0",
        "artifact_verification": {
            "verified": True,
            "checksum_fingerprint": artifacts.checksum_fingerprint,
            "verification_ms": verification_ms,
        },
        "runtime": {
            "device": cold.result.device,
            "cold_load_and_inference_ms": cold.result.runtime_ms,
            "warm_cached_request_ms": warm_cached.result.runtime_ms,
            "warm_uncached_image_inference_ms": warm_uncached.result.runtime_ms,
            "optical_geotiff_inference_ms": tiff.result.runtime_ms,
            "bi_temporal_inference_ms": temporal.result.runtime_ms,
            "peak_process_memory_mb": peak_memory_mb(),
            "mps_available": mps_available,
            "mps_inference_ms": None,
            "cpu_fallback_ms": None,
        },
        "checks": {
            "optical_png": cold.result.status,
            "optical_geotiff": tiff.result.status,
            "caption_reranking": cold.result.caption_consistency.model_dump(mode="json") if cold.result.caption_consistency else None,
            "vqa_consistency": cold.result.vqa_consistency.model_dump(mode="json") if cold.result.vqa_consistency else None,
            "controlled_vqa": {"supported": controlled.details.supported, "category": controlled.details.question_category.value, "answer_source": controlled.details.answer_source},
            "grounding_support": cold.result.grounding_support.model_dump(mode="json") if cold.result.grounding_support else None,
            "bi_temporal": temporal.result.semantic_comparison.model_dump(mode="json") if temporal.result.semantic_comparison else None,
            "raw_sar": {"status": raw_sar.status, "comparison_status": raw_sar.semantic_comparison.status if raw_sar.semantic_comparison else None},
            "embedding_exported": False,
            "full_report_generation": {"status": report.status, "formats": [item.format.value for item in report.artifacts], "artifact_count": len(report.artifacts)},
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

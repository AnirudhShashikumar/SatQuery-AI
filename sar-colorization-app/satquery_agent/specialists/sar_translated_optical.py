"""Selective optical evidence over an explicitly generated SAR representation."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ..image_ingestion import IngestedImage
from ..models import (
    DetectionConfidence,
    EvidenceItem,
    EvidenceLifecycleState,
    ImageFormat,
    ImageMetadata,
    ImageModality,
    Modality,
    RepresentationType,
    SarSceneResult,
    SarTranslatedOpticalAnalysis,
    SarWaterResult,
    TaskType,
)
from ..router import detect_task
from ..services.sar_translation_service import (
    DISCLOSURE,
    PIX2PIX_DISCLOSURE,
    SarTranslationError,
    evaluate_translation_eligibility,
    get_sar_translation_service,
    optical_specialists_enabled,
)
from ..services.sve_service import get_sve_service
from .captioner import CaptionerError, get_captioner
from .grounder import GrounderError, get_grounder
from .rsvqa_specialist import RSVQASpecialistError, classify_rsvqa_task, get_rsvqa_specialist
from .sar_optical_fusion import fuse_sar_optical_evidence
from .sar_scene import SarSceneAnalysisError, analyze_sar_scene


LOGGER = logging.getLogger("satquery.sar_translated_optical")


@dataclass
class SarTranslatedEvidenceRun:
    analysis: SarTranslatedOpticalAnalysis
    native_scene: Optional[SarSceneResult]
    steps: list[dict[str, Any]]


def _elapsed(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _step(tool: str, status: str, duration_ms: int = 0, **parameters: Any) -> dict[str, Any]:
    return {"tool": tool, "status": status, "duration_ms": max(0, duration_ms), "parameters": parameters}


def _generated_metadata(source: ImageMetadata, width: int, height: int, preview_url: Optional[str]) -> ImageMetadata:
    return source.model_copy(update={
        "file_id": str(uuid.uuid4()),
        "original_name": "generated-optical-like.png",
        "safe_name": "generated-optical-like.png",
        "format": ImageFormat.PNG,
        "mime_type": "image/png",
        "size_bytes": 0,
        "width": width,
        "height": height,
        "band_count": 3,
        "dtype": "uint8",
        "crs": None,
        "transform": None,
        "bounds": None,
        "nodata": None,
        "is_georeferenced": False,
        "preview_url": preview_url,
        "color_interpretation": ["red", "green", "blue"],
        "band_descriptions": ["generated_red", "generated_green", "generated_blue"],
        "band_statistics": [],
        "representation": RepresentationType.DISPLAY_PREVIEW,
        "auto_detected_modality": ImageModality.OPTICAL_RGB,
        "auto_detection_confidence": DetectionConfidence.UNAVAILABLE,
        "auto_detection_reason": "Generated optical-like representation; not an observed optical image.",
        "user_confirmed_modality": None,
        "effective_modality": ImageModality.OPTICAL_RGB,
        "modality_limitations": [DISCLOSURE],
        "warnings": [DISCLOSURE],
    })


def run_sar_translated_optical_evidence(
    *,
    ingested: IngestedImage,
    query: str,
    routed_task: TaskType,
    native_water: Optional[SarWaterResult],
    native_scene: Optional[SarSceneResult],
    automatic_preview: bool = False,
    evidence_run_id: Optional[str] = None,
) -> SarTranslatedEvidenceRun:
    """Run the optional branch synchronously so the API can isolate it in a worker thread."""
    steps: list[dict[str, Any]] = []
    supporting_scene = native_scene
    if supporting_scene is None:
        started = time.perf_counter()
        try:
            supporting_scene, _ = analyze_sar_scene(ingested.analysis_raster, ingested.metadata, TaskType.SAR_SCENE_ANALYSIS)
            steps.append(_step("sar_native_supporting_analysis", "success", _elapsed(started), method=supporting_scene.method))
        except SarSceneAnalysisError as error:
            steps.append(_step("sar_native_supporting_analysis", "failed", _elapsed(started), error_code=error.code))

    service = get_sar_translation_service()
    eligibility_started = time.perf_counter()
    eligibility = evaluate_translation_eligibility(
        ingested.analysis_raster,
        ingested.metadata,
        ingested.model_image,
        requested_model="pix2pix" if automatic_preview else None,
    )
    steps.append(_step(
        "sar_translation_eligibility",
        "success" if eligibility.eligible else "skipped",
        _elapsed(eligibility_started),
        eligible=eligibility.eligible,
        requested_model=eligibility.requested_model,
        selected_model=eligibility.selected_model,
        band_count=eligibility.band_count,
        finite_value_ratio=round(eligibility.finite_value_ratio, 6),
        reason=eligibility.reason,
    ))
    try:
        if automatic_preview:
            translated = service.translate(
                ingested.analysis_raster,
                ingested.metadata,
                ingested.model_image,
                automatic_preview=True,
            )
        else:
            translated = service.translate(ingested.analysis_raster, ingested.metadata, ingested.model_image)
    except Exception as caught:
        if isinstance(caught, SarTranslationError):
            error = caught
        else:
            LOGGER.exception("Unexpected SAR translation generation failure")
            error = SarTranslationError(
                "TRANSLATION_EXECUTION_FAILED",
                "The selected SAR translation model failed during generation; native SAR evidence was retained.",
            )
        generation_state = (
            EvidenceLifecycleState.NOT_ELIGIBLE
            if error.code in {"INELIGIBLE_INPUT", "INVALID_AUTOMATIC_PREVIEW"}
            else EvidenceLifecycleState.FAILED
        )
        analysis = fuse_sar_optical_evidence(
            query=query,
            translation=None,
            optical_evidence=None,
            native_water=native_water,
            native_scene=supporting_scene,
            translation_error=error.message,
            generation_state=generation_state,
            semantic_comparison_state=EvidenceLifecycleState.NOT_REQUESTED,
            evidence_products=[],
            attempted_model="Pix2Pix" if eligibility.selected_model == "pix2pix" else "SARFusionFormer" if eligibility.selected_model == "sarfusionformer" else None,
        )
        steps.append(_step(
            "sar_translation_inference",
            "skipped" if generation_state == EvidenceLifecycleState.NOT_ELIGIBLE else "failed",
            0,
            error_code=error.code,
            fallback_exhausted=generation_state == EvidenceLifecycleState.FAILED,
        ))
        steps.append(_step("translation_semantic_comparison", "skipped", 0, reason="generation_not_succeeded"))
        steps.append(_step("sar_optical_evidence_fusion", "success", 0, agreement=analysis.agreement))
        steps.append(_step("response_generation", "success", 0, source="native_sar_fallback"))
        return SarTranslatedEvidenceRun(analysis=analysis, native_scene=supporting_scene, steps=steps)

    for stage, duration in translated.stage_durations_ms.items():
        if stage == "sar_translation_eligibility":
            continue
        steps.append(_step(
            stage,
            getattr(translated, "stage_statuses", {}).get(stage, "success"),
            duration,
            model=translated.model_used,
            fallback_used=translated.fallback_used,
            color_correction_used=translated.color_correction_used,
        ))

    translation_payload = {
        "width": translated.width,
        "height": translated.height,
        "model_used": translated.model_used,
        "fallback_used": translated.fallback_used,
        "fallback_reason": translated.fallback_reason,
        "color_correction_used": translated.color_correction_used,
        "device": translated.device,
        "runtime_ms": translated.runtime_ms,
        "stage_durations_ms": translated.stage_durations_ms,
        "preprocessing_method": translated.preprocessing_method,
        "input_channel_interpretation": translated.input_channel_interpretation,
        "output_value_range": translated.output_value_range,
        "warnings": translated.warnings,
        "provenance": translated.provenance,
        "artifact_url": translated.artifact_url,
        "normalized_sar_preview_url": translated.normalized_sar_preview_url,
        "color_corrected_artifact_url": translated.color_corrected_artifact_url,
        "content_hash": translated.content_hash,
    }
    translation_payload["provenance"] = {
        **translation_payload["provenance"],
        "source_observation_id": ingested.metadata.file_id,
        "source_modality": ingested.metadata.effective_modality.value,
        "evidence_run_id": evidence_run_id,
        "evidence_type": "generated_optical_like",
        "generator": translated.model_used,
    }
    evidence_products: list[EvidenceItem] = []
    if translated.artifact_url and evidence_run_id:
        evidence_products.append(EvidenceItem(
            evidence_id=f"{evidence_run_id}:sar-translation:{translated.content_hash[:16]}",
            source_observation_id=ingested.metadata.file_id,
            source_role=ingested.metadata.observation_role,
            source_modality=ingested.metadata.effective_modality,
            evidence_type="generated_optical_like",
            evidence_run_id=evidence_run_id,
            generator=translated.model_used,
            status=EvidenceLifecycleState.SUCCEEDED,
            type="generated_optical_like_representation",
            label="Generated optical-like supporting evidence",
            description=PIX2PIX_DISCLOSURE if translated.model_used == "Pix2Pix" else DISCLOSURE,
            reference=translated.artifact_url,
        ))
    optical: dict[str, Any] = {"specialists_executed": []}
    metadata = _generated_metadata(ingested.metadata, translated.width, translated.height, translated.artifact_url)
    inferred_task, _, _ = detect_task(query)
    try:
        if optical_specialists_enabled():
            if get_sve_service().enabled:
                started = time.perf_counter()
                sve_call = get_sve_service().analyze(translated.image, translated.content_hash)
                if sve_call.result.available:
                    optical["scene_priors"] = [item.model_dump(mode="json") for item in sve_call.result.scene_priors]
                    optical["specialists_executed"].append("satquery_vision_encoder_v1")
                    steps.append(_step("translated_optical_sve", "success", _elapsed(started), generated_representation=True))
                else:
                    steps.append(_step("translated_optical_sve", "failed", _elapsed(started), reason=sve_call.result.status))

            if routed_task in {TaskType.SAR_SCENE_ANALYSIS, TaskType.CAPTIONING} or inferred_task == TaskType.CAPTIONING:
                started = time.perf_counter()
                try:
                    caption = get_captioner().describe(
                        translated.image, metadata, Modality.OPTICAL, ["generated_red", "generated_green", "generated_blue"],
                        "SAR-generated optical-like RGB representation",
                    )
                    optical["caption"] = caption.caption
                    optical["specialists_executed"].append("rs_captioner")
                    steps.append(_step("translated_optical_captioning", "success", _elapsed(started), generated_representation=True))
                except CaptionerError as error:
                    translated.warnings.append(error.message)
                    steps.append(_step("translated_optical_captioning", "failed", _elapsed(started), error_code=error.code))

            if inferred_task == TaskType.GROUNDING:
                started = time.perf_counter()
                try:
                    grounding = get_grounder().ground(
                        translated.image, query, metadata, Modality.OPTICAL,
                        ["generated_red", "generated_green", "generated_blue"],
                        "SAR-generated optical-like RGB representation",
                    )
                    optical["grounding"] = grounding.model_dump(mode="json")
                    optical["specialists_executed"].append("rs_grounder")
                    steps.append(_step("translated_optical_grounding", "success", _elapsed(started), generated_representation=True, source_coordinate_mapping=False))
                except GrounderError as error:
                    translated.warnings.append(error.message)
                    steps.append(_step("translated_optical_grounding", "failed", _elapsed(started), error_code=error.code))

            task_head = classify_rsvqa_task(query)
            if task_head is not None:
                started = time.perf_counter()
                try:
                    prediction = get_rsvqa_specialist().predict(translated.image, query, content_hash=translated.content_hash)
                    optical["vqa"] = {
                        "question": query,
                        "answer": prediction["answer"],
                        "confidence": prediction["confidence"],
                        "task": prediction["task"],
                        "logits": prediction["logits"],
                        "probabilities": prediction["probabilities"],
                        "model_used": prediction["model_used"],
                    }
                    optical["specialists_executed"].append("rsvqa_vqa_specialist")
                    steps.append(_step("translated_optical_vqa", "success", _elapsed(started), task_head=task_head, generated_representation=True))
                except RSVQASpecialistError as error:
                    translated.warnings.append(str(error))
                    steps.append(_step("translated_optical_vqa", "failed", _elapsed(started), task_head=task_head))
        else:
            steps.extend([
                _step("translated_optical_sve", "skipped", 0, reason="optical_specialists_disabled"),
                _step("translated_optical_captioning", "skipped", 0, reason="optical_specialists_disabled"),
                _step("translated_optical_grounding", "skipped", 0, reason="optical_specialists_disabled"),
                _step("translated_optical_vqa", "skipped", 0, reason="optical_specialists_disabled"),
            ])
    except Exception:
        LOGGER.exception("Unexpected translated-optical semantic stage failure")
        translated.warnings.append("Optional semantic analysis of the generated representation failed safely.")
        steps.append(_step(
            "translated_optical_semantic_dispatch",
            "failed",
            0,
            error_code="TRANSLATED_SEMANTIC_EXECUTION_FAILED",
        ))
    finally:
        translated.image.close()

    semantic_payload_present = any(optical.get(key) for key in ("caption", "scene_priors", "grounding", "vqa"))
    semantic_failure = any(
        item["status"] == "failed" and item["tool"].startswith("translated_optical_")
        for item in steps
    )
    if not evidence_products and (semantic_payload_present or semantic_failure):
        semantic_state = EvidenceLifecycleState.FAILED
        semantic_reason = "generated_evidence_not_published"
    elif semantic_payload_present:
        semantic_state = EvidenceLifecycleState.SUCCEEDED
        semantic_reason = "published_generated_evidence_interpreted"
    elif semantic_failure:
        semantic_state = EvidenceLifecycleState.FAILED
        semantic_reason = "no_translation_semantic_stage_succeeded"
    else:
        semantic_state = EvidenceLifecycleState.NOT_REQUESTED
        semantic_reason = "no_translation_semantic_comparison_requested"

    analysis = fuse_sar_optical_evidence(
        query=query,
        translation=translation_payload,
        optical_evidence=optical,
        native_water=native_water,
        native_scene=supporting_scene,
        generation_state=EvidenceLifecycleState.SUCCEEDED,
        semantic_comparison_state=semantic_state,
        evidence_products=evidence_products,
    )
    analysis.runtime_breakdown_ms.update({item["tool"]: item["duration_ms"] for item in steps})
    steps.append(_step(
        "translation_semantic_comparison",
        "success" if semantic_state == EvidenceLifecycleState.SUCCEEDED else "failed" if semantic_state == EvidenceLifecycleState.FAILED else "skipped",
        0,
        reason=semantic_reason,
        claim_used=semantic_state == EvidenceLifecycleState.SUCCEEDED,
    ))
    steps.append(_step("sar_optical_evidence_fusion", "success", 0, agreement=analysis.agreement, calibrated=False, translation_claim_used=semantic_state == EvidenceLifecycleState.SUCCEEDED))
    steps.append(_step("response_generation", "success", 0, source="translated_optical_specialist"))
    return SarTranslatedEvidenceRun(analysis=analysis, native_scene=supporting_scene, steps=steps)

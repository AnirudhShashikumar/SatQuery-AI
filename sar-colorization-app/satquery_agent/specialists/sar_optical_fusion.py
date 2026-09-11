"""Deterministic fusion of native SAR and generated-representation evidence."""

from __future__ import annotations

from typing import Any, Optional

from ..models import (
    Confidence,
    ConfidenceLevel,
    EvidenceItem,
    EvidenceLifecycleState,
    SarSceneResult,
    SarTranslatedOpticalAnalysis,
    SarWaterResult,
)
from ..services.sar_translation_service import DISCLOSURE, GROUNDING_DISCLOSURE, PIX2PIX_DISCLOSURE, RSVQA_DISCLOSURE


AGREEMENT_STATES = {
    "supported_by_both",
    "supported_by_native_sar_only",
    "suggested_by_translation_only",
    "conflicting_evidence",
    "insufficient_evidence",
    "translation_unavailable",
}
WATER_TERMS = ("water", "river", "lake", "reservoir", "flood")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _contains_water(value: str) -> bool:
    normalized = value.lower()
    return any(term in normalized for term in WATER_TERMS)


def _native_findings(
    water: Optional[SarWaterResult],
    scene: Optional[SarSceneResult],
) -> tuple[list[str], Optional[bool], float]:
    findings: list[str] = []
    native_water: Optional[bool] = None
    quality = 1.0
    if scene is not None:
        findings.append(
            "Direct SAR analysis indicates "
            f"{scene.valid_pixel_percent:.1f}% valid pixels, {scene.low_backscatter_percent:.1f}% low normalized "
            f"return, and a texture index of {scene.texture_index:.2f}."
        )
        quality = min(quality, scene.input_quality_score)
    if water is not None:
        native_water = bool(water.water_detected)
        if native_water:
            findings.append(
                "Direct SAR analysis indicates low-backscatter water candidates covering "
                f"approximately {water.image_area_percent:.2f}% of image pixels."
            )
        else:
            findings.append("Direct SAR analysis indicates no dependable low-backscatter water candidate was retained.")
        if water.input_quality_score is not None:
            quality = min(quality, water.input_quality_score)
    return findings, native_water, quality


def _translated_findings(optical: dict[str, Any]) -> tuple[list[str], Optional[bool]]:
    findings: list[str] = []
    water_signal: Optional[bool] = None
    caption = optical.get("caption")
    if isinstance(caption, str) and caption.strip():
        findings.append(f"The generated optical-like representation suggests: {caption.strip()}")
        if _contains_water(caption):
            water_signal = True
    vqa = optical.get("vqa")
    if isinstance(vqa, dict) and vqa.get("answer") is not None:
        question = str(vqa.get("question") or "")
        answer = str(vqa["answer"])
        findings.append(f"The generated optical-like representation suggests the VQA answer “{answer}”.")
        if _contains_water(question):
            normalized = answer.strip().lower()
            if normalized in {"yes", "true", "1", "present"}:
                water_signal = True
            elif normalized in {"no", "false", "0", "absent"}:
                water_signal = False
    grounding = optical.get("grounding")
    if isinstance(grounding, dict):
        target = str(grounding.get("target_phrase") or "target")
        count = int(grounding.get("accepted_detection_count") or 0)
        findings.append(
            "The generated optical-like representation suggests "
            f"{count} accepted grounding region(s) for “{target}”."
        )
        if _contains_water(target) and count > 0:
            water_signal = True
    priors = optical.get("scene_priors") or []
    if priors:
        labels = [str(item.get("label")) for item in priors[:3] if isinstance(item, dict) and item.get("label")]
        if labels:
            findings.append("The generated optical-like representation suggests scene-level priors: " + ", ".join(labels) + ".")
            if any(_contains_water(label) for label in labels):
                water_signal = True
    return _unique(findings), water_signal


def fuse_sar_optical_evidence(
    *,
    query: str,
    translation: Optional[dict[str, Any]],
    optical_evidence: Optional[dict[str, Any]],
    native_water: Optional[SarWaterResult],
    native_scene: Optional[SarSceneResult],
    translation_error: Optional[str] = None,
    generation_state: Optional[EvidenceLifecycleState] = None,
    semantic_comparison_state: EvidenceLifecycleState = EvidenceLifecycleState.NOT_REQUESTED,
    evidence_products: Optional[list[EvidenceItem]] = None,
    attempted_model: Optional[str] = None,
) -> SarTranslatedOpticalAnalysis:
    """Apply conservative explicit rules; no language model participates."""
    native_findings, native_water_signal, input_quality = _native_findings(native_water, native_scene)
    authoritative_generation_state = generation_state or (
        EvidenceLifecycleState.SUCCEEDED if translation is not None else EvidenceLifecycleState.FAILED
    )
    published_products = list(evidence_products or [])
    product_published = any(
        product.status == EvidenceLifecycleState.SUCCEEDED
        and product.evidence_type == "generated_optical_like"
        and product.generator
        and product.reference
        for product in published_products
    )
    semantic_evidence_valid = (
        authoritative_generation_state == EvidenceLifecycleState.SUCCEEDED
        and product_published
        and semantic_comparison_state == EvidenceLifecycleState.SUCCEEDED
    )
    optical = (optical_evidence or {}) if semantic_evidence_valid else {}
    translated_findings, translated_water_signal = _translated_findings(optical)
    water_question = _contains_water(query)

    if translation is None:
        state = "translation_unavailable"
    elif water_question and native_water_signal is not None and translated_water_signal is not None:
        state = "supported_by_both" if native_water_signal == translated_water_signal else "conflicting_evidence"
    elif native_findings and translated_findings:
        # Native intensity/texture findings do not independently validate arbitrary generated semantics.
        state = "suggested_by_translation_only"
    elif translated_findings:
        state = "suggested_by_translation_only"
    elif native_findings:
        state = "supported_by_native_sar_only"
    else:
        state = "insufficient_evidence"

    if state == "supported_by_both":
        direct = "Both native SAR evidence and the translated representation support the same water-related conclusion."
        level, score = ConfidenceLevel.MODERATE, 0.65
    elif state == "conflicting_evidence":
        generated_claim = translated_findings[0] if translated_findings else "a different conclusion"
        direct = (
            "The two analysis branches disagree. The translated representation suggests X, but this is not "
            "independently supported by the original SAR evidence. " + generated_claim.replace("The generated optical-like representation suggests", "Generated evidence")
        )
        level, score = ConfidenceLevel.LOW, 0.25
    elif state == "suggested_by_translation_only":
        direct = (
            translated_findings[0] + " This is generated-representation evidence only and is not independently "
            "supported by the original SAR measurement."
            if translated_findings else
            "The generated optical-like representation provides no dependable supporting observation."
        )
        level, score = ConfidenceLevel.LOW, 0.35
    elif state == "supported_by_native_sar_only":
        direct = native_findings[0] if native_findings else "Native SAR evidence is insufficient for a conclusion."
        level, score = ConfidenceLevel.LOW, 0.4
    elif state == "translation_unavailable":
        direct = (
            (native_findings[0] + " ") if native_findings else ""
        ) + "The generated optical-like representation is unavailable, so no translated evidence was used."
        level, score = ConfidenceLevel.LOW if native_findings else ConfidenceLevel.UNAVAILABLE, 0.3 if native_findings else None
    else:
        direct = "The available native SAR and translated evidence is insufficient for a supported conclusion."
        level, score = ConfidenceLevel.UNAVAILABLE, None

    if input_quality < 0.35 and score is not None:
        score = min(score, 0.25)
        level = ConfidenceLevel.LOW
    confidence = Confidence(
        level=level,
        score=score,
        reason="Deterministic evidence-agreement score; it is not a calibrated model probability.",
    )
    grounding = optical.get("grounding")
    vqa = optical.get("vqa")
    disclosure = PIX2PIX_DISCLOSURE if translation and translation.get("model_used") == "Pix2Pix" else DISCLOSURE
    warnings = list(translation.get("warnings") or []) if translation else []
    if translation_error:
        warnings.append(translation_error)
    limitations = [
        disclosure,
        "Native intensity and texture evidence does not establish arbitrary semantic object identity.",
        "Agreement is a deterministic evidence-consistency state, not ground-truth accuracy.",
    ]
    if input_quality < 0.35:
        warnings.append("Native SAR input quality is low; fused confidence was reduced.")
    return SarTranslatedOpticalAnalysis(
        status="translation_unavailable" if translation is None else "completed_with_limitations",
        generation_state=authoritative_generation_state,
        semantic_comparison_state=semantic_comparison_state,
        model=translation.get("model_used") if translation else attempted_model,
        device=translation.get("device") if translation else None,
        runtime_ms=int(translation.get("runtime_ms", 0)) if translation else 0,
        generated_width=translation.get("width") if translation else None,
        generated_height=translation.get("height") if translation else None,
        generated_preview_url=(translation.get("artifact_url") if translation and product_published else None),
        evidence_products=published_products if authoritative_generation_state == EvidenceLifecycleState.SUCCEEDED else [],
        normalized_sar_preview_url=translation.get("normalized_sar_preview_url") if translation else None,
        color_corrected_preview_url=translation.get("color_corrected_artifact_url") if translation else None,
        color_corrected=bool(translation and translation.get("color_correction_used")),
        fallback_used=bool(translation and translation.get("fallback_used")),
        fallback_reason=translation.get("fallback_reason") if translation else translation_error,
        preprocessing_method=translation.get("preprocessing_method") if translation else None,
        input_channel_interpretation=translation.get("input_channel_interpretation") if translation else None,
        output_value_range=list(translation.get("output_value_range") or []) if translation else [],
        content_hash=translation.get("content_hash") if translation else None,
        optical_caption=optical.get("caption"),
        optical_scene_priors=list(optical.get("scene_priors") or []),
        optical_grounding=grounding,
        optical_vqa=vqa,
        optical_specialists_executed=list(optical.get("specialists_executed") or []),
        native_sar_findings=native_findings,
        translated_findings=translated_findings,
        agreement=state,
        direct_answer=direct,
        confidence=confidence,
        disclosure=disclosure,
        grounding_disclosure=GROUNDING_DISCLOSURE if grounding else None,
        rsvqa_disclosure=RSVQA_DISCLOSURE if vqa else None,
        limitations=limitations,
        warnings=_unique(warnings),
        provenance=dict(translation.get("provenance") or {}) if translation else {},
        runtime_breakdown_ms=dict(translation.get("stage_durations_ms") or {}) if translation else {},
    )

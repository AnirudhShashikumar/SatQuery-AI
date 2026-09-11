"""Authoritative lifecycle checks for SAR-generated optical-like evidence."""

from __future__ import annotations

from typing import Any, Mapping

from .models import AgentResponse, Confidence, ConfidenceLevel, EvidenceLifecycleState


GENERATED_EVIDENCE_TYPES = {
    "generated_optical_like",
    "generated_optical_like_representation",
    "generated_representation_grounding",
}


def authoritative_translation_products_dict(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    analysis = response.get("sar_translated_optical_analysis")
    primary = response.get("primary_image_metadata")
    if not isinstance(analysis, Mapping) or not isinstance(primary, Mapping):
        return []
    if analysis.get("generation_state") != EvidenceLifecycleState.SUCCEEDED.value:
        return []
    products: list[dict[str, Any]] = []
    for product in analysis.get("evidence_products") or []:
        if not isinstance(product, dict):
            continue
        if (
            product.get("evidence_id")
            and product.get("status") == EvidenceLifecycleState.SUCCEEDED.value
            and product.get("evidence_type") == "generated_optical_like"
            and product.get("generator") == analysis.get("model")
            and product.get("evidence_run_id") == response.get("request_id")
            and product.get("source_observation_id") == primary.get("file_id")
            and str(product.get("source_modality") or "").startswith("sar")
            and product.get("reference")
        ):
            products.append(dict(product))
    return products


def normalize_translation_evidence(response: AgentResponse) -> AgentResponse:
    """Suppress mismatched/legacy products and prevent unsupported semantic claims."""
    analysis = response.sar_translated_optical_analysis
    if analysis is None:
        return response
    dumped = response.model_dump(mode="json")
    valid_products = authoritative_translation_products_dict(dumped)
    valid_ids = {item["evidence_id"] for item in valid_products if item.get("evidence_id")}
    had_generated_artifact = bool(
        analysis.generated_preview_url
        or analysis.evidence_products
        or any(item.evidence_type in GENERATED_EVIDENCE_TYPES or item.type in GENERATED_EVIDENCE_TYPES for item in response.evidence)
    )
    generated_evidence = [
        item for item in response.evidence
        if item.evidence_type not in GENERATED_EVIDENCE_TYPES and item.type not in GENERATED_EVIDENCE_TYPES
    ]
    generated_evidence.extend(
        item for item in analysis.evidence_products if item.evidence_id in valid_ids
    )

    generation_is_publishable = bool(valid_products)
    semantic_is_usable = generation_is_publishable and analysis.semantic_comparison_state == EvidenceLifecycleState.SUCCEEDED
    warning: str | None = None
    if not generation_is_publishable:
        warning = (
            "Legacy generated evidence provenance unavailable; the generated evidence product was suppressed."
            if had_generated_artifact and analysis.generation_state == EvidenceLifecycleState.NOT_REQUESTED
            else "Pix2Pix translation evidence was unavailable; native SAR evidence was retained."
        )
        direct_answer = " ".join(analysis.native_sar_findings[:1])
        direct_answer = (direct_answer + " " if direct_answer else "") + "The generated optical-like representation is unavailable, so no translated evidence was used."
        analysis = analysis.model_copy(update={
            "generated_preview_url": None,
            "normalized_sar_preview_url": None,
            "color_corrected_preview_url": None,
            "evidence_products": [],
            "optical_caption": None,
            "optical_scene_priors": [],
            "optical_grounding": None,
            "optical_vqa": None,
            "optical_specialists_executed": [],
            "translated_findings": [],
            "semantic_comparison_state": EvidenceLifecycleState.FAILED if analysis.semantic_comparison_state == EvidenceLifecycleState.FAILED else EvidenceLifecycleState.NOT_REQUESTED,
            "agreement": "translation_unavailable",
            "direct_answer": direct_answer,
        })
    elif not semantic_is_usable:
        direct_answer = " ".join(analysis.native_sar_findings[:1])
        direct_answer = (direct_answer + " " if direct_answer else "") + "Translation semantic comparison was unavailable, so no translation-derived claim was used."
        analysis = analysis.model_copy(update={
            "generated_preview_url": valid_products[0]["reference"],
            "evidence_products": [item for item in analysis.evidence_products if item.evidence_id in valid_ids],
            "optical_caption": None,
            "optical_scene_priors": [],
            "optical_grounding": None,
            "optical_vqa": None,
            "optical_specialists_executed": [],
            "translated_findings": [],
            "agreement": "supported_by_native_sar_only" if analysis.native_sar_findings else "insufficient_evidence",
            "direct_answer": direct_answer,
        })
    else:
        analysis = analysis.model_copy(update={
            "generated_preview_url": valid_products[0]["reference"],
            "evidence_products": [item for item in analysis.evidence_products if item.evidence_id in valid_ids],
        })

    warnings = list(dict.fromkeys(response.warnings + ([warning] if warning else [])))
    confidence = analysis.confidence
    if not semantic_is_usable:
        confidence = Confidence(
            level=ConfidenceLevel.LOW if analysis.native_sar_findings else ConfidenceLevel.UNAVAILABLE,
            score=min(confidence.score, 0.4) if confidence.score is not None and analysis.native_sar_findings else None,
            reason="Only native SAR evidence contributed to this answer; translation-derived semantics were not used.",
        )
        analysis = analysis.model_copy(update={"confidence": confidence})
    return response.model_copy(update={
        "answer": analysis.direct_answer if not semantic_is_usable else response.answer,
        "confidence": confidence if not semantic_is_usable else response.confidence,
        "evidence": generated_evidence,
        "warnings": warnings,
        "sar_translated_optical_analysis": analysis,
    })

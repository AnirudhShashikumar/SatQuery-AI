"""Transparent, rule-based query routing for the SatQuery foundation."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from .models import (
    AgentQueryRequest,
    ClassifiedQuery,
    ImageModality,
    InputMode,
    Modality,
    QuestionCategory,
    RequestedOutput,
    RoutingPlan,
    TaskType,
    ValidationStatus,
)
from .specialists.grounder import GrounderError, normalize_grounding_target
from .specialists.rsvqa_specialist import rsvqa_specialist_configured
from .specialists.vqa import get_vqa


TASK_RULES: Sequence[Tuple[TaskType, Sequence[str], str, str]] = (
    (TaskType.REPORT_GENERATION, ("generate report", "download report", "export analysis"), "report_generator", "The query explicitly requests an analysis report or export."),
    (TaskType.CHANGE_VQA, ("has built-up increased", "has vegetation decreased", "did water coverage change", "increased, decreased, or unchanged", "how much of the image changed", "how much of image changed", "where did the largest change", "where is the largest changed", "how many major changed regions", "mostly unchanged", "amount of change", "small, moderate, or large", "cause of change", "caused the change", "change caused"), "bitemporal_change_analyzer", "The query asks a controlled question about measured change between dates."),
    (TaskType.CHANGE_DESCRIPTION, ("what changed", "compare these dates", "describe the change"), "bitemporal_change_analyzer", "The query requests a description of bi-temporal change."),
    (TaskType.CROSS_MODAL_ANALYSIS, ("use both", "optical and sar", "combine the images", "jointly analyse", "jointly analyze", "using both modalities", "analyse the optical and sar", "analyze the optical and sar", "observations agree", "both modalities agree", "supported by both modalities", "structural information between", "percentage appears water-like", "percentage appears water like", "structurally complex regions", "evidence disagree", "joint evidence regions"), "cross_modal_optical_sar_analyzer", "The query explicitly requests joint optical-SAR analysis."),
    (TaskType.GROUNDING, ("highlight", "mark", "locate", "show me where", "show where", "show me", "show the", "show roads", "find", "identify the region containing", "where is", "where are"), "rs_grounder", "The query requests spatial localization of a supported concrete target."),
    (TaskType.CAPTIONING, ("describe", "summarize", "explain the scene", "what is visible"), "rs_captioner", "The query requests a scene-level description."),
    (TaskType.VQA, ("is there", "is a water", "water body visible", "water visible", "what is", "which", "how many", "where is", "who owns", "does this image show", "does this scene contain", "does this contain", "can you see", "are buildings", "is vegetation", "is this mainly", "what kind of scene", "how much of the image appears", "mostly built-up", "mostly built up"), "rs_vqa", "The query asks a controlled visual question about one image."),
)

PERMITTED_PARAMETER_NAMES = (
    "query",
    "input_mode",
    "primary_modality",
    "secondary_modality",
    "has_primary_image",
    "has_secondary_image",
    "primary_image_modality",
    "primary_representation",
    "primary_band_count",
)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


SAR_MODALITIES = {
    ImageModality.SAR_PREVIEW,
    ImageModality.SAR_VV,
    ImageModality.SAR_VH,
    ImageModality.SAR_VV_VH,
}
WATER_TERMS = ("water", "lake", "river", "reservoir", "flood")
LOCALIZATION_TERMS = ("highlight", "mark", "locate", "show", "where", "mask", "segment")


def classify_query(query: str, image_modality: ImageModality = ImageModality.AUTO) -> ClassifiedQuery:
    normalized = normalize_query(query)
    water_target = any(term in normalized for term in WATER_TERMS)
    localization = any(term in normalized for term in LOCALIZATION_TERMS)
    if image_modality in SAR_MODALITIES and water_target and localization:
        return ClassifiedQuery(
            task_type=TaskType.SAR_WATER_SEGMENTATION,
            target="water",
            requested_output=RequestedOutput.OVERLAY,
            requires_localization=True,
            requires_segmentation=True,
            requires_measurement=True,
        )
    if image_modality in SAR_MODALITIES and any(term in normalized for term in ("quality", "suitable", "usable", "noise", "contrast")):
        return ClassifiedQuery(task_type=TaskType.SAR_QUALITY_INSPECTION, requested_output=RequestedOutput.QUALITY_REPORT)
    if image_modality in SAR_MODALITIES and any(term in normalized for term in ("describe", "summarize", "backscatter", "scene")):
        return ClassifiedQuery(task_type=TaskType.SAR_SCENE_ANALYSIS, requested_output=RequestedOutput.TEXT)
    task, _, _ = detect_task(query)
    target = "water" if water_target else None
    return ClassifiedQuery(
        task_type=task,
        target=target,
        requested_output=RequestedOutput.LOCALIZATION if localization else RequestedOutput.TEXT,
        requires_localization=localization,
        requires_segmentation=False,
        requires_measurement=False,
    )


def detect_task(query: str) -> Tuple[TaskType, List[str], str]:
    normalized = normalize_query(query)
    # Pair-specific and localization requests remain more specific than the
    # single-image VQA vocabulary.  In particular, "how many changed regions"
    # is a bi-temporal change question, not an object-counting question.
    for task, phrases, tool, reason in TASK_RULES:
        if task in {TaskType.CAPTIONING, TaskType.VQA}:
            continue
        if any(phrase in normalized for phrase in phrases):
            if task == TaskType.GROUNDING:
                try:
                    normalize_grounding_target(query)
                except GrounderError:
                    continue
            return task, ["input_validator", tool], reason
    # Controlled RSVQA-style questions must be claimed before broad captioning
    # rules such as "describe" or "what is visible" get a chance to consume
    # them.  The VQA specialist performs the finer family classification.
    vqa_intent = get_vqa().classify_question(query)
    if vqa_intent.category != QuestionCategory.UNSUPPORTED:
        specialist_available = rsvqa_specialist_configured() and vqa_intent.category in {
            QuestionCategory.PRESENCE_VQA,
            QuestionCategory.COMPARISON_VQA,
            QuestionCategory.RURAL_URBAN_CLASSIFICATION,
            QuestionCategory.COUNT_VQA,
        }
        return (
            TaskType.VQA,
            ["input_validator", "rsvqa_vqa_specialist" if specialist_available else "rs_vqa"],
            (
                f"The query matches the exported {vqa_intent.category.value} RSVQA family; "
                "RSVQA Specialist v1 is the primary prediction source."
                if specialist_available else
                f"The query matches the controlled {vqa_intent.category.value} VQA family; "
                "the deterministic implementation is the configured fallback."
            ),
        )
    for task, phrases, tool, reason in TASK_RULES:
        if task not in {TaskType.CAPTIONING, TaskType.VQA}:
            continue
        if any(phrase in normalized for phrase in phrases):
            if task == TaskType.GROUNDING:
                try:
                    normalize_grounding_target(query)
                except GrounderError:
                    continue
            return task, ["input_validator", tool], reason
    return (
        TaskType.UNSUPPORTED,
        ["input_validator"],
        "No deterministic routing phrase matched the query.",
    )


def validate_compatibility(request: AgentQueryRequest, task: TaskType) -> ValidationStatus:
    errors: List[str] = []
    if not request.has_primary_image:
        errors.append("A primary image is required.")

    if request.input_mode in (InputMode.CROSS_MODAL, InputMode.BI_TEMPORAL):
        if not request.has_secondary_image:
            errors.append("A secondary image is required for the selected input mode.")
        if request.secondary_modality is None:
            errors.append("A secondary modality is required for the selected input mode.")

    if request.input_mode == InputMode.CROSS_MODAL and request.secondary_modality is not None:
        pair = {request.primary_modality, request.secondary_modality}
        optical_family = bool(pair.intersection({Modality.OPTICAL, Modality.MULTISPECTRAL}))
        if not (optical_family and Modality.SAR in pair):
            errors.append("Cross-modal analysis requires one optical or multispectral image and one SAR image.")

    if task in (TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA) and request.input_mode != InputMode.BI_TEMPORAL:
        errors.append("Change analysis requires the bi_temporal input mode with two co-registered dates.")
    if task == TaskType.CROSS_MODAL_ANALYSIS and request.input_mode != InputMode.CROSS_MODAL:
        errors.append("Cross-modal analysis requires the cross_modal input mode.")
    if task in (TaskType.CAPTIONING, TaskType.VQA, TaskType.GROUNDING) and request.input_mode != InputMode.SINGLE:
        errors.append("This initial workflow requires the single input mode.")

    return ValidationStatus(valid=not errors, errors=errors)


def route_query(request: AgentQueryRequest) -> RoutingPlan:
    classified = classify_query(request.query, request.primary_image_modality)
    task = classified.task_type
    if task == TaskType.SAR_WATER_SEGMENTATION:
        selected_tools = ["input_validator", "sar_water_segmenter"]
        reason = "The query requests a water mask/overlay and the effective modality is SAR, so the RGB-only grounder is incompatible and the SAR water specialist is selected."
    elif task in {TaskType.SAR_SCENE_ANALYSIS, TaskType.SAR_QUALITY_INSPECTION}:
        selected_tools = ["input_validator", "sar_scene_analyzer"]
        reason = "The effective modality is SAR and the query requests qualitative SAR intensity/quality analysis."
    else:
        task, selected_tools, reason = detect_task(request.query)
    validation = validate_compatibility(request, task)
    permitted: Dict[str, object] = {
        name: getattr(request, name) for name in PERMITTED_PARAMETER_NAMES
    }
    return RoutingPlan(
        detected_task=task,
        selected_tools=selected_tools,
        permitted_parameters=permitted,
        validation_status=validation,
        selection_reason=reason,
    )

"""FastAPI endpoints for deterministic SatQuery planning only."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .compatibility import validate_pair_compatibility
from .image_ingestion import (
    IngestedImage,
    ImageIngestionError,
    apply_modality_override,
    coarse_modality,
    ingest_upload,
    preview_file_path,
    remove_preview,
)
from .models import (
    AgentHealth,
    AgentQueryRequest,
    AgentResponse,
    AnalyticsResponse,
    AlignmentLevel,
    CaptionDetails,
    ComplianceResponse,
    ChangeAnalysisResponse,
    ChangeAnalysisStatus,
    ChangeEngine,
    ChangePreviewUrls,
    ComparisonAssessmentRequest,
    ComparisonAssessmentResponse,
    ComparisonItem,
    ComparisonItemSummary,
    ComparisonReportRequest,
    ComparisonReportResponse,
    ControlledVQAMethod,
    ControlledVQAResult,
    Confidence,
    ConfidenceLevel,
    CrossModalAnalysisResponse,
    CrossModalResult,
    CrossModalStatus,
    DemoManifest,
    EvidenceItem,
    EvidenceConsistency,
    ExecutionStep,
    ExecutionSummary,
    ImageMetadata,
    ImageInspectionResponse,
    ImageModality,
    ModelProvenance,
    GroundingResult,
    ImplementationStatus,
    InputMode,
    Modality,
    PairCompatibility,
    QuestionCategory,
    ReportRequest,
    ReportResponse,
    ResponseStatus,
    RepresentationType,
    SarWaterResult,
    SarSceneResult,
    SVEResult,
    SVESemanticComparison,
    SpecialistHealth,
    TaskType,
    ToolDefinition,
    ToolStatus,
    TTPResult,
    ValidationStatus,
)
from .analytics import (
    build_analytics,
    record_agent_response,
    record_change_response,
    record_cross_modal_response,
    record_report_generation,
    utc_now as analytics_utc_now,
)
from .registry import public_tool_registry, tool_definition
from .compliance import compliance_summary
from .demo import demo_manifest, demo_sample_path
from .diagnostics import diagnostic
from .reporting import (
    MISSION_STORE,
    artifact_file_path,
    cache_key_for,
    cached_response,
    generate_report,
    mark_fresh,
    result_is_reportable,
)
from .comparison import (
    COMPARISON_STORE,
    assess_many,
    generate_comparison_report,
    record_agent_result as record_comparison_agent_result,
    record_change_result as record_comparison_change_result,
    record_cross_modal_result as record_comparison_cross_modal_result,
)
from .router import classify_query, route_query
from .specialists.captioner import CaptionerError, MAX_NEW_TOKENS, NUM_BEAMS, SUPPORTED_MODALITIES, get_captioner
from .specialists.change_analysis import (
    ChangeAnalysisError,
    analyze_change,
    compare_binary_masks,
    requires_alignment,
    save_hybrid_previews,
    statistics_from_binary_mask,
)
from .specialists.cross_modal import CrossModalAnalysisError, get_cross_modal_analyzer
from .specialists.grounder import GrounderError, get_grounder, safe_grounding_parameters
from .specialists.sar_water import SarWaterAnalysisError, analyze_sar_water
from .specialists.sar_scene import SarSceneAnalysisError, analyze_sar_scene
from .specialists.rsvqa_specialist import (
    CHECKPOINT_FILENAME as RSVQA_CHECKPOINT_FILENAME,
    MODEL_NAME as RSVQA_MODEL_NAME,
    RSVQASpecialistError,
    classify_rsvqa_task,
    get_rsvqa_specialist,
)
from .specialists.single_image_evidence import (
    METHOD_ASSUMPTIONS as SINGLE_EVIDENCE_ASSUMPTIONS,
    METHOD_LIMITATIONS as SINGLE_EVIDENCE_LIMITATIONS,
    SingleImageEvidenceError,
    extract_single_image_evidence,
)
from .specialists.ttp_change import (
    ARCHITECTURE as TTP_ARCHITECTURE,
    CHECKPOINT_FINGERPRINT as TTP_CHECKPOINT_FINGERPRINT,
    CHECKPOINT_NAME as TTP_CHECKPOINT_NAME,
    DOMAIN_WARNING as TTP_DOMAIN_WARNING,
    MODEL_ID as TTP_MODEL_ID,
    TRAINING_DATASET as TTP_TRAINING_DATASET,
    TTP_CLIENT,
    TTPClientError,
    default_mode as ttp_default_mode,
    ttp_enabled,
)
from .specialists.vqa import (
    answer_change_question,
    answer_cross_modal_question,
    classify_cross_modal_question,
    get_vqa,
)
from .services.sve_service import SVECall, get_sve_service


router = APIRouter(prefix="/api/agent", tags=["satquery-agent"])


def _sve_execution_steps(call: SVECall) -> List[ExecutionStep]:
    status_lookup = {
        "success": ToolStatus.SUCCESS,
        "failed": ToolStatus.FAILED,
        "skipped": ToolStatus.SKIPPED,
    }
    return [
        ExecutionStep(
            tool=str(item.get("tool", "sve_fallback")),
            status=status_lookup.get(str(item.get("status")), ToolStatus.FAILED),
            duration_ms=max(0, int(item.get("duration_ms", 0))),
            parameters=dict(item.get("parameters") or {}),
        )
        for item in call.trace
    ]


def _sve_optical_eligible(metadata: ImageMetadata, modality: Modality) -> bool:
    return (
        modality in {Modality.OPTICAL, Modality.MULTISPECTRAL}
        and metadata.band_count in {3, 4}
        and metadata.effective_modality in {ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL}
    )


@router.get("/health", response_model=AgentHealth)
def agent_health() -> AgentHealth:
    ttp_status = SpecialistHealth(status="disabled")
    if ttp_enabled():
        try:
            health = TTP_CLIENT.health()
            ttp_status = SpecialistHealth(status=str(health.get("lifecycle", "unavailable")), device="cuda" if health.get("status") == "ready" else None)
        except TTPClientError:
            ttp_status = SpecialistHealth(status="unavailable")
    return AgentHealth(
        status="ok",
        module="satquery-agent",
        router="ready",
        registry="ready",
        specialists={"rs_captioner": get_captioner().health(), "rs_grounder": get_grounder().health(), "satquery_vision_encoder_v1": get_sve_service().health(), "rsvqa_vqa_specialist": get_rsvqa_specialist().health(), "ttp_change_detector": ttp_status},
    )


@router.get("/tools", response_model=List[ToolDefinition])
def agent_tools() -> List[ToolDefinition]:
    return public_tool_registry()


@router.post("/inspect", response_model=ImageInspectionResponse)
async def inspect_agent_image(image: UploadFile = File(...)) -> ImageInspectionResponse:
    """Inspect an upload without executing a specialist or inferring semantic content."""
    try:
        ingested = await ingest_upload(image)
    except ImageIngestionError as error:
        _raise_ingestion_error(error)
    ingested.model_image.close()
    return ImageInspectionResponse(
        metadata=ingested.metadata,
        content_hash_prefix=ingested.content_hash[:12],
        requires_modality_confirmation=(
            ingested.metadata.auto_detected_modality == ImageModality.UNKNOWN
            and ingested.metadata.auto_detection_confidence.value == "low"
        ),
    )


@router.get("/compliance", response_model=ComplianceResponse)
def agent_compliance() -> ComplianceResponse:
    return compliance_summary()


@router.get("/analytics", response_model=AnalyticsResponse)
def agent_analytics() -> AnalyticsResponse:
    """Return bounded, metadata-only runtime and research-readiness analytics."""
    return build_analytics()


@router.get("/demo", response_model=DemoManifest)
def agent_demo_manifest() -> DemoManifest:
    return demo_manifest()


@router.get("/demo/files/{sample_name}", response_class=FileResponse)
def agent_demo_file(sample_name: str) -> FileResponse:
    path = demo_sample_path(sample_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "DEMO_SAMPLE_UNAVAILABLE", "message": "Local demo mode is disabled or the approved sample is unavailable."},
        )
    media_type = "image/tiff" if path.suffix.lower() in {".tif", ".tiff"} else "image/png"
    return FileResponse(path=str(path), media_type=media_type, filename=path.name, headers={"Cache-Control": "private, no-store"})


@router.get("/previews/{preview_name}", response_class=FileResponse)
def agent_preview(preview_name: str) -> FileResponse:
    path = preview_file_path(preview_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PREVIEW_NOT_FOUND", "message": "The requested preview is unavailable."},
        )
    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=preview_name,
        headers={"Cache-Control": "private, no-store"},
        content_disposition_type="inline",
    )


@router.post("/report", response_model=ReportResponse)
def agent_report(request: ReportRequest) -> ReportResponse:
    try:
        response = generate_report(request.request_id, request.formats)
        COMPARISON_STORE.mark_report(request.request_id)
        record_report_generation(request.request_id, [artifact.format for artifact in response.artifacts])
        return response
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "REPORT_SOURCE_EXPIRED", "message": str(error.args[0])},
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "REPORT_NOT_AVAILABLE", "message": str(error)},
        ) from error


@router.get("/comparison-items", response_model=List[ComparisonItemSummary])
def comparison_items() -> List[ComparisonItemSummary]:
    """Return bounded, preview-safe summaries of recent authoritative results."""
    return COMPARISON_STORE.list()


@router.get("/comparison-items/{request_id}", response_model=ComparisonItem)
def comparison_item(request_id: str) -> ComparisonItem:
    state, record = COMPARISON_STORE.get(request_id)
    if record is None:
        status_code = 410 if state == "expired" else 404
        code = "COMPARISON_RESULT_EXPIRED" if state == "expired" else "COMPARISON_RESULT_NOT_FOUND"
        message = "The comparison result expired from bounded process-local history." if state == "expired" else "The comparison result ID is unknown."
        raise HTTPException(status_code=status_code, detail={"code": code, "message": message})
    return record.item


@router.post("/comparison-assessment", response_model=ComparisonAssessmentResponse)
def comparison_assessment(request: ComparisonAssessmentRequest) -> ComparisonAssessmentResponse:
    try:
        if len(set(request.request_ids)) != len(request.request_ids):
            raise ValueError("Select distinct result IDs.")
        return assess_many(request.request_ids)
    except LookupError as error:
        state, request_id = str(error).split(":", 1)
        raise HTTPException(
            status_code=410 if state == "expired" else 404,
            detail={"code": "COMPARISON_RESULT_EXPIRED" if state == "expired" else "COMPARISON_RESULT_NOT_FOUND", "message": f"Result {request_id} is unavailable or expired."},
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_COMPARISON_SELECTION", "message": str(error)}) from error


@router.post("/comparison-report", response_model=ComparisonReportResponse)
def comparison_report(request: ComparisonReportRequest) -> ComparisonReportResponse:
    try:
        return generate_comparison_report(request.request_ids, request.formats, request.user_note)
    except LookupError as error:
        state, request_id = str(error).split(":", 1)
        raise HTTPException(
            status_code=410 if state == "expired" else 404,
            detail={"code": "COMPARISON_RESULT_EXPIRED" if state == "expired" else "COMPARISON_RESULT_NOT_FOUND", "message": f"Result {request_id} is unavailable or expired."},
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_COMPARISON_REPORT", "message": str(error)}) from error


@router.get("/reports/{artifact_name}", response_class=FileResponse)
def agent_report_artifact(artifact_name: str) -> FileResponse:
    path = artifact_file_path(artifact_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "REPORT_ARTIFACT_EXPIRED", "message": "The report artifact is unavailable or expired."},
        )
    media_types = {".pdf": "application/pdf", ".json": "application/json", ".csv": "text/csv", ".zip": "application/zip"}
    return FileResponse(
        path=str(path),
        media_type=media_types[path.suffix.lower()],
        filename=path.name,
        headers={"Cache-Control": "private, no-store"},
        content_disposition_type="attachment",
    )


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


@router.post("/route", response_model=AgentResponse)
def agent_route(request: AgentQueryRequest) -> AgentResponse:
    """Preserve the JSON-only routing contract separately from file ingestion."""
    started = time.perf_counter()
    validation_started = time.perf_counter()
    plan = route_query(request)
    validation_duration = max(0, round((time.perf_counter() - validation_started) * 1000))
    steps = [
        ExecutionStep(
            tool="input_validator",
            status=ToolStatus.SUCCESS if plan.validation_status.valid else ToolStatus.FAILED,
            duration_ms=validation_duration,
        )
    ]
    warnings = list(plan.validation_status.errors)

    if not plan.validation_status.valid:
        for tool_id in plan.selected_tools[1:]:
            steps.append(ExecutionStep(tool=tool_id, status=ToolStatus.SKIPPED, duration_ms=0))
        status = ResponseStatus.FAILED
        confidence_reason = "Input validation failed; no specialist tool was executed."
    elif plan.detected_task == TaskType.UNSUPPORTED:
        status = ResponseStatus.NOT_IMPLEMENTED
        warnings.append("The query did not match a supported deterministic routing rule.")
        confidence_reason = "No supported task could be selected from the query."
    else:
        selected_tool = tool_definition(plan.selected_tools[-1])
        if selected_tool.status == ImplementationStatus.NOT_IMPLEMENTED:
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.NOT_IMPLEMENTED, duration_ms=0))
            status = ResponseStatus.NOT_IMPLEMENTED
            warnings.append(selected_tool.notes)
            confidence_reason = selected_tool.notes
        else:
            # The JSON route endpoint classifies only; pixel specialists run via multipart /query.
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.SKIPPED, duration_ms=0))
            status = ResponseStatus.NOT_IMPLEMENTED
            warnings.append("The routing-only endpoint does not execute imagery specialists; use multipart /api/agent/query with an uploaded image.")
            confidence_reason = "Routing succeeded; no image pixels are available on the routing-only endpoint."

    duration_ms = max(0, round((time.perf_counter() - started) * 1000))
    return AgentResponse(
        request_id=str(uuid.uuid4()),
        task=plan.detected_task,
        answer=None,
        confidence=Confidence(
            level=ConfidenceLevel.UNAVAILABLE,
            score=None,
            reason=confidence_reason,
        ),
        evidence=[],
        execution=ExecutionSummary(
            input_mode=request.input_mode,
            selected_tools=plan.selected_tools,
            steps=steps,
            duration_ms=duration_ms,
            permitted_parameters=plan.permitted_parameters,
            validation=plan.validation_status,
            selection_reason=plan.selection_reason,
        ),
        warnings=warnings,
        status=status,
    )


def _ingestion_steps(durations: List[Dict[str, int]]) -> List[ExecutionStep]:
    names = ("upload_received", "file_type_validation", "metadata_extraction", "preview_generation")
    return [
        ExecutionStep(
            tool=name,
            status=ToolStatus.SUCCESS,
            duration_ms=sum(item.get(name, 0) for item in durations),
        )
        for name in names
    ]


def _raise_ingestion_error(error: ImageIngestionError) -> None:
    raise HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    ) from error


def _order_cross_modal_inputs(
    primary_result: IngestedImage,
    primary_modality: Modality,
    secondary_result: IngestedImage,
    secondary_modality: Modality,
) -> tuple[IngestedImage, IngestedImage]:
    optical_family = {Modality.OPTICAL, Modality.MULTISPECTRAL}
    if primary_modality in optical_family and secondary_modality == Modality.SAR:
        return primary_result, secondary_result
    if secondary_modality in optical_family and primary_modality == Modality.SAR:
        return secondary_result, primary_result
    raise CrossModalAnalysisError(
        "INVALID_CROSS_MODALITIES",
        "Cross-modal analysis requires exactly one optical or multispectral input and one SAR input.",
    )


async def _execute_cross_modal(
    optical_result: IngestedImage,
    sar_result: IngestedImage,
    compatibility: PairCompatibility,
) -> CrossModalResult:
    return await run_in_threadpool(
        get_cross_modal_analyzer().analyze,
        optical_result.analysis_raster,
        sar_result.analysis_raster,
        optical_result.metadata,
        sar_result.metadata,
        compatibility,
    )


def _cross_modal_steps(
    optical_result: IngestedImage,
    sar_result: IngestedImage,
    compatibility: PairCompatibility,
    result: CrossModalResult,
    modality_validation_ms: int,
    compatibility_ms: int,
    modality_valid: bool = True,
) -> List[ExecutionStep]:
    full_analysis = result.statistics is not None
    analysis_status = ToolStatus.SUCCESS if full_analysis else ToolStatus.SKIPPED
    pair_status = ToolStatus.FAILED if result.status == CrossModalStatus.FAILED else ToolStatus.SUCCESS
    durations = result.stage_durations_ms
    return [
        ExecutionStep(
            tool="optical_upload_received",
            status=ToolStatus.SUCCESS,
            duration_ms=optical_result.durations_ms.get("upload_received", 0),
            parameters={"size_bytes": optical_result.metadata.size_bytes},
        ),
        ExecutionStep(
            tool="sar_upload_received",
            status=ToolStatus.SUCCESS,
            duration_ms=sar_result.durations_ms.get("upload_received", 0),
            parameters={"size_bytes": sar_result.metadata.size_bytes},
        ),
        ExecutionStep(
            tool="optical_metadata_extraction",
            status=ToolStatus.SUCCESS,
            duration_ms=optical_result.durations_ms.get("file_type_validation", 0)
            + optical_result.durations_ms.get("metadata_extraction", 0),
            parameters={
                "width": optical_result.metadata.width,
                "height": optical_result.metadata.height,
                "bands": optical_result.metadata.band_count,
                "georeferenced": optical_result.metadata.is_georeferenced,
            },
        ),
        ExecutionStep(
            tool="sar_metadata_extraction",
            status=ToolStatus.SUCCESS,
            duration_ms=sar_result.durations_ms.get("file_type_validation", 0)
            + sar_result.durations_ms.get("metadata_extraction", 0),
            parameters={
                "width": sar_result.metadata.width,
                "height": sar_result.metadata.height,
                "bands": sar_result.metadata.band_count,
                "georeferenced": sar_result.metadata.is_georeferenced,
            },
        ),
        ExecutionStep(
            tool="modality_validation",
            status=ToolStatus.SUCCESS if modality_valid else ToolStatus.FAILED,
            duration_ms=modality_validation_ms,
            parameters={"optical_modality": "optical_or_multispectral", "sar_modality": "sar"},
        ),
        ExecutionStep(
            tool="pair_compatibility_check",
            status=pair_status,
            duration_ms=compatibility_ms,
            parameters={
                "alignment_level": compatibility.alignment_level.value,
                "resampling_performed": False,
                "reprojection_performed": False,
            },
        ),
        ExecutionStep(
            tool="optical_preparation",
            status=analysis_status,
            duration_ms=durations.get("optical_preparation", 0),
            parameters={"max_analysis_dimension": 1024, "registration_performed": False},
        ),
        ExecutionStep(
            tool="sar_preparation",
            status=analysis_status,
            duration_ms=durations.get("sar_preparation", 0),
            parameters={"log_transform": "none", "calibrated_backscatter_claimed": False},
        ),
        ExecutionStep(
            tool="optical_evidence_extraction",
            status=analysis_status,
            duration_ms=durations.get("optical_evidence_extraction", 0),
            parameters={"method": "visible_color_brightness_edges_local_texture"},
        ),
        ExecutionStep(
            tool="sar_evidence_extraction",
            status=analysis_status,
            duration_ms=durations.get("sar_evidence_extraction", 0),
            parameters={"method": "relative_intensity_and_local_heterogeneity"},
        ),
        ExecutionStep(
            tool="joint_evidence_fusion",
            status=analysis_status,
            duration_ms=durations.get("joint_evidence_fusion", 0),
            parameters={"method": "boolean_intersection_and_modality_disagreement"},
        ),
        ExecutionStep(
            tool="region_extraction",
            status=analysis_status,
            duration_ms=durations.get("region_extraction", 0),
            parameters={"connectivity": 8, "minimum_region_pixels": 6},
        ),
        ExecutionStep(
            tool="preview_generation",
            status=analysis_status,
            duration_ms=durations.get("preview_generation", 0),
            parameters={"format": "png", "filesystem_paths_exposed": False},
        ),
        ExecutionStep(
            tool="response_generation",
            status=ToolStatus.SUCCESS,
            duration_ms=0,
            parameters={"template_generated_summary": True, "language_model_used": False},
        ),
    ]


def _response_status(result: CrossModalResult) -> ResponseStatus:
    return {
        CrossModalStatus.SUCCESS: ResponseStatus.SUCCESS,
        CrossModalStatus.PARTIAL: ResponseStatus.PARTIAL,
        CrossModalStatus.ALIGNMENT_REQUIRED: ResponseStatus.ALIGNMENT_REQUIRED,
        CrossModalStatus.FAILED: ResponseStatus.FAILED,
    }[result.status]


@router.post("/cross-modal", response_model=CrossModalAnalysisResponse)
async def agent_cross_modal_analysis(
    optical_image: UploadFile = File(...),
    sar_image: UploadFile = File(...),
    query: Optional[str] = Form(None),
    optical_modality: Modality = Form(Modality.OPTICAL),
    sar_modality: Modality = Form(Modality.SAR),
) -> CrossModalAnalysisResponse:
    """Run local deterministic evidence fusion for an optical-SAR pair."""
    started = time.perf_counter()
    analytics_started_at = analytics_utc_now()
    optical_result: Optional[IngestedImage] = None
    sar_result: Optional[IngestedImage] = None
    optical_metadata: Optional[ImageMetadata] = None
    sar_metadata: Optional[ImageMetadata] = None
    try:
        try:
            optical_result = await ingest_upload(optical_image)
            optical_metadata = optical_result.metadata
            sar_result = await ingest_upload(sar_image)
            sar_metadata = sar_result.metadata
        except ImageIngestionError as error:
            remove_preview(optical_metadata)
            remove_preview(sar_metadata)
            _raise_ingestion_error(error)
        assert optical_result is not None and sar_result is not None
        assert optical_metadata is not None and sar_metadata is not None

        modality_started = time.perf_counter()
        modality_valid = optical_modality in {Modality.OPTICAL, Modality.MULTISPECTRAL} and sar_modality == Modality.SAR
        modality_duration = max(0, round((time.perf_counter() - modality_started) * 1000))
        compatibility_started = time.perf_counter()
        compatibility = validate_pair_compatibility(
            input_mode=InputMode.CROSS_MODAL,
            primary_modality=optical_modality,
            secondary_modality=sar_modality,
            primary=optical_metadata,
            secondary=sar_metadata,
        )
        if not modality_valid:
            modality_error = "The optical_image field requires optical or multispectral modality and sar_image requires SAR modality."
            compatibility = compatibility.model_copy(
                update={
                    "compatible": False,
                    "alignment_level": AlignmentLevel.INCOMPATIBLE,
                    "errors": _unique(compatibility.errors + [modality_error]),
                }
            )
        compatibility_duration = max(0, round((time.perf_counter() - compatibility_started) * 1000))
        result = await _execute_cross_modal(optical_result, sar_result, compatibility)
        steps = _cross_modal_steps(
            optical_result,
            sar_result,
            compatibility,
            result,
            modality_duration,
            compatibility_duration,
            modality_valid=modality_valid,
        )
        if get_sve_service().enabled and _sve_optical_eligible(optical_metadata, optical_modality):
            sve_call = await run_in_threadpool(
                get_sve_service().analyze,
                optical_result.model_image,
                optical_result.content_hash,
            )
            raw_sar_skip = get_sve_service().unsupported_raw_sar_result()
            sve_result = sve_call.result.model_copy(update={
                "semantic_comparison": raw_sar_skip.semantic_comparison,
                "warning": raw_sar_skip.warning if sve_call.result.available else sve_call.result.warning,
            })
            result = result.model_copy(update={"sve_result": sve_result})
            steps.extend(_sve_execution_steps(sve_call))
            steps.append(ExecutionStep(
                tool="sve_eligibility_check",
                status=ToolStatus.SKIPPED,
                duration_ms=0,
                parameters={"input": "raw_sar", "reason": "validated_generated_rgb_required"},
            ))
        runtime_ms = max(0, round((time.perf_counter() - started) * 1000))
        result = result.model_copy(update={"runtime_ms": runtime_ms})
        response = CrossModalAnalysisResponse(
            request_id=str(uuid.uuid4()),
            optical_metadata=optical_metadata,
            sar_metadata=sar_metadata,
            compatibility=compatibility,
            result=result,
            execution=ExecutionSummary(
                input_mode=InputMode.CROSS_MODAL,
                selected_tools=["input_validator", "cross_modal_optical_sar_analyzer"],
                steps=steps,
                duration_ms=runtime_ms,
                permitted_parameters={
                    "query": query.strip() if query else None,
                    "optical_modality": optical_modality.value,
                    "sar_modality": sar_modality.value,
                    "registration": False,
                    "reprojection": False,
                    "resampling": False,
                    "language_model": False,
                },
                validation=ValidationStatus(
                    valid=result.status in {CrossModalStatus.SUCCESS, CrossModalStatus.PARTIAL},
                    errors=compatibility.errors,
                ),
                selection_reason="Multipart optical-SAR inputs are handled by the deterministic cross-modal evidence-fusion specialist.",
            ),
        )
        record_cross_modal_response(
            response,
            started_at=analytics_started_at,
            optical_modality=optical_modality.value,
            sar_modality=sar_modality.value,
        )
        record_comparison_cross_modal_result(
            response,
            primary_hash=optical_result.content_hash,
            secondary_hash=sar_result.content_hash,
            optical_modality=optical_modality,
            sar_modality=sar_modality,
        )
        return response
    except CrossModalAnalysisError as error:
        remove_preview(optical_metadata)
        remove_preview(sar_metadata)
        raise HTTPException(status_code=422, detail={"code": error.code, "message": error.message}) from error
    finally:
        if optical_result is not None:
            optical_result.model_image.close()
        if sar_result is not None:
            sar_result.model_image.close()


def _unsupported_vqa_details(question: str, reason: str) -> ControlledVQAResult:
    limitations = list(SINGLE_EVIDENCE_LIMITATIONS)
    return ControlledVQAResult(
        original_question=question,
        question_category=QuestionCategory.UNSUPPORTED,
        target_concept=None,
        answer_source="controlled question taxonomy",
        statistics_used={},
        evidence_references=[],
        method=ControlledVQAMethod(
            name="Controlled GeoVision evidence-grounded VQA",
            version="1.0",
            method_type="deterministic evidence-grounded VQA",
            uses_language_model=False,
            remote_sensing_adapted=False,
            assumptions=list(SINGLE_EVIDENCE_ASSUMPTIONS),
            limitations=limitations,
        ),
        confidence=Confidence(level=ConfidenceLevel.LOW, score=None, reason=reason),
        supported=False,
        limitations=limitations,
    )


async def _change_query_result(
    before_result: IngestedImage,
    after_result: IngestedImage,
    compatibility: PairCompatibility,
    before_date: str,
    after_date: str,
    modality: Modality,
) -> ChangeAnalysisResponse:
    started = time.perf_counter()
    before_metadata = before_result.metadata
    after_metadata = after_result.metadata
    steps = _change_ingestion_steps([before_result.durations_ms, after_result.durations_ms])
    steps.append(
        ExecutionStep(
            tool="pair_validation",
            status=ToolStatus.SUCCESS if compatibility.compatible or _alignment_outcome(compatibility) else ToolStatus.FAILED,
            duration_ms=0,
            parameters={"alignment_level": compatibility.alignment_level.value, "registration_performed": False},
        )
    )
    warnings = _unique(before_metadata.warnings + after_metadata.warnings + compatibility.warnings + compatibility.errors)
    previews = ChangePreviewUrls(before=before_metadata.preview_url, after=after_metadata.preview_url)
    statistics = None
    deterministic_statistics = None
    change_engine = ChangeEngine(
        mode="deterministic",
        primary_tool="deterministic_change_analyzer",
        supporting_tool=None,
        fallback_used=False,
        fallback_reason=None,
    )
    ttp_result: Optional[TTPResult] = None
    mask_comparison = None
    evidence_consistency = EvidenceConsistency(
        label="Unavailable",
        rationale=["TTP and deterministic masks were not both available for comparison."],
    )
    sve_result: Optional[SVEResult] = None
    if requires_alignment(compatibility):
        steps.append(ExecutionStep(
            tool="ttp_eligibility_check", status=ToolStatus.SKIPPED, duration_ms=0,
            parameters={"eligible": False, "reason": "alignment_required"},
        ))
        steps.extend(_skipped_change_steps())
        if _alignment_outcome(compatibility):
            status = ChangeAnalysisStatus.ALIGNMENT_REQUIRED
            validation = ValidationStatus(valid=False, errors=["Pixel alignment is required before change analysis."])
            warnings.append("Explicit registration, reprojection, or resampling is required; the service did not alter either image.")
        else:
            status = ChangeAnalysisStatus.FAILED
            validation = ValidationStatus(valid=False, errors=compatibility.errors or ["The image pair is incompatible."])
    else:
        analysis = await run_in_threadpool(
            analyze_change,
            before_result.analysis_raster,
            after_result.analysis_raster,
            before_metadata,
            after_metadata,
            compatibility,
        )
        statistics = analysis.statistics
        deterministic_statistics = analysis.statistics
        previews = analysis.previews
        steps.append(ExecutionStep(
            tool="deterministic_analysis", status=ToolStatus.SUCCESS,
            duration_ms=sum(analysis.durations_ms.values()),
            parameters={"changed_pixels": analysis.statistics.changed_pixels, "region_count": analysis.statistics.number_of_regions},
        ))
        for name in ("difference_computation", "thresholding", "morphology", "connected_components", "preview_generation"):
            steps.append(ExecutionStep(tool=name, status=ToolStatus.SUCCESS, duration_ms=analysis.durations_ms.get(name, 0)))
        warnings = _unique(warnings + analysis.warnings)
        status = ChangeAnalysisStatus.SUCCESS
        validation = ValidationStatus(valid=True, errors=[])
        remove_preview(before_metadata)
        remove_preview(after_metadata)
        before_metadata = before_metadata.model_copy(update={"preview_url": previews.before})
        after_metadata = after_metadata.model_copy(update={"preview_url": previews.after})
        requested_mode = ttp_default_mode()
        eligible = (
            requested_mode != "deterministic"
            and modality == Modality.OPTICAL
            and before_metadata.band_count in {3, 4}
            and after_metadata.band_count in {3, 4}
            and analysis.mask is not None
            and analysis.after_rgb is not None
        )
        eligibility_reason = None
        if requested_mode == "deterministic":
            eligibility_reason = "deterministic_mode_requested"
        elif modality != Modality.OPTICAL:
            eligibility_reason = "unsupported_modality"
        elif before_metadata.band_count not in {3, 4} or after_metadata.band_count not in {3, 4}:
            eligibility_reason = "unsupported_optical_bands"
        steps.append(ExecutionStep(
            tool="ttp_eligibility_check",
            status=ToolStatus.SUCCESS if eligible else ToolStatus.SKIPPED,
            duration_ms=0,
            parameters={"eligible": eligible, "reason": eligibility_reason, "alignment_level": compatibility.alignment_level.value},
        ))
        if eligible and not ttp_enabled():
            eligibility_reason = "ttp_disabled"
            eligible = False
            warnings.append("TTP is disabled; the deterministic change analyzer produced the result.")
        if eligible:
            try:
                health_started = time.perf_counter()
                health = await run_in_threadpool(TTP_CLIENT.health)
                steps.append(ExecutionStep(
                    tool="ttp_health_check", status=ToolStatus.SUCCESS,
                    duration_ms=max(0, round((time.perf_counter() - health_started) * 1000)),
                    parameters={"lifecycle": health.get("lifecycle"), "checkpoint_verified": True, "device": "cuda"},
                ))
                steps.append(ExecutionStep(
                    tool="ttp_request_preparation", status=ToolStatus.SUCCESS, duration_ms=0,
                    parameters={"width": before_metadata.width, "height": before_metadata.height, "exact_source_bytes": True},
                ))
                learned = await run_in_threadpool(
                    TTP_CLIENT.predict,
                    before_result.source_bytes,
                    after_result.source_bytes,
                    before_metadata.safe_name,
                    after_metadata.safe_name,
                    before_metadata.width,
                    before_metadata.height,
                    uuid.uuid4().hex,
                )
                steps.extend([
                    ExecutionStep(
                        tool="ttp_model_reuse" if learned.reused_model else "ttp_model_load",
                        status=ToolStatus.SUCCESS,
                        duration_ms=0 if learned.reused_model else learned.model_load_ms,
                        parameters={"reused_model": learned.reused_model},
                    ),
                    ExecutionStep(
                        tool="ttp_inference", status=ToolStatus.SUCCESS, duration_ms=learned.runtime_ms,
                        parameters={"changed_pixels": learned.changed_pixels, "region_count": learned.region_count},
                    ),
                    ExecutionStep(
                        tool="ttp_mask_validation", status=ToolStatus.SUCCESS, duration_ms=0,
                        parameters={"width": learned.mask.shape[1], "height": learned.mask.shape[0], "binary": True},
                    ),
                ])
                ttp_statistics = statistics_from_binary_mask(learned.mask, before_metadata.width, before_metadata.height)
                mask_comparison = compare_binary_masks(learned.mask, analysis.mask)
                previews = save_hybrid_previews(learned.mask, analysis.mask, analysis.after_rgb, previews)
                statistics = ttp_statistics
                change_engine = ChangeEngine(
                    mode="ttp" if requested_mode == "ttp" else "hybrid",
                    primary_tool="ttp_change_detector",
                    supporting_tool="deterministic_change_analyzer",
                    fallback_used=False,
                    fallback_reason=None,
                )
                ttp_result = TTPResult(
                    status="success",
                    changed_percentage=ttp_statistics.percentage_changed,
                    changed_pixels=ttp_statistics.changed_pixels,
                    region_count=ttp_statistics.number_of_regions,
                    largest_region_pixels=ttp_statistics.largest_connected_region,
                    runtime_ms=learned.runtime_ms,
                    model_load_ms=learned.model_load_ms,
                    model=TTP_MODEL_ID,
                    architecture=TTP_ARCHITECTURE,
                    training_dataset=TTP_TRAINING_DATASET,
                    checkpoint=TTP_CHECKPOINT_NAME,
                    checkpoint_fingerprint=TTP_CHECKPOINT_FINGERPRINT,
                    device=learned.device,
                    reused_model=learned.reused_model,
                    limitations=_unique(learned.limitations + [TTP_DOMAIN_WARNING, "TTP output is a model-generated binary change prediction and is not ground truth."]),
                    warnings=learned.warnings,
                )
                consistency_label = "Strong evidence consistency" if (mask_comparison.iou or 0) >= 0.65 else "Moderate evidence consistency" if (mask_comparison.iou or 0) >= 0.35 else "Limited evidence consistency"
                evidence_consistency = EvidenceConsistency(
                    label=consistency_label,
                    rationale=[
                        f"Pair compatibility: {compatibility.alignment_level.value}.",
                        f"TTP and deterministic mask IoU: {mask_comparison.iou:.3f}.",
                        f"Pixel agreement: {mask_comparison.agreement_percentage:.3f}%.",
                        "TTP execution and all evidence-artifact validation completed successfully.",
                    ],
                )
                warnings = _unique(warnings + learned.warnings + [TTP_DOMAIN_WARNING, mask_comparison.disclaimer, "TTP output is a model-generated binary change prediction and is not ground truth."])
                steps.extend([
                    ExecutionStep(tool="mask_comparison", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"iou": mask_comparison.iou, "agreement_percentage": mask_comparison.agreement_percentage}),
                    ExecutionStep(tool="evidence_fusion", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"primary_mask": "ttp", "supporting_mask": "deterministic", "mask_averaging": False}),
                ])
            except Exception as error:
                error_code = error.code if isinstance(error, (TTPClientError, ChangeAnalysisError)) else "INVALID_TTP_RESULT"
                public_message = error.public_message if isinstance(error, TTPClientError) else "TTP evidence validation failed; deterministic fallback was used."
                change_engine = ChangeEngine(
                    mode="deterministic_fallback",
                    primary_tool="deterministic_change_analyzer",
                    supporting_tool=None,
                    fallback_used=True,
                    fallback_reason=error_code.lower(),
                )
                ttp_result = TTPResult(status="failed", warnings=[public_message], limitations=[TTP_DOMAIN_WARNING])
                warnings = _unique(warnings + [public_message, TTP_DOMAIN_WARNING])
                steps.extend([
                    ExecutionStep(tool="ttp_health_check" if error_code in {"UNAVAILABLE", "UNHEALTHY", "TIMEOUT"} else "ttp_inference", status=ToolStatus.FAILED, duration_ms=0, parameters={"reason": error_code.lower()}),
                    ExecutionStep(tool="deterministic_fallback", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"reason": error_code.lower()}),
                ])
        elif eligibility_reason and eligibility_reason not in {"deterministic_mode_requested", "ttp_disabled"}:
            warnings.append(f"TTP was not called because the pair is not eligible ({eligibility_reason.replace('_', ' ')}).")
            change_engine = change_engine.model_copy(update={"fallback_reason": eligibility_reason})
    if get_sve_service().enabled and _sve_optical_eligible(before_metadata, modality) and _sve_optical_eligible(after_metadata, modality):
        sve_call = await run_in_threadpool(
            get_sve_service().compare,
            before_result.model_image,
            before_result.content_hash,
            after_result.model_image,
            after_result.content_hash,
            label="Before/after scene-level semantic similarity",
            disclaimer="Embedding differences are scene-level semantic evidence, not spatial change localization.",
        )
        sve_result = sve_call.result
        steps.extend(_sve_execution_steps(sve_call))
        if sve_result.warning:
            warnings.append(sve_result.warning)
    steps.append(ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"language_model_used": False}))
    runtime_ms = max(0, round((time.perf_counter() - started) * 1000))
    return ChangeAnalysisResponse(
        request_id=str(uuid.uuid4()), status=status, before_date=before_date, after_date=after_date,
        before_metadata=before_metadata, after_metadata=after_metadata, compatibility=compatibility,
        statistics=statistics, previews=previews,
        execution=ExecutionSummary(
            input_mode=InputMode.BI_TEMPORAL,
            selected_tools=(
                ["input_validator", "ttp_change_detector", "deterministic_change_analyzer"]
                if change_engine.mode in {"hybrid", "ttp"}
                else ["input_validator", "deterministic_change_analyzer"]
            ),
            steps=steps, duration_ms=runtime_ms,
            permitted_parameters={"before_date": before_date, "after_date": after_date, "modality": modality.value, "registration": False, "reprojection": False, "resampling": False},
            validation=validation,
            selection_reason=(
                "TTP is the primary learned detector and the deterministic analyzer supplies independent supporting evidence."
                if change_engine.mode in {"hybrid", "ttp"}
                else "The deterministic analyzer completed the request with the TTP eligibility or fallback state disclosed."
            ),
        ),
        runtime_ms=runtime_ms, warnings=_unique(warnings),
        change_engine=change_engine,
        ttp_result=ttp_result,
        deterministic_statistics=deterministic_statistics,
        mask_comparison=mask_comparison,
        evidence_consistency=evidence_consistency,
        sve_result=sve_result,
    )


@router.post("/query", response_model=AgentResponse)
async def agent_image_query(
    query: str = Form(...),
    input_mode: InputMode = Form(...),
    primary_modality: Modality = Form(...),
    primary_image_modality: Optional[ImageModality] = Form(None),
    primary_image: UploadFile = File(...),
    secondary_modality: Optional[Modality] = Form(None),
    secondary_image: Optional[UploadFile] = File(None),
    primary_date: Optional[str] = Form(None),
    secondary_date: Optional[str] = Form(None),
    use_cache: bool = Form(False),
    force_rerun: bool = Form(False),
) -> AgentResponse:
    """Ingest imagery, route deterministically, and execute connected local specialists."""
    started = time.perf_counter()
    analytics_started_at = analytics_utc_now()
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_QUERY", "message": "A natural-language query is required."},
        )
    pair_mode = input_mode in (InputMode.CROSS_MODAL, InputMode.BI_TEMPORAL)
    if pair_mode and secondary_image is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_SECONDARY_IMAGE", "message": "A secondary image is required for the selected input mode."},
        )
    if pair_mode and secondary_modality is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_SECONDARY_MODALITY", "message": "A secondary modality is required for the selected input mode."},
        )
    if input_mode == InputMode.BI_TEMPORAL and (not primary_date or not secondary_date):
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_TEMPORAL_DATES", "message": "Earlier and later dates are required for bi-temporal questions."},
        )

    primary_metadata: Optional[ImageMetadata] = None
    secondary_metadata: Optional[ImageMetadata] = None
    ingestion_durations: List[Dict[str, int]] = []
    try:
        primary_result = await ingest_upload(primary_image)
        requested_image_modality = primary_image_modality
        if requested_image_modality is None:
            if primary_modality == Modality.SAR:
                requested_image_modality = (
                    ImageModality.SAR_PREVIEW
                    if primary_result.metadata.representation == RepresentationType.DISPLAY_PREVIEW
                    else ImageModality.SAR_VV_VH if primary_result.metadata.band_count == 2 else ImageModality.SAR_VV
                )
            elif primary_modality == Modality.MULTISPECTRAL:
                requested_image_modality = ImageModality.MULTISPECTRAL
            elif primary_modality == Modality.OPTICAL:
                requested_image_modality = ImageModality.OPTICAL_RGB if primary_result.metadata.band_count >= 3 else ImageModality.OPTICAL_GRAYSCALE
            else:
                requested_image_modality = ImageModality.AUTO
        primary_metadata = apply_modality_override(primary_result.metadata, requested_image_modality)
        primary_result.metadata = primary_metadata
        ingestion_durations.append(primary_result.durations_ms)
        if pair_mode and secondary_image is not None:
            secondary_result = await ingest_upload(secondary_image)
            secondary_metadata = secondary_result.metadata
            ingestion_durations.append(secondary_result.durations_ms)
    except ImageIngestionError as error:
        remove_preview(primary_metadata)
        remove_preview(secondary_metadata)
        _raise_ingestion_error(error)

    if input_mode == InputMode.SINGLE:
        primary_modality = coarse_modality(primary_metadata.effective_modality)
    requires_modality_confirmation = bool(
        input_mode == InputMode.SINGLE
        and requested_image_modality == ImageModality.AUTO
        and primary_metadata.effective_modality == ImageModality.UNKNOWN
    )

    pair_compatibility: Optional[PairCompatibility] = None
    pair_compatibility_duration = 0
    steps = _ingestion_steps(ingestion_durations)
    steps.extend([
        ExecutionStep(
            tool="representation_detection",
            status=ToolStatus.SUCCESS,
            duration_ms=0,
            parameters={"representation": primary_metadata.representation.value},
        ),
        ExecutionStep(
            tool="modality_detection",
            status=ToolStatus.SUCCESS,
            duration_ms=0,
            parameters={
                "auto_detected": primary_metadata.auto_detected_modality.value,
                "confidence": primary_metadata.auto_detection_confidence.value,
                "effective_modality": primary_metadata.effective_modality.value,
            },
        ),
        ExecutionStep(
            tool="user_modality_override",
            status=ToolStatus.SUCCESS if primary_metadata.user_confirmed_modality else ToolStatus.SKIPPED,
            duration_ms=0,
            parameters={"confirmed_modality": primary_metadata.user_confirmed_modality.value if primary_metadata.user_confirmed_modality else None},
        ),
    ])
    if pair_mode and secondary_metadata is not None:
        compatibility_started = time.perf_counter()
        pair_compatibility = validate_pair_compatibility(
            input_mode=input_mode,
            primary_modality=primary_modality,
            secondary_modality=secondary_modality,
            primary=primary_metadata,
            secondary=secondary_metadata,
            primary_date=primary_date,
            secondary_date=secondary_date,
        )
        pair_compatibility_duration = max(0, round((time.perf_counter() - compatibility_started) * 1000))
        steps.append(
            ExecutionStep(
                tool="pair_compatibility_check",
                status=ToolStatus.SUCCESS if pair_compatibility.compatible else ToolStatus.FAILED,
                duration_ms=pair_compatibility_duration,
            )
        )

    routing_request = AgentQueryRequest(
        query=query.strip(),
        input_mode=input_mode,
        primary_modality=primary_modality,
        secondary_modality=secondary_modality if pair_mode else None,
        has_primary_image=True,
        has_secondary_image=secondary_metadata is not None,
        primary_image_modality=primary_metadata.effective_modality,
        primary_representation=primary_metadata.representation,
        primary_band_count=primary_metadata.band_count,
    )
    routing_started = time.perf_counter()
    plan = route_query(routing_request)
    diagnostic(
        "router",
        task=plan.detected_task.value,
        effective_modality=primary_metadata.effective_modality.value,
        representation=primary_metadata.representation.value,
        band_count=primary_metadata.band_count,
        selected=plan.selected_tools[-1],
        method="heuristic_candidate_detector" if plan.detected_task == TaskType.SAR_WATER_SEGMENTATION else None,
    )
    routing_duration = max(0, round((time.perf_counter() - routing_started) * 1000))
    steps.append(
        ExecutionStep(
            tool="query_routing",
            status=ToolStatus.SUCCESS,
            duration_ms=routing_duration,
        )
    )

    analysis_cache_key = cache_key_for(
        primary_hash=primary_result.content_hash,
        secondary_hash=secondary_result.content_hash if pair_mode and secondary_metadata is not None else None,
        task=plan.detected_task,
        normalized_query=query,
        safe_parameters={
            "input_mode": input_mode.value,
            "primary_modality": primary_modality.value,
            "auto_detected_modality": primary_metadata.auto_detected_modality.value,
            "user_modality_override": primary_metadata.user_confirmed_modality.value if primary_metadata.user_confirmed_modality else None,
            "effective_modality": primary_metadata.effective_modality.value,
            "representation": primary_metadata.representation.value,
            "band_count": primary_metadata.band_count,
            "secondary_modality": secondary_modality.value if secondary_modality else None,
            "primary_date": primary_date,
            "secondary_date": secondary_date,
            "selected_specialist": plan.selected_tools[-1],
            "specialist_version": tool_definition(plan.selected_tools[-1]).specialist_version if len(plan.selected_tools) > 1 else None,
            "change_engine_mode": ttp_default_mode() if plan.detected_task in {TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA} else None,
            "ttp_enabled": ttp_enabled() if plan.detected_task in {TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA} else None,
            "ttp_model_id": TTP_MODEL_ID if plan.detected_task in {TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA} else None,
            "ttp_checkpoint_fingerprint": TTP_CHECKPOINT_FINGERPRINT if plan.detected_task in {TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA} else None,
            "deterministic_analyzer_version": "bitemporal-change-1.0" if plan.detected_task in {TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA} else None,
            "sve_adapter_sha256": "a99c0bf0fb44044988ef1698483888c8a2e3a047d2d2d56478837575cf7626ea",
            "sve_enabled": get_sve_service().enabled,
            "preprocessing_version": "sar-preprocess-1.0" if plan.detected_task == TaskType.SAR_WATER_SEGMENTATION else None,
            "heuristic_version": "heuristic-sar-water-1.0" if plan.detected_task == TaskType.SAR_WATER_SEGMENTATION else None,
        },
    )
    if use_cache and not force_rerun:
        cached_record = MISSION_STORE.get_by_cache(analysis_cache_key)
        if cached_record is not None:
            remove_preview(primary_metadata)
            remove_preview(secondary_metadata)
            primary_result.model_image.close()
            if pair_mode and secondary_metadata is not None:
                secondary_result.model_image.close()
            response = cached_response(cached_record)
            MISSION_STORE.put(response, analysis_cache_key, cached_record.generated_at)
            record_comparison_agent_result(
                response,
                primary_hash=primary_result.content_hash,
                secondary_hash=secondary_result.content_hash if pair_mode and secondary_metadata is not None else None,
                primary_modality=primary_modality,
                secondary_modality=secondary_modality if pair_mode else None,
                primary_date=primary_date,
                secondary_date=secondary_date,
                query=query,
            )
            record_agent_response(
                response,
                started_at=analytics_started_at,
                primary_modality=primary_modality.value,
                secondary_modality=secondary_modality.value if secondary_modality else None,
            )
            return response

    validation_errors = list(plan.validation_status.errors)
    compatibility_warnings: List[str] = []
    if pair_compatibility is not None:
        if plan.detected_task not in (TaskType.CROSS_MODAL_ANALYSIS, TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA):
            validation_errors.extend(pair_compatibility.errors)
        compatibility_warnings.extend(pair_compatibility.warnings)
    validation = ValidationStatus(valid=not validation_errors, errors=_unique(validation_errors))
    metadata_warnings = list(primary_metadata.warnings)
    if secondary_metadata is not None:
        metadata_warnings.extend(secondary_metadata.warnings)
    warnings = _unique(validation.errors + compatibility_warnings + metadata_warnings)
    answer = None
    response_confidence = Confidence(
        level=ConfidenceLevel.UNAVAILABLE,
        score=None,
        reason="No connected specialist produced an answer.",
    )
    model_provenance = None
    caption_details = None
    cross_modal_result: Optional[CrossModalResult] = None
    change_analysis_result: Optional[ChangeAnalysisResponse] = None
    vqa_details: Optional[ControlledVQAResult] = None
    grounding_result: Optional[GroundingResult] = None
    sar_water_result: Optional[SarWaterResult] = None
    sar_scene_result: Optional[SarSceneResult] = None
    classified_query = classify_query(query, primary_metadata.effective_modality)
    response_evidence = []
    result_status = "COMPLETED"
    sve_result: Optional[SVEResult] = None

    if requires_modality_confirmation:
        steps.append(ExecutionStep(
            tool="compatibility_validation",
            status=ToolStatus.FAILED,
            duration_ms=0,
            parameters={"reason": "modality_confirmation_required", "specialist_executed": False},
        ))
        steps.append(ExecutionStep(tool="specialist_selection", status=ToolStatus.SKIPPED, duration_ms=0))
        status = ResponseStatus.FAILED
        result_status = "NEEDS_USER_CONFIRMATION"
        answer = "This single-band display image is ambiguous. Confirm SAR Preview, Optical Grayscale, Panchromatic, or another modality before specialist execution."
        warning = "A one-band PNG or JPEG does not contain enough authoritative metadata to distinguish SAR from optical grayscale or panchromatic imagery."
        warnings.append(warning)
        confidence_reason = "Modality-specific execution was intentionally stopped before specialist selection."
        plan = plan.model_copy(update={"selected_tools": ["input_validator"], "selection_reason": confidence_reason})
    elif not validation.valid:
        steps.append(ExecutionStep(tool="specialist_selection", status=ToolStatus.SKIPPED, duration_ms=0))
        for tool_id in plan.selected_tools[1:]:
            steps.append(ExecutionStep(tool=tool_id, status=ToolStatus.SKIPPED, duration_ms=0))
        status = ResponseStatus.FAILED
        result_status = "INVALID_INPUT"
        confidence_reason = "Input or pair validation failed; no specialist tool was executed."
    elif plan.detected_task == TaskType.UNSUPPORTED:
        steps.append(ExecutionStep(tool="specialist_selection", status=ToolStatus.SKIPPED, duration_ms=0))
        status = ResponseStatus.NOT_IMPLEMENTED
        result_status = "UNSUPPORTED_TASK"
        warnings.append("The query did not match a supported deterministic routing rule.")
        confidence_reason = "No supported task could be selected from the query."
    else:
        selection_started = time.perf_counter()
        selected_tool = tool_definition(plan.selected_tools[-1])
        capability_errors: List[str] = []
        semantic_unsupported = plan.detected_task == TaskType.VQA and primary_modality not in {
            Modality.OPTICAL,
            Modality.MULTISPECTRAL,
        }
        if not semantic_unsupported and selected_tool.supported_image_modalities and primary_metadata.effective_modality not in selected_tool.supported_image_modalities:
            capability_errors.append(f"effective modality {primary_metadata.effective_modality.value} is unsupported")
        if not semantic_unsupported and selected_tool.supported_representations and primary_metadata.representation not in selected_tool.supported_representations:
            capability_errors.append(f"representation {primary_metadata.representation.value} is unsupported")
        if not semantic_unsupported and selected_tool.minimum_bands is not None and primary_metadata.band_count < selected_tool.minimum_bands:
            capability_errors.append(f"requires at least {selected_tool.minimum_bands} bands")
        if not semantic_unsupported and selected_tool.maximum_bands is not None and primary_metadata.band_count > selected_tool.maximum_bands:
            capability_errors.append(f"supports at most {selected_tool.maximum_bands} bands")
        steps.append(ExecutionStep(
            tool="compatibility_validation",
            status=ToolStatus.FAILED if capability_errors else ToolStatus.SUCCESS,
            duration_ms=0,
            parameters={"specialist": selected_tool.id, "rejection_reasons": capability_errors},
        ))
        steps.append(
            ExecutionStep(
                tool="specialist_selection",
                status=ToolStatus.SUCCESS,
                duration_ms=max(0, round((time.perf_counter() - selection_started) * 1000)),
            )
        )
        if capability_errors:
            reason = f"{selected_tool.display_name} is incompatible with this input: " + "; ".join(capability_errors) + "."
            answer = "Unsupported for this input and available specialist set. " + reason
            warnings.append(reason)
            confidence_reason = reason
            status = ResponseStatus.NOT_IMPLEMENTED
            result_status = "UNSUPPORTED_INPUT"
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.SKIPPED, duration_ms=0, parameters={"execution_skipped": True}))
        elif plan.detected_task == TaskType.SAR_WATER_SEGMENTATION:
            try:
                sar_water_result, answer = await run_in_threadpool(analyze_sar_water, primary_result.analysis_raster, primary_metadata)
                diagnostic(
                    "sar_water_result",
                    method=sar_water_result.method,
                    version=sar_water_result.method_version,
                    candidate_pixels=sar_water_result.candidate_pixels,
                    image_area_percent=sar_water_result.image_area_percent,
                    region_count=len(sar_water_result.regions),
                    heuristic_reliability=sar_water_result.heuristic_reliability,
                    runtime_ms=sar_water_result.runtime_ms,
                    limitations=sar_water_result.limitations,
                )
                durations = sar_water_result.stage_durations_ms
                response_confidence = Confidence(
                    level=ConfidenceLevel.UNAVAILABLE,
                    score=None,
                    reason="The active SAR water method is a deterministic heuristic; no calibrated model confidence is available.",
                )
                confidence_reason = response_confidence.reason
                warnings = _unique(warnings + sar_water_result.warnings + sar_water_result.limitations)
                response_evidence = [
                    EvidenceItem(type=product.type, label=product.label, description=product.description, reference=product.reference)
                    for product in sar_water_result.evidence_products
                ]
                status = ResponseStatus.PARTIAL
                result_status = "COMPLETED_WITH_LIMITATIONS"
                steps = _ingestion_steps(ingestion_durations) + steps[4:] + [
                    ExecutionStep(tool="query_classification", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"task": classified_query.task_type.value, "target": classified_query.target, "requested_output": classified_query.requested_output.value}),
                    ExecutionStep(tool="sar_preprocessing", status=ToolStatus.SUCCESS, duration_ms=sum(durations.get(name, 0) for name in ("sar_input_validation", "sar_normalization", "sar_denoising")), parameters={"version": sar_water_result.preprocessing.version, "log_transform": False, "representation": primary_metadata.representation.value}),
                    ExecutionStep(tool="heuristic_segmentation", status=ToolStatus.SUCCESS, duration_ms=durations.get("heuristic_segmentation", 0), parameters={"method": sar_water_result.method, "version": sar_water_result.method_version, "threshold": sar_water_result.threshold, "trained_model": False}),
                    ExecutionStep(tool="post_processing", status=ToolStatus.SUCCESS, duration_ms=durations.get("morphological_postprocessing", 0), parameters={"candidate_pixels": sar_water_result.candidate_pixels}),
                    ExecutionStep(tool="connected_components", status=ToolStatus.SUCCESS, duration_ms=durations.get("connected_components", 0), parameters={"region_count": len(sar_water_result.regions)}),
                    ExecutionStep(tool="evidence_generation", status=ToolStatus.SUCCESS, duration_ms=durations.get("evidence_generation", 0), parameters={"product_count": len(sar_water_result.evidence_products), "geographic_area_reported": False}),
                    ExecutionStep(tool="result_assembly", status=ToolStatus.SUCCESS, duration_ms=0),
                    ExecutionStep(tool="export_record_creation", status=ToolStatus.SUCCESS, duration_ms=0),
                ]
            except SarWaterAnalysisError as error:
                answer = error.message
                warnings.append(error.message)
                confidence_reason = error.message
                status = ResponseStatus.FAILED
                result_status = "INVALID_INPUT" if error.code in {"NO_VALID_SAR_PIXELS", "INVALID_SAR_RANGE"} else "INFERENCE_FAILED"
                steps.append(ExecutionStep(tool="sar_water_segmenter", status=ToolStatus.FAILED, duration_ms=0, parameters={"error_code": error.code}))
        elif plan.detected_task in {TaskType.SAR_SCENE_ANALYSIS, TaskType.SAR_QUALITY_INSPECTION}:
            try:
                sar_scene_result, answer = await run_in_threadpool(analyze_sar_scene, primary_result.analysis_raster, primary_metadata, plan.detected_task)
                response_confidence = Confidence(level=ConfidenceLevel.UNAVAILABLE, score=None, reason="A deterministic SAR summary was used; no trained classifier probability is available.")
                confidence_reason = response_confidence.reason
                warnings = _unique(warnings + sar_scene_result.warnings + sar_scene_result.limitations)
                response_evidence = [EvidenceItem(type=product.type, label=product.label, description=product.description, reference=product.reference) for product in sar_scene_result.evidence_products]
                status = ResponseStatus.PARTIAL
                result_status = "COMPLETED_WITH_LIMITATIONS"
                durations = sar_scene_result.stage_durations_ms
                steps.extend([
                    ExecutionStep(tool="query_classification", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"task": plan.detected_task.value}),
                    ExecutionStep(tool="sar_preprocessing", status=ToolStatus.SUCCESS, duration_ms=sum(durations.get(name, 0) for name in ("sar_input_validation", "sar_normalization", "sar_denoising")), parameters={"version": sar_scene_result.preprocessing.version, "log_transform": False}),
                    ExecutionStep(tool="sar_scene_statistics", status=ToolStatus.SUCCESS, duration_ms=durations.get("sar_scene_statistics", 0), parameters={"trained_classifier": False, "texture_index": sar_scene_result.texture_index}),
                    ExecutionStep(tool="evidence_generation", status=ToolStatus.SUCCESS, duration_ms=0, parameters={"product_count": len(sar_scene_result.evidence_products)}),
                    ExecutionStep(tool="result_assembly", status=ToolStatus.SUCCESS, duration_ms=0),
                ])
            except SarSceneAnalysisError as error:
                answer = error.message
                warnings.append(error.message)
                confidence_reason = error.message
                status = ResponseStatus.FAILED
                result_status = "INVALID_INPUT" if error.code in {"NO_VALID_SAR_PIXELS", "INVALID_SAR_RANGE"} else "INFERENCE_FAILED"
                steps.append(ExecutionStep(tool="sar_scene_analyzer", status=ToolStatus.FAILED, duration_ms=0, parameters={"error_code": error.code}))
        elif plan.detected_task == TaskType.GROUNDING:
            try:
                grounding_result = await run_in_threadpool(
                    get_grounder().ground,
                    primary_result.model_image,
                    query,
                    primary_metadata,
                    primary_modality,
                    primary_result.bands_used,
                    primary_result.image_representation,
                )
                durations = grounding_result.stage_durations_ms
                detection_count = len(grounding_result.detections)
                rejected_count = len(grounding_result.rejected_candidates)
                candidate_count = detection_count + rejected_count
                rejection_reason_counts: Dict[str, int] = {}
                for candidate in grounding_result.rejected_candidates:
                    for reason in candidate.rejection_reasons:
                        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
                answer = (
                    f"Grounding DINO produced {detection_count} accepted localized region(s) for '{grounding_result.target_phrase}'. "
                    "Review the accepted model scores and annotated preview."
                    if detection_count
                    else grounding_result.empty_result_explanation
                )
                response_confidence = grounding_result.confidence
                confidence_reason = grounding_result.confidence.reason
                model_provenance = grounding_result.model
                warnings = _unique(warnings + grounding_result.warnings)
                response_evidence = [
                    EvidenceItem(
                        type="model_produced_grounding",
                        label="Grounding DINO annotated detections",
                        description="Model-produced candidate boxes; not ground truth.",
                        reference=grounding_result.annotated_preview_url,
                    )
                ] if grounding_result.annotated_preview_url else []
                status = ResponseStatus.SUCCESS if detection_count else ResponseStatus.PARTIAL
                steps = _ingestion_steps(ingestion_durations) + [
                    ExecutionStep(tool="query_routing", status=ToolStatus.SUCCESS, duration_ms=routing_duration),
                    ExecutionStep(tool="target_phrase_extraction", status=ToolStatus.SUCCESS, duration_ms=durations.get("target_phrase_extraction", 0), parameters={"target_phrase": grounding_result.target_phrase}),
                    ExecutionStep(tool="grounding_image_preparation", status=ToolStatus.SUCCESS, duration_ms=durations.get("grounding_image_preparation", 0), parameters={"representation": grounding_result.input.representation, "model_input_width": grounding_result.input.model_input_width, "model_input_height": grounding_result.input.model_input_height}),
                    ExecutionStep(tool="grounder_model_load", status=ToolStatus.SUCCESS, duration_ms=durations.get("grounder_model_load", 0), parameters={"checkpoint": grounding_result.model.checkpoint, "device": grounding_result.device, "reused": grounding_result.model_reused}),
                    ExecutionStep(tool="grounding_inference", status=ToolStatus.SUCCESS, duration_ms=durations.get("grounding_inference", 0), parameters={"model_produced": True}),
                    ExecutionStep(tool="grounding_postprocessing", status=ToolStatus.SUCCESS, duration_ms=durations.get("grounding_postprocessing", 0), parameters={"candidate_count": candidate_count, "box_threshold": safe_grounding_parameters()["box_threshold"], "text_threshold": safe_grounding_parameters()["text_threshold"]}),
                    ExecutionStep(tool="grounding_quality_filter", status=ToolStatus.SUCCESS, duration_ms=durations.get("grounding_quality_filter", 0), parameters={"candidate_count": candidate_count, "accepted_count": detection_count, "rejected_count": rejected_count, "minimum_score": grounding_result.quality_policy.minimum_alignment_score, "maximum_localized_area_ratio": grounding_result.quality_policy.maximum_localized_area_ratio, "rejection_reason_counts": rejection_reason_counts}),
                    ExecutionStep(tool="grounding_preview_generation", status=ToolStatus.SUCCESS if grounding_result.annotated_preview_url else ToolStatus.SKIPPED, duration_ms=durations.get("grounding_preview_generation", 0), parameters={"ground_truth_claimed": False, "mask_refinement": False}),
                    ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0),
                ]
            except GrounderError as error:
                unsupported = error.code in {
                    "EMPTY_GROUNDING_TARGET", "UNSUPPORTED_GROUNDING_QUERY", "UNSUPPORTED_GROUNDING_TARGET",
                    "UNSUPPORTED_GROUNDING_MODALITY", "UNSUPPORTED_GROUNDING_BANDS", "GROUNDER_UNAVAILABLE",
                }
                status = ResponseStatus.NOT_IMPLEMENTED if unsupported else ResponseStatus.FAILED
                answer = error.message
                warnings.append(error.message)
                confidence_reason = error.message
                steps.append(ExecutionStep(tool="rs_grounder", status=ToolStatus.NOT_IMPLEMENTED if unsupported else ToolStatus.FAILED, duration_ms=0))
        elif plan.detected_task == TaskType.VQA:
            classification_started = time.perf_counter()
            intent = get_vqa().classify_question(query)
            classification_duration = max(0, round((time.perf_counter() - classification_started) * 1000))
            if primary_modality not in {Modality.OPTICAL, Modality.MULTISPECTRAL}:
                reason = "Controlled single-image VQA supports optical or RGB-like multispectral imagery only; SAR VQA is not implemented."
                answer = reason
                vqa_details = _unsupported_vqa_details(query, reason)
                response_confidence = vqa_details.confidence
                status = ResponseStatus.NOT_IMPLEMENTED
                confidence_reason = reason
                warnings.append(reason)
                steps = _ingestion_steps(ingestion_durations) + [
                    ExecutionStep(tool="query_routing", status=ToolStatus.SUCCESS, duration_ms=routing_duration),
                    ExecutionStep(tool="question_classification", status=ToolStatus.FAILED, duration_ms=classification_duration),
                ] + [ExecutionStep(tool=name, status=ToolStatus.SKIPPED, duration_ms=0) for name in (
                    "optical_image_preparation", "evidence_extraction", "scene_summary_computation", "controlled_answer_generation", "evidence_preview_generation"
                )] + [ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0)]
            elif intent.category == QuestionCategory.UNSUPPORTED:
                reason = "The question is outside the controlled single-image VQA taxonomy."
                answer = "This question is unsupported. Ask about dominant scene type, water, vegetation, structural complexity, possible agriculture, relative coverage, or image metadata."
                vqa_details = _unsupported_vqa_details(query, reason)
                response_confidence = vqa_details.confidence
                status = ResponseStatus.NOT_IMPLEMENTED
                confidence_reason = reason
                warnings.append(reason)
                steps = _ingestion_steps(ingestion_durations) + [
                    ExecutionStep(tool="query_routing", status=ToolStatus.SUCCESS, duration_ms=routing_duration),
                    ExecutionStep(tool="question_classification", status=ToolStatus.FAILED, duration_ms=classification_duration, parameters={"category": "unsupported"}),
                ] + [ExecutionStep(tool=name, status=ToolStatus.SKIPPED, duration_ms=0) for name in (
                    "optical_image_preparation", "evidence_extraction", "scene_summary_computation", "controlled_answer_generation", "evidence_preview_generation"
                )] + [ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0)]
            else:
                specialist_answered = False
                if plan.selected_tools[-1] == "rsvqa_vqa_specialist" and classify_rsvqa_task(query):
                    try:
                        prediction = await run_in_threadpool(
                            get_rsvqa_specialist().predict,
                            primary_result.model_image,
                            query,
                            content_hash=primary_result.content_hash,
                        )
                        score = max(0.0, min(1.0, float(prediction["confidence"])))
                        level = (
                            ConfidenceLevel.HIGH if score >= 0.75 else
                            ConfidenceLevel.MODERATE if score >= 0.50 else
                            ConfidenceLevel.LOW
                        )
                        answer = str(prediction["answer"])
                        confidence_reason = "Maximum task-head softmax score; this value is not calibrated."
                        response_confidence = Confidence(level=level, score=score, reason=confidence_reason)
                        vqa_details = ControlledVQAResult(
                            original_question=query,
                            question_category=intent.category,
                            target_concept=intent.target,
                            answer_source=RSVQA_MODEL_NAME,
                            statistics_used={
                                "task": prediction["task"],
                                "logits": prediction["logits"],
                                "probabilities": prediction["probabilities"],
                            },
                            method=ControlledVQAMethod(
                                name=RSVQA_MODEL_NAME,
                                version="1.0.0",
                                method_type="trained RSVQA multi-task classifier",
                                uses_language_model=False,
                                remote_sensing_adapted=True,
                                assumptions=["Input is optical RGB or a scientifically rendered RGB-like image."],
                                limitations=["Softmax confidence is not calibrated.", "Count outputs are learned labels, not physical inventories."],
                            ),
                            confidence=response_confidence,
                            supported=True,
                            limitations=["Softmax confidence is not calibrated.", "Grounding and captioning are optional supporting evidence only."],
                        )
                        model_provenance = ModelProvenance(
                            tool_id="rsvqa_vqa_specialist",
                            checkpoint=RSVQA_CHECKPOINT_FILENAME,
                            base_architecture="OpenCLIP ViT-L-14 + frozen SatQuery Vision Encoder v1 adapter + learned fusion/task heads",
                            adaptation_dataset="RSVQA-LR official training split",
                            remote_sensing_adapted=True,
                            source="Verified local RSVQA Specialist v1 export",
                        )
                        specialist_answered = True
                        status = ResponseStatus.SUCCESS
                        steps = _ingestion_steps(ingestion_durations) + [
                            ExecutionStep(tool="query_routing", status=ToolStatus.SUCCESS, duration_ms=routing_duration),
                            ExecutionStep(tool="question_classification", status=ToolStatus.SUCCESS, duration_ms=classification_duration, parameters={"category": intent.category.value, "task_head": prediction["task"]}),
                            ExecutionStep(tool="sve_shared_encoder", status=ToolStatus.SUCCESS, duration_ms=int(prediction["encoder"].get("runtime_ms", 0)), parameters={"model_reused": prediction["encoder"].get("model_reused"), "image_cache_hit": prediction["encoder"].get("image_cache_hit"), "text_cache_hit": prediction["encoder"].get("text_cache_hit"), "device": prediction["encoder"].get("device")}),
                            ExecutionStep(tool="rsvqa_vqa_specialist", status=ToolStatus.SUCCESS, duration_ms=int(prediction.get("runtime_ms", 0)), parameters={"task_head": prediction["task"], "primary_prediction_source": True, "confidence_calibrated": False}),
                            ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0),
                        ]
                    except RSVQASpecialistError as error:
                        warnings.append(f"RSVQA Specialist v1 unavailable; deterministic VQA fallback used. {error}")
                        steps.append(ExecutionStep(tool="rsvqa_vqa_specialist", status=ToolStatus.FAILED, duration_ms=0, parameters={"fallback": "rs_vqa"}))
                        plan = plan.model_copy(update={
                            "selected_tools": ["input_validator", "rsvqa_vqa_specialist", "rs_vqa"],
                            "selection_reason": f"RSVQA Specialist v1 failed safely; deterministic controlled VQA was used. {error}",
                        })
                if not specialist_answered:
                    try:
                        extraction = await run_in_threadpool(
                            extract_single_image_evidence,
                            primary_result.analysis_raster,
                            primary_metadata,
                        )
                        answer_started = time.perf_counter()
                        controlled = get_vqa().answer(query, extraction.result, primary_metadata, intent)
                        answer_duration = max(0, round((time.perf_counter() - answer_started) * 1000))
                        answer = controlled.answer
                        vqa_details = controlled.details
                        response_confidence = controlled.details.confidence
                        confidence_reason = controlled.details.confidence.reason
                        warnings = _unique(warnings + extraction.result.warnings)
                        response_evidence = [
                            EvidenceItem(type="heuristic_support_preview", label="Computed VQA evidence", reference=reference)
                            for reference in controlled.details.evidence_references
                        ]
                        status = ResponseStatus.PARTIAL if response_confidence.level == ConfidenceLevel.LOW else ResponseStatus.SUCCESS
                        durations = extraction.stage_durations_ms
                        fallback_prefix = [step for step in steps if step.tool == "rsvqa_vqa_specialist" and step.status == ToolStatus.FAILED]
                        steps = _ingestion_steps(ingestion_durations) + [
                            ExecutionStep(tool="query_routing", status=ToolStatus.SUCCESS, duration_ms=routing_duration),
                            ExecutionStep(tool="question_classification", status=ToolStatus.SUCCESS, duration_ms=classification_duration, parameters={"category": intent.category.value, "target": intent.target}),
                        ] + fallback_prefix + [
                            ExecutionStep(tool="optical_image_preparation", status=ToolStatus.SUCCESS, duration_ms=durations.get("optical_image_preparation", 0), parameters={"normalization": "per-channel finite-pixel 2nd/98th percentile", "max_analysis_dimension": 1024}),
                            ExecutionStep(tool="evidence_extraction", status=ToolStatus.SUCCESS, duration_ms=durations.get("evidence_extraction", 0), parameters=extraction.threshold_parameters),
                            ExecutionStep(tool="scene_summary_computation", status=ToolStatus.SUCCESS, duration_ms=durations.get("scene_summary_computation", 0), parameters={"dominant_scene": extraction.result.dominant_scene}),
                            ExecutionStep(tool="controlled_answer_generation", status=ToolStatus.SUCCESS, duration_ms=answer_duration, parameters={"language_model": False, "answer_source": "computed_evidence", "fallback": bool(fallback_prefix)}),
                            ExecutionStep(tool="evidence_preview_generation", status=ToolStatus.SUCCESS, duration_ms=durations.get("evidence_preview_generation", 0), parameters={"ground_truth_claimed": False}),
                            ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0),
                        ]
                    except SingleImageEvidenceError as error:
                        answer = error.message
                        vqa_details = _unsupported_vqa_details(query, error.message)
                        response_confidence = vqa_details.confidence
                        status = ResponseStatus.NOT_IMPLEMENTED if error.code == "UNSUPPORTED_OPTICAL_BANDS" else ResponseStatus.FAILED
                        confidence_reason = error.message
                        warnings.append(error.message)
                        steps.append(ExecutionStep(tool="rs_vqa", status=ToolStatus.NOT_IMPLEMENTED if status == ResponseStatus.NOT_IMPLEMENTED else ToolStatus.FAILED, duration_ms=0))
        elif plan.detected_task == TaskType.CAPTIONING and primary_modality not in SUPPORTED_MODALITIES:
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.NOT_IMPLEMENTED, duration_ms=0))
            status = ResponseStatus.NOT_IMPLEMENTED
            warning = "The connected RSICD captioner supports optical and multispectral RGB representations only; SAR captioning is not implemented."
            warnings.append(warning)
            confidence_reason = warning
        elif selected_tool.status == ImplementationStatus.NOT_IMPLEMENTED:
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.NOT_IMPLEMENTED, duration_ms=0))
            status = ResponseStatus.NOT_IMPLEMENTED
            warnings.append(selected_tool.notes)
            confidence_reason = selected_tool.notes
        elif plan.detected_task in (TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA):
            assert secondary_metadata is not None and pair_compatibility is not None
            change_analysis_result = await _change_query_result(
                primary_result,
                secondary_result,
                pair_compatibility,
                primary_date or "",
                secondary_date or "",
                primary_modality,
            )
            controlled = answer_change_question(query, change_analysis_result)
            answer = controlled.answer
            vqa_details = controlled.details
            response_confidence = controlled.details.confidence
            confidence_reason = controlled.details.confidence.reason
            evidence_label = "TTP learned and deterministic supporting evidence" if change_analysis_result.change_engine and change_analysis_result.change_engine.mode in {"hybrid", "ttp"} else "Deterministic change evidence"
            response_evidence = [EvidenceItem(type="change_product", label=evidence_label, reference=reference) for reference in controlled.details.evidence_references]
            warnings = _unique(warnings + change_analysis_result.warnings)
            primary_metadata = change_analysis_result.before_metadata
            secondary_metadata = change_analysis_result.after_metadata
            steps = list(change_analysis_result.execution.steps)
            plan = plan.model_copy(update={
                "selected_tools": change_analysis_result.execution.selected_tools,
                "selection_reason": change_analysis_result.execution.selection_reason,
            })
            steps.insert(max(0, len(steps) - 1), ExecutionStep(tool="controlled_answer_generation", status=ToolStatus.SUCCESS if controlled.details.supported else ToolStatus.FAILED, duration_ms=0, parameters={"question_category": controlled.details.question_category.value, "language_model": False}))
            if not controlled.details.supported:
                status = ResponseStatus.NOT_IMPLEMENTED
            elif change_analysis_result.status == ChangeAnalysisStatus.SUCCESS:
                status = ResponseStatus.SUCCESS
            elif change_analysis_result.status == ChangeAnalysisStatus.ALIGNMENT_REQUIRED:
                status = ResponseStatus.ALIGNMENT_REQUIRED
            else:
                status = ResponseStatus.FAILED
        elif plan.detected_task == TaskType.CROSS_MODAL_ANALYSIS:
            assert secondary_modality is not None and secondary_metadata is not None
            assert pair_compatibility is not None
            modality_started = time.perf_counter()
            optical_result, sar_result = _order_cross_modal_inputs(
                primary_result,
                primary_modality,
                secondary_result,
                secondary_modality,
            )
            modality_duration = max(0, round((time.perf_counter() - modality_started) * 1000))
            cross_modal_result = await _execute_cross_modal(optical_result, sar_result, pair_compatibility)
            steps = _cross_modal_steps(
                optical_result,
                sar_result,
                pair_compatibility,
                cross_modal_result,
                modality_duration,
                pair_compatibility_duration,
            )
            optical_effective_modality = coarse_modality(optical_result.metadata.effective_modality)
            if get_sve_service().enabled and _sve_optical_eligible(optical_result.metadata, optical_effective_modality):
                sve_call = await run_in_threadpool(
                    get_sve_service().analyze,
                    optical_result.model_image,
                    optical_result.content_hash,
                )
                raw_sar_skip = get_sve_service().unsupported_raw_sar_result()
                sve_result = sve_call.result.model_copy(update={
                    "semantic_comparison": raw_sar_skip.semantic_comparison,
                    "warning": raw_sar_skip.warning if sve_call.result.available else sve_call.result.warning,
                })
                cross_modal_result = cross_modal_result.model_copy(update={"sve_result": sve_result})
                steps.extend(_sve_execution_steps(sve_call))
                steps.append(ExecutionStep(
                    tool="sve_eligibility_check",
                    status=ToolStatus.SKIPPED,
                    duration_ms=0,
                    parameters={"input": "raw_sar", "reason": "validated_generated_rgb_required"},
                ))
            controlled = answer_cross_modal_question(query, cross_modal_result)
            answer = controlled.answer
            vqa_details = controlled.details
            response_confidence = controlled.details.confidence
            warnings = _unique(warnings + cross_modal_result.warnings)
            status = _response_status(cross_modal_result) if controlled.details.supported else ResponseStatus.NOT_IMPLEMENTED
            confidence_reason = controlled.details.confidence.reason
            response_evidence = [
                EvidenceItem(type="preview", label="Optical evidence", reference=cross_modal_result.previews.optical_evidence),
                EvidenceItem(type="preview", label="SAR evidence", reference=cross_modal_result.previews.sar_evidence),
                EvidenceItem(type="preview", label="Joint evidence", reference=cross_modal_result.previews.joint_evidence),
            ] if cross_modal_result.statistics is not None else []
        elif plan.detected_task == TaskType.CAPTIONING:
            steps.append(ExecutionStep(tool="caption_input_validation", status=ToolStatus.SUCCESS, duration_ms=0))
            try:
                caption_result = await run_in_threadpool(
                    get_captioner().describe,
                    primary_result.model_image,
                    primary_metadata,
                    primary_modality,
                    primary_result.bands_used,
                    primary_result.image_representation,
                )
                steps.extend(
                    [
                        ExecutionStep(tool="model_loading_or_reuse", status=ToolStatus.SUCCESS, duration_ms=caption_result.model_load_ms),
                        ExecutionStep(tool="image_preparation", status=ToolStatus.SUCCESS, duration_ms=0),
                        ExecutionStep(tool="caption_inference", status=ToolStatus.SUCCESS, duration_ms=caption_result.runtime_ms),
                        ExecutionStep(tool=selected_tool.id, status=ToolStatus.SUCCESS, duration_ms=caption_result.runtime_ms),
                        ExecutionStep(tool="response_generation", status=ToolStatus.SUCCESS, duration_ms=0),
                    ]
                )
                answer = caption_result.caption
                response_confidence = caption_result.confidence
                model_provenance = caption_result.model
                caption_details = CaptionDetails(
                    modality=primary_modality,
                    device=caption_result.device,
                    runtime_ms=caption_result.runtime_ms,
                    model_load_ms=caption_result.model_load_ms,
                    model_reused=caption_result.reused_model,
                    image_representation=caption_result.image_representation,
                    bands_used=caption_result.bands_used,
                    limitations=get_captioner().limitations,
                )
                warnings.extend(caption_result.warnings)
                status = ResponseStatus.PARTIAL if caption_result.confidence.level == ConfidenceLevel.LOW else ResponseStatus.SUCCESS
                confidence_reason = caption_result.confidence.reason
            except CaptionerError as error:
                unsupported = error.code in {"UNSUPPORTED_CAPTION_MODALITY", "UNSUPPORTED_CAPTION_BANDS", "CAPTIONER_UNAVAILABLE"}
                steps.append(
                    ExecutionStep(
                        tool=selected_tool.id,
                        status=ToolStatus.NOT_IMPLEMENTED if unsupported else ToolStatus.FAILED,
                        duration_ms=0,
                    )
                )
                status = ResponseStatus.NOT_IMPLEMENTED if unsupported else ResponseStatus.FAILED
                warnings.append(error.message)
                confidence_reason = error.message
        else:
            steps.append(ExecutionStep(tool=selected_tool.id, status=ToolStatus.SKIPPED, duration_ms=0))
            status = ResponseStatus.NOT_IMPLEMENTED
            warnings.append("The selected tool has no SatQuery execution adapter yet.")
            confidence_reason = "Routing succeeded, but execution is not connected yet."

    if (
        input_mode == InputMode.SINGLE
        and get_sve_service().enabled
        and _sve_optical_eligible(primary_metadata, primary_modality)
        and plan.detected_task in {TaskType.CAPTIONING, TaskType.VQA, TaskType.GROUNDING}
        and result_status != "NEEDS_USER_CONFIRMATION"
    ):
        sve_call = await run_in_threadpool(
            get_sve_service().analyze,
            primary_result.model_image,
            primary_result.content_hash,
            captions=[answer] if plan.detected_task == TaskType.CAPTIONING and answer else [],
            vqa_category=vqa_details.question_category.value if vqa_details and vqa_details.supported else None,
            vqa_answer=answer if vqa_details and vqa_details.supported else None,
            grounding_target=grounding_result.target_phrase if grounding_result else None,
        )
        sve_result = sve_call.result
        steps.extend(_sve_execution_steps(sve_call))
        if sve_result.caption_consistency and sve_result.caption_consistency.reranked:
            answer = sve_result.caption_consistency.original_candidates[sve_result.caption_consistency.selected_candidate_index]
        if sve_result.vqa_consistency and sve_result.vqa_consistency.state == "disagreement":
            warnings.append(sve_result.vqa_consistency.explanation)
        if sve_result.grounding_support and sve_result.grounding_support.warning:
            warnings.append(sve_result.grounding_support.warning)
        if sve_result.warning:
            warnings.append(sve_result.warning)

    permitted_parameters = dict(plan.permitted_parameters)
    permitted_parameters.update({
        "primary_date": primary_date,
        "secondary_date": secondary_date,
        "representation": primary_metadata.representation.value,
        "auto_detected_modality": primary_metadata.auto_detected_modality.value,
        "user_confirmed_modality": primary_metadata.user_confirmed_modality.value if primary_metadata.user_confirmed_modality else None,
        "effective_modality": primary_metadata.effective_modality.value,
    })
    if caption_details is not None:
        permitted_parameters.update(
            {
                "max_new_tokens": MAX_NEW_TOKENS,
                "beam_count": NUM_BEAMS,
                "temperature": None,
                "device": caption_details.device,
                "image_representation": caption_details.image_representation,
                "selected_bands": caption_details.bands_used,
            }
        )
    if grounding_result is not None:
        permitted_parameters.update(
            {
                **safe_grounding_parameters(),
                "checkpoint": grounding_result.model.checkpoint,
                "model_input_width": grounding_result.input.model_input_width,
                "model_input_height": grounding_result.input.model_input_height,
                "target_phrase": grounding_result.target_phrase,
                "mask_refinement": False,
                "minimum_reliable_alignment_score": grounding_result.quality_policy.minimum_alignment_score,
                "maximum_localized_area_ratio": grounding_result.quality_policy.maximum_localized_area_ratio,
                "quality_policy_label": grounding_result.quality_policy.calibration_status,
            }
        )
    if sar_water_result is not None:
        permitted_parameters.update({
            "execution_method": sar_water_result.method,
            "specialist_version": sar_water_result.method_version,
            "preprocessing_version": sar_water_result.preprocessing.version,
            "normalized_threshold": sar_water_result.threshold,
            "model_checkpoint": None,
            "model_confidence": None,
            "heuristic_reliability": sar_water_result.heuristic_reliability,
            "geographic_area_reported": sar_water_result.geographic_area_square_meters is not None,
        })
    if result_status == "COMPLETED" and status == ResponseStatus.NOT_IMPLEMENTED:
        result_status = "UNSUPPORTED_TASK"
    elif result_status == "COMPLETED" and status == ResponseStatus.FAILED:
        result_status = "INFERENCE_FAILED"
    if response_confidence.level == ConfidenceLevel.UNAVAILABLE:
        response_confidence = Confidence(level=ConfidenceLevel.UNAVAILABLE, score=None, reason=confidence_reason)
    primary_result.model_image.close()
    if pair_mode and secondary_image is not None:
        secondary_result.model_image.close()
    response = AgentResponse(
        request_id=str(uuid.uuid4()),
        task=plan.detected_task,
        answer=answer,
        confidence=response_confidence,
        evidence=response_evidence,
        execution=ExecutionSummary(
            input_mode=input_mode,
            selected_tools=plan.selected_tools,
            steps=steps,
            duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
            permitted_parameters=permitted_parameters,
            validation=validation,
            selection_reason=plan.selection_reason,
        ),
        warnings=_unique(warnings),
        status=status,
        result_status=result_status,
        primary_image_metadata=primary_metadata,
        secondary_image_metadata=secondary_metadata,
        pair_compatibility=pair_compatibility,
        model=model_provenance,
        caption_details=caption_details,
        grounding_result=grounding_result,
        cross_modal_analysis=cross_modal_result,
        change_analysis=change_analysis_result,
        vqa_details=vqa_details,
        sar_water_analysis=sar_water_result,
        sar_scene_analysis=sar_scene_result,
        classified_query=classified_query,
        change_engine=change_analysis_result.change_engine if change_analysis_result else None,
        ttp_result=change_analysis_result.ttp_result if change_analysis_result else None,
        mask_comparison=change_analysis_result.mask_comparison if change_analysis_result else None,
        evidence_consistency=change_analysis_result.evidence_consistency if change_analysis_result else None,
        sve_result=sve_result or (change_analysis_result.sve_result if change_analysis_result else None),
    )
    storage_cache_key = analysis_cache_key
    if response.change_engine and response.change_engine.fallback_used:
        storage_cache_key = hashlib.sha256(
            f"{analysis_cache_key}|actual_engine=deterministic_fallback|reason={response.change_engine.fallback_reason}".encode("utf-8")
        ).hexdigest()
    response = mark_fresh(response, storage_cache_key)
    if result_is_reportable(response):
        MISSION_STORE.put(response, storage_cache_key, response.cache.original_generation_timestamp if response.cache else None)
        record_comparison_agent_result(
            response,
            primary_hash=primary_result.content_hash,
            secondary_hash=secondary_result.content_hash if pair_mode and secondary_metadata is not None else None,
            primary_modality=primary_modality,
            secondary_modality=secondary_modality if pair_mode else None,
            primary_date=primary_date,
            secondary_date=secondary_date,
            query=query,
        )
    record_agent_response(
        response,
        started_at=analytics_started_at,
        primary_modality=primary_modality.value,
        secondary_modality=secondary_modality.value if secondary_modality else None,
    )
    return response


def _change_ingestion_steps(durations: List[Dict[str, int]]) -> List[ExecutionStep]:
    return [
        ExecutionStep(
            tool="upload",
            status=ToolStatus.SUCCESS,
            duration_ms=sum(item.get("upload_received", 0) for item in durations),
        ),
        ExecutionStep(
            tool="metadata",
            status=ToolStatus.SUCCESS,
            duration_ms=sum(
                item.get("file_type_validation", 0)
                + item.get("metadata_extraction", 0)
                + item.get("preview_generation", 0)
                for item in durations
            ),
        ),
    ]


def _alignment_outcome(compatibility: PairCompatibility) -> bool:
    if compatibility.same_crs is False:
        return True
    if compatibility.bounds_overlap is False:
        return False
    return requires_alignment(compatibility) and compatibility.resampling_required


def _skipped_change_steps() -> List[ExecutionStep]:
    return [
        ExecutionStep(tool=name, status=ToolStatus.SKIPPED, duration_ms=0)
        for name in (
            "difference_computation",
            "thresholding",
            "morphology",
            "connected_components",
            "preview_generation",
        )
    ]


@router.post("/change", response_model=ChangeAnalysisResponse)
async def agent_change_analysis(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    before_date: str = Form(...),
    after_date: str = Form(...),
    modality: Modality = Form(Modality.OPTICAL),
) -> ChangeAnalysisResponse:
    """Compute deterministic evidence and, when eligible, TTP learned evidence."""
    analytics_started_at = analytics_utc_now()
    before_result = None
    after_result = None
    before_metadata: Optional[ImageMetadata] = None
    after_metadata: Optional[ImageMetadata] = None
    try:
        try:
            before_result = await ingest_upload(before_image)
            before_metadata = before_result.metadata
            after_result = await ingest_upload(after_image)
            after_metadata = after_result.metadata
        except ImageIngestionError as error:
            remove_preview(before_metadata)
            remove_preview(after_metadata)
            _raise_ingestion_error(error)

        assert before_result is not None and after_result is not None
        assert before_metadata is not None and after_metadata is not None
        compatibility = validate_pair_compatibility(
            input_mode=InputMode.BI_TEMPORAL,
            primary_modality=modality,
            secondary_modality=modality,
            primary=before_metadata,
            secondary=after_metadata,
            primary_date=before_date,
            secondary_date=after_date,
        )
        response = await _change_query_result(
            before_result, after_result, compatibility, before_date, after_date, modality
        )
        record_change_response(response, started_at=analytics_started_at, modality=modality.value)
        record_comparison_change_result(
            response,
            primary_hash=before_result.content_hash,
            secondary_hash=after_result.content_hash,
            modality=modality,
        )
        return response
    except ChangeAnalysisError as error:
        remove_preview(before_metadata)
        remove_preview(after_metadata)
        raise HTTPException(
            status_code=422,
            detail={"code": error.code, "message": error.message},
        ) from error
    finally:
        if before_result is not None:
            before_result.model_image.close()
        if after_result is not None:
            after_result.model_image.close()

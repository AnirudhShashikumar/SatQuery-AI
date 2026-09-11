"""Process-local, metadata-only research analytics for SatQuery.

No upload content, query text, filename, filesystem path, answer text, or model
output is retained here. The store intentionally resets when the backend does.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional

from .models import (
    AgentResponse,
    AnalyticsCache,
    AnalyticsCapability,
    AnalyticsDataset,
    AnalyticsExecution,
    AnalyticsPlatform,
    AnalyticsReports,
    AnalyticsResponse,
    AnalyticsSummary,
    AnalyticsSVE,
    AnalyticsTTP,
    AnalyticsTool,
    AnalyticsTraceStep,
    AnalyticsWorkflowMetric,
    ChangeAnalysisResponse,
    CrossModalAnalysisResponse,
    ImplementationStatus,
    ScientificTransparency,
)


MAX_HISTORY = 50
PROCESS_STARTED = time.monotonic()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _preview_count(value: Any) -> int:
    found = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and item.startswith("/api/agent/previews/"):
            found.add(item)

    visit(value)
    return len(found)


def _device_from_response(response: AgentResponse) -> Optional[str]:
    if response.sve_result is not None and response.sve_result.device:
        return response.sve_result.device
    if response.grounding_result is not None:
        return response.grounding_result.device
    if response.caption_details is not None:
        return response.caption_details.device
    return None


_SAFE_PARAMETER_KEYS = {
    "alignment_level", "connectivity", "detections", "device", "filesystem_paths_exposed",
    "format", "ground_truth_claimed", "image_representation", "language_model",
    "language_model_used", "mask_refinement", "max_analysis_dimension",
    "maximum_detections", "minimum_region_pixels", "model_input_height",
    "model_input_width", "normalization", "registration", "registration_performed",
    "reprojection", "resampling", "reused", "selected_bands", "template_generated_summary",
    "eligible", "reason", "checkpoint_verified", "width", "height", "binary",
    "changed_pixels", "region_count", "iou", "agreement_percentage", "primary_mask",
    "supporting_mask", "mask_averaging", "reused_model", "exact_source_bytes",
    "checksum_verified", "cache", "first_cache", "second_cache", "top_labels",
    "calibrated", "candidate_count", "selected_candidate_index", "consistency_state",
    "support_state", "input", "transform", "comparison", "influence_applied",
}
_LOCAL_PATH = re.compile(r"(?:/Users/|/home/|/root/|[A-Za-z]:\\)[^\s,;]+")


def _safe_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return _LOCAL_PATH.sub("[redacted local path]", value)[:600]


def _safe_trace_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in parameters.items():
        if key not in _SAFE_PARAMETER_KEYS:
            continue
        if isinstance(value, str):
            output[key] = _safe_text(value)
        elif value is None or isinstance(value, (bool, int, float)):
            output[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            output[key] = [(_safe_text(item) if isinstance(item, str) else item) for item in value[:12]]
    return output


def _safe_trace(steps: Iterable[Any]) -> List[AnalyticsTraceStep]:
    return [
        AnalyticsTraceStep(
            tool=str(step.tool),
            status=getattr(step.status, "value", str(step.status)),
            duration_ms=max(0, int(step.duration_ms)),
            parameters=_safe_trace_parameters(dict(getattr(step, "parameters", {}) or {})),
        )
        for step in steps
    ]


class AnalyticsStore:
    def __init__(self) -> None:
        self._history: Deque[AnalyticsExecution] = deque(maxlen=MAX_HISTORY)
        self._total_executions = 0
        self._successful_executions = 0
        self._report_requests = 0
        self._report_artifacts = 0
        self._report_formats: Counter[str] = Counter()
        self._lock = threading.RLock()

    def record(self, execution: AnalyticsExecution) -> None:
        with self._lock:
            self._history.appendleft(execution.model_copy(deep=True))
            self._total_executions += 1
            if execution.status in {"success", "partial"}:
                self._successful_executions += 1

    def record_report(self, request_id: str, formats: Iterable[str]) -> None:
        normalized = [getattr(value, "value", str(value)) for value in formats]
        with self._lock:
            self._report_requests += 1
            self._report_artifacts += len(normalized)
            self._report_formats.update(normalized)
            for index, execution in enumerate(self._history):
                if execution.request_id == request_id:
                    self._history[index] = execution.model_copy(update={"report_generated": True})
                    break

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "history": [item.model_copy(deep=True) for item in self._history],
                "total_executions": self._total_executions,
                "successful_executions": self._successful_executions,
                "report_requests": self._report_requests,
                "report_artifacts": self._report_artifacts,
                "report_formats": dict(self._report_formats),
            }

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._total_executions = 0
            self._successful_executions = 0
            self._report_requests = 0
            self._report_artifacts = 0
            self._report_formats.clear()


ANALYTICS_STORE = AnalyticsStore()


def record_agent_response(
    response: AgentResponse,
    *,
    started_at: str,
    primary_modality: Optional[str],
    secondary_modality: Optional[str],
) -> None:
    ANALYTICS_STORE.record(
        AnalyticsExecution(
            request_id=response.request_id,
            started_at=started_at,
            completed_at=utc_now(),
            task=response.task.value,
            input_mode=response.execution.input_mode.value,
            primary_modality=primary_modality,
            secondary_modality=secondary_modality,
            status=response.status.value,
            selected_tools=list(response.execution.selected_tools),
            duration_ms=response.execution.duration_ms,
            warning_count=len(response.warnings),
            output_count=_preview_count(response.model_dump(mode="json")),
            cache_status="hit" if response.cache and response.cache.cached else "fresh",
            device=_device_from_response(response),
            selection_reason=_safe_text(response.execution.selection_reason),
            confidence_level=response.confidence.level.value,
            confidence_reason=_safe_text(response.confidence.reason),
            warnings=[safe for warning in response.warnings if (safe := _safe_text(warning))],
            trace=_safe_trace(response.execution.steps),
        )
    )


def record_change_response(response: ChangeAnalysisResponse, *, started_at: str, modality: str) -> None:
    ANALYTICS_STORE.record(
        AnalyticsExecution(
            request_id=response.request_id,
            started_at=started_at,
            completed_at=utc_now(),
            task="change_analysis",
            input_mode=response.execution.input_mode.value,
            primary_modality=modality,
            secondary_modality=modality,
            status=response.status.value,
            selected_tools=list(response.execution.selected_tools),
            duration_ms=response.execution.duration_ms,
            warning_count=len(response.warnings),
            output_count=_preview_count(response.model_dump(mode="json")),
            cache_status="not_requested",
            selection_reason=_safe_text(response.execution.selection_reason),
            warnings=[safe for warning in response.warnings if (safe := _safe_text(warning))],
            trace=_safe_trace(response.execution.steps),
        )
    )


def record_cross_modal_response(
    response: CrossModalAnalysisResponse,
    *,
    started_at: str,
    optical_modality: str,
    sar_modality: str,
) -> None:
    ANALYTICS_STORE.record(
        AnalyticsExecution(
            request_id=response.request_id,
            started_at=started_at,
            completed_at=utc_now(),
            task="cross_modal_analysis",
            input_mode=response.execution.input_mode.value,
            primary_modality=optical_modality,
            secondary_modality=sar_modality,
            status=response.result.status.value,
            selected_tools=list(response.execution.selected_tools),
            duration_ms=response.execution.duration_ms,
            warning_count=len(response.result.warnings),
            output_count=_preview_count(response.model_dump(mode="json")),
            cache_status="not_requested",
            selection_reason=_safe_text(response.execution.selection_reason),
            confidence_level=response.result.confidence.level.value,
            confidence_reason=_safe_text(response.result.confidence.reason),
            warnings=[safe for warning in response.result.warnings if (safe := _safe_text(warning))],
            trace=_safe_trace(response.execution.steps),
        )
    )


def record_report_generation(request_id: str, formats: Iterable[Any]) -> None:
    ANALYTICS_STORE.record_report(request_id, formats)


def _hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "CUDA"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "Apple MPS"
    except Exception:
        pass
    return "CPU"


def _workflow_metrics(history: List[AnalyticsExecution]) -> List[AnalyticsWorkflowMetric]:
    task_order = [
        "captioning", "vqa", "grounding", "change_analysis", "change_description",
        "change_vqa", "cross_modal_analysis",
    ]
    output: List[AnalyticsWorkflowMetric] = []
    for task in task_order:
        runs = [item for item in history if item.task == task]
        last = runs[0] if runs else None
        output.append(
            AnalyticsWorkflowMetric(
                task=task,
                executions=len(runs),
                successful_executions=sum(item.status in {"success", "partial"} for item in runs),
                average_runtime_ms=(round(sum(item.duration_ms for item in runs) / len(runs), 1) if runs else None),
                last_runtime_ms=last.duration_ms if last else None,
                last_device=last.device if last else None,
                last_completed_at=last.completed_at if last else None,
            )
        )
    return output


def _datasets() -> List[AnalyticsDataset]:
    return [
        AnalyticsDataset(
            name="LEVIR-CD",
            usage_status="Active Model Training Provenance",
            used_by=["ttp_change_detector"],
            purpose="Building-change training provenance for the official TTP epoch_260 checkpoint.",
            source="https://github.com/KyanChen/TTP",
            note="Training provenance is disclosed as a domain limitation; GeoVision does not claim universal change-detection accuracy.",
        ),
        AnalyticsDataset(
            name="RSICD",
            usage_status="Active",
            used_by=["rs_captioner"],
            purpose="Remote-sensing adaptation provenance for optical image captioning.",
            source="https://huggingface.co/Gurveer05/blip-image-captioning-base-rscid-finetuned",
            note="The connected BLIP checkpoint declares RSICD adaptation; no local sample count is asserted.",
        ),
        AnalyticsDataset(
            name="SAR-to-optical training corpus",
            usage_status="Undeclared",
            used_by=["pix2pix_reconstruction", "sarfusionformer_analysis"],
            purpose="Training provenance for the checked-in SAR reconstruction checkpoints.",
            note="The exact training dataset is not declared in the checked-in runtime metadata, so SEN12MS-CR is not claimed.",
        ),
        AnalyticsDataset(
            name="RSVQA",
            usage_status="Evaluation Candidate / Not Used",
            used_by=[],
            purpose="Potential future remote-sensing VQA evaluation or adaptation.",
            note="Not connected and not used to fine-tune the current deterministic VQA specialist.",
        ),
        AnalyticsDataset(
            name="BigEarthNet.txt / BigEarthNet v2 Lithuania Summer",
            usage_status="Active Model Adaptation Provenance",
            used_by=["satquery_vision_encoder_v1"],
            purpose="Image-text adaptation provenance for scene-level Earth-observation embeddings.",
            sample_count=4008,
            note="The recorded training split contains 4,008 pairs; validation and test contain 2,291 and 2,053 pairs. Retrieval improved over generic OpenCLIP on validation but remains low in absolute terms.",
        ),
        AnalyticsDataset(
            name="Grounding DINO model provenance",
            usage_status="Model Provenance",
            used_by=["rs_grounder"],
            purpose="Official zero-shot open-vocabulary detector checkpoint provenance.",
            source="https://huggingface.co/IDEA-Research/grounding-dino-tiny",
            note="This is model provenance, not a GeoVision project dataset or remote-sensing fine-tuning claim.",
        ),
    ]


def _capabilities() -> List[AnalyticsCapability]:
    rows = [
        ("Optical / multispectral ingestion", "Available", "Validated PNG, JPEG, TIFF, and GeoTIFF ingestion."),
        ("SAR ingestion", "Available", "Validated SAR metadata and preview path."),
        ("Raster metadata", "Available", "Rasterio-backed CRS, transform, bounds, dtype, and nodata metadata."),
        ("Optical captioning", "Available", "RSICD-adapted BLIP specialist is registered."),
        ("Controlled optical VQA", "Available", "Deterministic evidence-grounded question taxonomy."),
        ("Text-guided grounding boxes", "Available", "Local Grounding DINO box and alignment-score output."),
        ("Grounding masks", "Not Implemented", "Mask refinement is explicitly not connected."),
        ("Bi-temporal change analysis", "Available", "Normalized difference, mask, morphology, regions, and overlays."),
        ("TTP hybrid optical change analysis", "Available when CUDA service is ready", "TTP learned mask is primary; deterministic mask remains independent supporting evidence with explicit fallback."),
        ("Change VQA", "Available", "Controlled answers use measured change products only."),
        ("Optical–SAR joint analysis", "Available", "Exactly aligned deterministic evidence fusion."),
        ("Automatic image registration", "Not Implemented", "Alignment-required is returned without silent registration."),
        ("SAR captioning", "Unsupported", "The optical captioner does not accept SAR imagery."),
        ("Single-image SAR VQA", "Unsupported", "The controlled single-image VQA taxonomy is optical-only."),
        ("Deterministic routing", "Available", "Explicit task and specialist selection rules."),
        ("Evidence and execution traces", "Available", "Safe previews, regions, warnings, confidence rationale, and ordered steps."),
        ("Remote-sensing scene embeddings", "Available when SVE is ready", "SatQuery Vision Encoder v1 supplies scene priors, similarity, consistency, and routing support without replacing specialist models."),
        ("Mission reports", "Available", "Backend-authoritative PDF, JSON, CSV, and ZIP artifacts."),
        ("Offline demo workflow", "Available", "Environment-gated local sample workflow; current activation is reported separately."),
    ]
    return [AnalyticsCapability(name=name, status=status, evidence=evidence) for name, status, evidence in rows]


def build_analytics() -> AnalyticsResponse:
    from .compliance import compliance_summary
    from .demo import demo_manifest
    from .registry import public_tool_registry
    from .reporting import ARTIFACT_STORE, MISSION_STORE
    from .specialists.captioner import get_captioner
    from .specialists.grounder import get_grounder
    from .specialists.ttp_change import TTP_CLIENT, TTPClientError, ttp_enabled
    from .services.sve_service import get_sve_service

    state = ANALYTICS_STORE.snapshot()
    history: List[AnalyticsExecution] = state["history"]
    compliance = compliance_summary()
    registry = public_tool_registry()
    cache = MISSION_STORE.snapshot()
    artifacts = ARTIFACT_STORE.snapshot()
    caption_health = get_captioner().health()
    grounder_health = get_grounder().health()
    sve_service = get_sve_service()
    sve_health = sve_service.health()
    lifecycle = {"rs_captioner": caption_health, "rs_grounder": grounder_health, "satquery_vision_encoder_v1": sve_health}
    failed_specialists = [name for name, health in lifecycle.items() if health.status == "failed"]
    demo = demo_manifest()
    ttp_health: Dict[str, Any] = {}
    if ttp_enabled():
        try:
            ttp_health = TTP_CLIENT.health()
            ttp_status = str(ttp_health.get("status", "unavailable"))
        except TTPClientError:
            ttp_status = "unavailable"
    else:
        ttp_status = "disabled"
    offline_requirements: List[str] = []
    if caption_health.status != "ready":
        offline_requirements.append("Load and verify the RSICD caption checkpoint before disconnecting if captioning will be used.")
    if grounder_health.status != "ready":
        offline_requirements.append("Load and verify the Grounding DINO checkpoint before disconnecting if grounding will be used.")
    if sve_health.status != "ready":
        offline_requirements.append("Load and verify SatQuery Vision Encoder v1 and its OpenCLIP backbone before disconnecting if scene embedding evidence will be used.")
    if not demo.enabled:
        offline_requirements.append("Set SATQUERY_DEMO_MODE=true to expose approved local demo samples.")
    offline_ready = not offline_requirements
    if ttp_enabled() and ttp_status != "ready":
        offline_requirements.append("Start, verify, and warm the persistent TTP CUDA service before the hybrid presentation workflow.")
        offline_ready = False

    last_by_tool: Dict[str, AnalyticsExecution] = {}
    tool_aliases = {"deterministic_change_analysis": "bitemporal_change_analyzer"}
    for execution in history:
        for tool_id in execution.selected_tools:
            last_by_tool.setdefault(tool_aliases.get(tool_id, tool_id), execution)
    tools = []
    for tool in registry:
        health = lifecycle.get(tool.id)
        last = last_by_tool.get(tool.id)
        tools.append(
            AnalyticsTool(
                id=tool.id,
                display_name=tool.display_name,
                implementation_status=tool.status.value,
                lifecycle_status=health.status if health else ("registered" if tool.status == ImplementationStatus.AVAILABLE else "not_implemented"),
                device=health.device if health else (last.device if last else None),
                last_runtime_ms=last.duration_ms if last else None,
                last_completed_at=last.completed_at if last else None,
                method_type=tool.method_type,
                checkpoint=tool.checkpoint,
                adaptation_dataset=tool.adaptation_dataset,
                remote_sensing_adapted=tool.remote_sensing_adapted,
                service_path=tool.service_path,
            )
        )

    hits = cache["hits"]
    misses = cache["misses"]
    lookups = hits + misses
    ttp_metrics = TTP_CLIENT.metrics()
    fallback_count = sum(
        1 for execution in history for step in execution.trace if step.tool == "deterministic_fallback"
    )
    ious = [
        float(step.parameters["iou"])
        for execution in history for step in execution.trace
        if step.tool == "mask_comparison" and isinstance(step.parameters.get("iou"), (int, float))
    ]
    runtime_versions = {
        name: _version(package)
        for name, package in {
            "fastapi": "fastapi", "torch": "torch", "torchvision": "torchvision",
            "transformers": "transformers", "rasterio": "rasterio",
            "open_clip": "open-clip-torch", "timm": "timm",
        }.items()
    }
    return AnalyticsResponse(
        generated_at=utc_now(),
        platform=AnalyticsPlatform(
            backend_status="degraded" if failed_specialists else "healthy",
            uptime_seconds=max(0, int(time.monotonic() - PROCESS_STARTED)),
            python_version=platform.python_version(),
            operating_system=platform.system(),
            architecture=platform.machine(),
            process_memory_mb=None,
            runtime_versions=runtime_versions,
            hardware_acceleration=_hardware(),
            demo_mode=demo.enabled,
            offline_ready=offline_ready,
            offline_readiness_requirements=offline_requirements,
        ),
        summary=AnalyticsSummary(
            total_executions_current_process=state["total_executions"],
            successful_executions_current_process=state["successful_executions"],
            registered_tools=len(registry),
            available_tools=sum(tool.status == ImplementationStatus.AVAILABLE for tool in registry),
            mandatory_satisfied=compliance.mandatory_satisfied,
            mandatory_total=compliance.mandatory_total,
            cache_hits_current_process=hits,
            cache_misses_current_process=misses,
            report_artifacts_generated_current_process=state["report_artifacts"],
        ),
        cache=AnalyticsCache(
            stored_results=cache["stored_results"],
            max_results=cache["max_results"],
            ttl_seconds=cache["ttl_seconds"],
            hits=hits,
            misses=misses,
            hit_rate_percent=round(hits * 100 / lookups, 1) if lookups else None,
        ),
        reports=AnalyticsReports(
            requests_generated_current_process=state["report_requests"],
            artifacts_generated_current_process=state["report_artifacts"],
            artifacts_currently_available=artifacts["stored_artifacts"],
            formats=state["report_formats"],
        ),
        tools=tools,
        datasets=_datasets(),
        capabilities=_capabilities(),
        workflow_metrics=_workflow_metrics(history),
        recent_executions=history,
        last_execution_trace=history[0].trace if history else [],
        scientific_transparency=ScientificTransparency(
            metric_source="Current backend process, public registry, compliance matrix, and completed SatQuery execution summaries.",
            history_retention="Metadata-only, newest first, maximum 50 executions; cleared on backend restart.",
            unavailable_value_policy="Unavailable or unmeasured values are null or explicitly labelled; zero is used only for measured counters.",
            caveats=[
                "No benchmark accuracy, dataset sample count, model quality score, or unexecuted runtime is inferred.",
                "Process memory is null because a safe current-RSS provider is not installed.",
                "Tool availability describes the registered implementation; lifecycle status separately reports lazy model state.",
                "History excludes uploaded content, filenames, queries, answers, hashes, paths, credentials, and environment values.",
            ],
        ),
        sve=AnalyticsSVE(**sve_service.metrics()),
        ttp=AnalyticsTTP(
            enabled=ttp_enabled(),
            service_status=ttp_status,
            model_load_count=int(ttp_health.get("model_load_count", 0)),
            model_reuse_count=int(ttp_metrics["model_reuse_count"] or 0),
            inference_count=int(ttp_metrics["inference_count"] or 0),
            failure_count=int(ttp_metrics["failure_count"] or 0),
            timeout_count=int(ttp_metrics["timeout_count"] or 0),
            oom_count=int(ttp_metrics["oom_count"] or 0),
            fallback_count=fallback_count,
            average_runtime_ms=ttp_metrics["average_runtime_ms"],
            average_mask_iou=round(sum(ious) / len(ious), 6) if ious else None,
        ),
    )

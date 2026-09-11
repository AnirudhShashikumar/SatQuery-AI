"""Backend-authoritative, bounded mission comparison records and reports."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .image_ingestion import preview_file_path, remove_preview_url, save_preview
from .models import (
    AgentResponse,
    ChangeAnalysisResponse,
    ComparabilityLevel,
    ComparabilityResult,
    ComparisonAssessmentResponse,
    ComparisonItem,
    ComparisonItemSummary,
    ComparisonLineageNode,
    ComparisonPreview,
    ComparisonReportResponse,
    ComparisonTask,
    CrossModalAnalysisResponse,
    InputMode,
    Modality,
    ReportArtifact,
    ReportFormat,
)
from .reporting import ARTIFACT_STORE, SCHEMA_VERSION
from .evidence_lifecycle import authoritative_translation_products_dict


DEFAULT_MAX_ITEMS = 32
DEFAULT_TTL_SECONDS = 30 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = " ".join(value.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def composite_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(":".join(values).encode("ascii")).hexdigest()


def _configured_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _safe(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(token in lowered for token in ("api_key", "secret", "password", "token", "local_path", "cache_path", "original_name", "safe_name", "filename")):
        return None
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        for name, nested in value.items():
            cleaned = _safe(nested, name)
            if cleaned is not None:
                output[name] = cleaned
        return output
    if isinstance(value, list):
        return [_safe(nested) for nested in value]
    if isinstance(value, str):
        if value.startswith(("/Users/", "/home/", "/root/", "C:\\")):
            return "[redacted local path]"
        return re.sub(r"(?i)(api[_ -]?key|secret|password|token)\s*[:=]\s*\S+", r"\1=[redacted]", value)
    return value


@dataclass
class ComparisonIdentity:
    primary_hash: str
    secondary_hash: Optional[str] = None
    component_hashes: Tuple[str, ...] = ()
    query_hash: Optional[str] = None
    target_hash: Optional[str] = None
    dates: Tuple[Optional[str], Optional[str]] = (None, None)
    safe_parameters: Dict[str, Any] = field(default_factory=dict)
    task_family: str = ""
    output_hashes: Tuple[str, ...] = ()

    def all_input_hashes(self) -> set[str]:
        return {value for value in (self.primary_hash, self.secondary_hash, *self.component_hashes) if value}

    def public(self) -> Dict[str, Any]:
        return {
            "primary_hash_prefix": self.primary_hash[:12],
            "secondary_hash_prefix": self.secondary_hash[:12] if self.secondary_hash else None,
            "component_hash_prefixes": [value[:12] for value in self.component_hashes],
            "dates": [value for value in self.dates if value],
            "safe_parameters": _safe(self.safe_parameters),
            "identity_basis": "SHA-256 content hashes, modality, dates, safe parameters, and task family; filenames are excluded.",
        }


@dataclass
class ComparisonRecord:
    item: ComparisonItem
    identity: ComparisonIdentity
    touched_at: float
    owned_preview_urls: Tuple[str, ...] = ()


class ComparisonStore:
    """Bounded process-local store with tombstones for explicit expiry responses."""

    def __init__(self) -> None:
        self.max_items = _configured_int("SATQUERY_COMPARISON_MAX_ITEMS", DEFAULT_MAX_ITEMS, 1)
        self.ttl_seconds = _configured_int("SATQUERY_COMPARISON_TTL_SECONDS", DEFAULT_TTL_SECONDS, 60)
        self._records: "OrderedDict[str, ComparisonRecord]" = OrderedDict()
        self._expired: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.RLock()

    def _discard(self, request_id: str, *, expired: bool = True) -> None:
        record = self._records.pop(request_id, None)
        if record is None:
            return
        for url in record.owned_preview_urls:
            remove_preview_url(url)
        if expired:
            self._expired[request_id] = time.time()
            self._expired.move_to_end(request_id)
            while len(self._expired) > self.max_items * 2:
                self._expired.popitem(last=False)

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for request_id, record in list(self._records.items()):
            if record.touched_at < cutoff:
                self._discard(request_id)
        while len(self._records) > self.max_items:
            self._discard(next(iter(self._records)))

    def put(self, item: ComparisonItem, identity: ComparisonIdentity, owned_preview_urls: Iterable[str] = ()) -> None:
        with self._lock:
            self._cleanup()
            if item.request_id in self._records:
                previous = self._records[item.request_id]
                for url in previous.owned_preview_urls:
                    if url not in owned_preview_urls:
                        remove_preview_url(url)
            item = ComparisonItem.model_validate(_safe(item.model_dump(mode="json")))
            item = item.model_copy(update={"input_identity": identity.public()})
            self._records[item.request_id] = ComparisonRecord(
                item=item.model_copy(deep=True),
                identity=copy.deepcopy(identity),
                touched_at=time.time(),
                owned_preview_urls=tuple(owned_preview_urls),
            )
            self._records.move_to_end(item.request_id)
            self._expired.pop(item.request_id, None)
            self._cleanup()

    def _lineage(self, record: ComparisonRecord) -> List[ComparisonLineageNode]:
        lineage: List[ComparisonLineageNode] = []
        for candidate in self._records.values():
            if candidate.item.request_id == record.item.request_id:
                continue
            if record.identity.primary_hash in candidate.identity.output_hashes:
                lineage.extend(candidate.item.lineage)
                lineage.append(ComparisonLineageNode(kind="derived_output", label=f"Derived from {candidate.item.display_name}", request_id=candidate.item.request_id, preview_url=(candidate.item.output_previews[0].url if candidate.item.output_previews else None)))
                break
        lineage.extend(record.item.lineage)
        seen = set()
        return [node for node in lineage if not ((node.kind, node.label, node.request_id) in seen or seen.add((node.kind, node.label, node.request_id)))]

    def get(self, request_id: str, *, touch: bool = True) -> Tuple[str, Optional[ComparisonRecord]]:
        with self._lock:
            self._cleanup()
            record = self._records.get(request_id)
            if record is None:
                return ("expired" if request_id in self._expired else "missing"), None
            if touch:
                record.touched_at = time.time()
                self._records.move_to_end(request_id)
            result = copy.deepcopy(record)
            result.item = result.item.model_copy(update={"lineage": self._lineage(record)})
            return "available", result

    def list(self) -> List[ComparisonItemSummary]:
        with self._lock:
            self._cleanup()
            output: List[ComparisonItemSummary] = []
            for record in reversed(self._records.values()):
                item = record.item
                thumbnail = (item.output_previews or item.input_previews or [None])[0]
                output.append(ComparisonItemSummary(
                    query=item.query,
                    answer_summary=item.answer,
                    request_id=item.request_id,
                    task=item.task,
                    display_name=item.display_name,
                    status=item.status,
                    created_at=item.created_at,
                    input_mode=item.input_mode,
                    modalities=item.modalities,
                    thumbnail=thumbnail,
                    execution_duration_ms=item.execution_duration_ms,
                    cached=item.cached,
                    report_available=item.report_available,
                    warning_count=len(item.warnings),
                    evidence_product_count=len(item.output_previews),
                ))
            return output

    def records(self, request_ids: Sequence[str]) -> List[ComparisonRecord]:
        records: List[ComparisonRecord] = []
        for request_id in request_ids:
            state, record = self.get(request_id)
            if record is None:
                raise LookupError(f"{state}:{request_id}")
            records.append(record)
        return records

    def mark_report(self, request_id: str) -> None:
        with self._lock:
            self._cleanup()
            record = self._records.get(request_id)
            if record:
                record.item = record.item.model_copy(update={"report_available": True})

    def clear(self) -> None:
        with self._lock:
            for request_id in list(self._records):
                self._discard(request_id, expired=False)
            self._expired.clear()

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._cleanup()
            return {"stored_results": len(self._records), "max_results": self.max_items, "ttl_seconds": self.ttl_seconds}


COMPARISON_STORE = ComparisonStore()


DISPLAY_NAMES = {
    ComparisonTask.CAPTIONING: "Optical Caption",
    ComparisonTask.VQA: "Controlled VQA",
    ComparisonTask.GROUNDING: "Grounding DINO",
    ComparisonTask.CHANGE: "Bi-Temporal Change Analysis",
    ComparisonTask.CHANGE_VQA: "Change VQA",
    ComparisonTask.CROSS_MODAL: "Optical–SAR Joint Analysis",
    ComparisonTask.PIX2PIX: "Pix2Pix Reconstruction",
    ComparisonTask.SARFUSIONFORMER: "SARFusionFormer Reconstruction",
    ComparisonTask.SAR_ANALYSIS: "Single-Image SAR Analysis",
}


def _task(value: str) -> ComparisonTask:
    alias = {
        "change_description": ComparisonTask.CHANGE,
        "cross_modal_analysis": ComparisonTask.CROSS_MODAL,
        "sar_water_segmentation": ComparisonTask.SAR_ANALYSIS,
        "sar_scene_analysis": ComparisonTask.SAR_ANALYSIS,
        "sar_quality_inspection": ComparisonTask.SAR_ANALYSIS,
    }.get(value)
    return alias or ComparisonTask(value)


def _preview(label: str, url: Optional[str], kind: str, metadata: Optional[Dict[str, Any]] = None, modality: Optional[Modality] = None) -> Optional[ComparisonPreview]:
    if not url:
        return None
    metadata = metadata or {}
    return ComparisonPreview(source_observation_id=metadata.get("file_id"), source_role=metadata.get("observation_role"), source_modality=metadata.get("auto_detected_modality"), label=f"{metadata['observation_role'].title()} source" if metadata.get("observation_role") and kind == "input" else label, url=url, kind=kind, width=metadata.get("width"), height=metadata.get("height"), modality=modality)


def _unique_previews(values: Iterable[Optional[ComparisonPreview]]) -> List[ComparisonPreview]:
    output: List[ComparisonPreview] = []
    seen = set()
    for value in values:
        if value is not None and value.url not in seen:
            seen.add(value.url)
            output.append(value)
    return output


def _agent_outputs(response: Dict[str, Any]) -> List[ComparisonPreview]:
    values: List[Optional[ComparisonPreview]] = []
    grounding = response.get("grounding_result") or {}
    values.append(_preview("Annotated detections", grounding.get("annotated_preview_url"), "evidence"))
    for index, detection in enumerate(grounding.get("detections") or [], start=1):
        values.append(_preview(f"Detection mask {index}", detection.get("mask_url"), "evidence"))
    change = response.get("change_analysis") or {}
    change_previews = change.get("previews") or {}
    engine_mode = ((response.get("change_engine") or change.get("change_engine") or {}).get("mode"))
    preferred_change_order = (
        ["ttp_raw_mask", "ttp_mask", "deterministic_mask", "agreement", "disagreement", "intersection", "union", "ttp_overlay", "difference"]
        if engine_mode in {"hybrid", "ttp"}
        else ["deterministic_mask", "mask", "deterministic_overlay", "overlay", "difference"]
    )
    for name in preferred_change_order:
        values.append(_preview(name.replace("_", " ").title(), change_previews.get(name), "evidence"))
    cross = response.get("cross_modal_analysis") or {}
    for product in cross.get("evidence_products") or []:
        if product.get("reference") and product.get("evidence_type") != "source":
            values.append(ComparisonPreview(label=product["label"], url=product["reference"], kind="evidence", source_observation_id=product.get("source_observation_id"), source_role=product.get("source_role"), source_modality=product.get("source_modality"), evidence_id=product.get("evidence_id")))
    for name, url in (cross.get("previews") or {}).items():
        if name not in {"optical", "sar"}:
            values.append(_preview(name.replace("_", " ").title(), url, "evidence"))
    single = ((response.get("vqa_details") or {}).get("single_image_evidence") or {})
    for name, url in (single.get("previews") or {}).items():
        values.append(_preview(name.replace("_", " ").title(), url, "evidence"))
    for product in ((response.get("sar_water_analysis") or {}).get("evidence_products") or []) + ((response.get("sar_scene_analysis") or {}).get("evidence_products") or []):
        values.append(_preview(product.get("label") or "SAR evidence", product.get("reference"), "evidence"))
    for product in authoritative_translation_products_dict(response):
        values.append(ComparisonPreview(
            label=product.get("label") or "Generated optical-like supporting evidence",
            url=product["reference"],
            kind="evidence",
            source_observation_id=product.get("source_observation_id"),
            source_role=product.get("source_role"),
            source_modality=product.get("source_modality"),
            evidence_id=product.get("evidence_id"),
        ))
    return _unique_previews(values)


def _agent_statistics(response: Dict[str, Any]) -> Dict[str, Any]:
    grounding = response.get("grounding_result") or {}
    change = response.get("change_analysis") or {}
    cross = response.get("cross_modal_analysis") or {}
    single = (((response.get("vqa_details") or {}).get("single_image_evidence")) or {})
    sar_water = response.get("sar_water_analysis") or {}
    sar_scene = response.get("sar_scene_analysis") or {}
    statistics: Dict[str, Any] = {}
    if grounding:
        statistics["grounding"] = {"detection_count": len(grounding.get("detections") or []), "top_alignment_score": (grounding.get("detections") or [{}])[0].get("score")}
    if change.get("statistics"):
        statistics["change"] = change["statistics"]
    if change.get("deterministic_statistics"):
        statistics["deterministic_change"] = change["deterministic_statistics"]
    if data_engine := (response.get("change_engine") or change.get("change_engine")):
        statistics["change_engine"] = data_engine
    if ttp := (response.get("ttp_result") or change.get("ttp_result")):
        statistics["ttp_change"] = ttp
    if mask_comparison := (response.get("mask_comparison") or change.get("mask_comparison")):
        statistics["mask_comparison"] = mask_comparison
    if semantic_summary := (response.get("semantic_change_summary") or change.get("semantic_change_summary")):
        statistics["semantic_change_summary"] = semantic_summary
    if cross.get("statistics"):
        statistics["cross_modal"] = cross["statistics"]
    if single.get("statistics"):
        statistics["single_image_evidence"] = single["statistics"]
    if (response.get("vqa_details") or {}).get("statistics_used"):
        statistics["controlled_answer"] = response["vqa_details"]["statistics_used"]
    if sar_water:
        statistics["sar_water"] = {name: sar_water.get(name) for name in ("water_detected", "image_area_percent", "candidate_pixels", "valid_pixels", "heuristic_reliability")}
    if sar_scene:
        statistics["sar_scene"] = {name: sar_scene.get(name) for name in ("valid_pixel_percent", "low_backscatter_percent", "mid_backscatter_percent", "high_backscatter_percent", "texture_index")}
    return _safe(statistics)


def _agent_limitations(response: Dict[str, Any]) -> List[str]:
    values = (
        ((response.get("caption_details") or {}).get("limitations") or [])
        + ((response.get("grounding_result") or {}).get("limitations") or [])
        + ((response.get("vqa_details") or {}).get("limitations") or [])
        + ((((response.get("cross_modal_analysis") or {}).get("method") or {}).get("limitations")) or [])
        + ((response.get("sar_water_analysis") or {}).get("limitations") or [])
        + ((response.get("sar_scene_analysis") or {}).get("limitations") or [])
        + ((response.get("ttp_result") or {}).get("limitations") or [])
    )
    return list(dict.fromkeys(values))


def record_agent_result(response: AgentResponse, *, primary_hash: str, secondary_hash: Optional[str], primary_modality: Modality, secondary_modality: Optional[Modality], primary_date: Optional[str], secondary_date: Optional[str], query: str) -> None:
    data = response.model_dump(mode="json")
    task = _task(data["task"])
    primary = data.get("primary_image_metadata") or {}
    secondary = data.get("secondary_image_metadata") or {}
    inputs = _unique_previews([
        _preview("Primary input", primary.get("preview_url"), "input", primary, primary_modality),
        _preview("Secondary input", secondary.get("preview_url"), "input", secondary, secondary_modality),
    ])
    model = data.get("model") or {}
    details = data.get("caption_details") or {}
    grounding = data.get("grounding_result") or {}
    device = details.get("device") or grounding.get("device")
    target = grounding.get("target_phrase")
    identity = ComparisonIdentity(
        primary_hash=primary_hash,
        secondary_hash=secondary_hash,
        query_hash=hash_text(query) if task in {ComparisonTask.VQA, ComparisonTask.CHANGE_VQA} else None,
        target_hash=hash_text(target) if task == ComparisonTask.GROUNDING else None,
        dates=(primary_date, secondary_date),
        safe_parameters={"input_mode": data["execution"]["input_mode"], "primary_modality": primary_modality.value, "secondary_modality": secondary_modality.value if secondary_modality else None, "date_order_supplied": bool(primary_date and secondary_date), "change_engine": (data.get("change_engine") or {}).get("mode")},
        task_family="single_image" if task in {ComparisonTask.CAPTIONING, ComparisonTask.VQA, ComparisonTask.GROUNDING, ComparisonTask.SAR_ANALYSIS} else "change" if task in {ComparisonTask.CHANGE, ComparisonTask.CHANGE_VQA} else "cross_modal",
    )
    lineage = [ComparisonLineageNode(kind="input", label="Input imagery", preview_url=inputs[0].url if inputs else None)]
    lineage.extend(ComparisonLineageNode(kind="tool", label=tool) for tool in data["execution"].get("selected_tools") or [])
    if _agent_outputs(data):
        lineage.append(ComparisonLineageNode(kind="evidence", label="Evidence products"))
    if data.get("answer"):
        lineage.append(ComparisonLineageNode(kind="answer", label="Answer"))
    item = ComparisonItem(
        query=query, request_id=data["request_id"], task=task, display_name=DISPLAY_NAMES[task], status=data["status"], created_at=(data.get("cache") or {}).get("original_generation_timestamp") or utc_now(), input_mode=InputMode(data["execution"]["input_mode"]), modalities=[value for value in (primary_modality, secondary_modality) if value], input_previews=inputs, output_previews=_agent_outputs(data), answer=data.get("answer"), statistics=_agent_statistics(data), confidence=_safe(data.get("confidence") or {}), provenance=_safe({"model": model, "method": (data.get("vqa_details") or {}).get("method"), "selection_reason": data["execution"].get("selection_reason"), "change_engine": data.get("change_engine"), "ttp": data.get("ttp_result")}), selected_tools=data["execution"].get("selected_tools") or [], execution_duration_ms=data["execution"].get("duration_ms"), device=device or (data.get("ttp_result") or {}).get("device"), warnings=_safe(data.get("warnings") or []), limitations=_agent_limitations(data), cached=bool((data.get("cache") or {}).get("cached")), input_identity=identity.public(), lineage=lineage, execution_summary=_safe({"steps": data["execution"].get("steps") or [], "validation": data["execution"].get("validation")}),
    )
    COMPARISON_STORE.put(item, identity)


def record_change_result(response: ChangeAnalysisResponse, *, primary_hash: str, secondary_hash: str, modality: Modality) -> None:
    data = response.model_dump(mode="json")
    before = data["before_metadata"]
    after = data["after_metadata"]
    engine = data.get("change_engine") or {}
    identity = ComparisonIdentity(primary_hash=primary_hash, secondary_hash=secondary_hash, dates=(data["before_date"], data["after_date"]), safe_parameters={"modality": modality.value, "date_order_supplied": True, "change_engine": engine.get("mode")}, task_family="change")
    inputs = _unique_previews([_preview("Before", data["previews"].get("before"), "input", before, modality), _preview("After", data["previews"].get("after"), "input", after, modality)])
    preview_order = ["ttp_raw_mask", "ttp_mask", "deterministic_mask", "agreement", "disagreement", "intersection", "union", "ttp_overlay", "difference"] if engine.get("mode") in {"hybrid", "ttp"} else ["deterministic_mask", "mask", "deterministic_overlay", "overlay", "difference"]
    outputs = _unique_previews(_preview(name.replace("_", " ").title(), data["previews"].get(name), "evidence", {"width": (data.get("statistics") or {}).get("analysis_width"), "height": (data.get("statistics") or {}).get("analysis_height")}) for name in preview_order)
    item = ComparisonItem(request_id=data["request_id"], task=ComparisonTask.CHANGE, display_name=DISPLAY_NAMES[ComparisonTask.CHANGE], status=data["status"], created_at=utc_now(), input_mode=InputMode.BI_TEMPORAL, modalities=[modality, modality], input_previews=inputs, output_previews=outputs, statistics=_safe({"change": data.get("statistics") or {}, "deterministic_change": data.get("deterministic_statistics") or {}, "ttp_change": data.get("ttp_result") or {}, "mask_comparison": data.get("mask_comparison") or {}}), confidence={"level": "unavailable", "score": None, "reason": ((data.get("evidence_consistency") or {}).get("disclosure") or "No calibrated change probability is produced.")}, provenance=_safe({"method": engine.get("mode") or "deterministic", "change_engine": engine, "ttp": data.get("ttp_result"), "registration": False, "reprojection": False}), selected_tools=data["execution"]["selected_tools"], execution_duration_ms=data["runtime_ms"], device=(data.get("ttp_result") or {}).get("device"), warnings=_safe(data.get("warnings") or []), limitations=["TTP and deterministic masks are model/method evidence, not ground truth.", "The workflow does not assign a semantic cause.", "Unaligned pairs are not silently registered."], input_identity=identity.public(), lineage=[ComparisonLineageNode(kind="input", label="Before and after imagery"), *[ComparisonLineageNode(kind="tool", label=tool) for tool in data["execution"]["selected_tools"]], ComparisonLineageNode(kind="evidence", label="Source-labelled learned and deterministic change evidence")], execution_summary=_safe({"steps": data["execution"].get("steps") or [], "compatibility": data.get("compatibility")}),)
    COMPARISON_STORE.put(item, identity, [preview.url for preview in inputs + outputs])


def record_cross_modal_result(response: CrossModalAnalysisResponse, *, primary_hash: str, secondary_hash: str, optical_modality: Modality, sar_modality: Modality) -> None:
    data = response.model_dump(mode="json")
    result = data["result"]
    identity = ComparisonIdentity(primary_hash=primary_hash, secondary_hash=secondary_hash, safe_parameters={"optical_modality": optical_modality.value, "sar_modality": sar_modality.value}, task_family="cross_modal")
    optical = data["optical_metadata"]
    sar = data["sar_metadata"]
    inputs = _unique_previews([_preview("Optical input", result["previews"].get("optical") or optical.get("preview_url"), "input", optical, optical_modality), _preview("SAR input", result["previews"].get("sar") or sar.get("preview_url"), "input", sar, sar_modality)])
    stats = result.get("statistics") or {}
    outputs = _unique_previews(_preview(name.replace("_", " ").title(), url, "evidence", {"width": stats.get("analysis_width"), "height": stats.get("analysis_height")}) for name, url in (result.get("previews") or {}).items() if name not in {"optical", "sar"})
    item = ComparisonItem(request_id=data["request_id"], task=ComparisonTask.CROSS_MODAL, display_name=DISPLAY_NAMES[ComparisonTask.CROSS_MODAL], status=result["status"], created_at=utc_now(), input_mode=InputMode.CROSS_MODAL, modalities=[optical_modality, sar_modality], input_previews=inputs, output_previews=outputs, statistics=_safe({"cross_modal": stats}), confidence=_safe(result.get("confidence") or {}), provenance=_safe({"method": result.get("method"), "preparation": {"optical": result.get("optical_preparation"), "sar": result.get("sar_preparation")}}), selected_tools=data["execution"]["selected_tools"], execution_duration_ms=data["execution"]["duration_ms"], warnings=_safe(result.get("warnings") or []), limitations=((result.get("method") or {}).get("limitations") or []), input_identity=identity.public(), lineage=[ComparisonLineageNode(kind="input", label="Optical and SAR pair"), ComparisonLineageNode(kind="tool", label="cross_modal_optical_sar_analyzer"), ComparisonLineageNode(kind="evidence", label="Agreement, disagreement, and joint evidence")], execution_summary=_safe({"steps": data["execution"].get("steps") or [], "compatibility": data.get("compatibility")}),)
    COMPARISON_STORE.put(item, identity, [preview.url for preview in inputs + outputs])


def store_model_preview(image: Image.Image, label: str, kind: str, modality: Optional[Modality] = None) -> ComparisonPreview:
    image = image.convert("RGB")
    return ComparisonPreview(label=label, url=save_preview(image), kind=kind, width=image.width, height=image.height, modality=modality)


def record_model_result(*, task: ComparisonTask, request_id: str, primary_hash: str, component_hashes: Sequence[str], input_previews: List[ComparisonPreview], output_previews: List[ComparisonPreview], output_hashes: Sequence[str], statistics: Dict[str, Any], execution_duration_ms: float, device: str, provenance: Dict[str, Any], warnings: Sequence[str], safe_parameters: Dict[str, Any]) -> None:
    identity = ComparisonIdentity(primary_hash=primary_hash, component_hashes=tuple(component_hashes), safe_parameters=safe_parameters, task_family="reconstruction", output_hashes=tuple(output_hashes))
    item = ComparisonItem(request_id=request_id, task=task, display_name=DISPLAY_NAMES[task], status="success", created_at=utc_now(), input_mode=InputMode.SINGLE, modalities=[Modality.SAR], input_previews=input_previews, output_previews=output_previews, statistics=_safe(statistics), confidence={"level": "unavailable", "score": None, "reason": "Reconstruction quality confidence is not calibrated; ground-truth metrics are shown only when supplied."}, provenance=_safe(provenance), selected_tools=[task.value], execution_duration_ms=max(0, round(execution_duration_ms)), device=device, warnings=_safe(list(warnings)), limitations=["Reconstruction outputs are generated products and are not ground truth.", "Runtime alone does not establish scientific accuracy."], input_identity=identity.public(), lineage=[ComparisonLineageNode(kind="input", label="SAR input", preview_url=input_previews[0].url if input_previews else None), ComparisonLineageNode(kind="tool", label=task.value), ComparisonLineageNode(kind="intermediate", label="Optical reconstruction", preview_url=output_previews[0].url if output_previews else None)], execution_summary={"measured_runtime_ms": max(0, round(execution_duration_ms))})
    COMPARISON_STORE.put(item, identity, [preview.url for preview in input_previews + output_previews])


def assess(left: ComparisonRecord, right: ComparisonRecord) -> ComparabilityResult:
    left_item, right_item = left.item, right.item
    left_hashes, right_hashes = left.identity.all_input_hashes(), right.identity.all_input_hashes()
    shared_any = bool(left_hashes & right_hashes)
    exact_pair = left.identity.primary_hash == right.identity.primary_hash and left.identity.secondary_hash == right.identity.secondary_hash
    same_family = left.identity.task_family == right.identity.task_family
    level = ComparabilityLevel.NOT_DIRECT
    reason = "The results use unrelated tasks or input identities; their task-specific measurements must not be compared as a shared score."
    warnings = ["Not directly comparable. Keep task-specific measurements in their own context."]

    reconstruction_pair = {left_item.task, right_item.task} == {ComparisonTask.PIX2PIX, ComparisonTask.SARFUSIONFORMER}
    if reconstruction_pair and shared_any:
        level, reason, warnings = ComparabilityLevel.DIRECT, "Pix2Pix and SARFusionFormer used a shared content-hash SAR identity.", ["Different reconstruction methods may produce different evidence; runtime does not identify a winner."]
    elif left_item.task == right_item.task and exact_pair:
        parameters_match = left_item.modalities == right_item.modalities and left.identity.safe_parameters == right.identity.safe_parameters
        if left_item.task in {ComparisonTask.VQA, ComparisonTask.CHANGE_VQA}:
            parameters_match = bool(left.identity.query_hash and left.identity.query_hash == right.identity.query_hash)
        if left_item.task == ComparisonTask.GROUNDING:
            parameters_match = bool(left.identity.target_hash and left.identity.target_hash == right.identity.target_hash)
        if left_item.task in {ComparisonTask.CHANGE, ComparisonTask.CHANGE_VQA}:
            parameters_match = parameters_match and left.identity.dates == right.identity.dates
        if parameters_match:
            level, reason, warnings = ComparabilityLevel.DIRECT, "The task, content-hash input identity, and required safe parameters match.", []
        else:
            level, reason, warnings = ComparabilityLevel.PARTIAL, "The imagery matches, but the question, target, dates, or other task-defining parameters differ.", ["Compare shared input evidence only; answer-specific values are not directly equivalent."]
    elif shared_any and {left_item.task, right_item.task} <= {ComparisonTask.CAPTIONING, ComparisonTask.VQA, ComparisonTask.GROUNDING, ComparisonTask.SAR_ANALYSIS}:
        level, reason, warnings = ComparabilityLevel.PARTIAL, "The same image is used by different single-image task families.", ["Caption, controlled answer, and grounding scores represent different constructs."]
    elif (left.identity.output_hashes and left.identity.output_hashes[0] in right_hashes) or (right.identity.output_hashes and right.identity.output_hashes[0] in left_hashes):
        level, reason, warnings = ComparabilityLevel.PARTIAL, "One result consumes a content-identical derived output from the other result.", ["Lineage is shared, but reconstruction and interpretation outputs remain different task types."]
    elif same_family and not shared_any:
        reason = "The task family matches, but content hashes show different input scenes or pairs."
        warnings = ["Not directly comparable because the input identity differs."]

    left_preview = (left_item.output_previews or [None])[0]
    right_preview = (right_item.output_previews or [None])[0]
    dimensions_match = bool(left_preview and right_preview and left_preview.width and left_preview.height and left_preview.width == right_preview.width and left_preview.height == right_preview.height)
    overlay = level == ComparabilityLevel.DIRECT and dimensions_match
    difference = overlay and (left_item.task == right_item.task or reconstruction_pair)
    if level == ComparabilityLevel.DIRECT and not dimensions_match and left_preview and right_preview:
        warnings.append("Overlay and difference view are disabled because output dimensions do not match or were not reported.")
    return ComparabilityResult(left_request_id=left_item.request_id, right_request_id=right_item.request_id, level=level, reason=reason, shared_inputs=shared_any, shared_task_family=same_family, warnings=warnings, overlay_allowed=overlay, difference_allowed=difference)


def assess_many(request_ids: Sequence[str]) -> ComparisonAssessmentResponse:
    records = COMPARISON_STORE.records(request_ids)
    assessments = [assess(left, right) for left, right in combinations(records, 2)]
    levels = {item.level for item in assessments}
    overall = ComparabilityLevel.NOT_DIRECT if ComparabilityLevel.NOT_DIRECT in levels else ComparabilityLevel.PARTIAL if ComparabilityLevel.PARTIAL in levels else ComparabilityLevel.DIRECT
    warnings = list(dict.fromkeys(warning for item in assessments for warning in item.warnings))
    return ComparisonAssessmentResponse(assessments=assessments, overall_level=overall, warnings=warnings)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _comparison_pdf(document: Dict[str, Any]) -> bytes:
    width, height, margin = 1240, 1754, 80
    pages: List[Image.Image] = []
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    y = margin

    def finish() -> None:
        nonlocal page, draw, y
        draw.text((margin, height - 48), f"GeoVision · Mission Comparison · Page {len(pages) + 1}", fill=(90, 100, 114), font=_font(18))
        pages.append(page)
        page = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(page)
        y = margin

    def text(value: Any, size: int = 23, bold: bool = False, color=(38, 47, 61), gap: int = 12) -> None:
        nonlocal y
        font = _font(size, bold)
        words = str(value if value not in (None, "") else "Unavailable").replace("—", "-").split()
        lines: List[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if not current or draw.textlength(trial, font=font) < width - margin * 2:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        needed = len(lines) * (size + 9) + gap
        if y + needed > height - 90:
            finish()
        for line in lines:
            draw.text((margin, y), line, fill=color, font=font)
            y += size + 9
        y += gap

    text("GEOVISION", 25, True, (10, 81, 122))
    text("Unified Mission Comparison", 48, True, (20, 29, 43), 22)
    text(f"Comparison ID: {document['comparison']['id']}")
    text(f"Generated: {document['comparison']['generated_at']}")
    text("Factual side-by-side review. No universal winner or cross-task quality score is produced.", 24, False, (99, 72, 18), 24)
    for item in document["selected_results"]:
        text(item["display_name"], 31, True, (10, 81, 122), 8)
        text(f"Task: {item['task']} · Status: {item['status']} · Runtime: {item.get('execution_duration_ms')} ms")
        text(f"Tools: {', '.join(item.get('selected_tools') or [])}", 20)
        text(f"Warnings: {len(item.get('warnings') or [])} · Evidence products: {len(item.get('output_previews') or [])}", 20)
        confidence = item.get("confidence") or {}
        text(f"Confidence basis: {confidence.get('level', 'unavailable')} · {confidence.get('reason', 'Unavailable')}", 20)
        if item.get("limitations"):
            text("Limitations: " + " | ".join(item["limitations"]), 19, False, (99, 72, 18))
    text("Comparability assessment", 32, True, (10, 81, 122), 10)
    for result in document["comparability"]:
        text(f"{result['level']}: {result['reason']}", 21)
    if document.get("user_note"):
        text("User note (user-authored)", 28, True, (10, 81, 122), 8)
        text(document["user_note"], 21)
    text("Factual conclusion", 32, True, (10, 81, 122), 10)
    for line in document["factual_conclusion"]:
        text(f"• {line}", 21)
    finish()
    stream = io.BytesIO()
    pages[0].save(stream, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return stream.getvalue()


def generate_comparison_report(request_ids: Sequence[str], formats: Sequence[ReportFormat], user_note: Optional[str]) -> ComparisonReportResponse:
    started = time.perf_counter()
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Select distinct result IDs for a comparison report.")
    records = COMPARISON_STORE.records(request_ids)
    assessment = assess_many(request_ids)
    comparison_id = str(uuid.uuid4())
    items = [record.item.model_copy(update={"lineage": COMPARISON_STORE._lineage(record)}).model_dump(mode="json") for record in records]
    runtimes = [{"request_id": item["request_id"], "runtime_ms": item.get("execution_duration_ms")} for item in items]
    document = _safe({
        "schema_version": SCHEMA_VERSION,
        "comparison": {"id": comparison_id, "generated_at": utc_now(), "selected_count": len(items), "authoritative_backend_values": True, "winner_declared": False},
        "selected_results": items,
        "input_identity_and_lineage": [{"request_id": item["request_id"], "input_identity": item["input_identity"], "lineage": item["lineage"]} for item in items],
        "comparability": [value.model_dump(mode="json") for value in assessment.assessments],
        "runtime_and_device": [{"request_id": item["request_id"], "runtime_ms": item.get("execution_duration_ms"), "device": item.get("device")} for item in items],
        "user_note": user_note.strip() if user_note and user_note.strip() else None,
        "factual_conclusion": [f"{len(items)} stored results were selected.", f"Pairwise comparability is {assessment.overall_level.value}.", f"Measured runtimes by request ID: {json.dumps(runtimes, separators=(',', ':'))}.", "No result is ranked; task-specific measurements retain their original meaning."],
        "limitations": ["Results and artifacts are bounded, process-local, and expire.", "Runtime is a performance observation, not a scientific accuracy score."],
    })
    metadata = json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")
    pdf = _comparison_pdf(document)
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("report/mission-comparison.pdf", pdf)
        archive.writestr("report/mission-comparison.json", metadata)
        for item in items:
            for index, preview in enumerate(item.get("input_previews", []) + item.get("output_previews", []), start=1):
                path = preview_file_path(preview["url"].rsplit("/", 1)[-1])
                if path:
                    label = re.sub(r"[^A-Za-z0-9_-]+", "_", preview["label"]).strip("_")[:60]
                    archive.write(path, f"evidence/{item['request_id']}/{index:02d}_{label}.png")
        archive.writestr("README.txt", "GeoVision Mission Comparison\n\nAll values are copied from authoritative stored backend results. No universal winner or cross-task quality score is generated. User notes, when present, are explicitly labelled.\n")
    payloads = {ReportFormat.PDF: pdf, ReportFormat.JSON: metadata, ReportFormat.ZIP: archive_stream.getvalue()}
    requested = list(dict.fromkeys(formats or [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.ZIP]))
    if any(value not in payloads for value in requested):
        raise ValueError("Comparison reports support PDF, JSON, and ZIP formats.")
    artifacts: List[ReportArtifact] = []
    for report_format in requested:
        name = f"GeoVision_SatQuery_comparison_{comparison_id}.{report_format.value}"
        path = ARTIFACT_STORE.put(name, payloads[report_format])
        artifacts.append(ReportArtifact(format=report_format, filename=name, url=f"/api/agent/reports/{name}", size_bytes=path.stat().st_size))
    for request_id in request_ids:
        COMPARISON_STORE.mark_report(request_id)
    return ComparisonReportResponse(comparison_id=comparison_id, status="success", schema_version=SCHEMA_VERSION, artifacts=artifacts, assessments=assessment.assessments, warnings=list(dict.fromkeys(assessment.warnings + ["Comparison artifacts are temporary and expire with the bounded local backend cache."])), runtime_ms=max(0, round((time.perf_counter() - started) * 1000)))

"""Bounded SatQuery result caching and local mission-report generation."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .image_ingestion import preview_file_path, remove_preview_url
from .evidence_lifecycle import normalize_translation_evidence
from .models import (
    AgentResponse,
    CacheMetadata,
    ReportArtifact,
    ReportFormat,
    ReportResponse,
    ResponseStatus,
    TaskType,
)


SCHEMA_VERSION = "1.0"
TOOL_VERSION = "satquery-sar-routing-1.2-pix2pix-evidence"
DEFAULT_MAX_RESULTS = 32
DEFAULT_TTL_SECONDS = 30 * 60
REPORT_DIR = Path(
    os.getenv("SATQUERY_REPORT_DIR", str(Path(tempfile.gettempdir()) / "geovision-satquery-reports"))
).resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
_PREVIEW_PATTERN = re.compile(r"^/api/agent/previews/([0-9a-f]{32}\.png)$")
_ARTIFACT_PATTERN = re.compile(r"^GeoVision_SatQuery_[a-z_]+_[0-9a-f-]{36}\.(pdf|json|csv|zip)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _configured_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def cache_key_for(
    *,
    primary_hash: str,
    secondary_hash: Optional[str],
    task: TaskType,
    normalized_query: str,
    safe_parameters: Dict[str, Any],
) -> str:
    payload = {
        "primary_hash": primary_hash,
        "secondary_hash": secondary_hash,
        "task": task.value,
        "query": " ".join(normalized_query.lower().split()),
        "tool_version": TOOL_VERSION,
        "parameters": safe_parameters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _preview_urls(value: Any) -> List[str]:
    output: List[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            output.extend(_preview_urls(nested))
    elif isinstance(value, list):
        for nested in value:
            output.extend(_preview_urls(nested))
    elif isinstance(value, str) and _PREVIEW_PATTERN.fullmatch(value):
        output.append(value)
    return list(dict.fromkeys(output))


@dataclass
class MissionRecord:
    response: AgentResponse
    cache_key: str
    generated_at: str
    touched_at: float


class MissionResultStore:
    """Process-local bounded store. Restarting the API intentionally clears it."""

    def __init__(self) -> None:
        self.max_items = _configured_int("SATQUERY_RESULT_CACHE_MAX_ITEMS", DEFAULT_MAX_RESULTS, 1)
        self.ttl_seconds = _configured_int("SATQUERY_RESULT_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS, 60)
        self._records: "OrderedDict[str, MissionRecord]" = OrderedDict()
        self._cache_index: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _discard(self, request_id: str) -> None:
        record = self._records.pop(request_id, None)
        if record is None:
            return
        if self._cache_index.get(record.cache_key) == request_id:
            self._cache_index.pop(record.cache_key, None)
        for url in _preview_urls(record.response.model_dump(mode="json")):
            remove_preview_url(url)

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for request_id, record in list(self._records.items()):
            if record.touched_at < cutoff:
                self._discard(request_id)
        while len(self._records) > self.max_items:
            self._discard(next(iter(self._records)))

    def put(self, response: AgentResponse, cache_key: str, generated_at: Optional[str] = None) -> None:
        with self._lock:
            self._cleanup()
            response = normalize_translation_evidence(response)
            request_id = response.request_id
            self._records[request_id] = MissionRecord(
                response=response.model_copy(deep=True),
                cache_key=cache_key,
                generated_at=generated_at or utc_now(),
                touched_at=time.time(),
            )
            self._records.move_to_end(request_id)
            self._cache_index[cache_key] = request_id
            self._cleanup()

    def get_by_request(self, request_id: str) -> Optional[MissionRecord]:
        with self._lock:
            self._cleanup()
            record = self._records.get(request_id)
            if record is None:
                return None
            record.touched_at = time.time()
            self._records.move_to_end(request_id)
            return copy.deepcopy(record)

    def get_by_cache(self, cache_key: str) -> Optional[MissionRecord]:
        with self._lock:
            self._cleanup()
            request_id = self._cache_index.get(cache_key)
            if request_id is None:
                self._misses += 1
                return None
            record = self.get_by_request(request_id)
            if record is None:
                self._misses += 1
                return None
            self._hits += 1
            return record

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._cleanup()
            return {
                "stored_results": len(self._records),
                "max_results": self.max_items,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
            }

    def clear(self) -> None:
        with self._lock:
            for request_id in list(self._records):
                self._discard(request_id)
            self._hits = 0
            self._misses = 0


MISSION_STORE = MissionResultStore()


def mark_fresh(response: AgentResponse, cache_key: str, generated_at: Optional[str] = None) -> AgentResponse:
    timestamp = generated_at or utc_now()
    return response.model_copy(
        update={
            "cache": CacheMetadata(
                cached=False,
                original_generation_timestamp=timestamp,
                retrieval_timestamp=timestamp,
                tool_version=TOOL_VERSION,
                cache_key_prefix=cache_key[:12],
            )
        }
    )


def cached_response(record: MissionRecord) -> AgentResponse:
    response = normalize_translation_evidence(record.response.model_copy(deep=True))
    return response.model_copy(
        update={
            "cache": CacheMetadata(
                cached=True,
                original_generation_timestamp=record.generated_at,
                retrieval_timestamp=utc_now(),
                tool_version=TOOL_VERSION,
                cache_key_prefix=record.cache_key[:12],
            )
        }
    )


def result_is_reportable(response: AgentResponse) -> bool:
    return response.status in {ResponseStatus.SUCCESS, ResponseStatus.PARTIAL, ResponseStatus.ALIGNMENT_REQUIRED}


def _sanitize(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(token in lowered for token in ("api_key", "secret", "password", "token", "local_path", "cache_path")):
        return None
    if isinstance(value, dict):
        return {name: _sanitize(item, name) for name, item in value.items() if _sanitize(item, name) is not None}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith(("/Users/", "/home/", "/root/", "C:\\")):
            return "[redacted local path]"
    return value


def _named_previews(value: Any, prefix: str = "evidence") -> List[Tuple[str, str]]:
    output: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for name, nested in value.items():
            output.extend(_named_previews(nested, f"{prefix}.{name}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            output.extend(_named_previews(nested, f"{prefix}.{index + 1}"))
    elif isinstance(value, str) and _PREVIEW_PATTERN.fullmatch(value):
        output.append((prefix.replace("_", " ").replace(".", " · ").title(), value))
    seen = set()
    return [(label, url) for label, url in output if not (url in seen or seen.add(url))]


def _report_previews(response: Dict[str, Any]) -> List[Tuple[str, str]]:
    preferred: List[Tuple[str, str]] = []
    grounding = response.get("grounding_result") or {}
    sar_water = response.get("sar_water_analysis") or {}
    sar_scene = response.get("sar_scene_analysis") or {}
    grounding_preview = grounding.get("annotated_preview_url")
    if isinstance(grounding_preview, str) and _PREVIEW_PATTERN.fullmatch(grounding_preview):
        preferred.append(("Grounding · Model-produced boxes and scores", grounding_preview))
    change = response.get("change_analysis") or {}
    for name, url in (change.get("previews") or {}).items():
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            preferred.append((f"Change · {name.replace('_', ' ').title()}", url))
    cross = response.get("cross_modal_analysis") or {}
    for product in cross.get("evidence_products") or []:
        url = product.get("reference")
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            preferred.append((product["label"], url))
    for name, url in (cross.get("previews") or {}).items():
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            preferred.append((f"Optical–SAR · {name.replace('_', ' ').title()}", url))
    single = ((response.get("vqa_details") or {}).get("single_image_evidence") or {})
    for name, url in (single.get("previews") or {}).items():
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            preferred.append((f"Single-image · {name.replace('_', ' ').title()}", url))
    for label, metadata in (("Primary input preview", response.get("primary_image_metadata")), ("Secondary input preview", response.get("secondary_image_metadata"))):
        url = (metadata or {}).get("preview_url")
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            role = (metadata or {}).get("observation_role")
            preferred.append((f"{role.title()} source" if role else label, url))
    for product in response.get("evidence") or []:
        url = product.get("reference") if isinstance(product, dict) else None
        if isinstance(url, str) and _PREVIEW_PATTERN.fullmatch(url):
            preferred.append((product.get("label") or "Published evidence", url))
    fallback_response = copy.deepcopy(response)
    fallback_response.pop("sar_translated_optical_analysis", None)
    fallback_response.pop("evidence", None)
    seen = set()
    return [(label, url) for label, url in preferred + _named_previews(fallback_response) if not (url in seen or seen.add(url))]


def _statistics(response: Dict[str, Any]) -> Dict[str, Any]:
    details = response.get("vqa_details") or {}
    single = details.get("single_image_evidence") or {}
    change = response.get("change_analysis") or {}
    cross = response.get("cross_modal_analysis") or {}
    grounding = response.get("grounding_result") or {}
    sar_water = response.get("sar_water_analysis") or {}
    sar_scene = response.get("sar_scene_analysis") or {}
    sve = (
        response.get("sve_result")
        or (change.get("sve_result") if isinstance(change, dict) else None)
        or (cross.get("sve_result") if isinstance(cross, dict) else None)
        or {}
    )
    detections = grounding.get("detections") or []
    rejected = grounding.get("rejected_candidates") or []
    return {
        "controlled_answer_statistics": details.get("statistics_used") or {},
        "single_image_evidence": single.get("statistics") or {},
        "change_statistics": change.get("statistics") or {},
        "semantic_change_summary": response.get("semantic_change_summary") or change.get("semantic_change_summary") or {},
        "deterministic_change_statistics": change.get("deterministic_statistics") or {},
        "ttp_change_statistics": response.get("ttp_result") or change.get("ttp_result") or {},
        "mask_comparison": response.get("mask_comparison") or change.get("mask_comparison") or {},
        "cross_modal_statistics": cross.get("statistics") or {},
        "grounding_statistics": {
            "target_phrase": grounding.get("target_phrase"),
            "detection_count": len(detections),
            "accepted_detection_count": len(detections),
            "rejected_candidate_count": len(rejected),
            "top_alignment_score": detections[0].get("score") if detections else None,
            "highest_rejected_alignment_score": max(
                (candidate.get("score") for candidate in rejected if candidate.get("score") is not None),
                default=None,
            ),
            "quality_policy": grounding.get("quality_policy"),
            "mask_refinement_connected": False,
        } if grounding else {},
        "sar_water_statistics": {name: sar_water.get(name) for name in ("water_detected", "image_area_percent", "geographic_area_square_meters", "geographic_area_method", "candidate_pixels", "valid_pixels", "heuristic_reliability", "input_quality_score", "threshold")} if sar_water else {},
        "sar_scene_statistics": {name: sar_scene.get(name) for name in ("valid_pixel_percent", "low_backscatter_percent", "mid_backscatter_percent", "high_backscatter_percent", "texture_index", "input_quality_score")} if sar_scene else {},
        "sve_scene_evidence": {
            "status": sve.get("status"),
            "device": sve.get("device"),
            "runtime_ms": sve.get("runtime_ms"),
            "top_scene_priors": sve.get("scene_priors") or [],
            "caption_consistency": sve.get("caption_consistency"),
            "vqa_consistency": sve.get("vqa_consistency"),
            "grounding_support": sve.get("grounding_support"),
            "semantic_comparison": sve.get("semantic_comparison"),
        } if sve else {},
    }


def build_report_document(response: AgentResponse, generated_at: str, artifact_names: Iterable[str]) -> Dict[str, Any]:
    safe_response = _sanitize(normalize_translation_evidence(response).model_dump(mode="json"))
    details = safe_response.get("vqa_details") or {}
    grounding = safe_response.get("grounding_result") or {}
    sve = (
        safe_response.get("sve_result")
        or (safe_response.get("change_analysis") or {}).get("sve_result")
        or (safe_response.get("cross_modal_analysis") or {}).get("sve_result")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "report": {
            "title": "GeoVision SatQuery Mission Report",
            "analysis_type": safe_response.get("task"),
            "request_id": safe_response.get("request_id"),
            "generated_timestamp": generated_at,
            "status": safe_response.get("status"),
            "result_status": safe_response.get("result_status"),
        },
        "input_summary": {
            "primary": safe_response.get("primary_image_metadata"),
            "secondary": safe_response.get("secondary_image_metadata"),
            "compatibility": safe_response.get("pair_compatibility"),
            "observations_by_role": {item["observation_role"]: item for item in (safe_response.get("primary_image_metadata"), safe_response.get("secondary_image_metadata")) if item and item.get("observation_role")},
            "evidence_products": (safe_response.get("cross_modal_analysis") or {}).get("evidence_products") or [],
        },
        "user_query": {
            "original_query": details.get("original_question") or grounding.get("original_query") or (safe_response.get("execution") or {}).get("permitted_parameters", {}).get("query"),
            "detected_task": safe_response.get("task"),
            "question_category": details.get("question_category"),
            "target_concept": details.get("target_concept") or grounding.get("target_phrase"),
        },
        "answer": {
            "text": safe_response.get("answer"),
            "source": details.get("answer_source") or ((safe_response.get("model") or {}).get("tool_id")),
            "status": safe_response.get("status"),
            "result_status": safe_response.get("result_status"),
            "warnings": safe_response.get("warnings") or [],
        },
        "evidence": {
            "items": safe_response.get("evidence") or [],
            "preview_products": [{"label": label, "url": url} for label, url in _report_previews(safe_response)],
            "disclaimer": "Evidence products are model-generated or deterministic heuristic products as labelled; none are ground truth.",
        },
        "statistics": _statistics(safe_response),
        "confidence": safe_response.get("confidence"),
        "specialist_provenance": {
            "selected_tools": (safe_response.get("execution") or {}).get("selected_tools") or [],
            "model": safe_response.get("model"),
            "caption_details": safe_response.get("caption_details"),
            "grounding": {
                "device": grounding.get("device"),
                "input": grounding.get("input"),
                "model_load_ms": grounding.get("model_load_ms"),
                "model_reused": grounding.get("model_reused"),
                "quality_policy": grounding.get("quality_policy"),
                "operational_threshold_disclaimer": grounding.get("operational_threshold_disclaimer"),
            } if grounding else None,
            "vqa_method": details.get("method"),
            "cross_modal_method": (safe_response.get("cross_modal_analysis") or {}).get("method"),
            "sar_water_method": (safe_response.get("sar_water_analysis") or {}).get("method"),
            "sar_water_method_version": (safe_response.get("sar_water_analysis") or {}).get("method_version"),
            "sar_preprocessing": ((safe_response.get("sar_water_analysis") or {}).get("preprocessing") or (safe_response.get("sar_scene_analysis") or {}).get("preprocessing")),
            "sar_translation": safe_response.get("sar_translated_optical_analysis"),
            "heuristic_reliability": (safe_response.get("sar_water_analysis") or {}).get("heuristic_reliability"),
            "model_confidence": (safe_response.get("sar_water_analysis") or {}).get("model_confidence"),
            "change_engine": safe_response.get("change_engine") or (safe_response.get("change_analysis") or {}).get("change_engine"),
            "ttp": safe_response.get("ttp_result") or (safe_response.get("change_analysis") or {}).get("ttp_result"),
            "mask_comparison": safe_response.get("mask_comparison") or (safe_response.get("change_analysis") or {}).get("mask_comparison"),
            "evidence_consistency": safe_response.get("evidence_consistency") or (safe_response.get("change_analysis") or {}).get("evidence_consistency"),
            "semantic_change_summary": safe_response.get("semantic_change_summary") or (safe_response.get("change_analysis") or {}).get("semantic_change_summary"),
            "runtime_ms": (safe_response.get("execution") or {}).get("duration_ms"),
            "satquery_vision_encoder": sve,
        },
        "execution_trace": safe_response.get("execution"),
        "scientific_limitations": list(dict.fromkeys(
            (details.get("limitations") or [])
            + ((safe_response.get("caption_details") or {}).get("limitations") or [])
            + (grounding.get("limitations") or [])
            + (((safe_response.get("cross_modal_analysis") or {}).get("method") or {}).get("limitations") or [])
            + ((safe_response.get("sar_water_analysis") or {}).get("limitations") or [])
            + ((safe_response.get("sar_scene_analysis") or {}).get("limitations") or [])
            + (((safe_response.get("ttp_result") or (safe_response.get("change_analysis") or {}).get("ttp_result") or {}).get("limitations")) or [])
            + ((sve or {}).get("limitations") or [])
            + [
                "Outputs are not ground truth and must be reviewed with source imagery and mission context.",
                "Learned change-detector output is a model-generated binary prediction and is not ground truth.",
                "No hidden geolocation, unsupported causal interpretation, or calibrated semantic probability is produced.",
                "SatQuery Vision Encoder v1 provides scene-level embedding evidence. It does not produce pixel-level segmentation, object grounding, calibrated probabilities, or ground truth.",
            ]
        )),
        "reproducibility": {
            "request_id": safe_response.get("request_id"),
            "tool_version": TOOL_VERSION,
            "method_configuration": (safe_response.get("execution") or {}).get("permitted_parameters") or {},
            "generated_artifact_list": list(artifact_names),
            "cache": safe_response.get("cache"),
            "restart_behavior": "Result and artifact caches are bounded, process-local, and cleared when the backend restarts.",
        },
        "authoritative_response": safe_response,
    }
    return report


def _csv_bytes(document: Dict[str, Any]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["section", "metric", "value", "region_id", "region_type", "area_pixels", "bbox"])
    for section, values in document["statistics"].items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if name in {"regions", "bounding_boxes"}:
                continue
            writer.writerow([section, name, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value, "", "", "", ""])
        for region in values.get("regions", []) if isinstance(values.get("regions"), list) else []:
            writer.writerow([section, "region", "", region.get("region_id"), region.get("type", "change"), region.get("area_pixels"), json.dumps(region.get("bbox_pixels") or region.get("bounding_box"))])
    response = document["authoritative_response"]
    for source in (
        ((response.get("vqa_details") or {}).get("single_image_evidence") or {}).get("regions") or [],
        (response.get("cross_modal_analysis") or {}).get("regions") or [],
    ):
        for region in source:
            writer.writerow(["regions", "region", "", region.get("region_id"), region.get("type"), region.get("area_pixels"), json.dumps(region.get("bbox_pixels") or region.get("bounding_box"))])
    grounding = response.get("grounding_result") or {}
    for index, detection in enumerate(grounding.get("detections") or [], start=1):
        value = {
            "label": detection.get("label"),
            "alignment_score": detection.get("score"),
            "bbox_normalized": detection.get("bbox_normalized"),
            "bbox_world": detection.get("bbox_world"),
            "crs": detection.get("crs"),
            "mask_url": detection.get("mask_url"),
            "source": detection.get("source"),
        }
        writer.writerow(["grounding_detections", "model_produced_detection", json.dumps(value, ensure_ascii=False), f"detection_{index}", detection.get("label"), "", json.dumps(detection.get("bbox_pixels"))])
    for index, candidate in enumerate(grounding.get("rejected_candidates") or [], start=1):
        value = {
            "label": candidate.get("label"),
            "alignment_score": candidate.get("score"),
            "box_area_ratio": candidate.get("box_area_ratio"),
            "rejection_reasons": candidate.get("rejection_reasons") or [],
            "quality": candidate.get("quality"),
            "source": candidate.get("source"),
        }
        writer.writerow(["grounding_rejected_candidates", "filtered_model_candidate", json.dumps(value, ensure_ascii=False), f"rejected_candidate_{index}", candidate.get("label"), candidate.get("quality", {}).get("box_area"), json.dumps(candidate.get("bbox_pixels"))])
    return stream.getvalue().encode("utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class _PdfPages:
    width = 1240
    height = 1754
    margin = 84
    bottom = 90

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.pages: List[Image.Image] = []
        self.page = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = self.margin

    def _finish(self) -> None:
        footer = f"GeoVision · SatQuery AI · {self.request_id} · Page {len(self.pages) + 1}"
        self.draw.text((self.margin, self.height - 52), footer, fill=(86, 96, 112), font=_font(18))
        self.pages.append(self.page)
        self.page = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = self.margin

    def ensure(self, height: int) -> None:
        if self.y + height > self.height - self.bottom:
            self._finish()

    def rule(self) -> None:
        self.draw.line((self.margin, self.y, self.width - self.margin, self.y), fill=(208, 216, 226), width=2)
        self.y += 28

    def text(self, value: Any, size: int = 25, bold: bool = False, color: Tuple[int, int, int] = (34, 42, 55), gap: int = 14) -> None:
        text = "Unavailable" if value is None or value == "" else str(value)
        text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        font = _font(size, bold)
        max_width = self.width - self.margin * 2
        lines: List[str] = []
        for paragraph in text.splitlines() or [""]:
            words = paragraph.split() or [""]
            current = words[0]
            for word in words[1:]:
                trial = current + " " + word
                if self.draw.textlength(trial, font=font) <= max_width:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        line_height = size + 10
        self.ensure(max(line_height, len(lines) * line_height + gap))
        for line in lines:
            self.draw.text((self.margin, self.y), line, fill=color, font=font)
            self.y += line_height
        self.y += gap

    def heading(self, value: str, level: int = 1) -> None:
        self.ensure(90)
        self.text(value, size=42 if level == 1 else 31, bold=True, color=(10, 81, 122), gap=20)

    def key_value(self, key: str, value: Any) -> None:
        self.text(f"{key}: {value if value not in (None, '') else 'Unavailable'}", size=22, gap=6)

    def image(self, label: str, path: Path) -> None:
        try:
            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((self.width - self.margin * 2, 520), Image.Resampling.LANCZOS)
                if preview.width < 360 and preview.height < 520:
                    scale = min(360 / max(1, preview.width), 520 / max(1, preview.height))
                    preview = preview.resize((max(1, round(preview.width * scale)), max(1, round(preview.height * scale))), Image.Resampling.NEAREST)
                self.ensure(preview.height + 90)
                self.text(label, size=23, bold=True, gap=8)
                x = self.margin + ((self.width - self.margin * 2 - preview.width) // 2)
                self.page.paste(preview, (x, self.y))
                self.draw.rectangle((x, self.y, x + preview.width, self.y + preview.height), outline=(190, 201, 214), width=2)
                self.y += preview.height + 26
        except OSError:
            self.text(f"{label}: preview unavailable.", size=21)

    def output(self) -> bytes:
        self._finish()
        stream = io.BytesIO()
        self.pages[0].save(stream, format="PDF", save_all=True, append_images=self.pages[1:], resolution=150.0)
        return stream.getvalue()


def _pdf_bytes(document: Dict[str, Any]) -> bytes:
    report = document["report"]
    response = document["authoritative_response"]
    pages = _PdfPages(report["request_id"])
    pages.text("GEOVISION", 25, True, (10, 81, 122), 12)
    pages.text("SatQuery AI Mission Report", 52, True, (18, 28, 43), 24)
    pages.key_value("Analysis type", report["analysis_type"])
    pages.key_value("Request ID", report["request_id"])
    pages.key_value("Generated", report["generated_timestamp"])
    pages.key_value("Status", report["status"])
    pages.rule()
    pages.text("Evidence-backed local remote-sensing analysis with an auditable execution trace.", 27, False, (65, 75, 90), 28)

    pages.heading("Input Summary")
    for label, metadata in (("Primary image", document["input_summary"]["primary"]), ("Secondary image", document["input_summary"]["secondary"])):
        if not metadata:
            continue
        role = metadata.get("observation_role")
        pages.text(f"{role.upper() if role == 'sar' else role.title()} observation" if role else label, 26, True)
        for key in ("file_id", "observation_role", "auto_detected_modality", "effective_modality", "safe_name", "format", "width", "height", "band_count", "available_band_names", "selected_visual_bands", "band_selection_reason", "dtype", "crs", "is_georeferenced", "bounds"):
            pages.key_value(key.replace("_", " ").title(), metadata.get(key))
    compatibility = document["input_summary"].get("compatibility")
    if compatibility:
        pages.text("Compatibility", 26, True)
        for key in ("compatible", "alignment_level", "scientific_classification", "same_dimensions", "same_crs", "same_transform", "same_resolution", "same_orientation", "nodata_compatible", "overlap_ratio", "resampling_required", "recommended_action"):
            pages.key_value(key.replace("_", " ").title(), compatibility.get(key))

    pages.heading("Query and Answer")
    pages.key_value("Original query", document["user_query"]["original_query"])
    pages.key_value("Detected task", document["user_query"]["detected_task"])
    pages.key_value("Question category", document["user_query"]["question_category"])
    pages.key_value("Target concept", document["user_query"]["target_concept"])
    pages.text(document["answer"]["text"] or "No textual answer was produced.", 30, True, (21, 72, 101), 18)
    pages.key_value("Answer source", document["answer"]["source"])

    pages.heading("Evidence Products")
    pages.text(document["evidence"]["disclaimer"], 21, False, (108, 77, 16), 20)
    for item in document["evidence"]["preview_products"]:
        match = _PREVIEW_PATTERN.fullmatch(item["url"])
        path = preview_file_path(match.group(1)) if match else None
        if path:
            pages.image(item["label"], path)

    pages.heading("Statistics")
    for section, values in document["statistics"].items():
        if not values:
            continue
        pages.text(section.replace("_", " ").title(), 25, True)
        for name, value in values.items():
            if name in {"regions", "bounding_boxes"}:
                pages.key_value(name.replace("_", " ").title(), len(value) if isinstance(value, list) else value)
            elif not isinstance(value, (dict, list)):
                pages.key_value(name.replace("_", " ").title(), value)

    grounding = response.get("grounding_result") or {}
    if grounding:
        pages.heading("Grounding Detections")
        pages.key_value("Target phrase", grounding.get("target_phrase"))
        pages.key_value("Accepted detection count", len(grounding.get("detections") or []))
        pages.key_value("Rejected candidate count", len(grounding.get("rejected_candidates") or []))
        policy = grounding.get("quality_policy") or {}
        pages.key_value("Minimum operational alignment score", policy.get("minimum_alignment_score"))
        pages.key_value("Maximum localized area ratio", policy.get("maximum_localized_area_ratio"))
        pages.text(grounding.get("operational_threshold_disclaimer"), 20, False, (108, 77, 16), 14)
        pages.text("Accepted boxes are model-produced Grounding DINO localization candidates and alignment scores, not ground truth or calibrated scientific probabilities. Mask refinement is not connected.", 20, False, (108, 77, 16), 14)
        for index, detection in enumerate(grounding.get("detections") or [], start=1):
            pages.text(f"Detection {index}: {detection.get('label')} · score {detection.get('score')}", 22, True, gap=4)
            pages.key_value("Pixel box", detection.get("bbox_pixels"))
            pages.key_value("Normalized box", detection.get("bbox_normalized"))
            pages.key_value("World box", detection.get("bbox_world"))
            pages.key_value("CRS", detection.get("crs"))
        for index, candidate in enumerate(grounding.get("rejected_candidates") or [], start=1):
            pages.text(f"Rejected candidate {index}: {candidate.get('label')} · score {candidate.get('score')}", 22, True, gap=4)
            pages.key_value("Source-pixel box", candidate.get("bbox_source_xyxy"))
            pages.key_value("Box area ratio", candidate.get("box_area_ratio"))
            pages.key_value("Rejection reasons", ", ".join(candidate.get("rejection_reasons") or []))
        if grounding.get("rejected_candidates"):
            pages.text("Rejected candidates are retained for audit only and are not described as localized objects or included as accepted visual evidence.", 20, False, (108, 77, 16), 14)

    pages.heading("Confidence and Provenance")
    confidence = document.get("confidence") or {}
    pages.key_value("Confidence level", confidence.get("level"))
    pages.key_value("Calibrated score", confidence.get("score"))
    pages.key_value("Rationale", confidence.get("reason"))
    provenance = document["specialist_provenance"]
    pages.key_value("Selected tools", ", ".join(provenance.get("selected_tools") or []))
    for name, value in (provenance.get("model") or {}).items():
        pages.key_value(name.replace("_", " ").title(), value)
    if provenance.get("change_engine"):
        pages.text("Hybrid change-engine provenance", 26, True)
        for name, value in provenance["change_engine"].items():
            pages.key_value(name.replace("_", " ").title(), value)
    if provenance.get("ttp"):
        learned_change_name = provenance["ttp"].get("model") or "Learned change detector"
        for name in ("status", "model", "architecture", "training_dataset", "checkpoint", "checkpoint_fingerprint", "device", "runtime_ms", "model_load_ms", "reused_model"):
            pages.key_value(f"{learned_change_name} {name.replace('_', ' ')}".title(), provenance["ttp"].get(name))
        pages.text(f"{learned_change_name} output is a model-generated binary change prediction and is not ground truth.", 20, False, (108, 77, 16), 14)
    if provenance.get("mask_comparison"):
        pages.key_value("Mask IoU (not accuracy)", provenance["mask_comparison"].get("iou"))
        pages.key_value("Pixel agreement", provenance["mask_comparison"].get("agreement_percentage"))
        pages.key_value("Pixel disagreement", provenance["mask_comparison"].get("disagreement_percentage"))
    if provenance.get("satquery_vision_encoder"):
        sve = provenance["satquery_vision_encoder"]
        pages.text("Remote-Sensing Adaptation", 26, True)
        for name in ("model", "model_version", "backbone", "adaptation_dataset", "adapter_checksum_fingerprint", "device", "runtime_ms", "status"):
            pages.key_value(f"SVE {name.replace('_', ' ')}".title(), sve.get(name))
        for prior in sve.get("scene_priors") or []:
            pages.key_value(f"Scene prior · {prior.get('label')}", prior.get("similarity"))
        caption_consistency = sve.get("caption_consistency") or {}
        if caption_consistency:
            pages.key_value("Caption consistency", caption_consistency.get("score"))
            pages.key_value("Selected caption candidate", caption_consistency.get("selected_candidate_index"))
        vqa_consistency = sve.get("vqa_consistency") or {}
        if vqa_consistency:
            pages.key_value("VQA evidence consistency", vqa_consistency.get("state"))
        pages.text(sve.get("disclaimer"), 20, False, (108, 77, 16), 14)
    pages.key_value("Runtime", f"{provenance.get('runtime_ms')} ms")
    pages.text("Confidence values are reported only where they are real; heuristic semantic outputs are not calibrated probabilities.", 20, False, (108, 77, 16))

    pages.heading("Observable Execution Trace")
    for index, step in enumerate((document.get("execution_trace") or {}).get("steps") or [], start=1):
        pages.text(f"{index}. {step.get('tool')} · {step.get('status')} · {step.get('duration_ms')} ms", 21, False, gap=4)
        if step.get("parameters"):
            pages.text(json.dumps(step["parameters"], sort_keys=True), 17, False, (86, 96, 112), 6)

    pages.heading("Scientific Limitations")
    for limitation in document["scientific_limitations"]:
        pages.text(f"• {limitation}", 21, False, gap=8)
    # Keep this compact closing section together instead of orphaning its
    # heading at the bottom of the preceding page.
    pages.ensure(500)
    pages.heading("Reproducibility Summary")
    for key, value in document["reproducibility"].items():
        formatted = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        pages.text(f"{key.replace('_', ' ').title()}: {formatted}", size=18, gap=4)
    return pages.output()


class ReportArtifactStore:
    def __init__(self) -> None:
        self.max_items = _configured_int("SATQUERY_REPORT_MAX_ARTIFACTS", 128, 4)
        self.ttl_seconds = _configured_int("SATQUERY_REPORT_TTL_SECONDS", DEFAULT_TTL_SECONDS, 60)
        self._items: "OrderedDict[str, Tuple[Path, float]]" = OrderedDict()
        self._lock = threading.RLock()

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for name, (path, touched) in list(self._items.items()):
            if touched < cutoff:
                path.unlink(missing_ok=True)
                self._items.pop(name, None)
        while len(self._items) > self.max_items:
            _, (path, _) = self._items.popitem(last=False)
            path.unlink(missing_ok=True)

    def put(self, name: str, data: bytes) -> Path:
        if not _ARTIFACT_PATTERN.fullmatch(name):
            raise ValueError("Unsafe report artifact filename.")
        with self._lock:
            self._cleanup()
            path = (REPORT_DIR / name).resolve()
            if path.parent != REPORT_DIR:
                raise ValueError("Unsafe report artifact path.")
            path.write_bytes(data)
            self._items[name] = (path, time.time())
            self._items.move_to_end(name)
            self._cleanup()
            return path

    def get(self, name: str) -> Optional[Path]:
        with self._lock:
            self._cleanup()
            item = self._items.get(name)
            if item is None or not item[0].is_file():
                return None
            self._items[name] = (item[0], time.time())
            self._items.move_to_end(name)
            return item[0]

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._cleanup()
            return {
                "stored_artifacts": len(self._items),
                "max_artifacts": self.max_items,
                "ttl_seconds": self.ttl_seconds,
            }


ARTIFACT_STORE = ReportArtifactStore()


def artifact_file_path(name: str) -> Optional[Path]:
    return ARTIFACT_STORE.get(name)


def _zip_bytes(document: Dict[str, Any], pdf: bytes, metadata: bytes, statistics: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("report/mission-report.pdf", pdf)
        archive.writestr("report/mission-report.json", metadata)
        archive.writestr("report/statistics-and-regions.csv", statistics)
        for index, item in enumerate(document["evidence"]["preview_products"], start=1):
            match = _PREVIEW_PATTERN.fullmatch(item["url"])
            path = preview_file_path(match.group(1)) if match else None
            if path:
                safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", item["label"]).strip("_")[:70]
                archive.write(path, f"evidence/{index:02d}_{safe_label}.png")
        published_references = {item["url"] for item in document["evidence"]["preview_products"]}
        evidence_manifest = {
            "schema_version": SCHEMA_VERSION,
            "request_id": document["report"]["request_id"],
            "evidence_products": [
                item for item in document["evidence"]["items"]
                if item.get("reference") in published_references
            ],
        }
        archive.writestr("evidence/manifest.json", json.dumps(evidence_manifest, indent=2, ensure_ascii=False))
        archive.writestr(
            "README.txt",
            "GeoVision SatQuery mission report package\n\n"
            f"Request ID: {document['report']['request_id']}\n"
            f"Task: {document['report']['analysis_type']}\n"
            "All evidence is model-generated or deterministic heuristic output as labelled; it is not ground truth.\n"
            "No original uploads, API keys, model-cache paths, local filesystem paths, environment values, or hidden reasoning are included.\n",
        )
    return stream.getvalue()


def generate_report(request_id: str, formats: List[ReportFormat]) -> ReportResponse:
    started = time.perf_counter()
    record = MISSION_STORE.get_by_request(request_id)
    if record is None:
        raise KeyError("The requested SatQuery result is unavailable or expired.")
    if not result_is_reportable(record.response):
        raise ValueError("This result does not contain a reportable successful, partial, or alignment outcome.")
    requested = list(dict.fromkeys(formats or [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.ZIP]))
    base = f"GeoVision_SatQuery_{record.response.task.value}_{record.response.request_id}"
    names = [f"{base}.{item.value}" for item in requested]
    generated_at = utc_now()
    document = build_report_document(record.response, generated_at, names)
    metadata = json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")
    statistics = _csv_bytes(document)
    pdf = _pdf_bytes(document)
    payloads = {
        ReportFormat.PDF: pdf,
        ReportFormat.JSON: metadata,
        ReportFormat.CSV: statistics,
        ReportFormat.ZIP: _zip_bytes(document, pdf, metadata, statistics),
    }
    artifacts: List[ReportArtifact] = []
    for report_format in requested:
        name = f"{base}.{report_format.value}"
        path = ARTIFACT_STORE.put(name, payloads[report_format])
        artifacts.append(
            ReportArtifact(
                format=report_format,
                filename=name,
                url=f"/api/agent/reports/{name}",
                size_bytes=path.stat().st_size,
            )
        )
    return ReportResponse(
        request_id=request_id,
        status="success",
        schema_version=SCHEMA_VERSION,
        artifacts=artifacts,
        warnings=["Report artifacts are temporary and expire with the bounded local backend cache."],
        runtime_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )

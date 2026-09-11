"""Truthful ISRO SIH 26167 readiness matrix for the local SatQuery MVP."""

from __future__ import annotations

from .models import ComplianceRequirement, ComplianceResponse
from .registry import tool_definition
from .reporting import utc_now


def compliance_summary() -> ComplianceResponse:
    caption_status = tool_definition("rs_captioner").status.value
    grounding_available = tool_definition("rs_grounder").status.value == "available"
    grounding_status = "optional_available" if grounding_available else "optional_not_implemented"
    rows = [
        ("Single optical image", "Validated optical/multispectral ingestion and analysis", "available", "input_validator", "ingestion and caption/VQA tests", "RGB-like interpretation is required for semantic workflows."),
        ("Single SAR input acceptance", "Validated SAR ingestion and metadata/preview generation", "available", "input_validator", "ingestion modality tests", "SAR captioning and single-image SAR VQA are not implemented."),
        ("GeoTIFF/TIFF support", "Rasterio metadata plus bounded display preparation", "available", "input_validator", "GeoTIFF/TIFF ingestion tests", "Display previews are not scientific products."),
        ("Single-image VQA", "RSVQA Specialist v1 for trained presence, comparison, count, and rural/urban answers with deterministic fallback", "available", "rsvqa_vqa_specialist", "specialist artifact, prediction, routing, API, confidence, fallback, and Smoke-50 tests", "Softmax confidence is not calibrated; count labels are not physical inventories."),
        ("Captioning", "RSICD-fine-tuned BLIP optical captioning", caption_status, "rs_captioner", "captioner contract and optional checkpoint test", "Requires the caption checkpoint to be cached for offline first use."),
        ("Bi-temporal change description", "Controlled summary over normalized difference and regions", "available", "bitemporal_change_analyzer", "change and change-VQA tests", "No semantic cause or land-cover transition is inferred."),
        ("Learned optical change evidence", "Official TTP learned mask with independent deterministic support and safe fallback", "available", "ttp_change_detector", "TTP client, hybrid, fallback, mask-validation, and registry tests", "Runtime readiness requires the separately deployed verified CUDA service; output is not ground truth."),
        ("Change VQA", "Questions answered from measured change statistics", "available", "bitemporal_change_analyzer", "change-VQA tests", "Requires compatible or explicitly pre-aligned inputs."),
        ("Optical–SAR joint analysis", "Deterministic exactly aligned evidence fusion", "available", "cross_modal_optical_sar_analyzer", "cross-modal tests", "Relative evidence is not calibrated backscatter or semantic ground truth."),
        ("Deterministic agent routing", "Rule-based task and specialist selection", "available", "input_validator", "router tests", "Unsupported queries fail explicitly."),
        ("Model/tool registry", "Public truthful specialist registry", "available", "input_validator", "registry tests", "Unavailable tools remain visible."),
        ("Evidence", "Safe UUID previews, masks, overlays, and regions", "available", "task specialists", "preview and artifact tests", "Products are labelled model-generated or heuristic."),
        ("Confidence", "Task-specific rationale with null scores where uncalibrated", "available", "task specialists", "confidence contract tests", "Heuristic VQA never reports high calibrated confidence."),
        ("Observable execution trace", "Ordered stages, statuses, timings, and safe parameters", "available", "task specialists", "trace completeness tests", "Timings vary by device and model first load."),
        ("Downloadable mission report", "Backend-authoritative PDF, JSON, CSV, and ZIP", "available", "report_generator", "mission report integration tests", "Artifacts and result records expire from bounded process-local storage."),
        ("Remote-sensing-adapted component", "RSVQA-LR-trained VQA specialist plus RSICD-fine-tuned BLIP captioner", "available", "rsvqa_vqa_specialist", "specialist checkpoint/provenance and caption provenance tests", "Each trained component remains limited to its documented task and input domain."),
        ("Text-guided grounding", "Official local Grounding DINO tiny zero-shot box grounding", grounding_status, "rs_grounder", "grounding unit, API, report, and real checkpoint tests", "Optical/RGB-like multispectral only; model-produced boxes are not ground truth and mask refinement is not connected."),
    ]
    requirements = [
        ComplianceRequirement(
            requirement=requirement,
            implementation=implementation,
            status=status,
            tool_or_model=tool,
            test_coverage=tests,
            limitation=limitation,
        )
        for requirement, implementation, status, tool, tests, limitation in rows
    ]
    mandatory = [item for item in requirements if not item.status.startswith("optional_")]
    return ComplianceResponse(
        generated_at=utc_now(),
        project="GeoVision · SatQuery AI · ISRO SIH 26167",
        requirements=requirements,
        mandatory_satisfied=sum(item.status == "available" for item in mandatory),
        mandatory_total=len(mandatory),
        optional_not_implemented=[] if grounding_available else ["Text-guided grounding"],
    )

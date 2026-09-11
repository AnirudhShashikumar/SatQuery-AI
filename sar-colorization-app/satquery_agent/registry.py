"""Public registry of specialist capabilities visible to SatQuery."""

from __future__ import annotations

from typing import Any, Dict, List

from .models import (
    ImageModality,
    ImplementationStatus,
    InputMode,
    Modality,
    RepresentationType,
    TaskType,
    ToolDefinition,
    SpecialistParameterSpec,
)
from .specialists.captioner import (
    ADAPTATION_DATASET,
    BASE_ARCHITECTURE,
    CHECKPOINT,
    LIMITATIONS,
    MODEL_LICENSE,
    MODEL_SOURCE,
    captioner_configured,
)
from .specialists.cross_modal import METHOD_LIMITATIONS
from .specialists.single_image_evidence import METHOD_LIMITATIONS as VQA_EVIDENCE_LIMITATIONS
from .specialists.rsvqa_specialist import (
    EXPECTED_CHECKPOINT_SHA256 as RSVQA_CHECKPOINT_SHA256,
    rsvqa_specialist_configured,
)
from .specialists import grounder as grounder_specialist
from .services.sve_service import LIMITATIONS as SVE_LIMITATIONS, sve_configured


TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        id="satquery_vision_encoder_v1",
        display_name="SatQuery Vision Encoder v1",
        supported_tasks=[TaskType.CAPTIONING, TaskType.VQA, TaskType.GROUNDING, TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA, TaskType.CROSS_MODAL_ANALYSIS],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        supported_input_modes=[InputMode.SINGLE, InputMode.CROSS_MODAL, InputMode.BI_TEMPORAL],
        status=ImplementationStatus.AVAILABLE if sve_configured() else ImplementationStatus.NOT_IMPLEMENTED,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint="sha256:a99c0bf0fb44",
        base_architecture="OpenCLIP ViT-L-14",
        adaptation_dataset="BigEarthNet.txt",
        source="Verified local SatQuery Vision Encoder v1 adapter bundle",
        limitations=SVE_LIMITATIONS,
        method_type="remote-sensing-adapted vision-language encoder",
        evidence_source="normalized 768-D OpenCLIP scene embeddings with controlled text prompts",
        evidence_outputs=["scene priors", "image-text similarity", "caption consistency", "routing support", "scene-level semantic comparison"],
        supported_image_modalities=[ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["scene_embedding_evidence", "similarity", "scene_priors"],
        specialist_version="1.0.0",
        notes="Scene-level supporting evidence only; no object grounding, segmentation, calibrated classification, or ground truth is claimed.",
    ),
    ToolDefinition(
        id="input_validator",
        display_name="Input Compatibility Validator",
        supported_tasks=list(TaskType),
        supported_modalities=list(Modality),
        supported_input_modes=list(InputMode),
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=False,
        service_path="/api/agent/query",
        supported_image_modalities=list(ImageModality),
        supported_representations=list(RepresentationType),
        supports_preview_inputs=True,
        output_types=["validation_record"],
        specialist_version="input-validator-2.0",
        notes="Validates routing metadata and input compatibility without inspecting image pixels.",
    ),
    ToolDefinition(
        id="rs_captioner",
        display_name="Remote-Sensing Captioner",
        supported_tasks=[TaskType.CAPTIONING],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE if captioner_configured() else ImplementationStatus.NOT_IMPLEMENTED,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint=CHECKPOINT,
        base_architecture=BASE_ARCHITECTURE,
        adaptation_dataset=ADAPTATION_DATASET,
        model_license=MODEL_LICENSE,
        source=MODEL_SOURCE,
        limitations=LIMITATIONS,
        supported_image_modalities=[ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["caption", "model_provenance"],
        specialist_version="rsicd-blip-captioner-1.0",
        notes="RSICD-fine-tuned BLIP captioning for single optical or RGB-like multispectral imagery.",
    ),
    ToolDefinition(
        id="rsvqa_vqa_specialist",
        display_name="SatQuery RSVQA Specialist v1",
        supported_tasks=[TaskType.VQA],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE if rsvqa_specialist_configured() else ImplementationStatus.NOT_IMPLEMENTED,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint=f"sha256:{RSVQA_CHECKPOINT_SHA256[:12]}",
        base_architecture="OpenCLIP ViT-L-14 + frozen SatQuery Vision Encoder v1 adapter + multi-task fusion head",
        adaptation_dataset="RSVQA-LR official training split",
        source="Verified local SatQuery RSVQA Specialist v1 export bundle",
        limitations=[
            "Supports RSVQA-style presence, comparison, count, and rural/urban questions only.",
            "Count predictions are benchmark-label classifications, not physical object inventories.",
            "Softmax confidence is not calibrated.",
            "Optical and rendered RGB inputs only; raw SAR is unsupported.",
        ],
        method_type="trained remote-sensing visual question answering",
        evidence_source="frozen SVE image/question embeddings and exported learned task heads",
        evidence_outputs=["task logits", "task softmax scores", "decoded vocabulary answer"],
        supported_question_categories=["presence_vqa", "comparison_vqa", "rural_urban_classification", "count_vqa"],
        supported_image_modalities=[ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["answer", "confidence", "task", "logits", "probabilities"],
        specialist_version="1.0.0",
        notes="Primary RSVQA-style answer source. Captioning, SVE priors, and Grounding DINO remain optional supporting evidence.",
    ),
    ToolDefinition(
        id="rs_vqa",
        display_name="Controlled Remote-Sensing Visual Question Answering",
        supported_tasks=[TaskType.VQA],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=False,
        service_path="/api/agent/query",
        method_type="deterministic evidence-grounded VQA",
        evidence_source="computed optical support maps and raster metadata",
        evidence_outputs=["water support", "vegetation support", "built-up/structural support", "agriculture support", "scene evidence overlay", "connected evidence regions"],
        supported_question_categories=["dominant_land_cover", "presence_water", "presence_buildings", "presence_vegetation", "presence_agriculture", "composition_built_up", "scene_type", "relative_coverage", "metadata_question"],
        limitations=VQA_EVIDENCE_LIMITATIONS,
        supported_image_modalities=[ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["answer", "support_maps", "evidence_overlay", "regions"],
        specialist_version="controlled-rs-vqa-1.0",
        notes="Local controlled VQA over deterministic evidence; no RSVQA fine-tuning or generic LLM is claimed. The system-level remote-sensing adaptation requirement is provided by the separate RSICD-adapted captioning specialist.",
    ),
    ToolDefinition(
        id="rs_grounder",
        display_name="Remote-Sensing Visual Grounder",
        supported_tasks=[TaskType.GROUNDING],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE if grounder_specialist.grounder_configured() else ImplementationStatus.NOT_IMPLEMENTED,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint=grounder_specialist.CHECKPOINT,
        base_architecture=grounder_specialist.BASE_ARCHITECTURE,
        adaptation_dataset=grounder_specialist.ADAPTATION_DATASET,
        model_license=grounder_specialist.MODEL_LICENSE,
        source=grounder_specialist.MODEL_SOURCE,
        limitations=grounder_specialist.LIMITATIONS,
        method_type="Grounding DINO open-vocabulary proposals with learned remote-sensing rescoring",
        evidence_source="Grounding DINO model-produced boxes with VRSBench-trained SatQuery Grounding Specialist v1.1 proposal scores",
        evidence_outputs=["annotated detection preview", "pixel bounding boxes", "normalized bounding boxes", "world-coordinate boxes when georeferenced"],
        notes="Grounding DINO proposals rescored by SatQuery Grounding Specialist v1.1 trained on VRSBench. Boxes are model-produced evidence, not ground truth; mask refinement is not connected.",
        supported_image_modalities=[ImageModality.OPTICAL_RGB, ImageModality.MULTISPECTRAL],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["bounding_boxes", "annotated_preview"],
        specialist_version="grounding-specialist-v1.1-step600",
    ),
    ToolDefinition(
        id="sar_water_segmenter",
        display_name="SAR Water Segmenter",
        supported_tasks=[TaskType.SAR_WATER_SEGMENTATION],
        supported_modalities=[Modality.SAR],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        method_type="deterministic low-backscatter candidate segmentation",
        evidence_source="value-preserving SAR preprocessing, image-adaptive threshold, morphology, and connected components",
        evidence_outputs=["normalized SAR preview", "binary candidate mask", "candidate overlay", "connected component outlines"],
        supported_image_modalities=[ImageModality.SAR_PREVIEW, ImageModality.SAR_VV, ImageModality.SAR_VH, ImageModality.SAR_VV_VH],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=1,
        maximum_bands=2,
        supports_preview_inputs=True,
        output_types=["binary_mask", "overlay", "regions", "image_area_percentage"],
        specialist_version="heuristic-sar-water-1.0",
        limitations=[
            "Heuristic candidates are not trained-model predictions or ground truth.",
            "Radar shadow and smooth non-water surfaces may produce similarly low returns.",
            "Geographic area is unavailable without a usable georeference.",
        ],
        notes="Conservative qualitative SAR water-candidate detector for single-channel previews and verified one/two-band SAR rasters. When explicitly enabled, learned SAR-to-optical visualization may be added as secondary interpretive evidence; generated images are not observed optical measurements.",
    ),
    ToolDefinition(
        id="sar_scene_analyzer",
        display_name="SAR Scene Analyzer",
        supported_tasks=[TaskType.SAR_SCENE_ANALYSIS, TaskType.SAR_QUALITY_INSPECTION],
        supported_modalities=[Modality.SAR],
        supported_input_modes=[InputMode.SINGLE],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        method_type="deterministic SAR intensity and input-quality inspection",
        evidence_source="finite-value distribution and local texture statistics",
        supported_image_modalities=[ImageModality.SAR_PREVIEW, ImageModality.SAR_VV, ImageModality.SAR_VH, ImageModality.SAR_VV_VH],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=1,
        maximum_bands=2,
        supports_preview_inputs=True,
        output_types=["intensity_summary", "input_quality", "normalized_preview"],
        specialist_version="deterministic-sar-scene-1.0",
        limitations=["Does not infer exact land-cover classes, object counts, or semantic identity from arbitrary SAR previews."],
        notes="Reports broad intensity, contrast, validity, and texture evidence without claiming a trained semantic classifier. Optional translated optical-like evidence is disclosed separately and is never treated as an observed optical measurement.",
    ),
    ToolDefinition(
        id="changerex_change_detector",
        display_name="ChangerEx Local Change Detector",
        supported_tasks=[TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA],
        supported_modalities=[Modality.OPTICAL],
        supported_input_modes=[InputMode.BI_TEMPORAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint="ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth",
        base_architecture="ChangerEx + IA-ResNetV1c-18 backbone",
        adaptation_dataset="LEVIR-CD",
        model_license="Open-CD project terms; verify official checkpoint terms with deployment records",
        source="https://github.com/likyoo/open-cd @ 09c03eb1077f06191c5448ea1fcf2f2f88ec98c2",
        method_type="deep_learning",
        evidence_source="changed-class probability map and binary learned mask from one lazily loaded, SHA-256-verified Pure-PyTorch model",
        evidence_outputs=["ChangerEx probability map", "binary learned mask", "overlay", "comparison products", "connected-region statistics"],
        limitations=[
            "Binary change only; no semantic cause or land-cover transition is inferred.",
            "LEVIR-CD building-change domain bias may reduce generalization.",
            "Inputs must be geometrically co-registered.",
            "Model output is not ground truth.",
        ],
        supported_image_modalities=[ImageModality.OPTICAL_RGB],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["probability_map", "binary_mask", "overlay", "regions", "statistics"],
        specialist_version="changerex-opencd-v1.1.0-da3f569306da",
        notes="Default local learned engine for aligned optical pairs. The validated Pure-PyTorch checkpoint uses one lazy process-wide singleton; deterministic evidence and automatic fallback remain active.",
    ),
    ToolDefinition(
        id="ttp_change_detector",
        display_name="TTP Learned Change Detector",
        supported_tasks=[TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA],
        supported_modalities=[Modality.OPTICAL],
        supported_input_modes=[InputMode.BI_TEMPORAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/agent/query",
        checkpoint="epoch_260.pth",
        base_architecture="SAM ViT-L + LoRA SiamEncoderDecoder",
        adaptation_dataset="LEVIR-CD",
        model_license="Apache-2.0 (official repository); verify checkpoint terms with deployment records",
        source="https://github.com/KyanChen/TTP @ 431377d; https://huggingface.co/KyanChen/TTP",
        method_type="deep_learning",
        evidence_source="changed-class probability decision and binary learned mask from one lazily loaded, verified, persistent CUDA model",
        evidence_outputs=["TTP probability-to-binary decision", "TTP raw learned mask", "TTP display mask", "TTP overlay", "connected-region statistics"],
        limitations=[
            "Binary change only; no semantic cause or land-cover transition is inferred.",
            "LEVIR-CD building-change domain bias may reduce generalization.",
            "A CUDA service and the exactly verified epoch_260.pth checkpoint are required.",
            "Model output is not ground truth.",
        ],
        supported_image_modalities=[ImageModality.OPTICAL_RGB],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=3,
        maximum_bands=4,
        supports_preview_inputs=True,
        output_types=["binary_mask", "overlay", "regions", "statistics"],
        specialist_version="ttp-431377d-epoch260-60294429b3d",
        notes="Optional alternate learned detector for validated aligned optical pairs. The official checkpoint is verified before one lazy singleton load; deterministic evidence remains independent and automatic fallback is enabled.",
    ),
    ToolDefinition(
        id="bitemporal_change_analyzer",
        display_name="Bi-Temporal Change Analyzer",
        supported_tasks=[TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR],
        supported_input_modes=[InputMode.BI_TEMPORAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=False,
        service_path="/api/agent/change",
        method_type="deterministic normalized-difference change analysis with controlled answers",
        evidence_source="computed binary change mask and connected-component statistics",
        supported_question_categories=["change_summary", "change_percentage", "largest_change", "change_region_count", "change_magnitude"],
        supported_image_modalities=[
            ImageModality.OPTICAL_RGB,
            ImageModality.OPTICAL_GRAYSCALE,
            ImageModality.PANCHROMATIC,
            ImageModality.SAR_PREVIEW,
            ImageModality.SAR_VV,
            ImageModality.SAR_VH,
            ImageModality.SAR_VV_VH,
            ImageModality.MULTISPECTRAL,
        ],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=1,
        supports_preview_inputs=True,
        output_types=["difference_map", "binary_mask", "overlay", "regions", "statistics"],
        specialist_version="bitemporal-change-1.0",
        notes="Local deterministic visual change analysis and controlled non-causal answers; no semantic cause is inferred.",
    ),
    ToolDefinition(
        id="cross_modal_optical_sar_analyzer",
        display_name="Optical-SAR Joint Analyzer",
        supported_tasks=[TaskType.CROSS_MODAL_ANALYSIS],
        supported_modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR],
        supported_input_modes=[InputMode.CROSS_MODAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=False,
        service_path="/api/agent/cross-modal",
        required_modalities={
            "optical_input": [Modality.OPTICAL, Modality.MULTISPECTRAL],
            "sar_input": [Modality.SAR],
        },
        method_type="deterministic evidence fusion",
        evidence_outputs=[
            "optical visible-spectrum evidence",
            "SAR relative-intensity evidence",
            "water-likelihood support",
            "built-up/structural-likelihood support",
            "visible-spectrum vegetation support",
            "cross-modal agreement and disagreement",
            "connected regions",
        ],
        limitations=METHOD_LIMITATIONS,
        supported_image_modalities=[
            ImageModality.OPTICAL_RGB,
            ImageModality.MULTISPECTRAL,
            ImageModality.SAR_PREVIEW,
            ImageModality.SAR_VV,
            ImageModality.SAR_VH,
            ImageModality.SAR_VV_VH,
        ],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=1,
        supports_preview_inputs=True,
        output_types=["support_maps", "agreement_map", "disagreement_map", "regions", "statistics"],
        specialist_version="optical-sar-evidence-fusion-1.0",
        notes="Local deterministic optical-SAR evidence fusion for exactly aligned pairs; no trained semantic classifier or hosted model is used.",
    ),
    ToolDefinition(
        id="pix2pix_reconstruction",
        display_name="Optical Reconstruction Specialist",
        supported_tasks=[TaskType.CROSS_MODAL_ANALYSIS],
        supported_modalities=[Modality.SAR],
        supported_input_modes=[InputMode.SINGLE, InputMode.CROSS_MODAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/pix2pix/infer",
        supported_image_modalities=[ImageModality.SAR_PREVIEW, ImageModality.SAR_VV, ImageModality.SAR_VH],
        supported_representations=[RepresentationType.DISPLAY_PREVIEW, RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=1,
        maximum_bands=1,
        supports_preview_inputs=True,
        output_types=["optical_like_reconstruction"],
        specialist_version="pix2pix-reconstruction-1.0",
        notes="Existing SAR-to-optical reconstruction endpoint; not automatically executed by SatQuery.",
    ),
    ToolDefinition(
        id="sarfusionformer_analysis",
        display_name="SARFusionFormer Reconstruction Specialist",
        supported_tasks=[TaskType.CROSS_MODAL_ANALYSIS],
        supported_modalities=[Modality.SAR],
        supported_input_modes=[InputMode.SINGLE, InputMode.CROSS_MODAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/sarfusionformer/infer",
        supported_image_modalities=[ImageModality.SAR_VV_VH],
        supported_representations=[RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=2,
        maximum_bands=2,
        output_types=["optical_like_reconstruction"],
        specialist_version="sarfusionformer-reconstruction-1.0",
        notes="Existing VV/VH reconstruction endpoint; not automatically executed by SatQuery.",
    ),
    ToolDefinition(
        id="color_corrector",
        display_name="Optional Reconstruction Color Corrector",
        supported_tasks=[TaskType.CROSS_MODAL_ANALYSIS],
        supported_modalities=[Modality.SAR],
        supported_input_modes=[InputMode.SINGLE, InputMode.CROSS_MODAL],
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=True,
        service_path="/api/sarfusionformer/infer",
        supported_image_modalities=[ImageModality.SAR_VV_VH],
        supported_representations=[RepresentationType.SCIENTIFIC_RASTER],
        minimum_bands=2,
        maximum_bands=2,
        output_types=["color_corrected_reconstruction"],
        specialist_version="optional-color-corrector-1.0",
        limitations=["Runs only as an explicitly requested post-processing step after SARFusionFormer reconstruction."],
        notes="Existing optional color-refinement stage; it is never selected for semantic SAR analysis.",
    ),
    ToolDefinition(
        id="report_generator",
        display_name="SatQuery Report Generator",
        supported_tasks=[TaskType.REPORT_GENERATION],
        supported_modalities=list(Modality),
        supported_input_modes=list(InputMode),
        status=ImplementationStatus.AVAILABLE,
        remote_sensing_adapted=False,
        service_path="/api/agent/report",
        method_type="backend-authoritative local document and artifact generation",
        evidence_source="bounded request-id result store and safe UUID preview products",
        evidence_outputs=["PDF mission report", "typed JSON", "statistics and regions CSV", "complete ZIP package"],
        limitations=[
            "Results and report artifacts are stored in bounded process-local memory and temporary storage.",
            "Backend restart or expiry requires the analysis to be run again.",
            "Reports contain prepared previews and evidence products, not original uploaded files.",
        ],
        supported_image_modalities=list(ImageModality),
        supported_representations=list(RepresentationType),
        supports_preview_inputs=True,
        output_types=["pdf", "json", "csv", "zip"],
        specialist_version="satquery-report-generator-1.0",
        notes="Generates local PDF, JSON, CSV, and ZIP mission artifacts from authoritative stored SatQuery results.",
    ),
]

PARAMETER_SCHEMAS: Dict[str, Dict[str, SpecialistParameterSpec]] = {
    "rs_grounder": {
        "box_threshold": SpecialistParameterSpec(type="float", default=0.18, minimum=0.0, maximum=1.0, description="Grounding DINO proposal threshold."),
        "text_threshold": SpecialistParameterSpec(type="float", default=0.15, minimum=0.0, maximum=1.0, description="Grounding text-token threshold."),
        "device": SpecialistParameterSpec(type="enum", default="auto", choices=["auto", "mps", "cpu", "cuda"]),
    },
    "sar_water_segmenter": {
        "threshold": SpecialistParameterSpec(type="float", default=0.35, minimum=0.0, maximum=1.0),
    },
    "changerex_change_detector": {
        "threshold": SpecialistParameterSpec(type="float", default=0.5, minimum=0.0, maximum=1.0),
        "device": SpecialistParameterSpec(type="enum", default="auto", choices=["auto", "mps", "cpu"]),
    },
    "ttp_change_detector": {
        "threshold": SpecialistParameterSpec(type="float", default=0.5, minimum=0.0, maximum=1.0),
        "device": SpecialistParameterSpec(type="enum", default="cuda", choices=["cuda"]),
    },
    "pix2pix_reconstruction": {
        "device": SpecialistParameterSpec(type="enum", default="auto", choices=["auto", "mps", "cpu", "cuda"]),
    },
    "sarfusionformer_analysis": {
        "device": SpecialistParameterSpec(type="enum", default="auto", choices=["auto", "mps", "cpu", "cuda"]),
    },
    "report_generator": {
        "formats": SpecialistParameterSpec(type="list", default=["pdf"], choices=["pdf", "json", "csv", "zip"]),
    },
}

TOOLS = [
    tool.model_copy(update={
        "allowed_parameters": PARAMETER_SCHEMAS.get(tool.id, {}),
        "defaults": {name: spec.default for name, spec in PARAMETER_SCHEMAS.get(tool.id, {}).items()},
        "constraints": ["Only declared parameters are accepted; checkpoint paths and executable values are forbidden."],
        "model_version": tool.specialist_version,
        "evidence_types": list(tool.evidence_outputs),
    })
    for tool in TOOLS
]

TOOL_BY_ID: Dict[str, ToolDefinition] = {tool.id: tool for tool in TOOLS}


class SpecialistParameterError(ValueError):
    pass


def validate_specialist_parameters(tool_id: str, supplied: Dict[str, Any], *, include_defaults: bool = True) -> Dict[str, Any]:
    """Validate a bounded parameter map without accepting code or checkpoint paths."""

    if tool_id not in TOOL_BY_ID:
        raise SpecialistParameterError(f"Unknown specialist: {tool_id}")
    schemas = TOOL_BY_ID[tool_id].allowed_parameters
    unknown = sorted(set(supplied) - set(schemas))
    if unknown:
        raise SpecialistParameterError(f"Unsupported parameter(s) for {tool_id}: {', '.join(unknown)}")
    output = dict(TOOL_BY_ID[tool_id].defaults) if include_defaults else {}
    for name, value in supplied.items():
        spec = schemas[name]
        valid_type = (
            (spec.type == "float" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (spec.type == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (spec.type == "boolean" and isinstance(value, bool))
            or (spec.type in {"string", "enum"} and isinstance(value, str))
            or (spec.type == "list" and isinstance(value, list))
        )
        if not valid_type:
            raise SpecialistParameterError(f"Parameter '{name}' must have type {spec.type}.")
        if spec.type == "list" and spec.choices and any(item not in spec.choices for item in value):
            raise SpecialistParameterError(f"Parameter '{name}' contains a value outside {spec.choices}.")
        if spec.type == "enum" and spec.choices and value not in spec.choices:
            raise SpecialistParameterError(f"Parameter '{name}' must be one of {spec.choices}.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.minimum is not None and value < spec.minimum:
                raise SpecialistParameterError(f"Parameter '{name}' must be at least {spec.minimum}.")
            if spec.maximum is not None and value > spec.maximum:
                raise SpecialistParameterError(f"Parameter '{name}' must be at most {spec.maximum}.")
        output[name] = float(value) if spec.type == "float" else value
    return output


def public_tool_registry() -> List[ToolDefinition]:
    """Return schema-filtered tool metadata with no secrets or filesystem paths."""
    return [tool_definition(tool.id) for tool in TOOLS]


def tool_definition(tool_id: str) -> ToolDefinition:
    tool = TOOL_BY_ID[tool_id]
    if tool_id == "rs_captioner":
        return tool.model_copy(update={"status": ImplementationStatus.AVAILABLE if captioner_configured() else ImplementationStatus.NOT_IMPLEMENTED})
    if tool_id == "rs_grounder":
        return tool.model_copy(update={"status": ImplementationStatus.AVAILABLE if grounder_specialist.get_grounder().is_available() else ImplementationStatus.NOT_IMPLEMENTED})
    return tool

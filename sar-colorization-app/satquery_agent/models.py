"""Shared backend contracts for the SatQuery agent foundation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InputMode(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"
    BI_TEMPORAL = "bi_temporal"


class ObservationRole(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class RepresentationType(str, Enum):
    SCIENTIFIC_RASTER = "scientific_raster"
    DISPLAY_PREVIEW = "display_preview"
    UNKNOWN_REPRESENTATION = "unknown_representation"


class ImageModality(str, Enum):
    AUTO = "auto"
    OPTICAL_RGB = "optical_rgb"
    OPTICAL_GRAYSCALE = "optical_grayscale"
    PANCHROMATIC = "panchromatic"
    SAR_PREVIEW = "sar_preview"
    SAR_VV = "sar_vv"
    SAR_VH = "sar_vh"
    SAR_VV_VH = "sar_vv_vh"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"


class EvidenceLifecycleState(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class DetectionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class TaskType(str, Enum):
    CAPTIONING = "captioning"
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL_ANALYSIS = "cross_modal_analysis"
    REPORT_GENERATION = "report_generation"
    SAR_WATER_SEGMENTATION = "sar_water_segmentation"
    SAR_SCENE_ANALYSIS = "sar_scene_analysis"
    SAR_QUALITY_INSPECTION = "sar_quality_inspection"
    UNSUPPORTED = "unsupported"


class RequestedOutput(str, Enum):
    TEXT = "text"
    LOCALIZATION = "localization"
    SEGMENTATION_MASK = "segmentation_mask"
    OVERLAY = "overlay"
    QUALITY_REPORT = "quality_report"


class ClassifiedQuery(BaseModel):
    task_type: TaskType
    target: Optional[str] = None
    requested_output: RequestedOutput = RequestedOutput.TEXT
    requires_localization: bool = False
    requires_segmentation: bool = False
    requires_measurement: bool = False


class ToolStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_IMPLEMENTED = "not_implemented"


class ImplementationStatus(str, Enum):
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"


class ReportFormat(str, Enum):
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    ZIP = "zip"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ALIGNMENT_REQUIRED = "alignment_required"
    FAILED = "failed"
    NOT_IMPLEMENTED = "not_implemented"


class ChangeAnalysisStatus(str, Enum):
    SUCCESS = "success"
    ALIGNMENT_REQUIRED = "alignment_required"
    FAILED = "failed"


class CrossModalStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ALIGNMENT_REQUIRED = "alignment_required"
    FAILED = "failed"


class QuestionCategory(str, Enum):
    # RSVQA-LR-compatible controlled question families.  These are deliberately
    # separate from the older descriptive VQA categories below: benchmark-facing
    # adapters need to know when an answer must be a token rather than prose.
    RURAL_URBAN_CLASSIFICATION = "rural_urban_classification"
    PRESENCE_VQA = "presence_vqa"
    COUNT_VQA = "count_vqa"
    COMPARISON_VQA = "comparison_vqa"
    DOMINANT_LAND_COVER = "dominant_land_cover"
    PRESENCE_WATER = "presence_water"
    PRESENCE_BUILDINGS = "presence_buildings"
    PRESENCE_VEGETATION = "presence_vegetation"
    PRESENCE_AGRICULTURE = "presence_agriculture"
    COMPOSITION_BUILT_UP = "composition_built_up"
    SCENE_TYPE = "scene_type"
    RELATIVE_COVERAGE = "relative_coverage"
    METADATA_QUESTION = "metadata_question"
    CHANGE_SUMMARY = "change_summary"
    CHANGE_PERCENTAGE = "change_percentage"
    LARGEST_CHANGE = "largest_change"
    CHANGE_REGION_COUNT = "change_region_count"
    CHANGE_MAGNITUDE = "change_magnitude"
    CHANGE_LOCATION = "change_location"
    BUILT_UP_CHANGE = "built_up_change"
    VEGETATION_CHANGE = "vegetation_change"
    WATER_CHANGE = "water_change"
    INFRASTRUCTURE_CHANGE = "infrastructure_change"
    NO_CHANGE_CHECK = "no_change_check"
    GENERAL_COMPARISON = "general_comparison"
    CROSS_MODAL_AGREEMENT = "cross_modal_agreement"
    CROSS_MODAL_WATER = "cross_modal_water"
    CROSS_MODAL_STRUCTURE = "cross_modal_structure"
    CROSS_MODAL_DISAGREEMENT = "cross_modal_disagreement"
    CROSS_MODAL_REGION_COUNT = "cross_modal_region_count"
    UNSUPPORTED = "unsupported"


class ImageFormat(str, Enum):
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    UNKNOWN = "unknown"


class AlignmentLevel(str, Enum):
    EXACT = "exact"
    GEOSPATIAL_OVERLAP = "geospatial_overlap"
    VISUAL_ONLY = "visual_only"
    INCOMPATIBLE = "incompatible"


class PairCompatibilityClass(str, Enum):
    EXACT_GRID_MATCH = "EXACT_GRID_MATCH"
    SAME_AREA_DIFFERENT_GRID = "SAME_AREA_DIFFERENT_GRID"
    REPROJECTION_REQUIRED = "REPROJECTION_REQUIRED"
    RESAMPLING_REQUIRED = "RESAMPLING_REQUIRED"
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    UNVERIFIABLE = "UNVERIFIABLE"


class AgentQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    input_mode: InputMode
    primary_modality: Modality
    secondary_modality: Optional[Modality] = None
    has_primary_image: bool
    has_secondary_image: bool
    primary_image_modality: ImageModality = ImageModality.AUTO
    primary_representation: RepresentationType = RepresentationType.UNKNOWN_REPRESENTATION
    primary_band_count: Optional[int] = Field(default=None, ge=1)


class ToolDefinition(BaseModel):
    id: str
    display_name: str
    supported_tasks: List[TaskType]
    supported_modalities: List[Modality]
    supported_input_modes: List[InputMode]
    status: ImplementationStatus
    remote_sensing_adapted: bool
    service_path: Optional[str] = None
    checkpoint: Optional[str] = None
    base_architecture: Optional[str] = None
    adaptation_dataset: Optional[str] = None
    model_license: Optional[str] = None
    source: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    required_modalities: Dict[str, List[Modality]] = Field(default_factory=dict)
    method_type: Optional[str] = None
    evidence_outputs: List[str] = Field(default_factory=list)
    evidence_source: Optional[str] = None
    supported_question_categories: List[str] = Field(default_factory=list)
    supported_image_modalities: List[ImageModality] = Field(default_factory=list)
    supported_representations: List[RepresentationType] = Field(default_factory=list)
    minimum_bands: Optional[int] = Field(default=None, ge=1)
    maximum_bands: Optional[int] = Field(default=None, ge=1)
    requires_georeference: bool = False
    supports_preview_inputs: bool = False
    output_types: List[str] = Field(default_factory=list)
    specialist_version: Optional[str] = None
    allowed_parameters: Dict[str, "SpecialistParameterSpec"] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    evidence_types: List[str] = Field(default_factory=list)
    notes: str


class SpecialistParameterSpec(BaseModel):
    type: str
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: List[Any] = Field(default_factory=list)
    description: str = ""


class ValidationStatus(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)


class RoutingPlan(BaseModel):
    detected_task: TaskType
    selected_tools: List[str]
    permitted_parameters: Dict[str, Any]
    validation_status: ValidationStatus
    selection_reason: str


class Confidence(BaseModel):
    level: ConfidenceLevel
    score: Optional[float] = None
    reason: str


class ModelProvenance(BaseModel):
    tool_id: str
    checkpoint: str
    base_architecture: str
    adaptation_dataset: str
    remote_sensing_adapted: bool
    license: Optional[str] = None
    source: str


class CaptionDetails(BaseModel):
    modality: Modality
    device: str
    runtime_ms: int = Field(ge=0)
    model_load_ms: int = Field(ge=0)
    model_reused: bool
    image_representation: str
    bands_used: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class CaptionResult(BaseModel):
    caption: str
    confidence: Confidence
    model: ModelProvenance
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    device: str
    image_representation: str
    bands_used: List[str] = Field(default_factory=list)
    model_load_ms: int = Field(ge=0)
    reused_model: bool = False


class GroundingCandidateQuality(BaseModel):
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    box_width: Optional[float] = None
    box_height: Optional[float] = None
    box_area: Optional[float] = None
    image_area: int = Field(gt=0)
    box_area_ratio: Optional[float] = None
    alignment_score: Optional[float] = None
    finite_score: bool
    finite_coordinates: bool
    positive_area: bool
    in_bounds: bool
    rejection_reasons: List[str] = Field(default_factory=list)


class GroundingQualityPolicy(BaseModel):
    minimum_alignment_score: float = Field(ge=0, le=1)
    maximum_localized_area_ratio: float = Field(gt=0, le=1)
    localized_targets: List[str]
    calibration_status: str


class GroundingDetection(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    bbox_pixels: List[int] = Field(min_length=4, max_length=4)
    bbox_normalized: List[float] = Field(min_length=4, max_length=4)
    bbox_world: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    crs: Optional[str] = None
    mask_url: Optional[str] = None
    source: str = "Grounding DINO"
    quality: GroundingCandidateQuality


class RejectedGroundingCandidate(BaseModel):
    label: str
    score: Optional[float] = Field(default=None, ge=0, le=1)
    bbox_source_xyxy: List[Optional[float]] = Field(min_length=4, max_length=4)
    bbox_pixels: Optional[List[int]] = Field(default=None, min_length=4, max_length=4)
    box_area_ratio: Optional[float] = None
    rejection_reasons: List[str] = Field(default_factory=list)
    quality: GroundingCandidateQuality
    source: str = "Grounding DINO"


class GroundingInputDetails(BaseModel):
    modality: Modality
    bands_used: List[str] = Field(default_factory=list)
    representation: str
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    model_input_width: int = Field(gt=0)
    model_input_height: int = Field(gt=0)
    normalization_method: str


class GroundingResult(BaseModel):
    original_query: str
    target_phrase: str
    detections: List[GroundingDetection] = Field(default_factory=list)
    accepted_detections: List[GroundingDetection] = Field(default_factory=list)
    rejected_candidates: List[RejectedGroundingCandidate] = Field(default_factory=list)
    accepted_detection_count: int = Field(ge=0)
    rejected_candidate_count: int = Field(ge=0)
    quality_policy: GroundingQualityPolicy
    empty_result_explanation: Optional[str] = None
    operational_threshold_disclaimer: str
    annotated_preview_url: Optional[str] = None
    confidence: Confidence
    model: ModelProvenance
    input: GroundingInputDetails
    device: str
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    model_load_ms: int = Field(ge=0)
    model_reused: bool = False
    stage_durations_ms: Dict[str, int] = Field(default_factory=dict, exclude=True)


class EvidenceItem(BaseModel):
    evidence_id: Optional[str] = None
    source_observation_id: Optional[str] = None
    source_observation_ids: List[str] = Field(default_factory=list)
    source_role: Optional[ObservationRole] = None
    source_modality: Optional[ImageModality] = None
    evidence_type: Optional[str] = None
    evidence_run_id: Optional[str] = None
    generator: Optional[str] = None
    status: Optional[EvidenceLifecycleState] = None
    type: str
    label: str
    description: Optional[str] = None
    reference: Optional[str] = None


class ExecutionStep(BaseModel):
    tool: str
    status: ToolStatus
    duration_ms: int = Field(ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExecutionSummary(BaseModel):
    input_mode: InputMode
    selected_tools: List[str]
    steps: List[ExecutionStep]
    duration_ms: int = Field(ge=0)
    permitted_parameters: Dict[str, Any]
    validation: ValidationStatus
    selection_reason: str


class RasterBounds(BaseModel):
    left: float
    bottom: float
    right: float
    top: float


class ImageMetadata(BaseModel):
    observation_role: Optional[ObservationRole] = None
    file_id: str
    original_name: str
    safe_name: str
    format: ImageFormat
    mime_type: str
    size_bytes: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtype: str
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds: Optional[RasterBounds] = None
    nodata: Optional[float] = None
    is_georeferenced: bool = False
    preview_url: Optional[str] = None
    color_interpretation: List[str] = Field(default_factory=list)
    band_descriptions: List[Optional[str]] = Field(default_factory=list)
    input_band_count: Optional[int] = Field(default=None, ge=1)
    available_band_names: List[str] = Field(default_factory=list)
    selected_visual_bands: List[str] = Field(default_factory=list)
    band_selection_reason: Optional[str] = None
    band_statistics: List["RasterBandStatistics"] = Field(default_factory=list)
    representation: RepresentationType = RepresentationType.UNKNOWN_REPRESENTATION
    auto_detected_modality: ImageModality = ImageModality.UNKNOWN
    auto_detection_confidence: DetectionConfidence = DetectionConfidence.UNAVAILABLE
    auto_detection_reason: str = "No modality decision was recorded."
    user_confirmed_modality: Optional[ImageModality] = None
    effective_modality: ImageModality = ImageModality.UNKNOWN
    modality_limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RasterBandStatistics(BaseModel):
    band: int = Field(ge=1)
    description: Optional[str] = None
    dtype: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    standard_deviation: Optional[float] = None
    percentile_1: Optional[float] = None
    percentile_5: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    nodata_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)


class ImageInspectionResponse(BaseModel):
    metadata: ImageMetadata
    content_hash_prefix: str
    requires_modality_confirmation: bool


class PairCompatibility(BaseModel):
    role_validation_status: Optional[str] = None
    optical_slot_detected_modality: Optional[ImageModality] = None
    sar_slot_detected_modality: Optional[ImageModality] = None
    role_match: Optional[bool] = None
    pair_valid: Optional[bool] = None
    compatible: bool
    alignment_level: AlignmentLevel
    same_dimensions: bool
    same_crs: Optional[bool] = None
    same_transform: Optional[bool] = None
    bounds_overlap: Optional[bool] = None
    overlap_ratio: Optional[float] = None
    resampling_required: bool = False
    scientific_classification: Optional[PairCompatibilityClass] = None
    primary_resolution: Optional[List[float]] = None
    secondary_resolution: Optional[List[float]] = None
    same_resolution: Optional[bool] = None
    same_orientation: Optional[bool] = None
    nodata_compatible: Optional[bool] = None
    recommended_action: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class PixelBoundingBox(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(ge=0)
    bottom: int = Field(ge=0)
    area_pixels: int = Field(gt=0)


class ChangeRegion(BaseModel):
    region_id: int = Field(gt=0)
    area_pixels: int = Field(gt=0)
    percentage_of_image: float = Field(ge=0, le=100)
    bounding_box: PixelBoundingBox


class ChangeStatistics(BaseModel):
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    total_pixels: int = Field(gt=0)
    changed_pixels: int = Field(ge=0)
    percentage_changed: float = Field(ge=0, le=100)
    largest_connected_region: int = Field(ge=0)
    number_of_regions: int = Field(ge=0)
    bounding_boxes: List[PixelBoundingBox] = Field(default_factory=list)
    regions: List[ChangeRegion] = Field(default_factory=list)
    normalized_threshold: float = Field(ge=0, le=1)


class ChangePreviewUrls(BaseModel):
    before: Optional[str] = None
    after: Optional[str] = None
    difference: Optional[str] = None
    mask: Optional[str] = None
    overlay: Optional[str] = None
    ttp_raw_mask: Optional[str] = None
    ttp_mask: Optional[str] = None
    ttp_overlay: Optional[str] = None
    deterministic_mask: Optional[str] = None
    deterministic_overlay: Optional[str] = None
    agreement: Optional[str] = None
    disagreement: Optional[str] = None
    intersection: Optional[str] = None
    union: Optional[str] = None
    top_regions: Optional[str] = None


class ChangeEngine(BaseModel):
    mode: str
    primary_tool: str
    supporting_tool: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class TTPResult(BaseModel):
    status: str
    changed_percentage: Optional[float] = Field(default=None, ge=0, le=100)
    changed_pixels: Optional[int] = Field(default=None, ge=0)
    region_count: Optional[int] = Field(default=None, ge=0)
    largest_region_pixels: Optional[int] = Field(default=None, ge=0)
    runtime_ms: Optional[int] = Field(default=None, ge=0)
    model_load_ms: Optional[int] = Field(default=None, ge=0)
    model: str = "TTP"
    architecture: str = "SAM ViT-L + LoRA SiamEncoderDecoder"
    training_dataset: str = "LEVIR-CD"
    checkpoint: str = "epoch_260.pth"
    checkpoint_fingerprint: str = "60294429b3d"
    device: Optional[str] = None
    reused_model: Optional[bool] = None
    limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class MaskComparison(BaseModel):
    intersection_pixels: int = Field(ge=0)
    union_pixels: int = Field(ge=0)
    iou: Optional[float] = Field(default=None, ge=0, le=1)
    agreement_percentage: float = Field(ge=0, le=100)
    disagreement_percentage: float = Field(ge=0, le=100)
    changed_class_agreement: Optional[float] = Field(default=None, ge=0, le=100)
    background_agreement: Optional[float] = Field(default=None, ge=0, le=100)
    disclaimer: str = "Mask agreement is evidence consistency, not ground-truth accuracy."


class EvidenceConsistency(BaseModel):
    label: str
    rationale: List[str] = Field(default_factory=list)
    calibrated_probability: bool = False
    disclosure: str = "This is not a calibrated probability."


class SVEScenePrior(BaseModel):
    label: str
    similarity: float = Field(ge=-1, le=1)


class SVECaptionConsistency(BaseModel):
    score: float = Field(ge=-1, le=1)
    selected_candidate_index: int = Field(ge=0)
    original_candidates: List[str] = Field(default_factory=list)
    original_candidate_order: List[int] = Field(default_factory=list)
    reranked_candidate_order: List[int] = Field(default_factory=list)
    candidate_scores: List[float] = Field(default_factory=list)
    reranked: bool = False
    disclosure: str = "Similarity is scene-level consistency evidence, not caption correctness."


class SVEVQAConsistency(BaseModel):
    state: str
    target_concept: Optional[str] = None
    similarity: Optional[float] = Field(default=None, ge=-1, le=1)
    explanation: str


class SVEGroundingSupport(BaseModel):
    state: str
    target: str
    related_scene_labels: List[str] = Field(default_factory=list)
    warning: Optional[str] = None
    disclosure: str = "Scene-level plausibility does not validate or reject Grounding DINO detections."


class SVESemanticComparison(BaseModel):
    label: str
    status: str
    similarity: Optional[float] = Field(default=None, ge=-1, le=1)
    prior_changes: List[Dict[str, Any]] = Field(default_factory=list)
    disclaimer: str


class SVEResult(BaseModel):
    available: bool
    status: str
    model: str = "SatQuery Vision Encoder v1"
    model_version: str = "1.0.0"
    backbone: str = "OpenCLIP ViT-L-14"
    pretrained_weights: str = "laion2b_s32b_b82k"
    adaptation_dataset: str = "BigEarthNet.txt"
    adapter_checksum_fingerprint: str = "sha256:a99c0bf0fb44"
    embedding_dimension: int = 768
    device: Optional[str] = None
    runtime_ms: Optional[int] = Field(default=None, ge=0)
    scene_priors: List[SVEScenePrior] = Field(default_factory=list)
    caption_consistency: Optional[SVECaptionConsistency] = None
    vqa_consistency: Optional[SVEVQAConsistency] = None
    grounding_support: Optional[SVEGroundingSupport] = None
    semantic_comparison: Optional[SVESemanticComparison] = None
    limitations: List[str] = Field(default_factory=list)
    warning: Optional[str] = None
    fallback: Optional[str] = None
    disclaimer: str = (
        "SatQuery Vision Encoder v1 provides scene-level embedding evidence. It does not produce "
        "pixel-level segmentation, object grounding, calibrated probabilities, or ground truth."
    )


class SemanticTransition(BaseModel):
    """Evidence-gated directional scene interpretation; never a physical measurement."""

    type: str
    confidence: str
    evidence: List[str] = Field(default_factory=list)


class BuiltUpRegionEvidence(BaseModel):
    """Image-relative evidence for one real connected change component."""

    region_id: int = Field(gt=0)
    relative_location: str
    bbox: List[int]
    pixel_area: int = Field(gt=0)
    relative_area_percent: float = Field(ge=0, le=100)
    change_strength: str
    compactness: float = Field(ge=0, le=1)
    before_structural_evidence: float = Field(ge=0, le=1)
    after_structural_evidence: float = Field(ge=0, le=1)
    before_built_up_evidence: float = Field(ge=0, le=1)
    after_built_up_evidence: float = Field(ge=0, le=1)
    directional_state: str
    semantic_support_level: str
    deterministic_overlap_percent: Optional[float] = Field(default=None, ge=0, le=100)


class BuiltUpChangeAssessment(BaseModel):
    state: str
    magnitude: str
    confidence: str
    confidence_factors: List[str] = Field(default_factory=list)
    regions: List[BuiltUpRegionEvidence] = Field(default_factory=list)
    overlay_label: str = "Changed region with built-up-like evidence"
    limitations: List[str] = Field(default_factory=list)


class SemanticChangeSummary(BaseModel):
    """Optional local interpretation layered over authoritative change evidence."""

    short_answer: str
    expanded_answer: str
    query_intent: str
    overall_change_level: str
    dominant_location: Optional[str] = None
    likely_change_type: Optional[str] = None
    evidence_strength: str
    stable_area_summary: Optional[str] = None
    changed_regions: List[str] = Field(default_factory=list)
    stable_regions: List[str] = Field(default_factory=list)
    likely_transitions: List[SemanticTransition] = Field(default_factory=list)
    supporting_facts: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    visual_structural_disagreement: bool = False
    built_up_assessment: Optional[BuiltUpChangeAssessment] = None
    generated_by: str = "local_semantic_interpreter"


class ChangeAnalysisResponse(BaseModel):
    request_id: str
    status: ChangeAnalysisStatus
    before_date: str
    after_date: str
    before_metadata: ImageMetadata
    after_metadata: ImageMetadata
    compatibility: PairCompatibility
    statistics: Optional[ChangeStatistics] = None
    previews: ChangePreviewUrls
    execution: ExecutionSummary
    runtime_ms: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    change_engine: Optional[ChangeEngine] = None
    ttp_result: Optional[TTPResult] = None
    deterministic_statistics: Optional[ChangeStatistics] = None
    mask_comparison: Optional[MaskComparison] = None
    evidence_consistency: Optional[EvidenceConsistency] = None
    sve_result: Optional[SVEResult] = None
    semantic_change_summary: Optional[SemanticChangeSummary] = None


class CrossModalSummary(BaseModel):
    optical_observations: List[str] = Field(default_factory=list)
    sar_observations: List[str] = Field(default_factory=list)
    joint_observations: List[str] = Field(default_factory=list)


class CrossModalEvidenceFact(BaseModel):
    """Conservative fact with explicit evidence origin for answer synthesis."""

    source: str
    kind: str
    statement: str
    supporting_region_ids: List[str] = Field(default_factory=list)
    calibrated_probability: bool = False


class CrossModalStatistics(BaseModel):
    analysis_width: Optional[int] = Field(default=None, gt=0)
    analysis_height: Optional[int] = Field(default=None, gt=0)
    source_width: Optional[int] = Field(default=None, gt=0)
    source_height: Optional[int] = Field(default=None, gt=0)
    water_likelihood_percent: Optional[float] = Field(default=None, ge=0, le=100)
    built_up_likelihood_percent: Optional[float] = Field(default=None, ge=0, le=100)
    vegetation_support_percent: Optional[float] = Field(default=None, ge=0, le=100)
    agreement_percent: Optional[float] = Field(default=None, ge=0, le=100)
    disagreement_percent: Optional[float] = Field(default=None, ge=0, le=100)
    valid_pixel_percent: Optional[float] = Field(default=None, ge=0, le=100)
    evaluated_candidate_pixels: Optional[int] = Field(default=None, ge=0)


class CrossModalRegionSupport(BaseModel):
    optical: bool
    sar: bool


class CrossModalRegion(BaseModel):
    region_id: str
    type: str
    area_pixels: int = Field(gt=0)
    area_percent: float = Field(ge=0, le=100)
    bbox_pixels: List[int] = Field(min_length=4, max_length=4)
    centroid_pixels: List[float] = Field(min_length=2, max_length=2)
    bbox_world: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    centroid_world: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    support: CrossModalRegionSupport


class CrossModalPreviewUrls(BaseModel):
    optical: Optional[str] = None
    sar: Optional[str] = None
    optical_evidence: Optional[str] = None
    sar_evidence: Optional[str] = None
    joint_evidence: Optional[str] = None
    water_likelihood: Optional[str] = None
    built_up_likelihood: Optional[str] = None
    vegetation_support: Optional[str] = None
    agreement: Optional[str] = None
    disagreement: Optional[str] = None
    joint_overlay: Optional[str] = None


class CrossModalPreparation(BaseModel):
    modality: Modality
    band_count: int = Field(gt=0)
    bands_used: List[str] = Field(default_factory=list)
    channel_interpretation: List[str] = Field(default_factory=list)
    stretch_method: str
    normalization: str
    invalid_pixel_handling: str
    resize_status: str
    log_transform: str


class CrossModalMethod(BaseModel):
    name: str
    version: str
    uses_trained_model: bool = False
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class CrossModalAgreement(BaseModel):
    category: str
    relative_location: str
    optical_support: str
    sar_support: str
    strength: str = "candidate"


class CrossModalResult(BaseModel):
    analysis_level: Optional[str] = None
    quantitative_metrics_available: bool = False
    agreements: List[CrossModalAgreement] = Field(default_factory=list)
    disagreements: List[str] = Field(default_factory=list)
    complementary_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence_products: List[EvidenceItem] = Field(default_factory=list)
    status: CrossModalStatus
    summary: CrossModalSummary
    statistics: Optional[CrossModalStatistics] = None
    regions: List[CrossModalRegion] = Field(default_factory=list)
    previews: CrossModalPreviewUrls
    confidence: Confidence
    method: CrossModalMethod
    optical_preparation: Optional[CrossModalPreparation] = None
    sar_preparation: Optional[CrossModalPreparation] = None
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    stage_durations_ms: Dict[str, int] = Field(default_factory=dict, exclude=True)
    sve_result: Optional[SVEResult] = None
    evidence_facts: Optional[List[CrossModalEvidenceFact]] = None


class CrossModalAnalysisResponse(BaseModel):
    request_id: str
    optical_metadata: ImageMetadata
    sar_metadata: ImageMetadata
    compatibility: PairCompatibility
    result: CrossModalResult
    execution: ExecutionSummary


class SingleImageEvidenceStatistics(BaseModel):
    water_support_percent: float = Field(ge=0, le=100)
    vegetation_support_percent: float = Field(ge=0, le=100)
    built_up_support_percent: float = Field(ge=0, le=100)
    barren_support_percent: float = Field(ge=0, le=100)
    agriculture_support_percent: float = Field(ge=0, le=100)
    edge_density_percent: float = Field(ge=0, le=100)
    valid_pixel_percent: float = Field(ge=0, le=100)


class SingleImageEvidenceRegion(BaseModel):
    region_id: str
    type: str
    area_pixels: int = Field(gt=0)
    area_percent: float = Field(ge=0, le=100)
    bbox_pixels: List[int] = Field(min_length=4, max_length=4)
    centroid_pixels: List[float] = Field(min_length=2, max_length=2)
    bbox_world: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    centroid_world: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)


class SingleImageEvidencePreviews(BaseModel):
    water_support: Optional[str] = None
    vegetation_support: Optional[str] = None
    built_up_support: Optional[str] = None
    agriculture_support: Optional[str] = None
    combined_overlay: Optional[str] = None


class SingleImageEvidenceMethod(BaseModel):
    name: str
    version: str
    uses_trained_classifier: bool = False
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class SingleImageEvidenceResult(BaseModel):
    statistics: SingleImageEvidenceStatistics
    dominant_scene: str
    regions: List[SingleImageEvidenceRegion] = Field(default_factory=list)
    previews: SingleImageEvidencePreviews
    warnings: List[str] = Field(default_factory=list)
    method: SingleImageEvidenceMethod
    low_information: bool = False
    runtime_ms: int = Field(ge=0)


class ControlledVQAMethod(BaseModel):
    name: str
    version: str
    method_type: str
    uses_language_model: bool = False
    remote_sensing_adapted: bool = False
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ControlledVQAResult(BaseModel):
    original_question: str
    question_category: QuestionCategory
    target_concept: Optional[str] = None
    answer_source: str
    statistics_used: Dict[str, Any] = Field(default_factory=dict)
    evidence_references: List[str] = Field(default_factory=list)
    method: ControlledVQAMethod
    confidence: Confidence
    supported: bool
    limitations: List[str] = Field(default_factory=list)
    single_image_evidence: Optional[SingleImageEvidenceResult] = None


class CacheMetadata(BaseModel):
    cached: bool = False
    original_generation_timestamp: str
    retrieval_timestamp: str
    tool_version: str
    cache_key_prefix: str


class SarEvidenceProduct(BaseModel):
    id: str
    type: str
    label: str
    description: str
    source: str
    authoritative: bool
    reference: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    generation_method: str


class SarRegion(BaseModel):
    region_id: int = Field(gt=0)
    area_pixels: int = Field(gt=0)
    image_area_percent: float = Field(ge=0, le=100)
    bounding_box: List[int] = Field(min_length=4, max_length=4)
    centroid: List[float] = Field(min_length=2, max_length=2)


class SarPreprocessingDetails(BaseModel):
    version: str
    input_value_domain: str
    log_transform_applied: bool
    normalization: str
    percentile_low: Optional[float] = None
    percentile_high: Optional[float] = None
    invalid_pixel_count: int = Field(ge=0)
    nodata_pixel_count: int = Field(ge=0)
    denoising: str
    resized: bool
    input_dtype: Optional[str] = None
    input_channel_count: Optional[int] = Field(default=None, ge=1)
    polarization_labels: List[str] = Field(default_factory=list)
    value_domain_reason: Optional[str] = None
    percentile_lows: List[float] = Field(default_factory=list)
    percentile_highs: List[float] = Field(default_factory=list)


class SarWaterResult(BaseModel):
    execution_status: str
    task: str = "sar_water_segmentation"
    target: str = "water"
    method: str
    method_version: str
    water_detected: bool
    image_area_percent: float = Field(ge=0, le=100)
    geographic_area_square_meters: Optional[float] = Field(default=None, ge=0)
    geographic_area_method: Optional[str] = None
    model_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    heuristic_reliability: Optional[float] = Field(default=None, ge=0, le=1)
    input_quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    threshold: Optional[float] = Field(default=None, ge=0, le=1)
    candidate_pixels: int = Field(ge=0)
    valid_pixels: int = Field(ge=0)
    regions: List[SarRegion] = Field(default_factory=list)
    evidence_products: List[SarEvidenceProduct] = Field(default_factory=list)
    preprocessing: SarPreprocessingDetails
    rationale: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    stage_durations_ms: Dict[str, int] = Field(default_factory=dict, exclude=True)


class SarSceneResult(BaseModel):
    execution_status: str
    task: str
    method: str
    method_version: str
    valid_pixel_percent: float = Field(ge=0, le=100)
    low_backscatter_percent: float = Field(ge=0, le=100)
    mid_backscatter_percent: float = Field(ge=0, le=100)
    high_backscatter_percent: float = Field(ge=0, le=100)
    normalized_mean: float = Field(ge=0, le=1)
    normalized_standard_deviation: float = Field(ge=0)
    texture_index: float = Field(ge=0, le=1)
    input_quality_score: float = Field(ge=0, le=1)
    evidence_products: List[SarEvidenceProduct] = Field(default_factory=list)
    preprocessing: SarPreprocessingDetails
    limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    stage_durations_ms: Dict[str, int] = Field(default_factory=dict, exclude=True)


class SarTranslatedOpticalAnalysis(BaseModel):
    enabled: bool = True
    status: str
    generation_state: EvidenceLifecycleState = EvidenceLifecycleState.NOT_REQUESTED
    semantic_comparison_state: EvidenceLifecycleState = EvidenceLifecycleState.NOT_REQUESTED
    model: Optional[str] = None
    device: Optional[str] = None
    runtime_ms: int = Field(default=0, ge=0)
    generated_width: Optional[int] = Field(default=None, gt=0)
    generated_height: Optional[int] = Field(default=None, gt=0)
    generated_preview_url: Optional[str] = None
    evidence_products: List[EvidenceItem] = Field(default_factory=list)
    normalized_sar_preview_url: Optional[str] = None
    color_corrected_preview_url: Optional[str] = None
    color_corrected: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    preprocessing_method: Optional[str] = None
    input_channel_interpretation: Optional[str] = None
    output_value_range: List[float] = Field(default_factory=list)
    content_hash: Optional[str] = None
    optical_caption: Optional[str] = None
    optical_scene_priors: List[Dict[str, Any]] = Field(default_factory=list)
    optical_grounding: Optional[Dict[str, Any]] = None
    optical_vqa: Optional[Dict[str, Any]] = None
    optical_specialists_executed: List[str] = Field(default_factory=list)
    native_sar_findings: List[str] = Field(default_factory=list)
    translated_findings: List[str] = Field(default_factory=list)
    agreement: str
    direct_answer: str
    confidence: Confidence
    disclosure: str
    grounding_disclosure: Optional[str] = None
    rsvqa_disclosure: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    runtime_breakdown_ms: Dict[str, int] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    request_id: str
    task: TaskType
    answer: Optional[str] = None
    confidence: Confidence
    evidence: List[EvidenceItem] = Field(default_factory=list)
    execution: ExecutionSummary
    warnings: List[str] = Field(default_factory=list)
    status: ResponseStatus
    result_status: str = "COMPLETED"
    primary_image_metadata: Optional[ImageMetadata] = None
    secondary_image_metadata: Optional[ImageMetadata] = None
    pair_compatibility: Optional[PairCompatibility] = None
    model: Optional[ModelProvenance] = None
    caption_details: Optional[CaptionDetails] = None
    grounding_result: Optional[GroundingResult] = None
    cross_modal_analysis: Optional[CrossModalResult] = None
    change_analysis: Optional[ChangeAnalysisResponse] = None
    vqa_details: Optional[ControlledVQAResult] = None
    sar_water_analysis: Optional[SarWaterResult] = None
    sar_scene_analysis: Optional[SarSceneResult] = None
    classified_query: Optional[ClassifiedQuery] = None
    cache: Optional[CacheMetadata] = None
    change_engine: Optional[ChangeEngine] = None
    ttp_result: Optional[TTPResult] = None
    mask_comparison: Optional[MaskComparison] = None
    evidence_consistency: Optional[EvidenceConsistency] = None
    sve_result: Optional[SVEResult] = None
    sar_translated_optical_analysis: Optional[SarTranslatedOpticalAnalysis] = None
    semantic_change_summary: Optional[SemanticChangeSummary] = None


class ReportRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.ZIP]
    )


class ReportArtifact(BaseModel):
    format: ReportFormat
    filename: str
    url: str
    size_bytes: int = Field(ge=0)


class ReportResponse(BaseModel):
    request_id: str
    status: str
    schema_version: str
    artifacts: List[ReportArtifact]
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)


class ComparisonTask(str, Enum):
    CAPTIONING = "captioning"
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE = "change"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL = "cross_modal"
    PIX2PIX = "pix2pix"
    SARFUSIONFORMER = "sarfusionformer"
    SAR_ANALYSIS = "sar_analysis"


class ComparabilityLevel(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    NOT_DIRECT = "not_direct"


class ComparisonPreview(BaseModel):
    source_observation_id: Optional[str] = None
    source_role: Optional[ObservationRole] = None
    source_modality: Optional[ImageModality] = None
    evidence_id: Optional[str] = None
    label: str
    url: str
    kind: str
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    modality: Optional[Modality] = None


class ComparisonLineageNode(BaseModel):
    kind: str
    label: str
    request_id: Optional[str] = None
    preview_url: Optional[str] = None


class ComparisonItem(BaseModel):
    query: Optional[str] = None
    request_id: str
    task: ComparisonTask
    display_name: str
    status: str
    created_at: str
    input_mode: InputMode
    modalities: List[Modality]
    input_previews: List[ComparisonPreview] = Field(default_factory=list)
    output_previews: List[ComparisonPreview] = Field(default_factory=list)
    answer: Optional[str] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    confidence: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    selected_tools: List[str] = Field(default_factory=list)
    execution_duration_ms: Optional[int] = Field(default=None, ge=0)
    device: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    report_available: bool = False
    cached: bool = False
    input_identity: Dict[str, Any] = Field(default_factory=dict)
    lineage: List[ComparisonLineageNode] = Field(default_factory=list)
    execution_summary: Dict[str, Any] = Field(default_factory=dict)


class ComparisonItemSummary(BaseModel):
    query: Optional[str] = None
    answer_summary: Optional[str] = None
    request_id: str
    task: ComparisonTask
    display_name: str
    status: str
    created_at: str
    input_mode: InputMode
    modalities: List[Modality]
    thumbnail: Optional[ComparisonPreview] = None
    execution_duration_ms: Optional[int] = Field(default=None, ge=0)
    cached: bool = False
    report_available: bool = False
    warning_count: int = Field(ge=0)
    evidence_product_count: int = Field(ge=0)


class ComparabilityResult(BaseModel):
    left_request_id: str
    right_request_id: str
    level: ComparabilityLevel
    reason: str
    shared_inputs: bool
    shared_task_family: bool
    warnings: List[str] = Field(default_factory=list)
    overlay_allowed: bool = False
    difference_allowed: bool = False


class ComparisonAssessmentRequest(BaseModel):
    request_ids: List[str] = Field(min_length=2, max_length=4)


class ComparisonAssessmentResponse(BaseModel):
    assessments: List[ComparabilityResult]
    overall_level: ComparabilityLevel
    warnings: List[str] = Field(default_factory=list)


class ComparisonReportRequest(ComparisonAssessmentRequest):
    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.JSON, ReportFormat.ZIP]
    )
    user_note: Optional[str] = Field(default=None, max_length=2000)


class ComparisonReportResponse(BaseModel):
    comparison_id: str
    status: str
    schema_version: str
    artifacts: List[ReportArtifact]
    assessments: List[ComparabilityResult]
    warnings: List[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)


class DemoSampleFile(BaseModel):
    role: str
    filename: str
    url: str
    mime_type: str


class DemoWorkflow(BaseModel):
    id: str
    title: str
    description: str
    input_mode: InputMode
    primary_modality: Modality
    secondary_modality: Optional[Modality] = None
    query: str
    primary_date: Optional[str] = None
    secondary_date: Optional[str] = None
    files: List[DemoSampleFile]


class DemoManifest(BaseModel):
    enabled: bool
    local_only: bool = True
    workflows: List[DemoWorkflow] = Field(default_factory=list)


class ComplianceRequirement(BaseModel):
    requirement: str
    implementation: str
    status: str
    tool_or_model: str
    test_coverage: str
    limitation: str
    readiness_status: Optional[str] = None
    specialists: List[str] = Field(default_factory=list)
    benchmark_evidence: Optional[str] = None
    evidence_status: Optional[str] = None


class ComplianceResponse(BaseModel):
    generated_at: str
    project: str
    requirements: List[ComplianceRequirement]
    mandatory_satisfied: int = Field(ge=0)
    mandatory_total: int = Field(ge=0)
    optional_not_implemented: List[str] = Field(default_factory=list)


class SpecialistHealth(BaseModel):
    status: str
    device: Optional[str] = None
    error: Optional[str] = None
    model_id: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    load_source: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    last_error: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    smoke_verified: Optional[bool] = Field(default=None, exclude_if=lambda value: value is None)


class AgentHealth(BaseModel):
    status: str
    module: str
    router: str
    registry: str
    specialists: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsTraceStep(BaseModel):
    tool: str
    status: str
    duration_ms: int = Field(ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsExecution(BaseModel):
    request_id: str
    started_at: str
    completed_at: str
    task: str
    input_mode: str
    primary_modality: Optional[str] = None
    secondary_modality: Optional[str] = None
    status: str
    selected_tools: List[str] = Field(default_factory=list)
    duration_ms: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    cache_status: str
    report_generated: bool = False
    device: Optional[str] = None
    selection_reason: Optional[str] = None
    confidence_level: Optional[str] = None
    confidence_reason: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    trace: List[AnalyticsTraceStep] = Field(default_factory=list)


class AnalyticsPlatform(BaseModel):
    backend_status: str
    uptime_seconds: int = Field(ge=0)
    python_version: str
    operating_system: str
    architecture: str
    process_memory_mb: Optional[float] = Field(default=None, ge=0)
    runtime_versions: Dict[str, Optional[str]] = Field(default_factory=dict)
    hardware_acceleration: str
    demo_mode: bool
    offline_ready: bool
    offline_readiness_requirements: List[str] = Field(default_factory=list)


class AnalyticsSummary(BaseModel):
    total_executions_current_process: int = Field(ge=0)
    successful_executions_current_process: int = Field(ge=0)
    registered_tools: int = Field(ge=0)
    available_tools: int = Field(ge=0)
    mandatory_satisfied: int = Field(ge=0)
    mandatory_total: int = Field(ge=0)
    cache_hits_current_process: int = Field(ge=0)
    cache_misses_current_process: int = Field(ge=0)
    report_artifacts_generated_current_process: int = Field(ge=0)


class AnalyticsCache(BaseModel):
    stored_results: int = Field(ge=0)
    max_results: int = Field(gt=0)
    ttl_seconds: int = Field(gt=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    hit_rate_percent: Optional[float] = Field(default=None, ge=0, le=100)


class AnalyticsReports(BaseModel):
    requests_generated_current_process: int = Field(ge=0)
    artifacts_generated_current_process: int = Field(ge=0)
    artifacts_currently_available: int = Field(ge=0)
    formats: Dict[str, int] = Field(default_factory=dict)


class AnalyticsTTP(BaseModel):
    enabled: bool
    service_status: str
    model_load_count: int = Field(ge=0)
    model_reuse_count: int = Field(ge=0)
    inference_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    average_runtime_ms: Optional[float] = Field(default=None, ge=0)
    average_mask_iou: Optional[float] = Field(default=None, ge=0, le=1)


class AnalyticsSVE(BaseModel):
    enabled: bool
    lifecycle_status: str
    model_load_count: int = Field(ge=0)
    model_reuse_count: int = Field(ge=0)
    inference_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    checksum_failure_count: int = Field(ge=0)
    cpu_fallback_count: int = Field(ge=0)
    mps_fallback_count: int = Field(ge=0)
    caption_reranking_count: int = Field(ge=0)
    vqa_agreement_count: int = Field(ge=0)
    vqa_disagreement_count: int = Field(ge=0)
    average_runtime_ms: Optional[float] = Field(default=None, ge=0)
    cache_hit_rate_percent: Optional[float] = Field(default=None, ge=0, le=100)
    device_usage: Dict[str, int] = Field(default_factory=dict)


class AnalyticsTool(BaseModel):
    id: str
    display_name: str
    implementation_status: str
    lifecycle_status: str
    device: Optional[str] = None
    last_runtime_ms: Optional[int] = Field(default=None, ge=0)
    last_completed_at: Optional[str] = None
    method_type: Optional[str] = None
    checkpoint: Optional[str] = None
    adaptation_dataset: Optional[str] = None
    remote_sensing_adapted: bool
    service_path: Optional[str] = None


class AnalyticsDataset(BaseModel):
    name: str
    usage_status: str
    used_by: List[str] = Field(default_factory=list)
    purpose: str
    sample_count: Optional[int] = Field(default=None, ge=0)
    source: Optional[str] = None
    note: str


class AnalyticsCapability(BaseModel):
    name: str
    status: str
    evidence: str


class AnalyticsWorkflowMetric(BaseModel):
    task: str
    executions: int = Field(ge=0)
    successful_executions: int = Field(ge=0)
    average_runtime_ms: Optional[float] = Field(default=None, ge=0)
    last_runtime_ms: Optional[int] = Field(default=None, ge=0)
    last_device: Optional[str] = None
    last_completed_at: Optional[str] = None


class ScientificTransparency(BaseModel):
    metric_source: str
    history_retention: str
    inference_behavior_changed: bool = False
    ai_generated_metrics: bool = False
    unavailable_value_policy: str
    caveats: List[str] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    generated_at: str
    platform: AnalyticsPlatform
    summary: AnalyticsSummary
    cache: AnalyticsCache
    reports: AnalyticsReports
    tools: List[AnalyticsTool]
    datasets: List[AnalyticsDataset]
    capabilities: List[AnalyticsCapability]
    workflow_metrics: List[AnalyticsWorkflowMetric]
    recent_executions: List[AnalyticsExecution]
    last_execution_trace: List[AnalyticsTraceStep]
    scientific_transparency: ScientificTransparency
    ttp: Optional[AnalyticsTTP] = None
    sve: Optional[AnalyticsSVE] = None

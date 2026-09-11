export type InputMode = "single" | "cross_modal" | "bi_temporal";
export type Modality = "optical" | "multispectral" | "sar" | "unknown";
export type ImageModality = "auto" | "optical_rgb" | "optical_grayscale" | "panchromatic" | "sar_preview" | "sar_vv" | "sar_vh" | "sar_vv_vh" | "multispectral" | "unknown";
export type RepresentationType = "scientific_raster" | "display_preview" | "unknown_representation";
export type DetectionConfidence = "high" | "medium" | "low" | "unavailable";
export type TaskType = "captioning" | "vqa" | "grounding" | "change_description" | "change_vqa" | "cross_modal_analysis" | "report_generation" | "sar_water_segmentation" | "sar_scene_analysis" | "sar_quality_inspection" | "unsupported";
export type ToolStatus = "pending" | "running" | "success" | "failed" | "skipped" | "not_implemented";
export type ConfidenceLevel = "high" | "moderate" | "low" | "unavailable";
export type AgentResponseStatus = "success" | "partial" | "alignment_required" | "failed" | "not_implemented";
export type ImageFormat = "geotiff" | "tiff" | "png" | "jpeg" | "unknown";
export type AlignmentLevel = "exact" | "geospatial_overlap" | "visual_only" | "incompatible";

export type AnalyticsTraceStep = { tool: string; status: string; duration_ms: number; parameters: Record<string, unknown> };
export type AnalyticsExecution = {
  request_id: string;
  started_at: string;
  completed_at: string;
  task: string;
  input_mode: string;
  primary_modality: string | null;
  secondary_modality: string | null;
  status: string;
  selected_tools: string[];
  duration_ms: number;
  warning_count: number;
  output_count: number;
  cache_status: string;
  report_generated: boolean;
  device: string | null;
  selection_reason: string | null;
  confidence_level: string | null;
  confidence_reason: string | null;
  warnings: string[];
  trace: AnalyticsTraceStep[];
};

export type AgentHealth = {
  status: string;
  module: string;
  router: string;
  registry: string;
  specialists: Record<string, {
    status: string; device: string | null; error: string | null; enabled?: boolean;
    selected_model?: string; fallback_available?: boolean; color_corrector_available?: boolean;
    load_count?: number; reuse_count?: number; loaded?: boolean; lifecycle?: string;
    warnings?: string[]; errors?: string[];
    model_id?: string | null; load_source?: string | null; last_error?: string | null; smoke_verified?: boolean | null;
  }>;
};
export type AnalyticsResponse = {
  generated_at: string;
  platform: {
    backend_status: string;
    uptime_seconds: number;
    python_version: string;
    operating_system: string;
    architecture: string;
    process_memory_mb: number | null;
    runtime_versions: Record<string, string | null>;
    hardware_acceleration: string;
    demo_mode: boolean;
    offline_ready: boolean;
    offline_readiness_requirements: string[];
  };
  summary: {
    total_executions_current_process: number;
    successful_executions_current_process: number;
    registered_tools: number;
    available_tools: number;
    mandatory_satisfied: number;
    mandatory_total: number;
    cache_hits_current_process: number;
    cache_misses_current_process: number;
    report_artifacts_generated_current_process: number;
  };
  cache: {
    stored_results: number;
    max_results: number;
    ttl_seconds: number;
    hits: number;
    misses: number;
    hit_rate_percent: number | null;
  };
  reports: {
    requests_generated_current_process: number;
    artifacts_generated_current_process: number;
    artifacts_currently_available: number;
    formats: Record<string, number>;
  };
  tools: Array<{
    id: string;
    display_name: string;
    implementation_status: string;
    lifecycle_status: string;
    device: string | null;
    last_runtime_ms: number | null;
    last_completed_at: string | null;
    method_type: string | null;
    checkpoint: string | null;
    adaptation_dataset: string | null;
    remote_sensing_adapted: boolean;
    service_path: string | null;
  }>;
  datasets: Array<{
    name: string;
    usage_status: string;
    used_by: string[];
    purpose: string;
    sample_count: number | null;
    source: string | null;
    note: string;
  }>;
  capabilities: Array<{ name: string; status: string; evidence: string }>;
  workflow_metrics: Array<{
    task: string;
    executions: number;
    successful_executions: number;
    average_runtime_ms: number | null;
    last_runtime_ms: number | null;
    last_device: string | null;
    last_completed_at: string | null;
  }>;
  recent_executions: AnalyticsExecution[];
  last_execution_trace: AnalyticsTraceStep[];
  scientific_transparency: {
    metric_source: string;
    history_retention: string;
    inference_behavior_changed: boolean;
    ai_generated_metrics: boolean;
    unavailable_value_policy: string;
    caveats: string[];
  };
  ttp?: {
    enabled: boolean; service_status: string; model_load_count: number; model_reuse_count: number;
    inference_count: number; failure_count: number; timeout_count: number; oom_count: number;
    fallback_count: number; average_runtime_ms: number | null; average_mask_iou: number | null;
  } | null;
  sve?: {
    enabled: boolean; lifecycle_status: string; model_load_count: number; model_reuse_count: number;
    inference_count: number; failure_count: number; checksum_failure_count: number;
    cpu_fallback_count: number; mps_fallback_count: number; caption_reranking_count: number;
    vqa_agreement_count: number; vqa_disagreement_count: number; average_runtime_ms: number | null;
    cache_hit_rate_percent: number | null; device_usage: Record<string, number>;
  } | null;
};

export type AgentQueryRequest = {
  query: string;
  input_mode: InputMode;
  primary_modality: Modality;
  secondary_modality: Modality | null;
  has_primary_image: boolean;
  has_secondary_image: boolean;
};

export type AgentValidation = { valid: boolean; errors: string[] };

export type RasterBounds = { left: number; bottom: number; right: number; top: number };

export type ObservationRole = "optical" | "sar";
export type CrossModalEvidenceProduct = { type: string; label: string; reference?: string | null; description?: string | null; evidence_id?: string | null; source_observation_id?: string | null; source_observation_ids?: string[]; source_role?: ObservationRole | null; source_modality?: ImageModality | null; evidence_type?: string | null };

export type ImageMetadata = {
  observation_role?: ObservationRole | null;
  file_id: string;
  original_name: string;
  safe_name: string;
  format: ImageFormat;
  mime_type: string;
  size_bytes: number;
  width: number;
  height: number;
  band_count: number;
  dtype: string;
  crs: string | null;
  transform: number[] | null;
  bounds: RasterBounds | null;
  nodata: number | null;
  is_georeferenced: boolean;
  preview_url: string | null;
  color_interpretation: string[];
  band_descriptions?: Array<string | null>;
  input_band_count?: number | null;
  available_band_names?: string[];
  selected_visual_bands?: string[];
  band_selection_reason?: string | null;
  band_statistics?: Array<{
    band: number; description: string | null; dtype: string; minimum: number | null; maximum: number | null;
    mean: number | null; standard_deviation: number | null; percentile_1: number | null; percentile_5: number | null;
    percentile_50: number | null; percentile_95: number | null; percentile_99: number | null;
    nan_count: number; inf_count: number; nodata_count: number; valid_count: number;
  }>;
  representation?: RepresentationType;
  auto_detected_modality?: ImageModality;
  auto_detection_confidence?: DetectionConfidence;
  auto_detection_reason?: string;
  user_confirmed_modality?: ImageModality | null;
  effective_modality?: ImageModality;
  modality_limitations?: string[];
  warnings: string[];
};

export type ImageInspectionResponse = { metadata: ImageMetadata; content_hash_prefix: string; requires_modality_confirmation: boolean };

export type PairCompatibility = {
  role_validation_status?: "match" | "mismatch" | "ambiguous" | null;
  role_match?: boolean | null;
  pair_valid?: boolean | null;
  optical_slot_detected_modality?: ImageModality | null;
  sar_slot_detected_modality?: ImageModality | null;
  compatible: boolean;
  alignment_level: AlignmentLevel;
  same_dimensions: boolean;
  same_crs: boolean | null;
  same_transform: boolean | null;
  bounds_overlap: boolean | null;
  overlap_ratio: number | null;
  resampling_required: boolean;
  scientific_classification?: "EXACT_GRID_MATCH" | "SAME_AREA_DIFFERENT_GRID" | "REPROJECTION_REQUIRED" | "RESAMPLING_REQUIRED" | "PARTIAL_OVERLAP" | "INSUFFICIENT_OVERLAP" | "UNVERIFIABLE" | null;
  primary_resolution?: number[] | null;
  secondary_resolution?: number[] | null;
  same_resolution?: boolean | null;
  same_orientation?: boolean | null;
  nodata_compatible?: boolean | null;
  recommended_action?: string | null;
  warnings: string[];
  errors: string[];
};

export type AgentExecutionStep = {
  tool: string;
  status: ToolStatus;
  duration_ms: number;
  parameters: Record<string, unknown>;
};

export type ModelProvenance = {
  tool_id: string;
  checkpoint: string;
  base_architecture: string;
  adaptation_dataset: string;
  remote_sensing_adapted: boolean;
  license: string | null;
  source: string;
};

export type CaptionDetails = {
  modality: Modality;
  device: string;
  runtime_ms: number;
  model_load_ms: number;
  model_reused: boolean;
  image_representation: string;
  bands_used: string[];
  limitations: string[];
};

export type SVEResult = {
  available: boolean;
  status: "success" | "loading" | "ready" | "disabled" | "unavailable" | "checksum_failure" | "unsupported_input" | "model_error" | string;
  model: string;
  model_version: string;
  backbone: string;
  pretrained_weights: string;
  adaptation_dataset: string;
  adapter_checksum_fingerprint: string;
  embedding_dimension: number;
  device: string | null;
  runtime_ms: number | null;
  scene_priors: Array<{ label: string; similarity: number }>;
  caption_consistency: {
    score: number; selected_candidate_index: number; original_candidates: string[];
    original_candidate_order: number[]; reranked_candidate_order: number[]; candidate_scores: number[];
    reranked: boolean; disclosure: string;
  } | null;
  vqa_consistency: { state: string; target_concept: string | null; similarity: number | null; explanation: string } | null;
  grounding_support: { state: string; target: string; related_scene_labels: string[]; warning: string | null; disclosure: string } | null;
  semantic_comparison: {
    label: string; status: string; similarity: number | null;
    prior_changes: Array<{ label: string; before_similarity: number; after_similarity: number; difference: number }>;
    disclaimer: string;
  } | null;
  limitations: string[];
  warning: string | null;
  fallback: string | null;
  disclaimer: string;
};

export type GroundingDetection = {
  label: string;
  score: number;
  bbox_pixels: [number, number, number, number];
  bbox_normalized: [number, number, number, number];
  bbox_world: [number, number, number, number] | null;
  crs: string | null;
  mask_url: string | null;
  source: string;
  quality: GroundingCandidateQuality;
};

export type GroundingCandidateQuality = {
  source_width: number;
  source_height: number;
  box_width: number | null;
  box_height: number | null;
  box_area: number | null;
  image_area: number;
  box_area_ratio: number | null;
  alignment_score: number | null;
  finite_score: boolean;
  finite_coordinates: boolean;
  positive_area: boolean;
  in_bounds: boolean;
  rejection_reasons: string[];
};

export type RejectedGroundingCandidate = {
  label: string;
  score: number | null;
  bbox_source_xyxy: Array<number | null>;
  bbox_pixels: [number, number, number, number] | null;
  box_area_ratio: number | null;
  rejection_reasons: string[];
  quality: GroundingCandidateQuality;
  source: string;
};

export type GroundingResult = {
  original_query: string;
  target_phrase: string;
  detections: GroundingDetection[];
  accepted_detections: GroundingDetection[];
  rejected_candidates: RejectedGroundingCandidate[];
  accepted_detection_count: number;
  rejected_candidate_count: number;
  quality_policy: {
    minimum_alignment_score: number;
    maximum_localized_area_ratio: number;
    localized_targets: string[];
    calibration_status: string;
  };
  empty_result_explanation: string | null;
  operational_threshold_disclaimer: string;
  annotated_preview_url: string | null;
  confidence: { level: ConfidenceLevel; score: number | null; reason: string };
  model: ModelProvenance;
  input: {
    modality: Modality;
    bands_used: string[];
    representation: string;
    original_width: number;
    original_height: number;
    model_input_width: number;
    model_input_height: number;
    normalization_method: string;
  };
  device: string;
  warnings: string[];
  limitations: string[];
  runtime_ms: number;
  model_load_ms: number;
  model_reused: boolean;
};

export type SarEvidenceProduct = {
  id: string; type: string; label: string; description: string; source: string; authoritative: boolean;
  reference: string; width: number; height: number; generation_method: string;
};
export type SarRegion = { region_id: number; area_pixels: number; image_area_percent: number; bounding_box: [number, number, number, number]; centroid: [number, number] };
export type SarPreprocessingDetails = {
  version: string; input_value_domain: string; log_transform_applied: boolean; normalization: string;
  percentile_low: number | null; percentile_high: number | null; invalid_pixel_count: number; nodata_pixel_count: number;
  denoising: string; resized: boolean;
  input_dtype?: string | null; input_channel_count?: number | null; polarization_labels?: string[];
  value_domain_reason?: string | null; percentile_lows?: number[]; percentile_highs?: number[];
};
export type SarWaterResult = {
  execution_status: string; task: string; target: string; method: string; method_version: string; water_detected: boolean;
  image_area_percent: number; geographic_area_square_meters?: number | null; geographic_area_method?: string | null;
  model_confidence: number | null; heuristic_reliability: number | null; input_quality_score: number | null;
  threshold: number | null; candidate_pixels: number; valid_pixels: number; regions: SarRegion[]; evidence_products: SarEvidenceProduct[];
  preprocessing: SarPreprocessingDetails; rationale: string[]; limitations: string[]; warnings: string[]; runtime_ms: number;
};
export type SarSceneResult = {
  execution_status: string; task: string; method: string; method_version: string; valid_pixel_percent: number;
  low_backscatter_percent: number; mid_backscatter_percent: number; high_backscatter_percent: number;
  normalized_mean: number; normalized_standard_deviation: number; texture_index: number; input_quality_score: number;
  evidence_products: SarEvidenceProduct[]; preprocessing: SarPreprocessingDetails; limitations: string[]; warnings: string[]; runtime_ms: number;
};
export type SarTranslatedOpticalAnalysis = {
  enabled: boolean; status: string; model: string | null; device: string | null; runtime_ms: number;
  generation_state: "NOT_REQUESTED" | "NOT_ELIGIBLE" | "RUNNING" | "SUCCEEDED" | "FAILED";
  semantic_comparison_state: "NOT_REQUESTED" | "NOT_ELIGIBLE" | "RUNNING" | "SUCCEEDED" | "FAILED";
  generated_width: number | null; generated_height: number | null; generated_preview_url: string | null;
  evidence_products: AgentEvidenceItem[];
  normalized_sar_preview_url: string | null; color_corrected_preview_url: string | null;
  color_corrected: boolean; fallback_used: boolean; fallback_reason: string | null;
  preprocessing_method: string | null; input_channel_interpretation: string | null;
  output_value_range: number[]; content_hash: string | null; optical_caption: string | null;
  optical_scene_priors: Array<Record<string, unknown>>; optical_grounding: Record<string, unknown> | null;
  optical_vqa: Record<string, unknown> | null; optical_specialists_executed: string[];
  native_sar_findings: string[]; translated_findings: string[]; agreement: string; direct_answer: string;
  confidence: { level: ConfidenceLevel; score: number | null; reason: string };
  disclosure: string; grounding_disclosure: string | null; rsvqa_disclosure: string | null;
  limitations: string[]; warnings: string[]; provenance: Record<string, unknown>;
  runtime_breakdown_ms: Record<string, number>;
};

export type AgentEvidenceItem = {
  evidence_id?: string | null; source_observation_id?: string | null; source_observation_ids?: string[];
  source_role?: "optical" | "sar" | null; source_modality?: ImageModality | null; evidence_type?: string | null;
  evidence_run_id?: string | null; generator?: string | null;
  status?: "NOT_REQUESTED" | "NOT_ELIGIBLE" | "RUNNING" | "SUCCEEDED" | "FAILED" | null;
  type: string; label: string; description?: string | null; reference?: string | null;
};

export type AgentResponse = {
  request_id: string;
  task: TaskType;
  answer: string | null;
  confidence: { level: ConfidenceLevel; score: number | null; reason: string };
  evidence: AgentEvidenceItem[];
  execution: {
    input_mode: InputMode;
    selected_tools: string[];
    steps: AgentExecutionStep[];
    duration_ms: number;
    permitted_parameters: Record<string, unknown>;
    validation: AgentValidation;
    selection_reason: string;
  };
  warnings: string[];
  status: AgentResponseStatus;
  result_status?: string;
  primary_image_metadata: ImageMetadata | null;
  secondary_image_metadata: ImageMetadata | null;
  pair_compatibility: PairCompatibility | null;
  model: ModelProvenance | null;
  caption_details: CaptionDetails | null;
  grounding_result: GroundingResult | null;
  cross_modal_analysis: CrossModalResult | null;
  change_analysis: ChangeAnalysisResponse | null;
  vqa_details: ControlledVQAResult | null;
  sar_water_analysis?: SarWaterResult | null;
  sar_scene_analysis?: SarSceneResult | null;
  sar_translated_optical_analysis?: SarTranslatedOpticalAnalysis | null;
  classified_query?: { task_type: TaskType; target: string | null; requested_output: string; requires_localization: boolean; requires_segmentation: boolean; requires_measurement: boolean } | null;
  cache: {
    cached: boolean;
    original_generation_timestamp: string;
    retrieval_timestamp: string;
    tool_version: string;
    cache_key_prefix: string;
  } | null;
  change_engine?: ChangeEngine | null;
  ttp_result?: TTPResult | null;
  mask_comparison?: MaskComparison | null;
  evidence_consistency?: EvidenceConsistency | null;
  sve_result?: SVEResult | null;
  semantic_change_summary?: SemanticChangeSummary | null;
};

export type QuestionCategory =
  | "dominant_land_cover" | "presence_water" | "presence_buildings" | "presence_vegetation"
  | "presence_agriculture" | "composition_built_up" | "scene_type" | "relative_coverage"
  | "metadata_question" | "change_summary" | "change_percentage" | "largest_change"
  | "change_region_count" | "change_magnitude" | "change_location" | "built_up_change"
  | "vegetation_change" | "water_change" | "infrastructure_change" | "no_change_check"
  | "general_comparison" | "cross_modal_agreement" | "cross_modal_water"
  | "cross_modal_structure" | "cross_modal_disagreement" | "cross_modal_region_count" | "unsupported";

export type SingleImageEvidenceStatistics = {
  water_support_percent: number;
  vegetation_support_percent: number;
  built_up_support_percent: number;
  barren_support_percent: number;
  agriculture_support_percent: number;
  edge_density_percent: number;
  valid_pixel_percent: number;
};

export type SingleImageEvidenceRegion = {
  region_id: string;
  type: string;
  area_pixels: number;
  area_percent: number;
  bbox_pixels: [number, number, number, number];
  centroid_pixels: [number, number];
  bbox_world: [number, number, number, number] | null;
  centroid_world: [number, number] | null;
};

export type SingleImageEvidenceResult = {
  statistics: SingleImageEvidenceStatistics;
  dominant_scene: string;
  regions: SingleImageEvidenceRegion[];
  previews: {
    water_support: string | null;
    vegetation_support: string | null;
    built_up_support: string | null;
    agriculture_support: string | null;
    combined_overlay: string | null;
  };
  warnings: string[];
  method: { name: string; version: string; uses_trained_classifier: boolean; assumptions: string[]; limitations: string[] };
  low_information: boolean;
  runtime_ms: number;
};

export type ControlledVQAResult = {
  original_question: string;
  question_category: QuestionCategory;
  target_concept: string | null;
  answer_source: string;
  statistics_used: Record<string, unknown>;
  evidence_references: string[];
  method: {
    name: string;
    version: string;
    method_type: string;
    uses_language_model: boolean;
    remote_sensing_adapted: boolean;
    assumptions: string[];
    limitations: string[];
  };
  confidence: { level: ConfidenceLevel; score: number | null; reason: string };
  supported: boolean;
  limitations: string[];
  single_image_evidence: SingleImageEvidenceResult | null;
};

export type AgentImageQueryRequest = {
  query: string;
  inputMode: InputMode;
  primaryModality: Modality;
  primaryImageModality?: ImageModality;
  secondaryModality: Modality | null;
  primaryImage: File;
  secondaryImage?: File;
  primaryDate?: string;
  secondaryDate?: string;
  signal?: AbortSignal;
  useCache?: boolean;
  forceRerun?: boolean;
};

export type ReportFormat = "pdf" | "json" | "csv" | "zip";
export type ReportResponse = {
  request_id: string;
  status: string;
  schema_version: string;
  artifacts: Array<{ format: ReportFormat; filename: string; url: string; size_bytes: number }>;
  warnings: string[];
  runtime_ms: number;
};

export type ComparisonTask = "captioning" | "vqa" | "grounding" | "change" | "change_vqa" | "cross_modal" | "pix2pix" | "sarfusionformer" | "sar_analysis";
export type ComparabilityLevel = "direct" | "partial" | "not_direct";
export type ComparisonPreview = { label: string; url: string; kind: string; width: number | null; height: number | null; modality: Modality | null };
export type ComparisonLineageNode = { kind: string; label: string; request_id: string | null; preview_url: string | null };
export type ComparisonItemSummary = {
  request_id: string; task: ComparisonTask; display_name: string; status: string; created_at: string;
  input_mode: InputMode; modalities: Modality[]; thumbnail: ComparisonPreview | null;
  execution_duration_ms: number | null; cached: boolean; report_available: boolean;
  warning_count: number; evidence_product_count: number;
};
export type ComparisonItem = {
  request_id: string; task: ComparisonTask; display_name: string; status: string; created_at: string;
  input_mode: InputMode; modalities: Modality[]; input_previews: ComparisonPreview[]; output_previews: ComparisonPreview[];
  answer: string | null; statistics: Record<string, unknown>; confidence: Record<string, unknown>; provenance: Record<string, unknown>;
  selected_tools: string[]; execution_duration_ms: number | null; device: string | null; warnings: string[]; limitations: string[];
  report_available: boolean; cached: boolean; input_identity: Record<string, unknown>; lineage: ComparisonLineageNode[]; execution_summary: Record<string, unknown>;
};
export type ComparabilityResult = {
  left_request_id: string; right_request_id: string; level: ComparabilityLevel; reason: string;
  shared_inputs: boolean; shared_task_family: boolean; warnings: string[]; overlay_allowed: boolean; difference_allowed: boolean;
};
export type ComparisonAssessment = { assessments: ComparabilityResult[]; overall_level: ComparabilityLevel; warnings: string[] };
export type ComparisonReportResponse = {
  comparison_id: string; status: string; schema_version: string; artifacts: ReportResponse["artifacts"];
  assessments: ComparabilityResult[]; warnings: string[]; runtime_ms: number;
};

export type DemoWorkflow = {
  id: string;
  title: string;
  description: string;
  input_mode: InputMode;
  primary_modality: Modality;
  secondary_modality: Modality | null;
  query: string;
  primary_date: string | null;
  secondary_date: string | null;
  files: Array<{ role: "primary" | "secondary"; filename: string; url: string; mime_type: string }>;
};

export type DemoManifest = { enabled: boolean; local_only: boolean; workflows: DemoWorkflow[] };

export type ComplianceRequirement = {
  requirement: string;
  implementation: string;
  status: string;
  tool_or_model: string;
  test_coverage: string;
  limitation: string;
  readiness_status?: "VERIFIED" | "IMPLEMENTED" | "PARTIALLY VERIFIED" | "EVIDENCE GAP" | "NOT AVAILABLE" | string | null;
  specialists?: string[];
  benchmark_evidence?: string | null;
  evidence_status?: string | null;
};

export type ComplianceResponse = {
  generated_at: string;
  project: string;
  requirements: ComplianceRequirement[];
  mandatory_satisfied: number;
  mandatory_total: number;
  optional_not_implemented: string[];
};

export type ToolDefinition = {
  id: string;
  display_name: string;
  supported_tasks: TaskType[];
  supported_modalities: Modality[];
  supported_input_modes: InputMode[];
  status: "available" | "not_implemented";
  remote_sensing_adapted: boolean;
  service_path: string | null;
  checkpoint: string | null;
  base_architecture: string | null;
  adaptation_dataset: string | null;
  model_license: string | null;
  source: string | null;
  limitations: string[];
  required_modalities: Record<string, Modality[]>;
  method_type: string | null;
  evidence_outputs: string[];
  evidence_source: string | null;
  supported_question_categories: string[];
  notes: string;
  allowed_parameters?: Record<string, { type: string; default?: unknown; minimum?: number | null; maximum?: number | null; choices?: unknown[]; description?: string }>;
  defaults?: Record<string, unknown>;
  constraints?: string[];
  model_version?: string | null;
  evidence_types?: string[];
};

export type ChangeAnalysisStatus = "success" | "alignment_required" | "failed";

export type PixelBoundingBox = {
  left: number;
  top: number;
  right: number;
  bottom: number;
  area_pixels: number;
};

export type ChangeRegion = {
  region_id: number;
  area_pixels: number;
  percentage_of_image: number;
  bounding_box: PixelBoundingBox;
};

export type ChangeStatistics = {
  analysis_width: number;
  analysis_height: number;
  source_width: number;
  source_height: number;
  total_pixels: number;
  changed_pixels: number;
  percentage_changed: number;
  largest_connected_region: number;
  number_of_regions: number;
  bounding_boxes: PixelBoundingBox[];
  regions: ChangeRegion[];
  normalized_threshold: number;
};

export type ChangePreviewUrls = {
  before: string | null;
  after: string | null;
  difference: string | null;
  mask: string | null;
  overlay: string | null;
  ttp_raw_mask?: string | null;
  ttp_mask?: string | null;
  ttp_overlay?: string | null;
  deterministic_mask?: string | null;
  deterministic_overlay?: string | null;
  agreement?: string | null;
  disagreement?: string | null;
  intersection?: string | null;
  union?: string | null;
  top_regions?: string | null;
};

export type ChangeEngine = {
  mode: "deterministic" | "ttp" | "hybrid" | "deterministic_fallback" | string;
  primary_tool: string;
  supporting_tool: string | null;
  fallback_used: boolean;
  fallback_reason: string | null;
};

export type TTPResult = {
  status: string; changed_percentage: number | null; changed_pixels: number | null;
  region_count: number | null; largest_region_pixels: number | null; runtime_ms: number | null;
  model_load_ms: number | null; model: string; architecture: string; training_dataset: string;
  checkpoint: string; checkpoint_fingerprint: string; device: string | null; reused_model: boolean | null;
  limitations: string[]; warnings: string[];
};

export type MaskComparison = {
  intersection_pixels: number; union_pixels: number; iou: number | null;
  agreement_percentage: number; disagreement_percentage: number;
  changed_class_agreement: number | null; background_agreement: number | null; disclaimer: string;
};

export type EvidenceConsistency = {
  label: string; rationale: string[]; calibrated_probability: boolean; disclosure: string;
};

export type SemanticChangeSummary = {
  short_answer: string;
  expanded_answer: string;
  query_intent: string;
  overall_change_level: "minimal" | "localized" | "moderate" | "widespread" | string;
  dominant_location: string | null;
  likely_change_type: string | null;
  evidence_strength: "limited" | "moderate" | "strong" | string;
  stable_area_summary: string | null;
  changed_regions: string[];
  stable_regions: string[];
  likely_transitions: Array<{ type: string; confidence: string; evidence: string[] }>;
  supporting_facts: string[];
  caveats: string[];
  visual_structural_disagreement: boolean;
  built_up_assessment?: {
    state: string; magnitude: string; confidence: string; confidence_factors: string[];
    overlay_label: string; limitations: string[];
    regions: Array<{
      region_id: number; relative_location: string; bbox: number[]; pixel_area: number;
      relative_area_percent: number; change_strength: string; compactness: number;
      before_structural_evidence: number; after_structural_evidence: number;
      before_built_up_evidence: number; after_built_up_evidence: number;
      directional_state: string; semantic_support_level: string;
      deterministic_overlap_percent: number | null;
    }>;
  } | null;
  generated_by: "local_semantic_interpreter" | string;
};

export type ChangeAnalysisResponse = {
  request_id: string;
  status: ChangeAnalysisStatus;
  before_date: string;
  after_date: string;
  before_metadata: ImageMetadata;
  after_metadata: ImageMetadata;
  compatibility: PairCompatibility;
  statistics: ChangeStatistics | null;
  previews: ChangePreviewUrls;
  execution: AgentResponse["execution"];
  runtime_ms: number;
  warnings: string[];
  change_engine?: ChangeEngine | null;
  ttp_result?: TTPResult | null;
  deterministic_statistics?: ChangeStatistics | null;
  mask_comparison?: MaskComparison | null;
  evidence_consistency?: EvidenceConsistency | null;
  sve_result?: SVEResult | null;
  semantic_change_summary?: SemanticChangeSummary | null;
};

export type ChangeAnalysisRequest = {
  beforeImage: File;
  afterImage: File;
  beforeDate: string;
  afterDate: string;
  modality: Modality;
  signal?: AbortSignal;
};

export type CrossModalStatus = "success" | "partial" | "alignment_required" | "failed";

export type CrossModalStatistics = {
  analysis_width: number | null;
  analysis_height: number | null;
  source_width: number | null;
  source_height: number | null;
  water_likelihood_percent: number | null;
  built_up_likelihood_percent: number | null;
  vegetation_support_percent: number | null;
  agreement_percent: number | null;
  disagreement_percent: number | null;
  valid_pixel_percent: number | null;
  evaluated_candidate_pixels: number | null;
};

export type CrossModalRegion = {
  region_id: string;
  type: "water_likelihood" | "built_up_likelihood" | "disagreement";
  area_pixels: number;
  area_percent: number;
  bbox_pixels: [number, number, number, number];
  centroid_pixels: [number, number];
  bbox_world: [number, number, number, number] | null;
  centroid_world: [number, number] | null;
  support: { optical: boolean; sar: boolean };
};

export type CrossModalPreviews = {
  optical: string | null;
  sar: string | null;
  optical_evidence: string | null;
  sar_evidence: string | null;
  joint_evidence: string | null;
  water_likelihood: string | null;
  built_up_likelihood: string | null;
  vegetation_support: string | null;
  agreement: string | null;
  disagreement: string | null;
  joint_overlay: string | null;
};

export type CrossModalPreparation = {
  modality: Modality;
  band_count: number;
  bands_used: string[];
  channel_interpretation: string[];
  stretch_method: string;
  normalization: string;
  invalid_pixel_handling: string;
  resize_status: string;
  log_transform: string;
};

export type CrossModalResult = {
  analysis_level?: "qualitative" | "pixel_verified" | null;
  quantitative_metrics_available?: boolean;
  evidence_products?: CrossModalEvidenceProduct[];
  agreements?: Array<{ category: string; relative_location: string; optical_support: string; sar_support: string; strength: string }>;
  disagreements?: string[];
  complementary_findings?: string[];
  limitations?: string[];
  status: CrossModalStatus;
  summary: { optical_observations: string[]; sar_observations: string[]; joint_observations: string[] };
  statistics: CrossModalStatistics | null;
  regions: CrossModalRegion[];
  previews: CrossModalPreviews;
  confidence: { level: ConfidenceLevel; score: number | null; reason: string };
  method: { name: string; version: string; uses_trained_model: boolean; assumptions: string[]; limitations: string[] };
  optical_preparation: CrossModalPreparation | null;
  sar_preparation: CrossModalPreparation | null;
  warnings: string[];
  runtime_ms: number;
  sve_result?: SVEResult | null;
  evidence_facts?: Array<{ source: "optical" | "native_sar" | "fused" | string; kind: string; statement: string; supporting_region_ids: string[]; calibrated_probability: boolean }> | null;
};

export type CrossModalAnalysisResponse = {
  request_id: string;
  optical_metadata: ImageMetadata;
  sar_metadata: ImageMetadata;
  compatibility: PairCompatibility;
  result: CrossModalResult;
  execution: AgentResponse["execution"];
};

export type CrossModalAnalysisRequest = {
  opticalImage: File;
  sarImage: File;
  query?: string;
  opticalModality: "optical" | "multispectral";
  sarModality: "sar";
  signal?: AbortSignal;
};

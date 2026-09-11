export type PresentationIcon =
  | "welcome"
  | "problem"
  | "solution"
  | "inputs"
  | "workflow"
  | "understanding"
  | "grounding"
  | "change"
  | "joint"
  | "reports"
  | "comparison"
  | "analytics"
  | "architecture"
  | "compliance"
  | "closing";

export type PresentationSceneType = "single_image_understanding" | "bi_temporal_change" | "cross_modal_analysis" | "mission_report" | "mission_comparison" | "research_analytics" | "architecture_summary" | "compliance_summary" | "closing_summary";
export type PresentationSupportedAction = "captioning" | "vqa" | "change" | "cross_modal";
export type PresentationReportFormat = "pdf" | "json" | "zip";
export type PresentationEvidenceMode = "response_dependent";

export type PresenterNote = {
  say: string;
  show: string;
  watchFor: string;
};

export type PresentationStep = {
  number: number;
  title: string;
  subtitle: string;
  points: readonly string[];
  icon?: PresentationIcon;
  presenterNote?: PresenterNote;
  linkedRoute?: string;
  scene_type?: PresentationSceneType;
  demo_sample_id?: string;
  before_sample_id?: string;
  after_sample_id?: string;
  optical_sample_id?: string;
  sar_sample_id?: string;
  default_dates?: { before: string; after: string };
  supported_actions?: readonly PresentationSupportedAction[];
  default_query?: Partial<Record<PresentationSupportedAction, string>>;
  evidence_mode?: PresentationEvidenceMode;
  linked_tool_ids?: readonly string[];
  supported_formats?: readonly PresentationReportFormat[];
  linked_capabilities?: readonly string[];
};

export const presentationSteps = [
  {
    number: 1,
    title: "Welcome to SatQuery AI",
    subtitle: "Auditable Earth-observation intelligence for mission-ready decisions.",
    points: ["One interface for multimodal imagery", "Evidence-first specialist workflows", "Designed for traceability and scientific caution"],
    icon: "welcome",
    presenterNote: {
      say: "SatQuery AI turns Earth-observation imagery into auditable mission evidence.",
      show: "Introduce the single research workspace and the presentation progress bar.",
      watchFor: "Keep this opening focused on the mission problem, not individual models.",
    },
    linkedRoute: "/assistant",
  },
  {
    number: 2,
    title: "The Mission Problem",
    subtitle: "Remote-sensing teams must reason across formats, sensors, dates, and operational constraints.",
    points: ["Inputs vary in modality and metadata", "Not every image pair is safely comparable", "Results must remain inspectable and reproducible"],
    icon: "problem",
    presenterNote: {
      say: "The challenge is not just inference; it is selecting a valid workflow and preserving the evidence chain.",
      show: "Emphasize the three constraints around the central mission graphic.",
      watchFor: "Do not imply that every uploaded pair can be aligned automatically.",
    },
  },
  {
    number: 3,
    title: "The SatQuery AI Solution",
    subtitle: "A deterministic agent coordinates validation, specialist selection, evidence products, and reporting.",
    points: ["Validate before execution", "Route to a declared specialist", "Return outputs with an execution trace"],
    icon: "solution",
    presenterNote: {
      say: "SatQuery AI uses deterministic orchestration to keep specialist selection explainable.",
      show: "Trace the validate, route, and evidence sequence.",
      watchFor: "Avoid describing the router as an autonomous language model.",
    },
    linkedRoute: "/architecture",
  },
  {
    number: 4,
    title: "Supported Inputs",
    subtitle: "Common visual formats and geospatial rasters enter through one validated ingestion layer.",
    points: ["PNG and JPEG imagery", "TIFF and GeoTIFF rasters", "Single, paired, and bi-temporal modes"],
    icon: "inputs",
    presenterNote: {
      say: "The same upload layer handles ordinary images and geospatial rasters.",
      show: "Call out both file formats and input modes.",
      watchFor: "Metadata availability differs between ordinary images and GeoTIFFs.",
    },
    linkedRoute: "/assistant",
  },
  {
    number: 5,
    title: "Agentic Workflow",
    subtitle: "Every request follows a visible path from ingestion to a governed response.",
    points: ["Metadata and compatibility checks", "Task-aware specialist routing", "Runtime and warnings recorded at each stage"],
    icon: "workflow",
    presenterNote: {
      say: "The workflow is agentic in coordination and explicit in execution.",
      show: "Point from ingestion through routing to response evidence.",
      watchFor: "Judges may ask which routing choices are deterministic.",
    },
    linkedRoute: "/architecture",
  },
  {
    number: 6,
    title: "Single-Image Understanding",
    subtitle: "Specialists inspect one optical, multispectral, or SAR observation at a time.",
    points: ["Remote-sensing-adapted scene priors", "Caption and VQA evidence consistency", "Specialist provenance and limitations remain visible"],
    icon: "understanding",
    presenterNote: {
      say: "SatQuery Vision Encoder v1 is our OpenCLIP ViT-L/14 backbone adapted on BigEarthNet.txt image-text pairs. It provides Earth-observation-aware scene embeddings used for routing, caption consistency, retrieval, and evidence fusion. Specialist models remain responsible for grounding, change detection, and sensor-specific analysis.",
      show: "Load the approved optical sample, show the remote-sensing adapted badge and scene priors, then demonstrate caption consistency and one land-cover VQA evidence-consistency result before opening the execution trace.",
      watchFor: "Scene priors are similarity evidence, not calibrated probabilities or ground truth. Captioning remains RSICD-adapted and VQA remains deterministic.",
    },
    linkedRoute: "/assistant",
    scene_type: "single_image_understanding",
    demo_sample_id: "single_vqa",
    supported_actions: ["captioning", "vqa"],
    default_query: {
      captioning: "Describe the land cover and major objects visible in this image.",
      vqa: "What is the dominant land-cover type?",
    },
    evidence_mode: "response_dependent",
    linked_tool_ids: ["satquery_vision_encoder_v1", "rs_captioner", "rs_vqa"],
  },
  {
    number: 7,
    title: "Text-Guided Grounding",
    subtitle: "Natural-language targets are localized as inspectable candidate regions.",
    points: ["Prompt-target extraction", "Visible boxes and alignment scores", "No fabricated box when thresholds are not met"],
    icon: "grounding",
    presenterNote: {
      say: "Grounding connects a requested target to visible candidate regions.",
      show: "Explain that boxes and scores remain inspectable outputs.",
      watchFor: "Scores are model alignment scores, not calibrated probabilities.",
    },
    linkedRoute: "/assistant",
  },
  {
    number: 8,
    title: "Bi-Temporal Change Analysis",
    subtitle: "Compatible optical pairs combine a primary ChangerEx learned mask with independent deterministic evidence.",
    points: ["ChangerEx learned and deterministic masks", "Agreement, disagreement, area, and regions", "Provenance and automatic fallback remain visible"],
    icon: "change",
    presenterNote: {
      say: "SatQuery AI uses ChangerEx as the default local learned change detector for compatible optical pairs. An independent deterministic engine produces a second evidence stream. Agreement and disagreement are shown transparently, and the system does not infer the semantic cause of change.",
      show: "Run the analysis explicitly, then show the learned mask, deterministic difference, agreement map, change percentage, largest region, provenance, and compact execution trace.",
      watchFor: "ChangerEx output is not ground truth; incompatible pairs are never silently aligned and specialist failure uses the disclosed deterministic fallback.",
    },
    linkedRoute: "/assistant",
    scene_type: "bi_temporal_change",
    demo_sample_id: "change_vqa",
    before_sample_id: "change-before.png",
    after_sample_id: "change-after.png",
    default_dates: { before: "2025-01-01", after: "2025-02-01" },
    default_query: { change: "What changed between these dates?" },
    linked_tool_ids: ["ttp_change_detector", "bitemporal_change_analyzer"],
  },
  {
    number: 9,
    title: "Optical–SAR Joint Analysis",
    subtitle: "Complementary sensors are compared through declared structural and visual evidence.",
    points: ["Cross-modal compatibility validation", "Sensor-aware evidence layers", "Limitations remain attached to the result"],
    icon: "joint",
    presenterNote: {
      say: "Optical imagery contributes spectral and contextual information, while SAR contributes structural and low-backscatter evidence. SatQuery AI combines both only when the pair is compatible.",
      show: "Point to the separate optical and SAR evidence, followed by the agreement, disagreement, and joint overlay products.",
      watchFor: "These are likelihood and support maps, not calibrated semantic ground truth. Images are never silently aligned.",
    },
    linkedRoute: "/assistant",
    scene_type: "cross_modal_analysis",
    demo_sample_id: "cross_modal",
    optical_sample_id: "cross-optical.tif",
    sar_sample_id: "cross-sar.tif",
    supported_actions: ["cross_modal"],
    default_query: { cross_modal: "Use the optical and SAR images together to identify built-up and water-covered regions." },
    linked_tool_ids: ["cross_modal_optical_sar_analyzer"],
  },
  {
    number: 10,
    title: "Mission Reports",
    subtitle: "Operational outputs can be packaged with provenance, warnings, and traceable evidence.",
    points: ["Human-readable mission summary", "Machine-readable evidence package", "Request identity preserved across exports"],
    icon: "reports",
    presenterNote: {
      say: "Every result can be converted into a reproducible mission report generated from the authoritative backend record.",
      show: "Highlight the evidence, provenance, confidence, execution trace, and limitations included in the report.",
      watchFor: "Reports never trust browser-supplied statistics and do not include secrets or local filesystem paths.",
    },
    linkedRoute: "/reports",
    scene_type: "mission_report",
    linked_tool_ids: ["report_generator"],
    supported_formats: ["pdf", "json", "zip"],
  },
  {
    number: 11,
    title: "Mission Comparison",
    subtitle: "Results can be reviewed side by side only when their comparison relationship is valid.",
    points: ["Identity-aware result selection", "Direct and partial comparison rules", "Visual evidence with guarded difference views"],
    icon: "comparison",
    presenterNote: {
      say: "SatQuery AI compares authoritative stored results rather than browser-supplied values.",
      show: "Highlight shared input identity, comparability level, factual differences, and the absence of a universal winner.",
      watchFor: "Runtime differences are not treated as scientific accuracy differences, and unrelated workflows are explicitly marked as not directly comparable.",
    },
    linkedRoute: "/assistant/compare",
    scene_type: "mission_comparison",
    linked_capabilities: ["authoritative stored-result comparison"],
  },
  {
    number: 12,
    title: "Research Analytics",
    subtitle: "Operational evidence exposes platform health, specialist activity, and measured runtime behavior.",
    points: ["Live backend status", "Workflow and specialist evidence", "No substitute values when data is unavailable"],
    icon: "analytics",
    presenterNote: {
      say: "Analytics reports observable platform behavior rather than invented performance claims.",
      show: "Highlight health, activity, and evidence coverage.",
      watchFor: "Separate illustrative benchmark previews from measured analytics.",
    },
    linkedRoute: "/assistant/analytics",
    scene_type: "research_analytics",
    linked_capabilities: ["live research analytics"],
  },
  {
    number: 13,
    title: "Architecture and Auditability",
    subtitle: "The system makes routing, specialist provenance, safeguards, and evidence flow inspectable.",
    points: ["Layered agent architecture", "Live execution-path visualization", "Truthful scope and limitations"],
    icon: "architecture",
    presenterNote: {
      say: "Auditability is designed into the architecture, not added only at report time.",
      show: "Follow the layers from interface to governance.",
      watchFor: "Private reasoning is never exposed as an audit artifact.",
    },
    linkedRoute: "/architecture",
    scene_type: "architecture_summary",
    linked_capabilities: ["live agentic architecture"],
  },
  {
    number: 14,
    title: "SIH Compliance",
    subtitle: "Requirements are mapped to implementations, evidence, tests, and declared limitations.",
    points: ["Requirement-by-requirement matrix", "Implementation and test references", "Unavailable capabilities remain explicit"],
    icon: "compliance",
    presenterNote: {
      say: "The compliance view connects each mandatory requirement to inspectable implementation evidence.",
      show: "Point across requirement, implementation, test coverage, and limitation.",
      watchFor: "Do not mark planned capabilities as already available.",
    },
    linkedRoute: "/assistant/compliance",
    scene_type: "compliance_summary",
    linked_capabilities: ["authoritative SIH compliance matrix"],
  },
  {
    number: 15,
    title: "Ask Earth Anything.",
    subtitle: "SatQuery AI brings validated imagery, specialist intelligence, and auditable evidence into one mission workspace.",
    points: ["Scientifically cautious", "Operationally inspectable", "Ready for the guided SIH demonstration"],
    icon: "closing",
    presenterNote: {
      say: "SatQuery AI helps teams ask better questions while preserving the evidence behind every result.",
      show: "Return attention to the SatQuery AI identity and the completed progress bar.",
      watchFor: "Pause for questions before leaving Presentation Mode.",
    },
    linkedRoute: "/assistant",
    scene_type: "closing_summary",
    linked_capabilities: ["complete guided walkthrough"],
  },
] as const satisfies readonly PresentationStep[];

export const presentationStorageKeys = {
  step: "geovision-presentation-step-v1",
  notes: "geovision-presentation-notes-v1",
  returnRoute: "geovision-presentation-return-route-v1",
  singleImageAction: "geovision-presentation-single-image-action-v1",
  singleImageQuery: "geovision-presentation-single-image-query-v1",
  changeQuery: "geovision-presentation-change-query-v1",
  changeBeforeDate: "geovision-presentation-change-before-date-v1",
  changeAfterDate: "geovision-presentation-change-after-date-v1",
  crossModalQuery: "geovision-presentation-cross-modal-query-v1",
  comparisonReturn: "geovision-presentation-comparison-return-v1",
  comparisonRequestIds: "geovision-presentation-comparison-request-ids-v1",
  handoffRoute: "geovision-presentation-handoff-route-v1",
  handoffTheme: "geovision-presentation-handoff-theme-v1",
  handoffFullscreen: "geovision-presentation-handoff-fullscreen-v1",
} as const;

export type PresentationCommand = "next" | "previous" | "first" | "last" | "fullscreen" | "notes" | "exit";

export function clampPresentationStep(value: unknown, count = presentationSteps.length): number {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(Math.max(Math.trunc(parsed), 0), Math.max(count - 1, 0));
}

export function presentationCommandForKey(key: string): PresentationCommand | null {
  if (key === "ArrowRight" || key === " ") return "next";
  if (key === "ArrowLeft" || key === "Backspace") return "previous";
  if (key === "Home") return "first";
  if (key === "End") return "last";
  if (key.toLowerCase() === "f") return "fullscreen";
  if (key.toLowerCase() === "n") return "notes";
  if (key === "Escape") return "exit";
  return null;
}

export function isPresentationTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest("input, textarea, select, button, a, [contenteditable='true'], [role='textbox']"));
}

export function rememberPresentationReturnRoute(storage: Storage, route: string): void {
  const safeRoute = validReturnRoute(route);
  storage.setItem(presentationStorageKeys.returnRoute, safeRoute);
}

export function readPresentationReturnRoute(storage: Storage): string {
  return validReturnRoute(storage.getItem(presentationStorageKeys.returnRoute) ?? "");
}

export function readPresentationState(storage: Storage): { step: number; notesVisible: boolean } {
  return {
    step: clampPresentationStep(storage.getItem(presentationStorageKeys.step)),
    notesVisible: storage.getItem(presentationStorageKeys.notes) === "true",
  };
}

export function clearPresentationState(storage: Storage): void {
  storage.removeItem(presentationStorageKeys.step);
  storage.removeItem(presentationStorageKeys.notes);
  storage.removeItem(presentationStorageKeys.returnRoute);
  storage.removeItem(presentationStorageKeys.singleImageAction);
  storage.removeItem(presentationStorageKeys.singleImageQuery);
  storage.removeItem(presentationStorageKeys.changeQuery);
  storage.removeItem(presentationStorageKeys.changeBeforeDate);
  storage.removeItem(presentationStorageKeys.changeAfterDate);
  storage.removeItem(presentationStorageKeys.crossModalQuery);
  storage.removeItem(presentationStorageKeys.comparisonReturn);
  storage.removeItem(presentationStorageKeys.comparisonRequestIds);
  storage.removeItem(presentationStorageKeys.handoffRoute);
  storage.removeItem(presentationStorageKeys.handoffTheme);
  storage.removeItem(presentationStorageKeys.handoffFullscreen);
}

function validReturnRoute(route: string): string {
  if (!route.startsWith("/") || route.startsWith("//")) return "/assistant";
  const pathname = route.split(/[?#]/, 1)[0].replace(/\/+$/, "") || "/";
  if (pathname === "/presentation") return "/assistant";
  return route;
}

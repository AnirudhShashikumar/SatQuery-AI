import type { AgentResponse, DemoManifest, DemoWorkflow, PairCompatibility } from "@/types/agent";
import { changeEngineName } from "@/lib/scientific-presentation";

export const changeDefaultQuery = "What changed between these dates?";
export const changeDefaultDates = { before: "2025-01-01", after: "2025-02-01" } as const;

export type ApprovedChangePair = {
  workflow: DemoWorkflow;
  before: DemoWorkflow["files"][number];
  after: DemoWorkflow["files"][number];
  beforeDate: string;
  afterDate: string;
};

export type ChangeEvidencePreview = { label: string; path: string };

export function approvedChangePair(
  manifest: DemoManifest,
  workflowId: string,
  beforeSampleId: string,
  afterSampleId: string,
  fallbackDates: { before: string; after: string } = changeDefaultDates,
): ApprovedChangePair {
  if (!manifest.enabled) throw new Error("Local demo mode is disabled. Start the backend with SATQUERY_DEMO_MODE=true to use the approved presentation pair.");
  const workflow = manifest.workflows.find(item => item.id === workflowId);
  if (!workflow || workflow.input_mode !== "bi_temporal") throw new Error("The approved bi-temporal demo pair is unavailable from the local manifest.");
  const before = workflow.files.find(item => item.role === "primary" && item.filename === beforeSampleId);
  const after = workflow.files.find(item => item.role === "secondary" && item.filename === afterSampleId);
  if (!before || !after) throw new Error("The approved before/after demo files are missing from the local manifest.");
  return {
    workflow,
    before,
    after,
    beforeDate: workflow.primary_date || fallbackDates.before,
    afterDate: workflow.secondary_date || fallbackDates.after,
  };
}

export function changeEvidencePreviews(response: AgentResponse): ChangeEvidencePreview[] {
  const previews = response.change_analysis?.previews;
  if (!previews) return [];
  const engine = response.change_engine ?? response.change_analysis?.change_engine;
  const learned = response.ttp_result ?? response.change_analysis?.ttp_result;
  const engineName = changeEngineName(engine, learned);
  const hybrid = response.change_engine?.mode === "hybrid" || response.change_analysis?.change_engine?.mode === "hybrid" || response.change_engine?.mode === "ttp";
  const candidates: Array<ChangeEvidencePreview | null> = [
    previews.difference ? { label: "Difference Heatmap", path: previews.difference } : null,
    hybrid && previews.ttp_mask ? { label: "Learned Change Mask", path: previews.ttp_mask } : null,
    hybrid && previews.deterministic_mask ? { label: "Deterministic Difference", path: previews.deterministic_mask } : null,
    hybrid && previews.agreement ? { label: "Agreement", path: previews.agreement } : null,
    hybrid && previews.disagreement ? { label: "Disagreement", path: previews.disagreement } : null,
    hybrid && previews.ttp_overlay ? { label: `${engineName} Overlay`, path: previews.ttp_overlay } : null,
    !hybrid && previews.mask ? { label: "Binary mask", path: previews.mask } : null,
    !hybrid && previews.overlay ? { label: "Overlay", path: previews.overlay } : null,
  ];
  return candidates.filter((item): item is ChangeEvidencePreview => item !== null);
}

export function compatibilityLabel(compatibility: PairCompatibility | null): string {
  if (!compatibility) return "Compatibility pending";
  if (!compatibility.compatible) return compatibility.resampling_required ? "Alignment required" : "Incompatible pair";
  if (compatibility.alignment_level === "visual_only") return "Compatible · visual only";
  if (compatibility.alignment_level === "exact") return "Compatible · exact alignment";
  return "Compatible · geospatial overlap";
}

export function changeResponseDisclosure(response: AgentResponse): string | null {
  const change = response.change_analysis;
  const compatibility = change?.compatibility ?? response.pair_compatibility;
  if (response.status === "alignment_required" || change?.status === "alignment_required") {
    return "Alignment is required. SatQuery did not resize, register, reproject, or resample either observation.";
  }
  if (compatibility && !compatibility.compatible) {
    return "This pair is incompatible for pixel-level change analysis. No change statistics were fabricated and neither image was altered.";
  }
  if (compatibility?.alignment_level === "visual_only") {
    return "Visual-only compatibility: equal pixel dimensions support comparison, but geospatial correspondence is not independently verified.";
  }
  if (!change) return response.answer || response.warnings[0] || "The backend did not return a change-analysis result.";
  if (change.status === "failed") return change.warnings[0] || "Change analysis failed before measurable products could be created.";
  return null;
}

export function requestFailureMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : "The local SatQuery change-analysis request failed.";
  if (/fetch|network|backend|failed to connect/i.test(message)) return "The local SatQuery backend is offline. Start it on port 8010 and retry.";
  if (/expired|not found|404/i.test(message)) return "The result or preview has expired. Re-run analysis to generate a current evidence set.";
  return message;
}

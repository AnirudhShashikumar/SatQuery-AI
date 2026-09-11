import type { AnalyticsExecution } from "@/types/agent";

export type PathState = "active" | "completed" | "failed" | "skipped" | "neutral";
export const DEFAULT_ARCHITECTURE_TAB = "agentic" as const;

export type ArchitectureNode = {
  id: string;
  label: string;
  detail: string;
  matchTrace?: string[];
  matchTools?: string[];
  matchModes?: string[];
  matchModalities?: string[];
  matchTasks?: string[];
  evidence?: boolean;
  report?: boolean;
  confidence?: boolean;
  warnings?: boolean;
  traceRecord?: boolean;
  anyExecution?: boolean;
};

export const architectureLabel = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());

const toolAliases: Record<string, string> = {
  deterministic_change_analysis: "bitemporal_change_analyzer",
  deterministic_change_analyzer: "bitemporal_change_analyzer",
  deterministic_change_vqa: "bitemporal_change_analyzer",
};

export function canonicalToolId(value: string) {
  return toolAliases[value] ?? value;
}

function includesTerm(value: string, terms: string[]) {
  const normalized = value.toLowerCase();
  return terms.some(term => normalized === term || normalized.includes(term));
}

export function architectureNodeState(node: ArchitectureNode, execution?: AnalyticsExecution): PathState {
  if (!execution) return "neutral";
  const matchedSteps = execution.trace.filter(step => node.matchTrace && includesTerm(step.tool, node.matchTrace));
  const selected = node.matchTools?.some(tool => execution.selected_tools.map(canonicalToolId).includes(canonicalToolId(tool))) ?? false;
  const mode = node.matchModes?.includes(execution.input_mode) ?? false;
  const modalities = [execution.primary_modality, execution.secondary_modality].filter(Boolean) as string[];
  const modality = node.matchModalities?.some(value => modalities.includes(value)) ?? false;
  const inputMatch = node.matchModes && node.matchModalities ? mode && modality : mode || modality;
  const task = node.matchTasks?.includes(execution.task) ?? false;
  const evidence = Boolean(node.evidence && execution.output_count > 0);
  const report = Boolean(node.report && execution.report_generated);
  const confidence = Boolean(node.confidence && execution.confidence_level);
  const warnings = Boolean(node.warnings && execution.warning_count > 0);
  const traceRecord = Boolean(node.traceRecord && execution.trace.length > 0);
  const anyExecution = Boolean(node.anyExecution);
  if (!matchedSteps.length && !selected && !inputMatch && !task && !evidence && !report && !confidence && !warnings && !traceRecord && !anyExecution) return "neutral";
  if (matchedSteps.some(step => step.status === "running" || step.status === "pending")) return "active";
  if (matchedSteps.some(step => step.status === "failed") || execution.status === "failed" && (selected || task || anyExecution)) return "failed";
  if (matchedSteps.length && matchedSteps.every(step => step.status === "skipped" || step.status === "not_implemented")) return "skipped";
  return "completed";
}

export function buildExecutionSummary(execution: AnalyticsExecution) {
  const modalities = [execution.primary_modality, execution.secondary_modality].filter(Boolean).map(value => architectureLabel(value!)).join(" + ") || "Not reported";
  const lines = [
    "SatQuery AI observable execution summary",
    `Request ID: ${execution.request_id}`,
    `Completed: ${execution.completed_at}`,
    `Task: ${architectureLabel(execution.task)}`,
    `Input mode: ${architectureLabel(execution.input_mode)}`,
    `Modalities: ${modalities}`,
    `Status: ${architectureLabel(execution.status)}`,
    `Selected tools: ${execution.selected_tools.join(", ") || "None"}`,
    `Total runtime: ${execution.duration_ms} ms`,
    `Cache: ${architectureLabel(execution.cache_status)}`,
    `Evidence products: ${execution.output_count}`,
    `Warnings: ${execution.warning_count}`,
    `Report generated: ${execution.report_generated ? "Yes" : "No"}`,
  ];
  if (execution.selection_reason) lines.push(`Selection reason: ${execution.selection_reason}`);
  if (execution.confidence_level) lines.push(`Confidence: ${architectureLabel(execution.confidence_level)}${execution.confidence_reason ? ` — ${execution.confidence_reason}` : ""}`);
  lines.push("Trace:");
  execution.trace.forEach((step, index) => lines.push(`${index + 1}. ${step.tool} — ${step.status} — ${step.duration_ms} ms`));
  return lines.join("\n");
}

export function activePathSummary(execution?: AnalyticsExecution) {
  if (!execution) return "No completed workflow is available. Run a workflow in SatQuery Assistant to visualize its execution path.";
  return `${architectureLabel(execution.input_mode)} input routed to ${architectureLabel(execution.task)}. ${execution.selected_tools.length} selected tools and ${execution.trace.length} observable stages completed with status ${architectureLabel(execution.status)} in ${execution.duration_ms} milliseconds.`;
}

export function statusDescription(state: PathState) {
  return { active: "Currently active", completed: "Involved and completed", failed: "Involved and failed", skipped: "Involved but skipped", neutral: "Not involved in the last execution" }[state];
}

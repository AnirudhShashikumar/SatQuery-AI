import type { ComparabilityResult, ComparisonItem, ComparisonPreview } from "@/types/agent";

export type DifferenceFact = { label: string; values: Record<string, string>; caution?: string };

export const comparisonTaskLabel = (task: ComparisonItem["task"]) => ({
  captioning: "Captioning", vqa: "Controlled VQA", grounding: "Grounding", change: "Change analysis",
  change_vqa: "Change VQA", cross_modal: "Optical–SAR", pix2pix: "Pix2Pix", sarfusionformer: "SARFusionFormer",
  sar_analysis: "Single-Image SAR Analysis",
}[task]);

export const scalarEntries = (value: unknown, prefix = ""): Array<[string, string]> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const output: Array<[string, string]> = [];
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    const label = [...(prefix ? [prefix] : []), key].join(" · ").replaceAll("_", " ");
    if (typeof nested === "number") output.push([label, Number.isInteger(nested) ? nested.toLocaleString() : nested.toFixed(3)]);
    else if (typeof nested === "string" || typeof nested === "boolean") output.push([label, String(nested)]);
    else if (nested && typeof nested === "object" && !Array.isArray(nested)) output.push(...scalarEntries(nested, label));
  }
  return output.slice(0, 18);
};

const findNumber = (value: unknown, key: string): number | null => {
  if (!value || typeof value !== "object") return null;
  if (!Array.isArray(value) && typeof (value as Record<string, unknown>)[key] === "number") return (value as Record<string, number>)[key];
  for (const nested of Object.values(value as Record<string, unknown>)) {
    const found = findNumber(nested, key);
    if (found != null) return found;
  }
  return null;
};

export function factualDifferences(items: ComparisonItem[], assessments: ComparabilityResult[]): DifferenceFact[] {
  if (items.length < 2) return [];
  const values = (render: (item: ComparisonItem) => string) => Object.fromEntries(items.map(item => [item.request_id, render(item)]));
  const runtimes = items.map(item => item.execution_duration_ms).filter((value): value is number => value != null);
  const fastest = runtimes.length === items.length ? Math.min(...runtimes) : null;
  const facts: DifferenceFact[] = [
    { label: "Runtime", values: values(item => item.execution_duration_ms == null ? "Unavailable" : `${item.execution_duration_ms.toLocaleString()} ms${item.execution_duration_ms === fastest ? " · Faster" : ""}`), caution: "Runtime is not a scientific accuracy score." },
    { label: "Evidence products", values: values(item => `${item.output_previews.length}${item.output_previews.length === Math.max(...items.map(value => value.output_previews.length)) ? " · More evidence products" : ""}`) },
    { label: "Cache state", values: values(item => item.cached ? "Cached" : "Fresh") },
    { label: "Warnings", values: values(item => item.warnings.length.toString()) },
    { label: "Tool selection", values: values(item => item.selected_tools.join(" → ") || "Unavailable"), caution: new Set(items.map(item => item.selected_tools.join("|"))).size > 1 ? "Different method" : undefined },
    { label: "Confidence basis", values: values(item => `${String(item.confidence.level ?? "unavailable")} · ${String(item.confidence.reason ?? "No basis reported")}`), caution: new Set(items.map(item => String(item.confidence.reason))).size > 1 ? "Different confidence basis" : undefined },
    { label: "Report", values: values(item => item.report_available ? "Available" : "Not generated") },
  ];
  if (items.every(item => item.task === items[0].task) && ["change", "change_vqa"].includes(items[0].task)) {
    facts.push({ label: "Changed percentage", values: values(item => { const value = findNumber(item.statistics, "percentage_changed"); return value == null ? "Unavailable" : `${value.toFixed(3)}%`; }) });
    facts.push({ label: "Region count", values: values(item => { const value = findNumber(item.statistics, "number_of_regions"); return value == null ? "Unavailable" : value.toLocaleString(); }) });
  }
  if (items.every(item => item.task === "grounding")) facts.push({ label: "Detection count", values: values(item => { const value = findNumber(item.statistics, "detection_count"); return value == null ? "Unavailable" : value.toLocaleString(); }) });
  if (assessments.some(item => item.level === "not_direct")) facts.unshift({ label: "Comparability", values: values(() => "Not directly comparable"), caution: "Measurements remain task-specific." });
  return facts;
}

export const dimensionsMatch = (left?: ComparisonPreview | null, right?: ComparisonPreview | null) => Boolean(left?.width && left?.height && right?.width && right?.height && left.width === right.width && left.height === right.height);
export const pairAssessment = (assessments: ComparabilityResult[], leftId: string, rightId: string) => assessments.find(value => (value.left_request_id === leftId && value.right_request_id === rightId) || (value.left_request_id === rightId && value.right_request_id === leftId));

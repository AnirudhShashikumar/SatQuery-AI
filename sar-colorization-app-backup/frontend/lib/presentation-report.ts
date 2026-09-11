import { ApiRequestError } from "@/services/api";
import type { ComparisonItem, ReportFormat } from "@/types/agent";

export const latestSatQueryRequestKey = "satquery-latest-request-id";
export const presentationReportFormats = ["pdf", "json", "zip"] as const satisfies readonly ReportFormat[];

export type ReportContentsOverview = {
  label: string;
  value: string;
  caution?: boolean;
};

export function isReportableResult(item: ComparisonItem): boolean {
  return ["success", "partial", "alignment_required"].includes(item.status);
}

export function evidenceCount(item: ComparisonItem): number {
  return item.output_previews.length;
}

export function reportContentsOverview(item: ComparisonItem): ReportContentsOverview[] {
  const trace = Array.isArray(item.execution_summary.steps) ? item.execution_summary.steps.length : 0;
  const statistics = Object.keys(item.statistics).length;
  const provenance = Object.keys(item.provenance).length;
  return [
    { label: "Inputs", value: `${item.input_previews.length} stored input preview${item.input_previews.length === 1 ? "" : "s"}` },
    { label: "Query", value: "Backend-stored original query" },
    { label: "Answer", value: item.answer ? "Stored authoritative answer" : "No textual answer", caution: !item.answer },
    { label: "Evidence", value: `${evidenceCount(item)} evidence product${evidenceCount(item) === 1 ? "" : "s"}`, caution: evidenceCount(item) === 0 },
    { label: "Statistics", value: statistics ? `${statistics} task statistics section${statistics === 1 ? "" : "s"}` : "No statistics section", caution: statistics === 0 },
    { label: "Confidence", value: String(item.confidence.level ?? "Unavailable") },
    { label: "Provenance", value: provenance ? `${item.selected_tools.length} selected tool${item.selected_tools.length === 1 ? "" : "s"}` : "No provenance record", caution: provenance === 0 },
    { label: "Execution trace", value: `${trace} recorded step${trace === 1 ? "" : "s"}` },
    { label: "Limitations", value: `${item.limitations.length} declared limitation${item.limitations.length === 1 ? "" : "s"}` },
  ];
}

export function presentationReportError(error: unknown): string {
  if (error instanceof ApiRequestError) {
    if (["COMPARISON_RESULT_EXPIRED", "COMPARISON_RESULT_NOT_FOUND", "REPORT_SOURCE_EXPIRED"].includes(error.code)) {
      return "The latest authoritative result is unavailable or expired. Run a presentation analysis again before generating a report.";
    }
    if (error.code === "REPORT_NOT_AVAILABLE") return error.message;
  }
  const message = error instanceof Error ? error.message : "Mission report generation failed.";
  if (/fetch|network|backend|connect/i.test(message)) return "The local SatQuery backend is offline. Restart it on port 8010 and retry.";
  if (/expired|unavailable|not found|404|410/i.test(message)) return "The latest result or report artifact is unavailable or expired.";
  return message;
}

export function formatArtifactSize(sizeBytes: number): string {
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KB`;
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

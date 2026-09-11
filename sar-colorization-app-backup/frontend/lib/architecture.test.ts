import { describe, expect, it } from "vitest";
import type { AnalyticsExecution } from "@/types/agent";
import { activePathSummary, architectureLabel, architectureNodeState, buildExecutionSummary, canonicalToolId, DEFAULT_ARCHITECTURE_TAB } from "./architecture";

const run: AnalyticsExecution = {
  request_id: "request-1", started_at: "2026-01-01T00:00:00Z", completed_at: "2026-01-01T00:00:01Z",
  task: "change_analysis", input_mode: "bi_temporal", primary_modality: "optical", secondary_modality: "optical",
  status: "success", selected_tools: ["input_validator", "deterministic_change_analysis"], duration_ms: 18,
  warning_count: 1, output_count: 5, cache_status: "not_requested", report_generated: false, device: null,
  selection_reason: "Bi-temporal uploads use deterministic analysis.", confidence_level: null, confidence_reason: null,
  warnings: ["Measured warning"],
  trace: [
    { tool: "upload", status: "success", duration_ms: 1, parameters: {} },
    { tool: "pair_validation", status: "success", duration_ms: 2, parameters: { alignment_level: "exact" } },
    { tool: "morphology", status: "skipped", duration_ms: 0, parameters: {} },
  ],
};

describe("agentic architecture helpers", () => {
  it("defaults to the agentic system view", () => expect(DEFAULT_ARCHITECTURE_TAB).toBe("agentic"));
  it("labels backend identifiers", () => expect(architectureLabel("bi_temporal")).toBe("Bi Temporal"));
  it("maps direct change tool aliases to the registry specialist", () => expect(canonicalToolId("deterministic_change_analysis")).toBe("bitemporal_change_analyzer"));
  it("highlights the real input mode", () => expect(architectureNodeState({ id: "pair", label: "pair", detail: "", matchModes: ["bi_temporal"] }, run)).toBe("completed"));
  it("highlights an aliased selected specialist", () => expect(architectureNodeState({ id: "change", label: "change", detail: "", matchTools: ["bitemporal_change_analyzer"] }, run)).toBe("completed"));
  it("uses exact trace failure state", () => expect(architectureNodeState({ id: "failed", label: "failed", detail: "", matchTrace: ["pair_validation"] }, { ...run, trace: [{ ...run.trace[1], status: "failed" }] })).toBe("failed"));
  it("renders skipped stages distinctly", () => expect(architectureNodeState({ id: "skip", label: "skip", detail: "", matchTrace: ["morphology"] }, run)).toBe("skipped"));
  it("keeps uninvolved nodes neutral", () => expect(architectureNodeState({ id: "ground", label: "ground", detail: "", matchTools: ["rs_grounder"] }, run)).toBe("neutral"));
  it("does not claim a routing node for a direct endpoint trace", () => expect(architectureNodeState({ id: "route", label: "route", detail: "", matchTrace: ["query_routing"] }, run)).toBe("neutral"));
  it("builds an observable-only copy summary", () => {
    const summary = buildExecutionSummary(run);
    expect(summary).toContain("Request ID: request-1");
    expect(summary).toContain("pair_validation — success — 2 ms");
    expect(summary).not.toContain("hidden reasoning");
  });
  it("provides empty and populated screen-reader summaries", () => {
    expect(activePathSummary()).toContain("No completed workflow");
    expect(activePathSummary(run)).toContain("3 observable stages");
  });
});

import { describe, expect, it } from "vitest";
import { dimensionsMatch, factualDifferences, scalarEntries } from "./mission-comparison";
import type { ComparisonItem } from "@/types/agent";

const result = (id: string, runtime: number, task: ComparisonItem["task"] = "change"): ComparisonItem => ({
  request_id: id, task, display_name: id, status: "success", created_at: "2026-01-01T00:00:00Z", input_mode: "bi_temporal", modalities: ["optical"], input_previews: [], output_previews: [{ label: "mask", url: id, kind: "evidence", width: 64, height: 64, modality: null }], answer: null, statistics: { change: { percentage_changed: runtime, number_of_regions: 2 } }, confidence: { level: "unavailable", reason: "deterministic" }, provenance: {}, selected_tools: [task], execution_duration_ms: runtime, device: "cpu", warnings: [], limitations: [], report_available: false, cached: false, input_identity: {}, lineage: [], execution_summary: {},
});

describe("mission comparison facts", () => {
  it("labels measured runtime without declaring a winner", () => {
    const facts = factualDifferences([result("a", 10), result("b", 20)], []);
    expect(facts.find(value => value.label === "Runtime")?.values.a).toContain("Faster");
    expect(JSON.stringify(facts).toLowerCase()).not.toContain("winner");
  });
  it("only exposes shared change statistics for the same task", () => {
    expect(factualDifferences([result("a", 10), result("b", 20)], []).some(value => value.label === "Changed percentage")).toBe(true);
    expect(factualDifferences([result("a", 10), result("b", 20, "grounding")], []).some(value => value.label === "Changed percentage")).toBe(false);
  });
  it("checks exact preview dimensions for overlay", () => {
    expect(dimensionsMatch(result("a", 1).output_previews[0], result("b", 2).output_previews[0])).toBe(true);
    expect(dimensionsMatch(result("a", 1).output_previews[0], { ...result("b", 2).output_previews[0], width: 32 })).toBe(false);
  });
  it("flattens only factual scalar statistics", () => {
    expect(scalarEntries({ change: { percentage_changed: 2.5 }, regions: [{ id: 1 }] })).toEqual([["change · percentage changed", "2.500"]]);
  });
});

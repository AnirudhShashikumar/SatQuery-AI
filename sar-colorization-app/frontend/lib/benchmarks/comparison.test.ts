import { describe, expect, it } from "vitest";
import { comparisonCompatibility } from "./comparison";
import { benchmarkData } from "./data";

describe("benchmark comparison guard", () => {
  const rsvqa = benchmarkData.records.find(record => record.evaluation.status === "verified_test")!;
  it("blocks different datasets or splits", () => {
    const validation = benchmarkData.records.find(record => record.benchmark_id.includes("validation.epoch8"))!;
    expect(comparisonCompatibility(rsvqa, validation)).toEqual({ compatible: false, reason: "Runs must use the same dataset and split.", sharedMetricIds: [] });
  });

  it("blocks cross-task ranking", () => {
    const grounding = benchmarkData.preferred.find(record => record.specialist_id === "grounding")!;
    expect(comparisonCompatibility(rsvqa, grounding).reason).toContain("different tasks");
  });

  it("allows a common metric only under the same protocol context", () => {
    const peer = structuredClone(rsvqa);
    peer.benchmark_id = "peer";
    peer.metrics = [rsvqa.metrics[0]];
    const result = comparisonCompatibility(rsvqa, peer);
    expect(result.compatible).toBe(true);
    expect(result.sharedMetricIds).toEqual([rsvqa.metrics[0].id]);
  });
});

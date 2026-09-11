import { describe, expect, it } from "vitest";
import { benchmarkData } from "./data";
import { validateBenchmarkRecord } from "./validation";

describe("benchmark runtime validation", () => {
  it("loads every canonical record and preferred specialist without warnings", () => {
    expect(benchmarkData.records).toHaveLength(9);
    expect(benchmarkData.preferred).toHaveLength(7);
    expect(benchmarkData.warnings).toEqual([]);
  });

  it("rejects malformed records safely", () => {
    const invalid = validateBenchmarkRecord({ schema_version: "1.0.0", benchmark_id: "broken" });
    expect(invalid.valid).toBe(false);
    if (!invalid.valid) expect(invalid.errors).toContain("model is required");
  });

  it("rejects private artifact paths", () => {
    const source = structuredClone(benchmarkData.records[0]);
    source.provenance.source_artifacts = ["/Users/private/result.json"];
    const result = validateBenchmarkRecord(source);
    expect(result.valid).toBe(false);
    if (!result.valid) expect(result.errors).toContain("provenance exposes a private path");
  });

  it("keeps missing metrics missing and numeric zero numeric", () => {
    const unavailable = benchmarkData.records.find(record => record.specialist_id === "captioning")!;
    expect(unavailable.metrics).toEqual([]);
    const record = structuredClone(benchmarkData.records[0]);
    record.metrics[0].value = 0;
    const validated = validateBenchmarkRecord(record);
    expect(validated.valid).toBe(true);
    if (validated.valid) expect(validated.value.metrics[0].value).toBe(0);
  });

  it("selects the official RSVQA test result while preserving validation history", () => {
    const preferred = benchmarkData.preferred.find(record => record.specialist_id === "rsvqa")!;
    expect(preferred.evaluation.status).toBe("verified_test");
    expect(benchmarkData.historyBySpecialist.rsvqa.map(record => record.evaluation.status)).toEqual(["verified_test", "verified_validation"]);
  });
});

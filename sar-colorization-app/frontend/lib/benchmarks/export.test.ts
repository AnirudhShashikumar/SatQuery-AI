import { describe, expect, it } from "vitest";
import { benchmarkData } from "./data";
import { benchmarkPdfLines, benchmarkRecordsToCsv, buildBenchmarkJsonExport } from "./export";

describe("benchmark exports", () => {
  it("exports JSON with provenance, timestamps, and selected records", () => {
    const result = buildBenchmarkJsonExport(benchmarkData.suite, benchmarkData.preferred, "2026-09-01T00:00:00Z");
    expect(result.exported_at).toBe("2026-09-01T00:00:00Z");
    expect(result.records[0].provenance.source_artifacts.length).toBeGreaterThan(0);
    expect(result.suite.source_commit).toHaveLength(40);
  });

  it("exports normalized CSV with one row per metric and unavailable specialists", () => {
    const csv = benchmarkRecordsToCsv(benchmarkData.preferred);
    expect(csv).toContain('"specialist_id","model_name"');
    expect(csv).toContain('"exact_match_accuracy"');
    expect(csv).toContain('"captioning"');
    expect(csv).not.toContain("/Users/");
  });

  it("builds PDF content with status and scientific footnote", () => {
    const lines = benchmarkPdfLines(benchmarkData.suite, benchmarkData.preferred, "2026-09-01T00:00:00Z");
    expect(lines.join("\n")).toContain("Verified test");
    expect(lines.at(-1)).toContain("Smoke and operational records do not establish model accuracy");
  });
});

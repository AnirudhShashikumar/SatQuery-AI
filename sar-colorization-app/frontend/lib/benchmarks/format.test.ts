import { describe, expect, it } from "vitest";
import { formatMetricValue, statusMeta } from "./format";

describe("benchmark metric formatting", () => {
  it("distinguishes ratios, percentages, zero, and unavailable values", () => {
    expect(formatMetricValue(.710515, "ratio")).toBe("71.05%");
    expect(formatMetricValue(71.0515, "percentage")).toBe("71.05%");
    expect(formatMetricValue(0, "ratio")).toBe("0.00%");
    expect(formatMetricValue(null, "ratio")).toBe("Unavailable");
  });

  it("labels validation, smoke, and external evidence without conflating them", () => {
    expect(statusMeta.verified_validation.label).toBe("Verified validation");
    expect(statusMeta.smoke_only.description).toContain("not benchmark accuracy");
    expect(statusMeta.external_reported.description).toContain("not reproduced");
  });
});

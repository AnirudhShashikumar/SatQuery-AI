import { describe, expect, it } from "vitest";
import { analyticsLabel, formatAnalyticsDuration, formatUptime, isPositiveStatus, runtimeBarPercent } from "./analytics";

describe("research analytics presentation helpers", () => {
  it("formats task identifiers", () => expect(analyticsLabel("cross_modal_analysis")).toBe("Cross Modal Analysis"));
  it("keeps unavailable runtime explicit", () => expect(formatAnalyticsDuration(null)).toBe("Not measured"));
  it("formats measured runtimes without inventing precision", () => {
    expect(formatAnalyticsDuration(42)).toBe("42 ms");
    expect(formatAnalyticsDuration(2500)).toBe("2.50 s");
  });
  it("formats process uptime", () => expect(formatUptime(90061)).toBe("1d 1h"));
  it("bounds runtime bars and keeps a visible measured minimum", () => {
    expect(runtimeBarPercent(null, 100)).toBe(0);
    expect(runtimeBarPercent(1, 100)).toBe(3);
    expect(runtimeBarPercent(200, 100)).toBe(100);
  });
  it("separates positive states from unsupported states", () => {
    expect(isPositiveStatus("Available")).toBe(true);
    expect(isPositiveStatus("Not Implemented")).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import { swapTemporalDates } from "./temporal-inputs";

describe("swapTemporalDates", () => {
  it("swaps the earlier and later date in one state value", () => {
    expect(swapTemporalDates({ primary: "2025-03-15", secondary: "2024-02-10" })).toEqual({
      primary: "2024-02-10",
      secondary: "2025-03-15",
    });
  });
});

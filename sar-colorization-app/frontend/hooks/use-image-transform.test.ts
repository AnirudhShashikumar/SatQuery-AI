import { describe, expect, it } from "vitest";
import { MAX_SCALE, MIN_SCALE, ZOOM_STEP, nextScale } from "./use-image-transform";

describe("image transform scale", () => {
  it("advances one toolbar click by one zoom step", () => expect(nextScale(1, ZOOM_STEP)).toBe(1.25));
  it("preserves precision and clamps to the viewer range", () => {
    expect(nextScale(1.5, ZOOM_STEP)).toBe(1.75);
    expect(nextScale(MAX_SCALE, ZOOM_STEP)).toBe(MAX_SCALE);
    expect(nextScale(MIN_SCALE, -ZOOM_STEP)).toBe(MIN_SCALE);
  });
});

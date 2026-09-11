import { describe, expect, it } from "vitest";
import { isModelSlug, modelCatalog } from "./model-catalog";

describe("model catalog", () => {
  it("provides one consistent, complete contract for every model navigation page", () => {
    expect(Object.keys(modelCatalog)).toEqual(["vision-encoder", "pix2pix", "sarfusionformer", "change-detection", "color-corrector"]);
    for (const model of Object.values(modelCatalog)) {
      expect(model.name).toBeTruthy();
      expect(model.role).toBeTruthy();
      expect(model.checkpoint).toBeTruthy();
      expect(model.inputs.length).toBeGreaterThan(0);
      expect(model.limitations.length).toBeGreaterThan(0);
      expect(model.actionHref).toMatch(/^\//);
    }
  });

  it("surfaces both learned and deterministic change-analysis components without fabricated metrics", () => {
    const change = modelCatalog["change-detection"];
    expect(change.supportingStack).toContain("TTP learned change detector");
    expect(change.supportingStack).toContain("Deterministic normalized-difference analyzer");
    expect(JSON.stringify(change)).not.toMatch(/accuracy|precision|recall|f1/i);
  });

  it("rejects unknown dynamic model routes", () => {
    expect(isModelSlug("pix2pix")).toBe(true);
    expect(isModelSlug("presentation")).toBe(false);
  });
});

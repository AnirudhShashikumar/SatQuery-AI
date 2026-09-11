import { beforeEach, describe, expect, it, vi } from "vitest";
import { geoVisionResults, sourceFingerprint } from "./geovision-result-store";

const asset = (url: string) => ({ url, origin: "current-session" as const, mimeType: "image/png" });

beforeEach(() => {
  sessionStorage.clear();
  geoVisionResults.clearAllResults();
});

describe("GeoVision shared result store", () => {
  it("keeps a Pix2Pix result when no later successful result is stored", () => {
    geoVisionResults.setPix2PixResult({ output: asset("pix-one"), checkpointName: "pix2pix_gen_180.pth" });
    expect(geoVisionResults.getState().pix2pix.output?.url).toBe("pix-one");
    expect(geoVisionResults.getState().pix2pix.checkpointName).toBe("pix2pix_gen_180.pth");
  });

  it("keeps results independently and clears only the requested model", () => {
    geoVisionResults.setPix2PixResult({ output: asset("pix") });
    geoVisionResults.setSarFusionFormerResult({ rawOutput: asset("sar") });
    geoVisionResults.clearPix2PixResult();
    expect(geoVisionResults.getState().pix2pix.output).toBeUndefined();
    expect(geoVisionResults.getState().sarfusionformer.rawOutput?.url).toBe("sar");
  });

  it("persists a versioned session snapshot across a module reload", async () => {
    geoVisionResults.setSarFusionFormerResult({ rawOutput: asset("sar-refresh") });
    expect(JSON.parse(sessionStorage.getItem("geovision-result-store-v1") ?? "{}").version).toBe(1);
    vi.resetModules();
    const fresh = await import("./geovision-result-store");
    fresh.geoVisionResults.hydrate();
    expect(fresh.geoVisionResults.getState().sarfusionformer.rawOutput?.url).toBe("sar-refresh");
  });

  it("tracks the SAR comparison mode without replacing the result", () => {
    geoVisionResults.setSarFusionFormerResult({ rawOutput: asset("raw"), enhancedOutput: asset("enhanced") });
    geoVisionResults.setSarComparisonMode("enhanced");
    expect(geoVisionResults.getState().comparison.sarOutputMode).toBe("enhanced");
    expect(geoVisionResults.getState().sarfusionformer.rawOutput?.url).toBe("raw");
  });

  it("creates stable source fingerprints for source-match checks", async () => {
    const first = new File(["same"], "scene.png", { type: "image/png", lastModified: 1 });
    const second = new File(["same"], "scene.png", { type: "image/png", lastModified: 1 });
    expect(await sourceFingerprint(first)).toBe(await sourceFingerprint(second));
  });
});

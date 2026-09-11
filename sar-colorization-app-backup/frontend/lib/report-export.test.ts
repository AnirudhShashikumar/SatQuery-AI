import { describe, expect, it } from "vitest";
import { createMetadataJson, createMetricsCsv, csvEscape, safeTimestamp, type ExportReport } from "./report-export";

const readBlob = (blob: Blob) => new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onerror = () => reject(reader.error); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });

const report: ExportReport = {
  id: "GV-TEST", generatedAt: "2026-08-15T00:00:00.000Z", source: "pix2pix", version: "1.0",
  scientificDisclaimer: "Reconstruction is not exact ground truth.",
  models: [{ name: "Pix2Pix", checkpoint: "pix2pix_gen_180.pth", inputFile: "scene,one.png", outputMode: "optical", metrics: { psnr: 24.8, ssim: .84, rgbL1: .071, inferenceTimeMs: 318 }, images: [{ label: "Prediction", fileName: "pix2pix_prediction.png", asset: { url: "data:image/png;base64,abc", origin: "current-session", mimeType: "image/png", width: 256, height: 256 } }] }],
};

describe("report exports", () => {
  it("escapes CSV values and creates one model row", async () => {
    expect(csvEscape('scene, "one"')).toBe('"scene, ""one"""');
    const csv = await readBlob(createMetricsCsv(report));
    expect(csv).toContain("report_id");
    expect(csv).toContain('"scene,one.png"');
    expect(csv.split("\n")).toHaveLength(2);
  });

  it("exports metadata without image source URLs or secrets", async () => {
    const json = await readBlob(createMetadataJson(report));
    expect(json).toContain("pix2pix_gen_180.pth");
    expect(json).not.toContain("data:image");
    expect(json).not.toMatch(/api[_-]?key/i);
  });

  it("creates filesystem-safe timestamps", () => {
    expect(safeTimestamp(new Date("2026-08-15T01:02:03.444Z"))).toBe("2026-08-15_01-02-03");
  });
});

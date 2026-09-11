import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationCrossModalResult, PresentationCrossModalScene } from "./presentation-cross-modal-scene";
import { approvedCrossModalPair, createApprovedRasterPreview, crossModalDefaultQuery } from "@/lib/presentation-cross-modal";
import { presentationSteps, presentationStorageKeys } from "@/lib/presentation";
import { ApiRequestError, getDemoManifest, getHealth, loadDemoFile, runAgentImageQuery } from "@/services/api";
import type { AgentResponse, CrossModalResult, DemoManifest, ImageMetadata, PairCompatibility } from "@/types/agent";

vi.mock("@/services/api", async importOriginal => {
  const original = await importOriginal<typeof import("@/services/api")>();
  return { ...original, getDemoManifest: vi.fn(), getHealth: vi.fn(), loadDemoFile: vi.fn(), runAgentImageQuery: vi.fn() };
});

vi.mock("@/lib/presentation-cross-modal", async importOriginal => {
  const original = await importOriginal<typeof import("@/lib/presentation-cross-modal")>();
  return { ...original, createApprovedRasterPreview: vi.fn() };
});

const manifest: DemoManifest = {
  enabled: true,
  local_only: true,
  workflows: [{
    id: "cross_modal", title: "Optical–SAR", description: "Approved exactly aligned pair", input_mode: "cross_modal",
    primary_modality: "optical", secondary_modality: "sar", query: "Where do both modalities agree?",
    primary_date: null, secondary_date: null,
    files: [
      { role: "primary", filename: "cross-optical.tif", url: "/optical.tif", mime_type: "image/tiff" },
      { role: "secondary", filename: "cross-sar.tif", url: "/sar.tif", mime_type: "image/tiff" },
    ],
  }],
};

const exact: PairCompatibility = {
  compatible: true, alignment_level: "exact", same_dimensions: true, same_crs: true, same_transform: true,
  bounds_overlap: true, overlap_ratio: 1, resampling_required: false, warnings: [], errors: [],
};

function metadata(name: string, bands: number, dtype: string): ImageMetadata {
  return {
    file_id: name, original_name: name, safe_name: name, format: "geotiff", mime_type: "image/tiff", size_bytes: 1024,
    width: 128, height: 128, band_count: bands, dtype, crs: "EPSG:4326", transform: [0.01, 0, 70, 0, -0.01, 20],
    bounds: { left: 70, bottom: 18.72, right: 71.28, top: 20 }, nodata: null, is_georeferenced: true,
    preview_url: `/${name}.png`, color_interpretation: [], warnings: [],
  };
}

function crossResult(status: CrossModalResult["status"] = "success", withStatistics = true): CrossModalResult {
  return {
    status,
    summary: withStatistics ? {
      optical_observations: ["Optical evidence used RGB bands."],
      sar_observations: ["SAR evidence used two relative-intensity bands."],
      joint_observations: ["Joint water-like support covers 8.250% of valid pixels.", "Joint structural-likelihood support covers 14.500% of valid pixels.", "Agreement and disagreement were measured from candidate pixels."],
    } : { optical_observations: [], sar_observations: [], joint_observations: [status === "alignment_required" ? "Explicit alignment is required before fusion." : status === "partial" ? "Only side-by-side visual inspection is available." : "The pair is incompatible and was not analyzed."] },
    statistics: withStatistics ? {
      analysis_width: 128, analysis_height: 128, source_width: 128, source_height: 128,
      water_likelihood_percent: 8.25, built_up_likelihood_percent: 14.5, vegetation_support_percent: 31.75,
      agreement_percent: 63.2, disagreement_percent: 36.8, valid_pixel_percent: 99.5, evaluated_candidate_pixels: 1200,
    } : null,
    regions: withStatistics ? [{ region_id: "water-1", type: "water_likelihood", area_pixels: 240, area_percent: 1.465, bbox_pixels: [2, 4, 20, 18], centroid_pixels: [11, 11], bbox_world: null, centroid_world: null, support: { optical: true, sar: true } }] : [],
    previews: withStatistics ? {
      optical: "/optical.png", sar: "/sar.png", optical_evidence: "/optical-evidence.png", sar_evidence: "/sar-evidence.png",
      joint_evidence: "/joint.png", water_likelihood: "/water.png", built_up_likelihood: "/structural.png",
      vegetation_support: "/vegetation.png", agreement: "/agreement.png", disagreement: "/disagreement.png", joint_overlay: "/overlay.png",
    } : { optical: "/optical.png", sar: "/sar.png", optical_evidence: null, sar_evidence: null, joint_evidence: null, water_likelihood: null, built_up_likelihood: null, vegetation_support: null, agreement: null, disagreement: null, joint_overlay: null },
    confidence: { level: withStatistics ? "moderate" : "unavailable", score: null, reason: withStatistics ? "Exactly aligned deterministic evidence." : "Pixel-level fusion unavailable." },
    method: { name: "Deterministic optical-SAR evidence fusion", version: "1", uses_trained_model: false, assumptions: ["Exact alignment."], limitations: ["Likelihood maps are not semantic ground truth."] },
    optical_preparation: null, sar_preparation: null, warnings: [], runtime_ms: 18,
  };
}

function response(options: { cached?: boolean; compatibility?: PairCompatibility; status?: CrossModalResult["status"]; withStatistics?: boolean } = {}): AgentResponse {
  const compatibility = options.compatibility ?? exact;
  const result = crossResult(options.status ?? "success", options.withStatistics ?? true);
  const responseStatus = result.status === "alignment_required" ? "alignment_required" : result.status === "failed" ? "failed" : result.status === "partial" ? "partial" : "success";
  return {
    request_id: "request-cross", task: "cross_modal_analysis", answer: null,
    confidence: result.confidence, evidence: [], status: responseStatus, warnings: [],
    primary_image_metadata: metadata("cross-optical.tif", 3, "uint8"), secondary_image_metadata: metadata("cross-sar.tif", 2, "float32"),
    pair_compatibility: compatibility, model: null, caption_details: null, grounding_result: null, cross_modal_analysis: result, change_analysis: null, vqa_details: null,
    execution: { input_mode: "cross_modal", selected_tools: ["input_validator", "cross_modal_optical_sar_analyzer"], steps: [
      { tool: "pair_compatibility_validation", status: "success", duration_ms: 2, parameters: {} },
      { tool: "optical_evidence_extraction", status: "success", duration_ms: 3, parameters: {} },
      { tool: "sar_evidence_extraction", status: "success", duration_ms: 3, parameters: {} },
      { tool: "joint_evidence_fusion", status: "success", duration_ms: 4, parameters: {} },
    ], duration_ms: 22, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Joint optical-SAR analysis." },
    cache: { cached: options.cached ?? false, original_generation_timestamp: "2026-08-24T00:00:00Z", retrieval_timestamp: "2026-08-24T00:00:00Z", tool_version: "1", cache_key_prefix: "cross" },
  };
}

let container: HTMLDivElement;
let root: Root;

async function flush() { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); }
function button(name: string) { return Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name)) as HTMLButtonElement | undefined; }
async function click(element: Element | undefined) { expect(element).toBeTruthy(); await act(async () => { element!.dispatchEvent(new MouseEvent("click", { bubbles: true })); await Promise.resolve(); }); }
async function loadPair() { await click(button("Load Approved Pair")); await flush(); }
async function renderScene() { await act(async () => { root.render(createElement(PresentationCrossModalScene, { step: presentationSteps[8] })); }); await flush(); }

beforeEach(async () => {
  sessionStorage.clear();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.mocked(getHealth).mockResolvedValue({ status: "ok" } as never);
  vi.mocked(getDemoManifest).mockResolvedValue(manifest);
  vi.mocked(loadDemoFile).mockImplementation(async (_path, filename, mime) => new File([filename], filename, { type: mime }));
  vi.mocked(createApprovedRasterPreview).mockImplementation(async (_file, modality) => ({
    url: `data:image/png;base64,${modality}`,
    metadata: { width: 128, height: 128, bandCount: modality === "sar" ? 2 : 3, dtype: modality === "sar" ? "float32" : "uint8", crs: "EPSG:4326", transformKey: "same", bounds: [70, 18.72, 71.28, 20] },
  }));
  vi.mocked(runAgentImageQuery).mockResolvedValue(response());
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await renderScene();
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

describe("PresentationCrossModalScene", () => {
  it("renders before load, isolates its query state, and does not analyze automatically", () => {
    expect(container.textContent).toContain("Optical and SAR samples are not loaded");
    expect((container.querySelector("#presentation-cross-modal-query") as HTMLInputElement).value).toBe(crossModalDefaultQuery);
    expect(button("Run Joint Analysis")?.disabled).toBe(true);
    expect(runAgentImageQuery).not.toHaveBeenCalled();
    expect(sessionStorage.getItem(presentationStorageKeys.singleImageQuery)).toBeNull();
    expect(sessionStorage.getItem(presentationStorageKeys.changeQuery)).toBeNull();
  });

  it("loads both labeled source images, metadata, and preflight compatibility without analysis", async () => {
    await loadPair();
    expect(container.querySelector('img[alt="Optical approved cross-modal observation"]')).toBeTruthy();
    expect(container.querySelector('img[alt="SAR approved cross-modal observation"]')).toBeTruthy();
    expect(container.textContent).toContain("Optical · Optical");
    expect(container.textContent).toContain("SAR · Synthetic aperture radar");
    expect(container.textContent).toContain("128 × 128 · 3 bands · uint8 · EPSG:4326");
    expect(container.textContent).toContain("128 × 128 · 2 bands · float32 · EPSG:4326");
    expect(container.textContent).toContain("Preflight · exact candidate");
    expect(loadDemoFile).toHaveBeenCalledTimes(2);
    expect(runAgentImageQuery).not.toHaveBeenCalled();
  });

  it("starts only on click and blocks duplicate requests", async () => {
    await loadPair();
    let resolveRequest!: (value: AgentResponse) => void;
    vi.mocked(runAgentImageQuery).mockReturnValue(new Promise(resolve => { resolveRequest = resolve; }));
    const run = button("Run Joint Analysis")!;
    await act(async () => {
      run.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      run.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    expect(runAgentImageQuery).toHaveBeenCalledTimes(1);
    expect(button("Analyzing")?.disabled).toBe(true);
    expect(vi.mocked(runAgentImageQuery).mock.calls[0][0]).toMatchObject({ inputMode: "cross_modal", query: crossModalDefaultQuery, useCache: true, forceRerun: false });
    await act(async () => { resolveRequest(response()); await Promise.resolve(); });
  });

  it("renders the exact real-result contract, structured answer, statistics, regions, confidence, trace, and at least six evidence products", async () => {
    await loadPair();
    await click(button("Run Joint Analysis"));
    await flush();
    expect(container.textContent).toContain("Joint water-like support covers 8.250%");
    expect(container.textContent).toContain("Joint structural-likelihood support covers 14.500%");
    expect(container.textContent).toContain("Water likelihood8.250%");
    expect(container.textContent).toContain("Structural likelihood14.500%");
    expect(container.textContent).toContain("Agreement63.200%");
    expect(container.textContent).toContain("Disagreement36.800%");
    expect(container.textContent).toContain("1 region");
    expect(container.textContent).toContain("Moderate · no calibrated score");
    expect(container.textContent).toContain("joint_evidence_fusion");
    expect(container.textContent).toContain("Fresh Analysis");
    expect(container.textContent).toContain("Compatible · exact alignment");
    expect(container.querySelectorAll('img[alt$="from real optical–SAR analysis"]')).toHaveLength(9);
  });

  it("shows cached/fresh state, bypasses cache on rerun, and resets the result and query", async () => {
    vi.mocked(runAgentImageQuery).mockResolvedValueOnce(response({ cached: true })).mockResolvedValueOnce(response());
    await loadPair();
    await click(button("Run Joint Analysis"));
    await flush();
    expect(container.textContent).toContain("Cached Result");
    await click(button("Re-run Analysis"));
    await flush();
    expect(vi.mocked(runAgentImageQuery).mock.calls[1][0].forceRerun).toBe(true);
    expect(container.textContent).toContain("Fresh Analysis");
    const query = container.querySelector("#presentation-cross-modal-query") as HTMLInputElement;
    await act(async () => { query.value = "edited"; query.dispatchEvent(new Event("input", { bubbles: true })); });
    await click(button("Reset Scene"));
    expect(container.textContent).not.toContain("Fresh Analysis");
    expect((container.querySelector("#presentation-cross-modal-query") as HTMLInputElement).value).toBe(crossModalDefaultQuery);
    expect(container.querySelector('img[alt="Optical approved cross-modal observation"]')).toBeTruthy();
  });

  const guardedCases: Array<[CrossModalResult["status"], PairCompatibility, string]> = [
    ["alignment_required", { ...exact, alignment_level: "geospatial_overlap", same_transform: false, resampling_required: true }, "Alignment is required"],
    ["partial", { ...exact, alignment_level: "visual_only", same_crs: null, same_transform: null }, "Visual-only comparison"],
    ["failed", { ...exact, compatible: false, alignment_level: "incompatible", same_dimensions: false, errors: ["Dimensions differ"] }, "pair is incompatible"],
  ];

  it.each(guardedCases)("renders the %s compatibility outcome without fabricating statistics", async (status, compatibility, disclosure) => {
    await act(async () => { root.render(createElement(PresentationCrossModalResult, { response: response({ status, compatibility, withStatistics: false }) })); });
    expect(container.textContent).toContain(disclosure);
    expect(container.textContent).toContain("Unavailable");
    expect(container.textContent).not.toContain("8.250%");
  });

  it("shows demo, sample, preview, request, backend, and expired-preview failure states", async () => {
    vi.mocked(getDemoManifest).mockResolvedValueOnce({ ...manifest, enabled: false, workflows: [] });
    await loadPair();
    expect(container.textContent).toContain("Local demo mode is disabled");

    vi.mocked(getDemoManifest).mockResolvedValue(manifest);
    vi.mocked(createApprovedRasterPreview).mockRejectedValueOnce(new Error("preview failed"));
    await loadPair();
    expect(container.textContent).toContain("preview failed");
    expect(container.textContent).toContain("Source preview unavailable");

    vi.mocked(runAgentImageQuery).mockRejectedValueOnce(new ApiRequestError("request failure", "REQUEST_FAILED", 400));
    await click(button("Run Joint Analysis"));
    await flush();
    expect(container.textContent).toContain("request failure");

    vi.mocked(runAgentImageQuery).mockResolvedValue(response());
    await click(button("Run Joint Analysis"));
    await flush();
    const preview = container.querySelector('img[alt^="Optical evidence"]')!;
    await act(async () => { preview.dispatchEvent(new Event("error", { bubbles: true })); });
    expect(container.textContent).toContain("Preview unavailable or expired");

    vi.mocked(getHealth).mockRejectedValue(new TypeError("Failed to fetch"));
    await act(async () => { root.unmount(); root = createRoot(container); });
    await renderScene();
    expect(container.querySelector('[aria-label="Backend offline"]')).toBeTruthy();
  });

  it("rejects missing optical and SAR manifest entries with distinct messages", () => {
    const withoutOptical = { ...manifest, workflows: [{ ...manifest.workflows[0], files: manifest.workflows[0].files.filter(file => file.role !== "primary") }] };
    const withoutSar = { ...manifest, workflows: [{ ...manifest.workflows[0], files: manifest.workflows[0].files.filter(file => file.role !== "secondary") }] };
    expect(() => approvedCrossModalPair(withoutOptical, "cross_modal", "cross-optical.tif", "cross-sar.tif")).toThrow("optical sample is missing");
    expect(() => approvedCrossModalPair(withoutSar, "cross_modal", "cross-optical.tif", "cross-sar.tif")).toThrow("SAR sample is missing");
  });

  it("aborts and drops a stale completion after leaving the stage", async () => {
    await loadPair();
    let resolveRequest!: (value: AgentResponse) => void;
    vi.mocked(runAgentImageQuery).mockReturnValue(new Promise(resolve => { resolveRequest = resolve; }));
    await click(button("Run Joint Analysis"));
    await act(async () => root.unmount());
    resolveRequest(response());
    await flush();
    expect(sessionStorage.getItem("satquery-latest-request-id")).toBeNull();
    root = createRoot(container);
  });
});

describe("Optical–SAR presentation layout contract", () => {
  const css = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");

  it("keeps source, answer, and statistics in a bounded 1024px projector layout", () => {
    expect(css).toContain("@media (min-width: 1024px)");
    expect(css).toContain(".presentation-cross-modal-scene { display: grid; height: min(55dvh, 505px)");
    expect(css).toContain(".presentation-cross-modal-result { flex: 1 1 auto; }");
  });

  it("stacks source images and releases result scrolling on mobile", () => {
    expect(css).toContain(".presentation-cross-modal-source > div.grid { grid-template-columns: minmax(0, 1fr); }");
    expect(css).toContain(".presentation-cross-modal-result { max-height: none; overflow-y: visible; }");
  });

  it("provides explicit dark and light theme surfaces", () => {
    expect(css).toContain(".presentation-cross-modal-workspace { border-color: rgba(255,255,255,.08)");
    expect(css).toContain("html.light .presentation-cross-modal-workspace { border-color: #dbe3ed");
    expect(css).toContain("html.light .presentation-cross-modal-result h2 { color: #111827; }");
  });
});

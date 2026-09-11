import { afterEach, describe, expect, it, vi } from "vitest";
import { runAgentImageQuery } from "@/services/api";
import type { AgentResponse, DemoManifest, PairCompatibility } from "@/types/agent";
import {
  approvedChangePair,
  changeDefaultDates,
  changeDefaultQuery,
  changeEvidencePreviews,
  changeResponseDisclosure,
  compatibilityLabel,
  requestFailureMessage,
} from "./presentation-change";

afterEach(() => vi.unstubAllGlobals());

const compatibility: PairCompatibility = {
  compatible: true,
  alignment_level: "visual_only",
  same_dimensions: true,
  same_crs: null,
  same_transform: null,
  bounds_overlap: null,
  overlap_ratio: null,
  resampling_required: false,
  warnings: ["Visual-only pair."],
  errors: [],
};

const manifest: DemoManifest = {
  enabled: true,
  local_only: true,
  workflows: [{
    id: "change_vqa",
    title: "Bi-temporal change analysis",
    description: "Approved pair",
    input_mode: "bi_temporal",
    primary_modality: "optical",
    secondary_modality: "optical",
    query: "How much changed?",
    primary_date: "2025-01-01",
    secondary_date: "2025-02-01",
    files: [
      { role: "primary", filename: "change-before.png", url: "/before.png", mime_type: "image/png" },
      { role: "secondary", filename: "change-after.png", url: "/after.png", mime_type: "image/png" },
    ],
  }],
};

function response(overrides: Partial<AgentResponse> = {}): AgentResponse {
  return {
    request_id: "change-request-1",
    task: "change_description",
    answer: "Approximately 12.5% of the analysis grid changed across 2 connected regions. The largest region contains 80 pixels. The cause and land-cover type of change are not inferred.",
    confidence: { level: "moderate", score: null, reason: "The summary is templated directly from deterministic change statistics." },
    evidence: [],
    execution: { input_mode: "bi_temporal", selected_tools: ["input_validator", "bitemporal_change_analyzer"], steps: [{ tool: "difference_computation", status: "success", duration_ms: 3, parameters: {} }], duration_ms: 18, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Measured change." },
    warnings: [],
    status: "success",
    primary_image_metadata: null,
    secondary_image_metadata: null,
    pair_compatibility: compatibility,
    model: null,
    caption_details: null,
    grounding_result: null,
    cross_modal_analysis: null,
    change_analysis: {
      request_id: "inner-change-1",
      status: "success",
      before_date: "2025-01-01",
      after_date: "2025-02-01",
      before_metadata: null as never,
      after_metadata: null as never,
      compatibility,
      statistics: {
        analysis_width: 100, analysis_height: 100, source_width: 100, source_height: 100,
        total_pixels: 10000, changed_pixels: 1250, percentage_changed: 12.5,
        largest_connected_region: 80, number_of_regions: 2,
        bounding_boxes: [{ left: 1, top: 2, right: 11, bottom: 10, area_pixels: 80 }],
        regions: [{ region_id: 1, area_pixels: 80, percentage_of_image: 0.8, bounding_box: { left: 1, top: 2, right: 11, bottom: 10, area_pixels: 80 } }],
        normalized_threshold: 0.2,
      },
      previews: { before: "/before-preview.png", after: "/after-preview.png", difference: "/difference.png", mask: "/mask.png", overlay: "/overlay.png" },
      execution: { input_mode: "bi_temporal", selected_tools: ["deterministic_change_analysis"], steps: [], duration_ms: 15, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Change engine." },
      runtime_ms: 15,
      warnings: [],
    },
    vqa_details: null,
    cache: { cached: false, original_generation_timestamp: "2026-08-24T00:00:00Z", retrieval_timestamp: "2026-08-24T00:00:00Z", tool_version: "1", cache_key_prefix: "change" },
    ...overrides,
  };
}

describe("Bi-Temporal presentation configuration", () => {
  it("uses the controlled default question and approved dates", () => {
    expect(changeDefaultQuery).toBe("What changed between these dates?");
    expect(changeDefaultDates).toEqual({ before: "2025-01-01", after: "2025-02-01" });
  });

  it("selects the approved before/after pair by exact IDs", () => {
    const pair = approvedChangePair(manifest, "change_vqa", "change-before.png", "change-after.png");
    expect(pair.before.filename).toBe("change-before.png");
    expect(pair.after.filename).toBe("change-after.png");
    expect([pair.beforeDate, pair.afterDate]).toEqual(["2025-01-01", "2025-02-01"]);
  });

  it("rejects disabled demo mode, missing workflows, and missing files", () => {
    expect(() => approvedChangePair({ ...manifest, enabled: false }, "change_vqa", "change-before.png", "change-after.png")).toThrow("demo mode is disabled");
    expect(() => approvedChangePair(manifest, "missing", "change-before.png", "change-after.png")).toThrow("unavailable");
    expect(() => approvedChangePair(manifest, "change_vqa", "missing.png", "change-after.png")).toThrow("files are missing");
  });
});

describe("real change-result presentation mapping", () => {
  it("maps only actual difference, mask, and overlay references", () => {
    expect(changeEvidencePreviews(response())).toEqual([
      { label: "Difference Heatmap", path: "/difference.png" },
      { label: "Binary mask", path: "/mask.png" },
      { label: "Overlay", path: "/overlay.png" },
    ]);
  });

  it("does not invent missing preview products", () => {
    const next = response();
    next.change_analysis!.previews.mask = null;
    expect(changeEvidencePreviews(next).map(item => item.label)).toEqual(["Difference Heatmap", "Overlay"]);
  });

  it("maps source-labelled hybrid evidence without presenting either mask as ground truth", () => {
    const next = response({ change_engine: { mode: "hybrid", primary_tool: "ttp_change_detector", supporting_tool: "deterministic_change_analyzer", fallback_used: false, fallback_reason: null } });
    next.change_analysis!.change_engine = next.change_engine;
    Object.assign(next.change_analysis!.previews, {
      ttp_mask: "/ttp-mask.png", deterministic_mask: "/deterministic-mask.png",
      agreement: "/agreement.png", disagreement: "/disagreement.png", ttp_overlay: "/ttp-overlay.png",
    });
    expect(changeEvidencePreviews(next).map(item => item.label)).toEqual([
      "Difference Heatmap", "Learned Change Mask", "Deterministic Difference", "Agreement", "Disagreement", "TTP Overlay",
    ]);
  });

  it("discloses visual-only compatibility", () => {
    expect(compatibilityLabel(compatibility)).toBe("Compatible · visual only");
    expect(changeResponseDisclosure(response())).toContain("geospatial correspondence is not independently verified");
  });

  it("reports alignment-required without suggesting silent registration", () => {
    const next = response({ status: "alignment_required" });
    next.change_analysis!.status = "alignment_required";
    expect(changeResponseDisclosure(next)).toContain("did not resize, register, reproject, or resample");
  });

  it("reports incompatible pairs without fabricated statistics", () => {
    const next = response({ status: "failed" });
    next.change_analysis!.compatibility = { ...compatibility, compatible: false, alignment_level: "incompatible", same_dimensions: false };
    next.change_analysis!.statistics = null;
    expect(changeResponseDisclosure(next)).toContain("No change statistics were fabricated");
  });

  it("maps offline and expired-result failures clearly", () => {
    expect(requestFailureMessage(new TypeError("Failed to fetch"))).toContain("backend is offline");
    expect(requestFailureMessage(new Error("Result not found (404)"))).toContain("expired");
  });
});

describe("existing cached multipart change-query contract", () => {
  it("sends the pair, dates, controlled question, and cache flag only when called", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response()), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    await runAgentImageQuery({
      query: changeDefaultQuery,
      inputMode: "bi_temporal",
      primaryModality: "optical",
      secondaryModality: "optical",
      primaryImage: new File(["before"], "change-before.png", { type: "image/png" }),
      secondaryImage: new File(["after"], "change-after.png", { type: "image/png" }),
      primaryDate: changeDefaultDates.before,
      secondaryDate: changeDefaultDates.after,
      useCache: true,
      forceRerun: false,
    });
    const form = fetchMock.mock.calls[0][1].body as FormData;
    expect(form.get("query")).toBe(changeDefaultQuery);
    expect(form.get("input_mode")).toBe("bi_temporal");
    expect((form.get("primary_image") as File).name).toBe("change-before.png");
    expect((form.get("secondary_image") as File).name).toBe("change-after.png");
    expect(form.get("primary_date")).toBe("2025-01-01");
    expect(form.get("secondary_date")).toBe("2025-02-01");
    expect(form.get("use_cache")).toBe("true");
    expect(form.get("force_rerun")).toBe("false");
  });

  it("uses the existing force-rerun flag to bypass cache", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response()), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    await runAgentImageQuery({ query: changeDefaultQuery, inputMode: "bi_temporal", primaryModality: "optical", secondaryModality: "optical", primaryImage: new File(["a"], "a.png"), secondaryImage: new File(["b"], "b.png"), primaryDate: "2025-01-01", secondaryDate: "2025-02-01", useCache: true, forceRerun: true });
    expect((fetchMock.mock.calls[0][1].body as FormData).get("force_rerun")).toBe("true");
  });
});

export { manifest as changePresentationManifest, response as changePresentationResponse };

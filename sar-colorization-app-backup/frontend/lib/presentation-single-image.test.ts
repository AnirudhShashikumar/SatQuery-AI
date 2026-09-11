import { afterEach, describe, expect, it, vi } from "vitest";
import { runAgentImageQuery } from "@/services/api";
import type { AgentResponse, DemoManifest } from "@/types/agent";
import {
  approvedSingleImageWorkflow,
  responseFailureMessage,
  singleImageDefaultQueries,
  vqaEvidencePreviews,
} from "./presentation-single-image";

afterEach(() => vi.unstubAllGlobals());

const manifest: DemoManifest = {
  enabled: true,
  local_only: true,
  workflows: [{
    id: "single_vqa",
    title: "Single-image understanding",
    description: "Approved sample",
    input_mode: "single",
    primary_modality: "optical",
    secondary_modality: null,
    query: "Is water visible?",
    primary_date: null,
    secondary_date: null,
    files: [{ role: "primary", filename: "single-optical.png", url: "/api/agent/demo/files/single-optical.png", mime_type: "image/png" }],
  }],
};

function response(overrides: Partial<AgentResponse> = {}): AgentResponse {
  return {
    request_id: "request-1",
    task: "vqa",
    answer: "Vegetation-dominant mixed scene.",
    confidence: { level: "moderate", score: null, reason: "Computed evidence supports this controlled answer." },
    evidence: [],
    execution: { input_mode: "single", selected_tools: ["input_validator", "rs_vqa"], steps: [], duration_ms: 12, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Controlled question." },
    warnings: [],
    status: "success",
    primary_image_metadata: null,
    secondary_image_metadata: null,
    pair_compatibility: null,
    model: null,
    caption_details: null,
    grounding_result: null,
    cross_modal_analysis: null,
    change_analysis: null,
    vqa_details: null,
    cache: null,
    ...overrides,
  };
}

describe("Single-Image presentation configuration", () => {
  it("uses the required captioning and VQA default queries", () => {
    expect(singleImageDefaultQueries.captioning).toBe("Describe the land cover and major objects visible in this image.");
    expect(singleImageDefaultQueries.vqa).toBe("What is the dominant land-cover type?");
  });

  it("selects only the approved single-image workflow", () => {
    expect(approvedSingleImageWorkflow(manifest, "single_vqa").files[0].filename).toBe("single-optical.png");
    expect(() => approvedSingleImageWorkflow({ ...manifest, enabled: false, workflows: [] }, "single_vqa")).toThrow("demo mode is disabled");
    expect(() => approvedSingleImageWorkflow(manifest, "missing")).toThrow("unavailable");
  });

  it("maps only real VQA evidence preview references", () => {
    const result = response({
      vqa_details: {
        original_question: "What is the dominant land-cover type?", question_category: "dominant_land_cover", target_concept: null,
        answer_source: "computed_evidence", statistics_used: {}, evidence_references: [], supported: true, limitations: [],
        method: { name: "Controlled VQA", version: "1", method_type: "deterministic", uses_language_model: false, remote_sensing_adapted: false, assumptions: [], limitations: [] },
        confidence: { level: "moderate", score: null, reason: "Evidence support." },
        single_image_evidence: {
          statistics: { water_support_percent: 1, vegetation_support_percent: 2, built_up_support_percent: 3, barren_support_percent: 4, agriculture_support_percent: 5, edge_density_percent: 6, valid_pixel_percent: 100 },
          dominant_scene: "vegetation", regions: [], warnings: [], low_information: false, runtime_ms: 4,
          previews: { water_support: "/water.png", vegetation_support: null, built_up_support: "/built.png", agriculture_support: null, combined_overlay: "/overlay.png" },
          method: { name: "Evidence", version: "1", uses_trained_classifier: false, assumptions: [], limitations: [] },
        },
      },
    });
    expect(vqaEvidencePreviews(result)).toEqual([
      { label: "Water support", path: "/water.png" },
      { label: "Structural support", path: "/built.png" },
      { label: "Combined evidence overlay", path: "/overlay.png" },
    ]);
  });

  it("provides a judge-safe caption checkpoint fallback", () => {
    const unavailable = response({ status: "not_implemented", answer: null, warnings: ["Captioner unavailable because checkpoint is not cached."] });
    expect(responseFailureMessage(unavailable, "captioning")).toContain("Switch to Ask a Question");
  });
});

describe("existing multipart query contract", () => {
  it("submits the approved file without analysis automation and preserves cache flags", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response()), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["image"], "single-optical.png", { type: "image/png" });
    await runAgentImageQuery({ query: singleImageDefaultQueries.captioning, inputMode: "single", primaryModality: "optical", secondaryModality: null, primaryImage: file, useCache: true, forceRerun: false });
    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const form = options.body as FormData;
    expect(options.method).toBe("POST");
    expect(form.get("query")).toBe(singleImageDefaultQueries.captioning);
    expect(form.get("input_mode")).toBe("single");
    expect(form.get("primary_modality")).toBe("optical");
    expect((form.get("primary_image") as File).name).toBe("single-optical.png");
    expect(form.get("use_cache")).toBe("true");
    expect(form.get("force_rerun")).toBe("false");
  });

  it("uses the existing force-rerun flag to bypass cache", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response()), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    await runAgentImageQuery({ query: singleImageDefaultQueries.vqa, inputMode: "single", primaryModality: "optical", secondaryModality: null, primaryImage: new File(["image"], "sample.png", { type: "image/png" }), useCache: true, forceRerun: true });
    const form = fetchMock.mock.calls[0][1].body as FormData;
    expect(form.get("force_rerun")).toBe("true");
  });
});

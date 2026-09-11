import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AssistantResultExperience, evidenceProducts } from "./assistant-result-experience";
import type { AgentResponse, SingleImageEvidenceRegion } from "@/types/agent";

const regions: SingleImageEvidenceRegion[] = Array.from({ length: 7 }, (_, index) => ({
  region_id: `water-${index + 1}`,
  type: "water_support",
  area_pixels: 700 - index * 50,
  area_percent: 7 - index * .5,
  bbox_pixels: [index, index, 20 + index, 30 + index],
  centroid_pixels: [10 + index, 15 + index],
  bbox_world: null,
  centroid_world: null,
}));

function response(): AgentResponse {
  return {
    request_id: "presentation-result-1",
    task: "vqa",
    answer: "Water body likely present.",
    confidence: { level: "moderate", score: null, reason: "Measured water-support evidence is consistent with the controlled answer." },
    evidence: [{ type: "heuristic_support", label: "Water support", description: "Water-support regions were detected.", reference: "/water.png" }],
    execution: {
      input_mode: "single",
      selected_tools: ["input_validator", "rs_vqa"],
      steps: [
        { tool: "upload", status: "success", duration_ms: 3, parameters: {} },
        { tool: "input_validator", status: "success", duration_ms: 4, parameters: {} },
        { tool: "rs_vqa", status: "success", duration_ms: 59, parameters: {} },
      ],
      duration_ms: 66,
      permitted_parameters: {},
      validation: { valid: true, errors: [] },
      selection_reason: "Presence question detected.",
    },
    warnings: ["Visible-spectrum evidence only."],
    status: "success",
    primary_image_metadata: {
      file_id: "source", original_name: "source.png", safe_name: "source.png", format: "png", mime_type: "image/png", size_bytes: 1024,
      width: 256, height: 256, band_count: 3, dtype: "uint8", crs: null, transform: null, bounds: null, nodata: null,
      is_georeferenced: false, preview_url: "/source.png", color_interpretation: ["r", "g", "b"], warnings: [],
    },
    secondary_image_metadata: null,
    pair_compatibility: null,
    model: null,
    caption_details: null,
    grounding_result: null,
    cross_modal_analysis: null,
    change_analysis: null,
    vqa_details: {
      original_question: "Is a water body visible?",
      question_category: "presence_water",
      target_concept: "water",
      answer_source: "deterministic_single_image_evidence",
      statistics_used: { water_support_percent: 14.9 },
      evidence_references: ["Water support"],
      method: { name: "Controlled VQA", version: "1", method_type: "deterministic", uses_language_model: false, remote_sensing_adapted: true, assumptions: [], limitations: ["Not NDVI."] },
      confidence: { level: "moderate", score: null, reason: "Measured support." },
      supported: true,
      limitations: ["Heuristic VQA."],
      single_image_evidence: {
        statistics: { water_support_percent: 14.9, vegetation_support_percent: 21, built_up_support_percent: 11, barren_support_percent: 4, agriculture_support_percent: 8, edge_density_percent: 6, valid_pixel_percent: 100 },
        dominant_scene: "water",
        regions,
        previews: { water_support: "/water.png", vegetation_support: "/vegetation.png", built_up_support: "/built.png", agriculture_support: "/agriculture.png", combined_overlay: "/combined.png" },
        warnings: [],
        method: { name: "Evidence", version: "1", uses_trained_classifier: false, assumptions: [], limitations: [] },
        low_information: false,
        runtime_ms: 40,
      },
    },
    cache: null,
  };
}

function changeResponse(withSemantic = true): AgentResponse {
  const base = response();
  const statistics = {
    analysis_width: 100, analysis_height: 100, source_width: 100, source_height: 100,
    total_pixels: 10_000, changed_pixels: 111, percentage_changed: 1.11,
    largest_connected_region: 80, number_of_regions: 1,
    bounding_boxes: [{ left: 70, top: 5, right: 90, bottom: 25, area_pixels: 400 }],
    regions: [{ region_id: 1, area_pixels: 80, percentage_of_image: .8, bounding_box: { left: 70, top: 5, right: 90, bottom: 25, area_pixels: 400 } }],
    normalized_threshold: .5,
  };
  const semantic = withSemantic ? {
    short_answer: "The main change is concentrated in the northeast.",
    expanded_answer: "The learned evidence identifies localized change in the northeast. Most of the remaining scene is comparatively stable.",
    query_intent: "change_summary", overall_change_level: "localized", dominant_location: "northeast",
    likely_change_type: "unknown_change", evidence_strength: "moderate", stable_area_summary: "Most of the scene is stable.",
    changed_regions: ["northeast"], stable_regions: ["central", "western"], likely_transitions: [],
    supporting_facts: ["The primary change evidence identifies localized change.", "The largest connected region is centered in the northeast."],
    caveats: ["The binary change detector does not identify the real-world cause of change."],
    visual_structural_disagreement: true, generated_by: "local_semantic_interpreter" as const,
  } : null;
  return {
    ...base,
    task: "change_vqa",
    answer: semantic?.expanded_answer ?? "Approximately 1.1% of the analysis grid changed.",
    execution: { ...base.execution, input_mode: "bi_temporal", selected_tools: ["changerex_change_detector", "deterministic_change_analyzer"], steps: [...base.execution.steps, { tool: "semantic_change_interpretation", status: "success", duration_ms: 1, parameters: {} }] },
    vqa_details: null,
    change_engine: { mode: "hybrid", primary_tool: "changerex_change_detector", supporting_tool: "deterministic_change_analyzer", fallback_used: false, fallback_reason: null },
    semantic_change_summary: semantic,
    change_analysis: {
      request_id: "change-inner", status: "success", before_date: "2025-01-01", after_date: "2025-02-01",
      before_metadata: base.primary_image_metadata!, after_metadata: base.primary_image_metadata!, compatibility: { compatible: true, alignment_level: "visual_only", same_dimensions: true, same_crs: null, same_transform: null, bounds_overlap: null, overlap_ratio: null, resampling_required: false, warnings: [], errors: [] },
      statistics, deterministic_statistics: { ...statistics, changed_pixels: 3900, percentage_changed: 39 },
      previews: { before: "/before.png", after: "/after.png", difference: "/difference.png", mask: "/mask.png", overlay: "/overlay.png", ttp_mask: "/learned.png", deterministic_mask: "/deterministic.png" },
      execution: { ...base.execution, input_mode: "bi_temporal" }, runtime_ms: 90, warnings: [],
      change_engine: { mode: "hybrid", primary_tool: "changerex_change_detector", supporting_tool: "deterministic_change_analyzer", fallback_used: false, fallback_reason: null },
      semantic_change_summary: semantic,
    },
  };
}

function sarTranslationResponse(generation: "SUCCEEDED" | "FAILED", semantic: "SUCCEEDED" | "FAILED" = "SUCCEEDED"): AgentResponse {
  const item = response();
  item.task = "sar_scene_analysis";
  item.vqa_details = null;
  item.evidence = [];
  item.primary_image_metadata = {
    ...item.primary_image_metadata!, file_id: "sar-source", preview_url: "/sar-source.png",
    effective_modality: "sar_preview", auto_detected_modality: "sar_preview",
  };
  item.execution = {
    ...item.execution,
    selected_tools: ["sar_scene_analyzer"],
    steps: [
      { tool: "sar_translation_inference", status: generation === "SUCCEEDED" ? "success" : "failed", duration_ms: 3, parameters: {} },
      { tool: "translation_semantic_comparison", status: semantic === "SUCCEEDED" ? "success" : "failed", duration_ms: 1, parameters: {} },
    ],
  };
  item.sar_scene_analysis = {
    execution_status: "completed_with_limitations", task: "sar_scene_analysis", method: "native_sar_statistics",
    method_version: "1", valid_pixel_percent: 100, low_backscatter_percent: 20, mid_backscatter_percent: 50,
    high_backscatter_percent: 30, normalized_mean: .5, normalized_standard_deviation: .2, texture_index: .4,
    input_quality_score: .8, evidence_products: [],
    preprocessing: { version: "1", input_value_domain: "display", log_transform_applied: false, normalization: "percentile", percentile_low: 0, percentile_high: 255, invalid_pixel_count: 0, nodata_pixel_count: 0, denoising: "none", resized: false },
    limitations: [], warnings: [], runtime_ms: 2,
  };
  item.sar_translated_optical_analysis = {
    enabled: true, status: generation === "SUCCEEDED" ? "completed_with_limitations" : "translation_unavailable",
    generation_state: generation, semantic_comparison_state: generation === "FAILED" ? "NOT_REQUESTED" : semantic,
    model: "Pix2Pix", device: "cpu", runtime_ms: 4, generated_width: 256, generated_height: 256,
    generated_preview_url: "/generated.png", normalized_sar_preview_url: null, color_corrected_preview_url: null,
    evidence_products: [{
      evidence_id: "pix-product", source_observation_id: "sar-source", source_modality: "sar_preview",
      evidence_type: "generated_optical_like", evidence_run_id: item.request_id, generator: "Pix2Pix",
      status: "SUCCEEDED", type: "generated_optical_like_representation", label: "Generated optical-like representation",
      description: "Learned optical-like representation generated from SAR; not observed optical imagery and not ground truth.", reference: "/generated.png",
    }],
    color_corrected: false, fallback_used: false, fallback_reason: null, preprocessing_method: "test",
    input_channel_interpretation: "SAR preview", output_value_range: [0, 1], content_hash: "abc",
    optical_caption: semantic === "SUCCEEDED" ? "generated scene" : null, optical_scene_priors: [],
    optical_grounding: null, optical_vqa: null, optical_specialists_executed: semantic === "SUCCEEDED" ? ["rs_captioner"] : [],
    native_sar_findings: ["Native SAR evidence remains available."], translated_findings: semantic === "SUCCEEDED" ? ["Translation suggests a generated scene."] : [],
    agreement: semantic === "SUCCEEDED" ? "suggested_by_translation_only" : "supported_by_native_sar_only",
    direct_answer: "Native SAR evidence remains available.", confidence: { level: "low", score: .3, reason: "Supporting evidence only." },
    disclosure: "Learned optical-like representation generated from SAR; not observed optical imagery and not ground truth.",
    grounding_disclosure: null, rsvqa_disclosure: null, limitations: [], warnings: [], provenance: {}, runtime_breakdown_ms: {},
  };
  return item;
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function renderResult(value = response(), query = "Is a water body visible?") {
  await act(async () => root.render(createElement(AssistantResultExperience, {
    result: value,
    changeResult: value.change_analysis ?? undefined,
    query,
    children: createElement("div", { "data-testid": "full-audit" }, "Full specialist region table"),
  })));
}

function clickButton(name: string) {
  const button = Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name));
  expect(button).toBeTruthy();
  act(() => button!.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

describe("AssistantResultExperience", () => {
  it("shows a five-second executive summary with persistent report and compare actions", async () => {
    await renderResult();
    expect(container.textContent).toContain("Water body likely present.");
    expect(container.textContent).toContain("Moderate");
    expect(container.textContent).toContain("66 ms");
    expect(container.textContent).toContain("Remote Sensing VQA Specialist");
    expect(container.textContent).toContain("What determines confidence?");
    expect(container.textContent).toContain("This is not a calibrated probability.");
    expect(container.querySelector('button[aria-label="Download PDF mission report"]')).not.toBeNull();
    expect(container.querySelector('button[aria-label="Download JSON mission report"]')).not.toBeNull();
    expect(container.querySelector('button[aria-label="Download Full ZIP mission report"]')).not.toBeNull();
    expect(container.querySelector('a[href="/assistant/compare"]')).not.toBeNull();
    expect(container.querySelector(".result-sticky-header")).not.toBeNull();
  });

  it("keeps evidence, statistics, rationale, and the compact timeline visible in Judge View", async () => {
    await renderResult();
    expect(container.textContent).toContain("Evidence");
    expect(container.textContent).toContain("Key Statistics");
    expect(container.textContent).toContain("Why This Answer?");
    expect(container.textContent).toContain("Presence question detected.");
    expect(container.textContent).toContain("Execution Timeline");
    expect(Array.from(container.querySelectorAll(".result-timeline-groups > details > summary")).map(item => item.textContent)).toEqual(expect.arrayContaining([expect.stringContaining("Input"), expect.stringContaining("Validation"), expect.stringContaining("Model Inference"), expect.stringContaining("Evidence Fusion"), expect.stringContaining("Response")]));
    expect(container.textContent).toContain("Upload");
    expect(container.textContent).not.toContain("Full specialist region table");
  });

  it("lazy-renders full scientific detail and defaults region tables to five entries", async () => {
    await renderResult();
    clickButton("Research View");
    expect(container.textContent).toContain("Show All Regions (7)");
    expect(container.querySelectorAll("tbody tr")).toHaveLength(5);
    expect(container.textContent).not.toContain("Full specialist region table");
    clickButton("Complete scientific record");
    expect(container.textContent).toContain("Full specialist region table");
    clickButton("Show All Regions");
    expect(container.querySelectorAll("tbody tr")).toHaveLength(7);
  });

  it("puts the natural bi-temporal interpretation before technical statistics in Judge View", async () => {
    await renderResult(changeResponse(), "What changed?");
    expect(container.textContent).toContain("Human-readable interpretation");
    expect(container.textContent).toContain("localized change in the northeast");
    expect(container.textContent).toContain("Unknown Change");
    expect(container.textContent).toContain("Key Statistics");
    const text = container.textContent ?? "";
    expect(text.indexOf("Human-readable interpretation")).toBeLessThan(text.indexOf("Key Statistics"));
    expect(text).not.toContain("new buildings");
  });

  it("preserves exact regions and scientific fields in Research View", async () => {
    await renderResult(changeResponse(), "What changed?");
    clickButton("Research View");
    expect(container.textContent).toContain("Pixel bounds");
    expect(container.textContent).toContain("39.000%");
    expect(container.textContent).toContain("Semantic Change Interpretation");
  });

  it("renders an old stored bi-temporal result without a semantic field", async () => {
    await renderResult(changeResponse(false), "What changed?");
    expect(container.textContent).toContain("Approximately 1.1% of the analysis grid changed.");
    expect(container.textContent).not.toContain("Human-readable interpretation");
    expect(container.textContent).toContain("Key Statistics");
  });

  it("uses a mobile-safe responsive semantic facts grid", async () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });
    await renderResult(changeResponse(), "What changed?");
    expect(container.querySelector("#semantic-change-heading")?.closest("section")?.querySelector("dl")?.className).toContain("sm:grid-cols-3");
  });
});

describe("Pix2Pix evidence lifecycle", () => {
  it("shows a succeeded run as used evidence with disclosure and blend controls", async () => {
    const item = sarTranslationResponse("SUCCEEDED");
    expect(evidenceProducts(item).map(product => product.url)).toContain("/generated.png");
    await renderResult(item, "Describe this SAR scene.");
    expect(container.textContent).toContain("Pix2Pix-generated representation");
    expect(container.textContent).toContain("used");
    expect(container.textContent).toContain("not observed optical imagery and not ground truth");
    expect(container.querySelector('[aria-label="View Generated optical-like representation, Supporting Evidence"]')).not.toBeNull();
    const opacity = Array.from(container.querySelectorAll("button")).find(button => button.textContent === "Opacity")!;
    await act(async () => opacity.click());
    expect(container.querySelector('[aria-label="Evidence image viewer"]')?.textContent).toContain("Original image + Generated optical-like representation · Visual blend");
  });

  it("suppresses a failed run even when stale preview fields are present", async () => {
    const item = sarTranslationResponse("FAILED");
    expect(evidenceProducts(item).map(product => product.url)).not.toContain("/generated.png");
    await renderResult(item, "Describe this SAR scene.");
    expect(container.textContent).toContain("Pix2Pix-generated representation");
    expect(container.textContent).toContain("failed");
    expect(container.textContent).toContain("Legacy generated evidence provenance unavailable");
    expect(container.querySelector('[aria-label="View Generated optical-like representation, Supporting Evidence"]')).toBeNull();
    expect(Array.from(container.querySelectorAll("button")).some(button => button.textContent === "Opacity")).toBe(false);
    expect(container.textContent).not.toContain("Agreement: Suggested By Translation Only");
  });

  it("keeps successful generation used while marking semantic comparison separately failed", async () => {
    const item = sarTranslationResponse("SUCCEEDED", "FAILED");
    await renderResult(item, "Describe this SAR scene.");
    expect(container.textContent).toContain("Pix2Pix-generated representation");
    expect(container.textContent).toContain("Translation semantic comparison");
    expect(container.textContent).toContain("No failed semantic claim was used");
    expect(container.textContent).not.toContain("Agreement: Supported By Native Sar Only");
  });
});


describe("Cross-modal evidence provenance", () => {
  function paired(): AgentResponse {
    const item = response();
    item.cross_modal_analysis = {
      status: "success", analysis_level: "qualitative", quantitative_metrics_available: false,
      summary: { optical_observations: [], sar_observations: [], joint_observations: [] },
      statistics: null, regions: [],
      previews: { optical: null, sar: null, optical_evidence: null, sar_evidence: null, joint_evidence: null, water_likelihood: null, built_up_likelihood: null, vegetation_support: null, agreement: null, disagreement: null, joint_overlay: null },
      confidence: item.confidence, method: { name: "Native evidence", version: "1.1", uses_trained_model: false, assumptions: [], limitations: [] },
      optical_preparation: null, sar_preparation: null, warnings: [], runtime_ms: 1,
    };
    item.primary_image_metadata = { ...item.primary_image_metadata!, observation_role: "sar", file_id: "sar-id", preview_url: "/sar.png" };
    item.secondary_image_metadata = { ...item.primary_image_metadata!, observation_role: "optical", file_id: "optical-id", preview_url: "/optical.png" };
    return item;
  }
  it("resolves Optical and SAR source tabs from provenance despite inverted storage order", () => {
    const item = paired();
    item.cross_modal_analysis!.evidence_products = [
      { type: "preview", label: "Optical source", reference: "/optical-product.png", source_role: "optical", source_observation_id: "optical-id", evidence_type: "source" },
      { type: "preview", label: "SAR source", reference: "/sar-product.png", source_role: "sar", source_observation_id: "sar-id", evidence_type: "source" },
    ];
    const products = evidenceProducts(item);
    expect(products.find(product => product.label === "SAR source")!.url).toBe("/sar-product.png");
    expect(products.find(product => product.label === "Optical source")!.url).toBe("/optical-product.png");
  });
  it("renders only SAR pixels on the SAR source tab and names both images in an explicit blend", async () => {
    const item = paired();
    await act(async () => root.render(createElement(AssistantResultExperience, { result: item, query: "Where do both modalities agree?", children: null })));
    const viewer = container.querySelector('[aria-label="Evidence image viewer"]')!;
    expect(viewer.querySelectorAll("img")).toHaveLength(1);
    expect(viewer.querySelector("img")!.getAttribute("src")).toContain("/sar.png");
    const opacity = Array.from(container.querySelectorAll("button")).find(button => button.textContent === "Opacity")!;
    await act(async () => opacity.click());
    expect(viewer.textContent).toContain("Optical source + SAR source");
    const sarTab = container.querySelector<HTMLButtonElement>('[aria-label="View SAR source, Primary Evidence"]')!;
    await act(async () => sarTab.click());
    expect(viewer.querySelectorAll("img")).toHaveLength(1);
    expect(viewer.querySelector("img")!.getAttribute("alt")).toBe("SAR source");
  });
  it("uses role metadata when products are absent and never invents legacy roles from index", () => {
    const item = paired();
    expect(evidenceProducts(item).find(product => product.label === "SAR source")!.url).toBe("/sar.png");
    delete item.primary_image_metadata!.observation_role;
    delete item.secondary_image_metadata!.observation_role;
    expect(evidenceProducts(item).some(product => product.label === "SAR source")).toBe(false);
    item.cross_modal_analysis!.previews.sar = "/legacy-sar.png";
    expect(evidenceProducts(item).find(product => product.label === "SAR source")!.url).toBe("/legacy-sar.png");
  });
});

import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AssistantResultExperience } from "./assistant-result-experience";
import type { AgentResponse } from "@/types/agent";

const result: AgentResponse = {
  request_id: "sar-result-1", task: "sar_water_segmentation", answer: "Probable low-backscatter water candidates cover 13.8% of image pixels.",
  confidence: { level: "unavailable", score: null, reason: "A heuristic was used; no calibrated model confidence is available." },
  evidence: [], status: "partial", result_status: "COMPLETED_WITH_LIMITATIONS", warnings: [],
  execution: { input_mode: "single", selected_tools: ["input_validator", "sar_water_segmenter"], duration_ms: 41, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "SAR water mask requested.", steps: [{ tool: "sar_preprocessing", status: "success", duration_ms: 8, parameters: {} }, { tool: "heuristic_segmentation", status: "success", duration_ms: 11, parameters: {} }, { tool: "evidence_generation", status: "success", duration_ms: 6, parameters: {} }] },
  primary_image_metadata: { file_id: "sar", original_name: "sar.png", safe_name: "sar.png", format: "png", mime_type: "image/png", size_bytes: 1024, width: 256, height: 256, band_count: 1, dtype: "uint8", crs: null, transform: null, bounds: null, nodata: null, is_georeferenced: false, preview_url: "/source.png", color_interpretation: ["l"], representation: "display_preview", auto_detected_modality: "unknown", auto_detection_confidence: "low", auto_detection_reason: "Ambiguous grayscale display.", user_confirmed_modality: "sar_preview", effective_modality: "sar_preview", modality_limitations: [], warnings: [] },
  secondary_image_metadata: null, pair_compatibility: null, model: null, caption_details: null, grounding_result: null, cross_modal_analysis: null, change_analysis: null, vqa_details: null,
  sar_water_analysis: { execution_status: "completed_with_limitations", task: "sar_water_segmentation", target: "water", method: "heuristic_candidate_detector", method_version: "heuristic-sar-water-1.0", water_detected: true, image_area_percent: 13.8, model_confidence: null, heuristic_reliability: .74, input_quality_score: .81, threshold: .22, candidate_pixels: 9044, valid_pixels: 65536, regions: [{ region_id: 1, area_pixels: 9044, image_area_percent: 13.8, bounding_box: [30, 90, 220, 230], centroid: [125, 160] }], evidence_products: [{ id: "source", type: "source_preview", label: "Original source preview", description: "Source display.", source: "sar_water_segmenter", authoritative: true, reference: "/source.png", width: 256, height: 256, generation_method: "ingestion" }, { id: "mask", type: "binary_candidate_mask", label: "Binary water-candidate mask", description: "Computed mask.", source: "sar_water_segmenter", authoritative: true, reference: "/mask.png", width: 256, height: 256, generation_method: "heuristic" }, { id: "overlay", type: "candidate_overlay", label: "Water-candidate overlay", description: "Computed overlay.", source: "sar_water_segmenter", authoritative: true, reference: "/overlay.png", width: 256, height: 256, generation_method: "heuristic" }], preprocessing: { version: "sar-preprocess-1.0", input_value_domain: "unknown", log_transform_applied: false, normalization: "1/99 percentile", percentile_low: 4, percentile_high: 242, invalid_pixel_count: 0, nodata_pixel_count: 0, denoising: "median", resized: false }, rationale: ["Low-backscatter candidates."], limitations: ["Heuristic, not a trained model.", "No geographic area without georeference."], warnings: [], runtime_ms: 35 },
  sar_scene_analysis: null, classified_query: { task_type: "sar_water_segmentation", target: "water", requested_output: "overlay", requires_localization: true, requires_segmentation: true, requires_measurement: true }, cache: null,
};

let container: HTMLDivElement;
let root: Root;
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container); });
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

describe("single-image SAR result experience", () => {
  it("shows qualitative representation, evidence, reliability, and honest limitations", async () => {
    await act(async () => root.render(createElement(AssistantResultExperience, { result, query: "Highlight the water body.", children: createElement("div") })));
    expect(container.textContent).toContain("COMPLETED WITH LIMITATIONS");
    expect(container.textContent).toContain("Display Preview — Qualitative Analysis Only");
    expect(container.textContent).toContain("Binary water-candidate mask");
    expect(container.textContent).toContain("Water-candidate overlay");
    expect(container.textContent).toContain("74.0% · not model confidence");
    expect(container.textContent).toContain("Model confidence");
    expect(container.textContent).toContain("Not available");
    expect(container.textContent).toContain("No geographic area without georeference.");
    expect(container.textContent).toContain("Sar Water Segmenter");
  });
});

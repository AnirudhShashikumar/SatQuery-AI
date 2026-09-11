import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { GroundingResultPanel } from "./assistant-workspace";
import { AssistantResultExperience } from "./assistant-result-experience";
import type { AgentResponse, GroundingCandidateQuality, GroundingResult } from "@/types/agent";

const quality: GroundingCandidateQuality = {
  source_width: 256, source_height: 256, box_width: 256.000015, box_height: 255.999862,
  box_area: 65535.968512, image_area: 65536, box_area_ratio: 0.99999952,
  alignment_score: 0.37422514, finite_score: true, finite_coordinates: true,
  positive_area: true, in_bounds: false,
  rejection_reasons: ["alignment_score_below_minimum", "localized_box_area_ratio_above_maximum"],
};

function response(accepted = false): AgentResponse {
  const detection = {
    label: "water body", score: 0.75, bbox_pixels: [40, 50, 130, 140] as [number, number, number, number],
    bbox_normalized: [0.15625, 0.195313, 0.507813, 0.546875] as [number, number, number, number],
    bbox_world: null, crs: null, mask_url: null, source: "Grounding DINO",
    quality: { ...quality, box_width: 90, box_height: 90, box_area: 8100, box_area_ratio: 0.12359619, alignment_score: 0.75, in_bounds: true, rejection_reasons: [] },
  };
  return {
    request_id: "grounding-quality-request", task: "grounding",
    answer: accepted ? "Grounding DINO produced 1 candidate region(s) for 'water body'." : "No reliable localized region was found for 'water body'. The model produced a weak scene-level match rather than a precise object region.",
    confidence: accepted ? { level: "moderate", score: 0.75, reason: "Alignment score." } : { level: "unavailable", score: null, reason: "No candidate met the reliability gate." },
    evidence: accepted ? [{ type: "model_produced_grounding", label: "Accepted grounding", reference: "/api/agent/previews/accepted.png" }] : [],
    execution: { input_mode: "single", selected_tools: ["input_validator", "rs_grounder"], steps: [{ tool: "grounding_quality_filter", status: "success", duration_ms: 1, parameters: { accepted_count: accepted ? 1 : 0, rejected_count: accepted ? 0 : 1 } }], duration_ms: 24, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Grounding request." },
    warnings: [], status: accepted ? "success" : "partial",
    primary_image_metadata: { file_id: "image", original_name: "scene.png", safe_name: "scene.png", format: "png", mime_type: "image/png", size_bytes: 100, width: 256, height: 256, band_count: 3, dtype: "uint8", crs: null, transform: null, bounds: null, nodata: null, is_georeferenced: false, preview_url: "/api/agent/previews/source.png", color_interpretation: ["r", "g", "b"], warnings: [] },
    secondary_image_metadata: null, pair_compatibility: null,
    model: { tool_id: "rs_grounder", checkpoint: "IDEA-Research/grounding-dino-tiny", base_architecture: "Grounding DINO (Swin-T)", adaptation_dataset: "Not applicable", remote_sensing_adapted: false, license: "Apache-2.0", source: "https://huggingface.co/IDEA-Research/grounding-dino-tiny" },
    caption_details: null,
    grounding_result: {
      original_query: "Highlight the water body.", target_phrase: "water body",
      detections: accepted ? [detection] : [], accepted_detections: accepted ? [detection] : [],
      rejected_candidates: accepted ? [] : [{ label: "water body", score: 0.37422514, bbox_source_xyxy: [0.021347, 0.053055, 256.021362, 256.052917], bbox_pixels: [0, 0, 256, 256], box_area_ratio: 0.99999952, rejection_reasons: quality.rejection_reasons, quality, source: "Grounding DINO" }],
      accepted_detection_count: accepted ? 1 : 0, rejected_candidate_count: accepted ? 0 : 1,
      quality_policy: { minimum_alignment_score: 0.45, maximum_localized_area_ratio: 0.85, localized_targets: ["water body"], calibration_status: "Operational reliability gates pending benchmark calibration." },
      empty_result_explanation: accepted ? null : "No reliable localized region was found for 'water body'. The model produced a weak scene-level match rather than a precise object region.",
      operational_threshold_disclaimer: "Operational reliability gates pending benchmark calibration.",
      annotated_preview_url: accepted ? "/api/agent/previews/accepted.png" : null,
      confidence: accepted ? { level: "moderate", score: 0.75, reason: "Alignment score." } : { level: "unavailable", score: null, reason: "No candidate met the reliability gate." },
      model: { tool_id: "rs_grounder", checkpoint: "IDEA-Research/grounding-dino-tiny", base_architecture: "Grounding DINO (Swin-T)", adaptation_dataset: "Not applicable", remote_sensing_adapted: false, license: "Apache-2.0", source: "https://huggingface.co/IDEA-Research/grounding-dino-tiny" },
      input: { modality: "optical", bands_used: ["r", "g", "b"], representation: "RGB", original_width: 256, original_height: 256, model_input_width: 800, model_input_height: 800, normalization_method: "Official processor" },
      device: "cpu", warnings: [], limitations: ["Zero-shot detector."], runtime_ms: 22, model_load_ms: 0, model_reused: true,
    },
    cross_modal_analysis: null, change_analysis: null, vqa_details: null, cache: null,
  };
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

describe("GroundingResultPanel reliability state", () => {
  it("renders a legacy cached Grounding result when optional quality arrays are absent", async () => {
    const legacy = response();
    const grounding = legacy.grounding_result as Partial<GroundingResult>;
    delete grounding.rejected_candidates;
    delete grounding.rejected_candidate_count;
    delete grounding.accepted_detection_count;
    await act(async () => root.render(createElement(AssistantResultExperience, { result: legacy, query: "Highlight the water body.", children: createElement("div", null, "Legacy audit") })));
    expect(container.textContent).toContain("No reliable localized region was found");
    expect(container.textContent).toContain("Accepted regions");
    expect(container.textContent).toContain("Rejected candidates");
    expect(container.textContent).not.toContain("Cannot read properties of undefined");
  });

  it("renders the rejected full-frame candidate as an honest empty localization state", async () => {
    await act(async () => root.render(createElement(GroundingResultPanel, { response: response() })));
    expect(container.textContent).toContain("No reliable localization found");
    expect(container.textContent).toContain("possible scene-level match");
    expect(container.textContent).toContain("0 accepted · 1 rejected");
    expect(container.textContent).toContain("0.3742");
    expect(container.textContent).toContain("100.00%");
    expect(container.textContent).toContain("Alignment Score Below Minimum");
    expect(container.textContent).toContain("Operational reliability gates pending benchmark calibration");
    expect(container.textContent).not.toContain("1 detection");
    expect(container.querySelector("img")?.getAttribute("alt")).toContain("without rejected annotations");
    expect(Array.from(container.querySelectorAll("button")).some(button => button.textContent?.includes("labels and scores"))).toBe(false);
  });

  it("preserves the existing successful rendering for an accepted localized candidate", async () => {
    await act(async () => root.render(createElement(GroundingResultPanel, { response: response(true) })));
    expect(container.textContent).toContain("Text-Guided Grounding");
    expect(container.textContent).toContain("1 detection");
    expect(container.textContent).toContain("0.7500");
    expect(container.textContent).not.toContain("No reliable localization found");
  });
});

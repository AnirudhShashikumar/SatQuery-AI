import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AssistantResultExperience } from "./assistant-result-experience";
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

async function renderResult() {
  await act(async () => root.render(createElement(AssistantResultExperience, {
    result: response(),
    query: "Is a water body visible?",
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
    expect(container.textContent).toContain("Rs Vqa");
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
});

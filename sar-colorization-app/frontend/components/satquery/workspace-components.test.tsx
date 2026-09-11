import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModeSelector } from "./mode-selector";
import { ErrorState } from "./error-state";
import { EvidenceSources } from "./evidence-sources";
import type { AgentResponse } from "@/types/agent";

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

function response(): AgentResponse {
  return {
    request_id: "source-test",
    task: "vqa",
    answer: "Water is visible.",
    confidence: { level: "moderate", score: null, reason: "Published reason." },
    evidence: [],
    execution: {
      input_mode: "single",
      selected_tools: ["input_validator", "rs_vqa", "sve"],
      steps: [
        { tool: "input_validator", status: "success", duration_ms: 1, parameters: {} },
        { tool: "evidence_extraction", status: "success", duration_ms: 2, parameters: {} },
        { tool: "rs_vqa", status: "success", duration_ms: 3, parameters: {} },
      ],
      duration_ms: 6,
      permitted_parameters: {},
      validation: { valid: true, errors: [] },
      selection_reason: "Presence question detected.",
    },
    warnings: [], status: "success", primary_image_metadata: null, secondary_image_metadata: null,
    pair_compatibility: null, model: null, caption_details: null, grounding_result: null,
    cross_modal_analysis: null, change_analysis: null,
    vqa_details: {
      original_question: "Is water visible?", question_category: "presence_water", target_concept: "water",
      answer_source: "rsvqa_specialist", statistics_used: {}, evidence_references: [],
      method: { name: "RSVQA", version: "1", method_type: "neural", uses_language_model: false, remote_sensing_adapted: true, assumptions: [], limitations: [] },
      confidence: { level: "moderate", score: null, reason: "Published reason." }, supported: true, limitations: [], single_image_evidence: null,
    },
    cache: null,
  };
}

describe("world-class workspace components", () => {
  it("makes all workflows visible and dispatches a keyboard-accessible mode choice", async () => {
    const onChange = vi.fn();
    await act(async () => root.render(createElement(ModeSelector, { value: "single", onChange })));
    const buttons = Array.from(container.querySelectorAll("button"));
    expect(buttons.map(button => button.textContent)).toEqual(expect.arrayContaining([
      expect.stringContaining("Single Image"), expect.stringContaining("Optical + SAR"), expect.stringContaining("Bi-temporal"),
    ]));
    expect(buttons[0].getAttribute("aria-pressed")).toBe("true");
    act(() => buttons[1].dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onChange).toHaveBeenCalledWith("cross_modal");
  });

  it("explains an unsupported format and provides a recovery action", async () => {
    const retry = vi.fn();
    await act(async () => root.render(createElement(ErrorState, { message: "File type is unsupported.", onRetry: retry })));
    expect(container.textContent).toContain("Choose a GeoTIFF, TIFF, PNG, or JPEG");
    const button = container.querySelector("button");
    act(() => button!.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(retry).toHaveBeenCalledOnce();
  });

  it("summarizes source-level evidence without repeating low-level trace steps", async () => {
    await act(async () => root.render(createElement(EvidenceSources, { result: response() })));
    expect(container.textContent).toContain("Visual question answering");
    expect(container.textContent).toContain("SatQuery Vision Encoder");
    expect(container.textContent).not.toContain("Evidence Extraction");
    expect(container.querySelectorAll(".sq-source-card")).toHaveLength(2);
  });
});

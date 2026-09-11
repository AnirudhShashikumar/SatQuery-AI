import { beforeEach, describe, expect, it } from "vitest";
import {
  clampPresentationStep,
  clearPresentationState,
  isPresentationTypingTarget,
  presentationCommandForKey,
  presentationSteps,
  presentationStorageKeys,
  readPresentationReturnRoute,
  readPresentationState,
  rememberPresentationReturnRoute,
} from "./presentation";

beforeEach(() => sessionStorage.clear());

describe("presentation definition", () => {
  it("contains the complete ordered static sequence", () => {
    expect(presentationSteps).toHaveLength(15);
    expect(presentationSteps.map(step => step.number)).toEqual(Array.from({ length: 15 }, (_, index) => index + 1));
    expect(presentationSteps[0].title).toBe("Welcome to SatQuery AI");
    expect(presentationSteps.at(-1)?.title).toBe("Ask Earth Anything.");
  });

  it("keeps every stage concise and presentation ready", () => {
    for (const step of presentationSteps) {
      expect(step.points.length).toBeGreaterThanOrEqual(2);
      expect(step.points.length).toBeLessThanOrEqual(4);
      expect(step.presenterNote?.say).toBeTruthy();
      expect(step.presenterNote?.show).toBeTruthy();
      expect(step.presenterNote?.watchFor).toBeTruthy();
    }
  });

  it("keeps completed scenes unchanged while completing the remaining presentation summaries", () => {
    const interactive = presentationSteps.filter(step => "scene_type" in step);
    expect(interactive).toHaveLength(9);
    expect(interactive[0]).toMatchObject({
      number: 6,
      scene_type: "single_image_understanding",
      demo_sample_id: "single_vqa",
      supported_actions: ["captioning", "vqa"],
      evidence_mode: "response_dependent",
      linked_tool_ids: ["satquery_vision_encoder_v1", "rs_captioner", "rs_vqa"],
    });
    expect(interactive[0].presenterNote).toEqual({
      say: "SatQuery Vision Encoder v1 is our OpenCLIP ViT-L/14 backbone adapted on BigEarthNet.txt image-text pairs. It provides Earth-observation-aware scene embeddings used for routing, caption consistency, retrieval, and evidence fusion. Specialist models remain responsible for grounding, change detection, and sensor-specific analysis.",
      show: "Load the approved optical sample, show the remote-sensing adapted badge and scene priors, then demonstrate caption consistency and one land-cover VQA evidence-consistency result before opening the execution trace.",
      watchFor: "Scene priors are similarity evidence, not calibrated probabilities or ground truth. Captioning remains RSICD-adapted and VQA remains deterministic.",
    });
    expect(interactive[1]).toMatchObject({
      number: 8,
      scene_type: "bi_temporal_change",
      demo_sample_id: "change_vqa",
      before_sample_id: "change-before.png",
      after_sample_id: "change-after.png",
      default_dates: { before: "2025-01-01", after: "2025-02-01" },
      default_query: { change: "What changed between these dates?" },
      linked_tool_ids: ["ttp_change_detector", "bitemporal_change_analyzer"],
    });
    expect(interactive[2]).toMatchObject({
      number: 9,
      scene_type: "cross_modal_analysis",
      demo_sample_id: "cross_modal",
      optical_sample_id: "cross-optical.tif",
      sar_sample_id: "cross-sar.tif",
      supported_actions: ["cross_modal"],
      default_query: { cross_modal: "Use the optical and SAR images together to identify built-up and water-covered regions." },
      linked_tool_ids: ["cross_modal_optical_sar_analyzer"],
    });
    expect(interactive[2].presenterNote).toEqual({
      say: "Optical imagery contributes spectral and contextual information, while SAR contributes structural and low-backscatter evidence. SatQuery AI combines both only when the pair is compatible.",
      show: "Point to the separate optical and SAR evidence, followed by the agreement, disagreement, and joint overlay products.",
      watchFor: "These are likelihood and support maps, not calibrated semantic ground truth. Images are never silently aligned.",
    });
    expect(interactive[3]).toMatchObject({
      number: 10,
      scene_type: "mission_report",
      linked_tool_ids: ["report_generator"],
      supported_formats: ["pdf", "json", "zip"],
    });
    expect(interactive[3].presenterNote).toEqual({
      say: "Every result can be converted into a reproducible mission report generated from the authoritative backend record.",
      show: "Highlight the evidence, provenance, confidence, execution trace, and limitations included in the report.",
      watchFor: "Reports never trust browser-supplied statistics and do not include secrets or local filesystem paths.",
    });
    expect(interactive[4]).toMatchObject({
      number: 11,
      scene_type: "mission_comparison",
      linkedRoute: "/assistant/compare",
      linked_capabilities: ["authoritative stored-result comparison"],
    });
    expect(interactive[4].presenterNote).toEqual({
      say: "SatQuery AI compares authoritative stored results rather than browser-supplied values.",
      show: "Highlight shared input identity, comparability level, factual differences, and the absence of a universal winner.",
      watchFor: "Runtime differences are not treated as scientific accuracy differences, and unrelated workflows are explicitly marked as not directly comparable.",
    });
    expect(interactive[5]).toMatchObject({ number: 12, scene_type: "research_analytics", linkedRoute: "/assistant/analytics", linked_capabilities: ["live research analytics"] });
    expect(interactive[6]).toMatchObject({ number: 13, scene_type: "architecture_summary", linkedRoute: "/architecture", linked_capabilities: ["live agentic architecture"] });
    expect(interactive[7]).toMatchObject({ number: 14, scene_type: "compliance_summary", linkedRoute: "/assistant/compliance", linked_capabilities: ["authoritative SIH compliance matrix"] });
    expect(interactive[8]).toMatchObject({ number: 15, scene_type: "closing_summary", linkedRoute: "/assistant", linked_capabilities: ["complete guided walkthrough"] });
  });
});

describe("presentation keyboard mapping", () => {
  it.each([
    ["ArrowRight", "next"], [" ", "next"], ["ArrowLeft", "previous"], ["Backspace", "previous"],
    ["Home", "first"], ["End", "last"], ["f", "fullscreen"], ["F", "fullscreen"],
    ["n", "notes"], ["N", "notes"], ["Escape", "exit"],
  ])("maps %s to %s", (key, command) => expect(presentationCommandForKey(key)).toBe(command));

  it("does not map unrelated keys", () => expect(presentationCommandForKey("Enter")).toBeNull());

  it("recognizes typing and interactive targets", () => {
    for (const element of [document.createElement("input"), document.createElement("textarea"), document.createElement("select"), document.createElement("button"), document.createElement("a")]) {
      expect(isPresentationTypingTarget(element)).toBe(true);
    }
    expect(isPresentationTypingTarget(document.createElement("div"))).toBe(false);
  });
});

describe("presentation session state", () => {
  it("clamps restored steps to the valid sequence", () => {
    expect(clampPresentationStep(-10)).toBe(0);
    expect(clampPresentationStep(999)).toBe(14);
    expect(clampPresentationStep("7")).toBe(7);
    expect(clampPresentationStep("invalid")).toBe(0);
  });

  it("starts at the first step with notes hidden", () => {
    expect(readPresentationState(sessionStorage)).toEqual({ step: 0, notesVisible: false });
  });

  it("restores current step and notes visibility", () => {
    sessionStorage.setItem(presentationStorageKeys.step, "6");
    sessionStorage.setItem(presentationStorageKeys.notes, "true");
    sessionStorage.setItem(presentationStorageKeys.changeQuery, "What changed?");
    sessionStorage.setItem(presentationStorageKeys.changeBeforeDate, "2025-01-01");
    sessionStorage.setItem(presentationStorageKeys.changeAfterDate, "2025-02-01");
    sessionStorage.setItem(presentationStorageKeys.crossModalQuery, "Use both images");
    expect(readPresentationState(sessionStorage)).toEqual({ step: 6, notesVisible: true });
  });

  it("stores an exact safe return route and rejects presentation or external destinations", () => {
    rememberPresentationReturnRoute(sessionStorage, "/assistant/compare?sort=recent#evidence");
    expect(readPresentationReturnRoute(sessionStorage)).toBe("/assistant/compare?sort=recent#evidence");
    rememberPresentationReturnRoute(sessionStorage, "/presentation?step=2");
    expect(readPresentationReturnRoute(sessionStorage)).toBe("/assistant");
    rememberPresentationReturnRoute(sessionStorage, "https://example.com");
    expect(readPresentationReturnRoute(sessionStorage)).toBe("/assistant");
    expect(sessionStorage.getItem(presentationStorageKeys.changeQuery)).toBeNull();
    expect(sessionStorage.getItem(presentationStorageKeys.changeBeforeDate)).toBeNull();
    expect(sessionStorage.getItem(presentationStorageKeys.changeAfterDate)).toBeNull();
    expect(sessionStorage.getItem(presentationStorageKeys.crossModalQuery)).toBeNull();
  });

  it("clears only presentation-owned keys", () => {
    sessionStorage.setItem("unrelated", "keep");
    sessionStorage.setItem(presentationStorageKeys.step, "4");
    sessionStorage.setItem(presentationStorageKeys.notes, "true");
    rememberPresentationReturnRoute(sessionStorage, "/reports");
    clearPresentationState(sessionStorage);
    expect(sessionStorage.getItem("unrelated")).toBe("keep");
    expect(readPresentationState(sessionStorage)).toEqual({ step: 0, notesVisible: false });
    expect(readPresentationReturnRoute(sessionStorage)).toBe("/assistant");
  });
});

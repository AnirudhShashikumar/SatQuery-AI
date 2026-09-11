import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationChangeScene } from "./presentation-change-scene";
import { presentationSteps } from "@/lib/presentation";
import type { AgentResponse, DemoManifest, PairCompatibility } from "@/types/agent";
import { getDemoManifest, getHealth, loadDemoFile, runAgentImageQuery } from "@/services/api";

vi.mock("@/services/api", async importOriginal => {
  const original = await importOriginal<typeof import("@/services/api")>();
  return {
    ...original,
    getDemoManifest: vi.fn(),
    getHealth: vi.fn(),
    loadDemoFile: vi.fn(),
    runAgentImageQuery: vi.fn(),
  };
});

const manifest: DemoManifest = {
  enabled: true,
  local_only: true,
  workflows: [{
    id: "change_vqa", title: "Change", description: "Approved", input_mode: "bi_temporal",
    primary_modality: "optical", secondary_modality: "optical", query: "How much changed?",
    primary_date: "2025-01-01", secondary_date: "2025-02-01",
    files: [
      { role: "primary", filename: "change-before.png", url: "/before.png", mime_type: "image/png" },
      { role: "secondary", filename: "change-after.png", url: "/after.png", mime_type: "image/png" },
    ],
  }],
};

const compatibility: PairCompatibility = {
  compatible: true, alignment_level: "visual_only", same_dimensions: true, same_crs: null, same_transform: null,
  bounds_overlap: null, overlap_ratio: null, resampling_required: false, warnings: [], errors: [],
};

function result(cached = false): AgentResponse {
  return {
    request_id: "request-change", task: "change_description",
    answer: "Approximately 12.5% of the analysis grid changed across 2 connected region(s). The largest region contains 80 pixels. The cause and land-cover type of change are not inferred.",
    confidence: { level: "moderate", score: null, reason: "Templated directly from deterministic change statistics." },
    evidence: [], status: "success", warnings: [], primary_image_metadata: null, secondary_image_metadata: null,
    pair_compatibility: compatibility, model: null, caption_details: null, grounding_result: null, cross_modal_analysis: null, vqa_details: null,
    execution: { input_mode: "bi_temporal", selected_tools: ["input_validator", "bitemporal_change_analyzer"], steps: [{ tool: "difference_computation", status: "success", duration_ms: 2, parameters: {} }], duration_ms: 14, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Change." },
    change_analysis: {
      request_id: "inner", status: "success", before_date: "2025-01-01", after_date: "2025-02-01",
      before_metadata: null as never, after_metadata: null as never, compatibility,
      statistics: { analysis_width: 100, analysis_height: 100, source_width: 100, source_height: 100, total_pixels: 10000, changed_pixels: 1250, percentage_changed: 12.5, largest_connected_region: 80, number_of_regions: 2, bounding_boxes: [], regions: [{ region_id: 1, area_pixels: 80, percentage_of_image: 0.8, bounding_box: { left: 1, top: 1, right: 11, bottom: 9, area_pixels: 80 } }], normalized_threshold: 0.2 },
      previews: { before: "/before-result.png", after: "/after-result.png", difference: "/difference.png", mask: "/mask.png", overlay: "/overlay.png" },
      execution: { input_mode: "bi_temporal", selected_tools: [], steps: [], duration_ms: 12, permitted_parameters: {}, validation: { valid: true, errors: [] }, selection_reason: "Change." },
      runtime_ms: 12, warnings: [],
    },
    cache: { cached, original_generation_timestamp: "2026-08-24T00:00:00Z", retrieval_timestamp: "2026-08-24T00:00:00Z", tool_version: "1", cache_key_prefix: "change" },
  };
}

let container: HTMLDivElement;
let root: Root;
let objectUrlIndex = 0;

async function flush() {
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

function button(name: string) {
  return Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name)) as HTMLButtonElement | undefined;
}

async function click(element: Element | undefined) {
  expect(element).toBeTruthy();
  await act(async () => { element!.dispatchEvent(new MouseEvent("click", { bubbles: true })); await Promise.resolve(); });
}

async function loadPair() {
  await click(button("Load Approved Pair"));
  await flush();
}

beforeEach(async () => {
  sessionStorage.clear();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => `blob:pair-${++objectUrlIndex}`) });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  vi.mocked(getHealth).mockResolvedValue({ status: "ok" } as never);
  vi.mocked(getDemoManifest).mockResolvedValue(manifest);
  vi.mocked(loadDemoFile).mockImplementation(async (_path, filename, mime) => new File([filename], filename, { type: mime }));
  vi.mocked(runAgentImageQuery).mockResolvedValue(result());
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => { root.render(createElement(PresentationChangeScene, { step: presentationSteps[7] })); });
  await flush();
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("PresentationChangeScene", () => {
  it("renders the pre-load stage with analysis disabled", () => {
    expect(container.textContent).toContain("Before/after pair is not loaded");
    expect(button("Run Change Analysis")?.disabled).toBe(true);
    expect(runAgentImageQuery).not.toHaveBeenCalled();
  });

  it("loads both approved samples and dates without starting analysis", async () => {
    await loadPair();
    expect(container.querySelector('img[alt="Before approved bi-temporal observation"]')).toBeTruthy();
    expect(container.querySelector('img[alt="After approved bi-temporal observation"]')).toBeTruthy();
    expect((container.querySelector("#presentation-change-before-date") as HTMLInputElement).value).toBe("2025-01-01");
    expect((container.querySelector("#presentation-change-after-date") as HTMLInputElement).value).toBe("2025-02-01");
    expect(loadDemoFile).toHaveBeenCalledTimes(2);
    expect(runAgentImageQuery).not.toHaveBeenCalled();
  });

  it("starts only on click and blocks duplicate requests", async () => {
    await loadPair();
    let resolveRequest!: (value: AgentResponse) => void;
    vi.mocked(runAgentImageQuery).mockReturnValue(new Promise(resolve => { resolveRequest = resolve; }));
    const run = button("Run Change Analysis")!;
    await act(async () => {
      run.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      run.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    expect(runAgentImageQuery).toHaveBeenCalledTimes(1);
    expect(button("Analyzing")?.disabled).toBe(true);
    await act(async () => { resolveRequest(result()); await Promise.resolve(); });
  });

  it("renders the real result contract, statistics, controlled answer, and all evidence products", async () => {
    await loadPair();
    await click(button("Run Change Analysis"));
    await flush();
    expect(container.textContent).toContain("12.5% of the analysis grid changed");
    expect(container.textContent).toContain("12.500%");
    expect(container.textContent).toContain("Regions");
    expect(container.textContent).toContain("80 px · 0.80%");
    expect(container.textContent).toContain("Fresh Analysis");
    expect(container.textContent).toContain("Compatible · visual only");
    expect(container.querySelector('img[alt^="Difference map"]')).toBeTruthy();
    expect(container.querySelector('img[alt^="Binary mask"]')).toBeTruthy();
    expect(container.querySelector('img[alt^="Overlay"]')).toBeTruthy();
    expect(container.textContent).toContain("difference_computation");
  });

  it("renders cached state and reruns with the cache bypass contract", async () => {
    vi.mocked(runAgentImageQuery).mockResolvedValueOnce(result(true)).mockResolvedValueOnce(result(false));
    await loadPair();
    await click(button("Run Change Analysis"));
    await flush();
    expect(container.textContent).toContain("Cached Result");
    await click(button("Re-run Analysis"));
    await flush();
    expect(vi.mocked(runAgentImageQuery).mock.calls[1][0].forceRerun).toBe(true);
    expect(container.textContent).toContain("Fresh Analysis");
  });

  it("reset clears result and restores the controlled defaults while keeping the pair", async () => {
    await loadPair();
    await click(button("Run Change Analysis"));
    await flush();
    await click(button("Reset Scene"));
    expect(container.textContent).not.toContain("Fresh Analysis");
    expect(container.querySelector('img[alt="Before approved bi-temporal observation"]')).toBeTruthy();
    expect((container.querySelector("#presentation-change-query") as HTMLInputElement).value).toBe("What changed between these dates?");
  });

  it("shows backend-offline state and drops a stale completion after unmount", async () => {
    vi.mocked(getHealth).mockRejectedValue(new TypeError("Failed to fetch"));
    await act(async () => { root.unmount(); root = createRoot(container); root.render(createElement(PresentationChangeScene, { step: presentationSteps[7] })); });
    await flush();
    expect(container.querySelector('[aria-label="Backend offline"]')).toBeTruthy();

    vi.mocked(getHealth).mockResolvedValue({ status: "ok" } as never);
    await loadPair();
    let resolveRequest!: (value: AgentResponse) => void;
    vi.mocked(runAgentImageQuery).mockReturnValue(new Promise(resolve => { resolveRequest = resolve; }));
    await click(button("Run Change Analysis"));
    await act(async () => root.unmount());
    resolveRequest(result());
    await flush();
    expect(sessionStorage.getItem("satquery-latest-request-id")).toBeNull();
    root = createRoot(container);
  });
});

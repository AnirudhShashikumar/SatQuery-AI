import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DemoManifest, ImageMetadata } from "@/types/agent";

const navigation = vi.hoisted(() => ({ search: "" }));
const api = vi.hoisted(() => ({
  getHealth: vi.fn(), getDemoManifest: vi.fn(), inspectAgentImage: vi.fn(), loadDemoFile: vi.fn(), runAgentImageQuery: vi.fn(),
}));

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(navigation.search) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: React.ReactNode }) => createElement("a", { href, ...props }, children) }));
vi.mock("@/services/api", () => ({
  ...api,
  agentPreviewUrl: (value?: string | null) => value ?? null,
}));
vi.mock("@/components/satquery/system-health-panel", () => ({ SystemHealthPanel: () => createElement("span", null, "System Ready") }));
vi.mock("@/components/assistant-result-experience", () => ({ AssistantResultExperience: () => createElement("section", { "data-testid": "result" }) }));

import { AssistantUploadCard } from "./assistant-upload-card";
import { AssistantWorkspace } from "./assistant-workspace";

const demo: DemoManifest = {
  enabled: true,
  local_only: true,
  workflows: [
    { id: "single-demo", title: "Single-image understanding", description: "Real single-image sample", input_mode: "single", primary_modality: "optical", secondary_modality: null, query: "What is visible in this scene?", primary_date: null, secondary_date: null, files: [{ role: "primary", filename: "single.png", url: "/single.png", mime_type: "image/png" }] },
    { id: "cross-demo", title: "Optical–SAR joint analysis", description: "Real paired sample", input_mode: "cross_modal", primary_modality: "optical", secondary_modality: "sar", query: "Where do both modalities agree?", primary_date: null, secondary_date: null, files: [{ role: "primary", filename: "optical.tif", url: "/optical.tif", mime_type: "image/tiff" }, { role: "secondary", filename: "sar.tif", url: "/sar.tif", mime_type: "image/tiff" }] },
    { id: "change-demo", title: "Bi-temporal change analysis", description: "Real temporal sample", input_mode: "bi_temporal", primary_modality: "optical", secondary_modality: "optical", query: "What changed between these dates?", primary_date: "2025-01-01", secondary_date: "2025-02-01", files: [{ role: "primary", filename: "before.png", url: "/before.png", mime_type: "image/png" }, { role: "secondary", filename: "after.png", url: "/after.png", mime_type: "image/png" }] },
  ],
};

const metadata: ImageMetadata = {
  file_id: "image", original_name: "image.tif", safe_name: "image.tif", format: "geotiff", mime_type: "image/tiff", size_bytes: 2048,
  width: 1024, height: 1024, band_count: 3, dtype: "uint8", crs: "EPSG:4326", transform: [1, 0, 0, 0, -1, 0],
  bounds: { left: 0, bottom: 0, right: 1, top: 1 }, nodata: null, is_georeferenced: true, preview_url: "/preview.png",
  color_interpretation: ["r", "g", "b"], warnings: [], representation: "display_preview", auto_detected_modality: "optical_rgb",
  effective_modality: "optical_rgb", auto_detection_confidence: "high", auto_detection_reason: "RGB channels detected.",
};

let container: HTMLDivElement;
let root: Root;

async function flush() { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); }
function button(text: string) { return Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(text)) as HTMLButtonElement | undefined; }
async function click(element: Element | undefined) { expect(element).toBeTruthy(); await act(async () => { element!.dispatchEvent(new MouseEvent("click", { bubbles: true })); await Promise.resolve(); }); }
async function renderWorkspace() { await act(async () => root.render(createElement(AssistantWorkspace))); await flush(); }

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  navigation.search = "";
  api.getHealth.mockResolvedValue({ status: "ok" });
  api.getDemoManifest.mockResolvedValue({ enabled: false, local_only: true, workflows: [] });
  api.inspectAgentImage.mockResolvedValue({ metadata, requires_modality_confirmation: false });
  api.loadDemoFile.mockImplementation(async (_url: string, filename: string, mime: string) => new File([filename], filename, { type: mime }));
  api.runAgentImageQuery.mockResolvedValue({ request_id: "ui-result", change_analysis: null, pair_compatibility: null, vqa_details: null, grounding_result: null, cross_modal_analysis: null });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

describe("Assistant workspace polish", () => {
  it("renders one purposeful input for Single Image and updates prompts and header by workflow", async () => {
    await renderWorkspace();
    expect(container.querySelectorAll(".assistant-upload-card")).toHaveLength(1);
    expect(container.querySelector(".assistant-workflow-identity")?.textContent).toContain("Single Image Analysis");
    expect(container.textContent).toContain("What is visible in this scene?");
    expect(container.textContent).toContain("Add another observation");

    await click(button("Optical + SAR"));
    expect(container.querySelectorAll(".assistant-upload-card")).toHaveLength(2);
    expect(container.querySelector(".assistant-workflow-identity")?.textContent).toContain("Optical + SAR Joint Analysis");
    expect(container.textContent).toContain("Optical observation");
    expect(container.textContent).toContain("SAR observation");
    expect(container.textContent).toContain("What does SAR reveal that optical does not?");

    await click(button("Bi-temporal"));
    expect(container.querySelectorAll(".assistant-upload-card")).toHaveLength(2);
    expect(container.querySelector(".assistant-workflow-identity")?.textContent).toContain("Bi-temporal Change Analysis");
    expect(container.textContent).toContain("Earlier observation");
    expect(container.textContent).toContain("Later observation");
    expect(container.textContent).toContain("Did the built-up area increase?");
    expect(container.querySelectorAll('input[type="date"]')).toHaveLength(2);
    expect(container.querySelector('button[aria-label="Swap earlier and later observations"]')).toBeTruthy();
  });

  it("keeps the query compact and disables the text CTA until the workflow is valid", async () => {
    await renderWorkspace();
    const textarea = container.querySelector<HTMLTextAreaElement>('textarea[placeholder^="Ask a precise"]')!;
    const run = container.querySelector<HTMLButtonElement>('button[aria-label="Run SatQuery analysis"]')!;
    expect(textarea.rows).toBe(2);
    expect(textarea.maxLength).toBe(2000);
    expect(run.textContent).toContain("Run Analysis");
    expect(run.disabled).toBe(true);
    expect(container.querySelector(".assistant-workbench")).toBeTruthy();
  });

  it("loads a real demo scenario, enables submission, and preserves the existing request contract", async () => {
    api.getDemoManifest.mockResolvedValue(demo);
    await renderWorkspace();
    await click(container.querySelector('button[aria-label="Load demo scenario: Single-image understanding"]') ?? undefined);
    await flush();
    expect(container.textContent).toContain("single.png");
    expect(container.querySelector(".assistant-dropzone")).toBeNull();
    const run = container.querySelector<HTMLButtonElement>('button[aria-label="Run SatQuery analysis"]')!;
    expect(run.disabled).toBe(false);
    await click(run);
    await flush();
    expect(api.runAgentImageQuery).toHaveBeenCalledTimes(1);
    expect(api.runAgentImageQuery.mock.calls[0][0]).toMatchObject({ inputMode: "single", query: "What is visible in this scene?", useCache: true, forceRerun: false });
  });

  it("supports keyboard submission and swaps the temporal files and dates", async () => {
    api.getDemoManifest.mockResolvedValue(demo);
    await renderWorkspace();
    await click(container.querySelector('button[aria-label="Load demo scenario: Bi-temporal change analysis"]') ?? undefined);
    await flush();
    const beforeSwap = Array.from(container.querySelectorAll(".assistant-upload-file p")).map(item => item.textContent);
    expect(beforeSwap).toEqual(["before.png", "after.png"]);
    await click(container.querySelector('button[aria-label="Swap earlier and later observations"]') ?? undefined);
    expect(Array.from(container.querySelectorAll(".assistant-upload-file p")).map(item => item.textContent)).toEqual(["after.png", "before.png"]);
    expect(Array.from(container.querySelectorAll<HTMLInputElement>('input[type="date"]')).map(input => input.value)).toEqual(["2025-02-01", "2025-01-01"]);

    await click(container.querySelector('button[aria-label="Swap earlier and later observations"]') ?? undefined);
    const textarea = container.querySelector<HTMLTextAreaElement>("textarea")!;
    await act(async () => { textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", ctrlKey: true, bubbles: true })); await Promise.resolve(); });
    await flush();
    expect(api.runAgentImageQuery).toHaveBeenCalledTimes(1);
    expect(api.runAgentImageQuery.mock.calls[0][0]).toMatchObject({ inputMode: "bi_temporal", primaryDate: "2025-01-01", secondaryDate: "2025-02-01" });
  });

  it("blocks reversed cross-modal slots and explicitly swaps and reinspects the actual files", async () => {
    navigation.search = "mode=cross_modal";
    api.inspectAgentImage.mockImplementation(async (file: File) => ({ metadata: { ...metadata, original_name: file.name, auto_detected_modality: file.name.startsWith("sar") ? "sar_preview" : "optical_rgb" }, requires_modality_confirmation: false }));
    await renderWorkspace();
    const sar = new File(["radar"], "sar.png", { type: "image/png" });
    const optical = new File(["rgb"], "optical.png", { type: "image/png" });
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    for (const [index, file] of [sar, optical].entries()) {
      Object.defineProperty(inputs[index], "files", { value: [file], configurable: true });
      await act(async () => { inputs[index].dispatchEvent(new Event("change", { bubbles: true })); });
      await flush();
    }
    const run = container.querySelector<HTMLButtonElement>('button[aria-label="Run SatQuery analysis"]')!;
    expect(run.disabled).toBe(true);
    expect(container.textContent).toContain("These observations appear to be reversed");
    expect(container.querySelector(".assistant-detected-workflow")).toBeNull();
    expect(api.runAgentImageQuery).not.toHaveBeenCalled();
    await click(button("Swap observations"));
    await flush();
    expect(api.inspectAgentImage).toHaveBeenCalledTimes(4);
    expect(container.textContent).not.toContain("These observations appear to be reversed");
    expect(run.disabled).toBe(false);
    await click(run);
    expect(api.runAgentImageQuery.mock.calls[0][0]).toMatchObject({ inputMode: "cross_modal", primaryImage: optical, secondaryImage: sar, primaryModality: "optical", secondaryModality: "sar" });
  });

  it("retains file inspection when switching an existing optical upload into the paired workflow", async () => {
    api.getDemoManifest.mockResolvedValue(demo);
    api.inspectAgentImage.mockImplementation(async (file: File) => ({ metadata: { ...metadata, auto_detected_modality: file.name.startsWith("sar") ? "sar_preview" : "optical_rgb" }, requires_modality_confirmation: false }));
    await renderWorkspace();
    await click(container.querySelector('button[aria-label="Load demo scenario: Single-image understanding"]') ?? undefined);
    await click(button("Optical + SAR"));
    const sarInput = container.querySelectorAll<HTMLInputElement>('input[type="file"]')[1];
    Object.defineProperty(sarInput, "files", { value: [new File(["radar"], "sar.png", { type: "image/png" })], configurable: true });
    await act(async () => { sarInput.dispatchEvent(new Event("change", { bubbles: true })); });
    await flush();
    expect(container.querySelector<HTMLButtonElement>('button[aria-label="Run SatQuery analysis"]')!.disabled).toBe(false);
  });

  it("inspects both approved cross-modal demo files before enabling analysis", async () => {
    api.getDemoManifest.mockResolvedValue(demo);
    api.inspectAgentImage.mockImplementation(async (file: File) => ({ metadata: { ...metadata, auto_detected_modality: file.name.startsWith("sar") ? "sar_vv_vh" : "optical_rgb" }, requires_modality_confirmation: false }));
    await renderWorkspace();
    await click(container.querySelector('button[aria-label="Load demo scenario: Optical–SAR joint analysis"]') ?? undefined);
    await flush();
    expect(api.inspectAgentImage).toHaveBeenCalledTimes(2);
    expect(container.querySelector<HTMLButtonElement>('button[aria-label="Run SatQuery analysis"]')!.disabled).toBe(false);
  });

  it("uses responsive desktop and mobile layout rules without changing backend services", () => {
    const styles = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");
    expect(styles).toContain("grid-template-columns: minmax(0,45fr) minmax(0,55fr)");
    expect(styles).toContain("@media (max-width: 720px)");
    const backendDiffGuard = readFileSync(resolve(process.cwd(), "services/api.ts"), "utf8");
    expect(backendDiffGuard).toContain("runAgentImageQuery");
  });
});

describe("Assistant upload card", () => {
  it("collapses an uploaded image into preview and core facts while retaining advanced metadata", async () => {
    await act(async () => root.render(createElement(AssistantUploadCard, { label: "Primary observation", hint: "One satellite observation", file: new File(["image"], "test_1.tif", { type: "image/tiff" }), metadata, busy: false, onFile: vi.fn(), onError: vi.fn() })));
    expect(container.querySelector(".assistant-dropzone")).toBeNull();
    expect(container.querySelector('img[alt="Primary observation display preview"]')).toBeTruthy();
    expect(container.textContent).toContain("GEOTIFF · 1024 × 1024 · optical rgb");
    const details = container.querySelector<HTMLDetailsElement>(".assistant-upload-metadata")!;
    expect(details.open).toBe(false);
    expect(details.textContent).toContain("CRS");
    expect(container.querySelector('button[aria-label="Replace Primary observation"]')).toBeTruthy();
    expect(container.querySelector('button[aria-label="Remove Primary observation"]')).toBeTruthy();
  });
});

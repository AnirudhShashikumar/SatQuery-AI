import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationComparisonScene } from "./presentation-comparison-scene";
import { clearPresentationComparisonHandoff, hasPresentationComparisonReturn, missionComparisonStageIndex, readPresentationComparisonRequestIds } from "@/lib/presentation-comparison";
import { presentationStorageKeys } from "@/lib/presentation";
import { ApiRequestError, assessComparison, getComparisonItem, getComparisonItems } from "@/services/api";
import type { ComparabilityLevel, ComparisonAssessment, ComparisonItem, ComparisonItemSummary } from "@/types/agent";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
vi.mock("@/services/api", async importOriginal => {
  const original = await importOriginal<typeof import("@/services/api")>();
  return { ...original, assessComparison: vi.fn(), getComparisonItem: vi.fn(), getComparisonItems: vi.fn() };
});

const summary = (id: string, created: string, overrides: Partial<ComparisonItemSummary> = {}): ComparisonItemSummary => ({
  request_id: id, task: "cross_modal", display_name: `Mission ${id}`, status: "success", created_at: created,
  input_mode: "cross_modal", modalities: ["optical", "sar"], thumbnail: null, execution_duration_ms: id === "new" ? 25 : 41,
  cached: id === "new", report_available: id === "old", warning_count: 0, evidence_product_count: 9, ...overrides,
});
const detail = (value: ComparisonItemSummary): ComparisonItem => ({
  request_id: value.request_id, task: value.task, display_name: value.display_name, status: value.status, created_at: value.created_at,
  input_mode: value.input_mode, modalities: value.modalities, input_previews: [], output_previews: Array.from({ length: value.evidence_product_count }, (_, index) => ({ label: `Evidence ${index}`, url: `/evidence-${index}.png`, kind: "evidence", width: 10, height: 10, modality: null })),
  answer: null, statistics: {}, confidence: { level: "moderate", reason: "Stored basis" }, provenance: {}, selected_tools: ["cross_modal_optical_sar_analyzer"],
  execution_duration_ms: value.execution_duration_ms, device: "cpu", warnings: [], limitations: [], report_available: value.report_available,
  cached: value.cached, input_identity: {}, lineage: [], execution_summary: {},
});
const assessment = (level: ComparabilityLevel, sharedInputs = true, sharedTask = true): ComparisonAssessment => ({
  overall_level: level, warnings: [], assessments: [{ left_request_id: "new", right_request_id: "old", level,
    reason: level === "direct" ? "The authoritative input identity and task family match." : level === "partial" ? "The task family matches but input identity differs." : "These workflows have no common inputs or task family.",
    shared_inputs: sharedInputs, shared_task_family: sharedTask, warnings: [], overlay_allowed: level === "direct", difference_allowed: level === "direct" }],
});

let container: HTMLDivElement;
let root: Root;
async function flush() { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); }
async function renderScene() { await act(async () => { root.render(createElement(PresentationComparisonScene, { onContinue: vi.fn() })); }); await flush(); }
function button(name: string) { return Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name)); }
async function click(element: Element | undefined) { expect(element).toBeTruthy(); await act(async () => { element!.dispatchEvent(new MouseEvent("click", { bubbles: true })); await Promise.resolve(); }); await flush(); }

beforeEach(() => {
  sessionStorage.clear(); push.mockReset(); Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const values = [summary("old", "2026-08-24T12:00:00Z"), summary("new", "2026-08-24T12:01:00Z")];
  vi.mocked(getComparisonItems).mockResolvedValue(values);
  vi.mocked(getComparisonItem).mockImplementation(id => Promise.resolve(detail(values.find(item => item.request_id === id)!)));
  vi.mocked(assessComparison).mockResolvedValue(assessment("direct"));
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.clearAllMocks(); });

describe("PresentationComparisonScene", () => {
  it("shows the empty state when fewer than two authoritative results exist", async () => {
    vi.mocked(getComparisonItems).mockResolvedValue([summary("new", "2026-08-24T12:01:00Z")]);
    await renderScene();
    expect(container.textContent).toContain("Two stored results are required");
    expect(assessComparison).not.toHaveBeenCalled();
  });

  it("loads the two newest results and renders every requested summary field", async () => {
    await renderScene();
    expect(getComparisonItem).toHaveBeenNthCalledWith(1, "new");
    expect(getComparisonItem).toHaveBeenNthCalledWith(2, "old");
    for (const value of ["Optical–SAR", "Success", "Cross Modal", "25 ms", "9 products", "Cached", "Fresh", "Available", "Not generated"]) expect(container.textContent).toContain(value);
  });

  it.each([
    ["direct", true, true, "Direct"],
    ["partial", false, true, "Partial"],
    ["not_direct", false, false, "Not directly comparable"],
  ] as const)("renders the %s authoritative assessment", async (level, sharedInputs, sharedTask, label) => {
    vi.mocked(assessComparison).mockResolvedValue(assessment(level, sharedInputs, sharedTask));
    await renderScene();
    expect(container.textContent).toContain(label);
    expect(container.textContent).toContain(level === "direct" ? "Shared input identity: Yes" : "Shared input identity: No");
    expect(container.textContent).toContain(level === "not_direct" ? "No common input identity" : level === "partial" ? "task family matches" : "input identity and task family match");
  });

  it("renders factual differences without a universal winner or runtime-as-accuracy claim", async () => {
    await renderScene();
    for (const fact of ["Runtime", "Evidence products", "Cache state", "Tool selection", "Report"]) expect(container.textContent).toContain(fact);
    expect(container.textContent).toContain("Runtime is not a scientific accuracy score");
    expect(container.textContent).toContain("No universal winner is inferred");
  });

  it("refreshes on click without starting an analysis", async () => {
    await renderScene();
    expect(getComparisonItems).toHaveBeenCalledTimes(1);
    await click(button("Refresh"));
    expect(getComparisonItems).toHaveBeenCalledTimes(2);
  });

  it("distinguishes backend offline, expired results, and assessment failure", async () => {
    vi.mocked(getComparisonItems).mockRejectedValueOnce(new TypeError("Failed to fetch")); await renderScene();
    expect(container.textContent).toContain("backend is offline");
    await act(async () => root.unmount()); root = createRoot(container);
    vi.mocked(getComparisonItems).mockResolvedValue([summary("new", "2026-08-24T12:01:00Z"), summary("old", "2026-08-24T12:00:00Z")]);
    vi.mocked(getComparisonItem).mockRejectedValueOnce(new ApiRequestError("Expired", "COMPARISON_RESULT_EXPIRED", 410)); await renderScene();
    expect(container.textContent).toContain("has expired");
    await act(async () => root.unmount()); root = createRoot(container);
    vi.mocked(getComparisonItem).mockImplementation(id => Promise.resolve(detail(summary(id, "2026-08-24T12:01:00Z"))));
    vi.mocked(assessComparison).mockRejectedValueOnce(new ApiRequestError("Unavailable", "BACKEND_UNAVAILABLE", 503)); await renderScene();
    expect(container.textContent).toContain("comparability assessment could not be completed");
  });

  it("drops stale completions after leaving the stage", async () => {
    let resolveHistory!: (value: ComparisonItemSummary[]) => void;
    vi.mocked(getComparisonItems).mockReturnValue(new Promise(resolve => { resolveHistory = resolve; }));
    await act(async () => { root.render(createElement(PresentationComparisonScene, { onContinue: vi.fn() })); await Promise.resolve(); });
    await act(async () => root.unmount()); resolveHistory([summary("new", "2026-08-24T12:01:00Z"), summary("old", "2026-08-24T12:00:00Z")]); await flush();
    expect(container.textContent).toBe(""); root = createRoot(container);
  });

  it("stores Stage 11 and request IDs before opening the full workspace in the same tab", async () => {
    await renderScene(); await click(button("Open Full Comparison"));
    expect(sessionStorage.getItem(presentationStorageKeys.step)).toBe(String(missionComparisonStageIndex));
    expect(hasPresentationComparisonReturn(sessionStorage)).toBe(true);
    expect(readPresentationComparisonRequestIds(sessionStorage)).toEqual(["new", "old"]);
    expect(push).toHaveBeenCalledWith("/assistant/compare");
    clearPresentationComparisonHandoff(sessionStorage);
    expect(sessionStorage.getItem(presentationStorageKeys.step)).toBe("10");
    expect(hasPresentationComparisonReturn(sessionStorage)).toBe(false);
  });
});

describe("Mission Comparison presentation layout and return contract", () => {
  const css = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");
  const workspace = readFileSync(resolve(process.cwd(), "components/mission-comparison-workspace.tsx"), "utf8");
  const handoff = readFileSync(resolve(process.cwd(), "components/presentation-page-handoff.tsx"), "utf8");
  it("keeps both cards and assessment visible in the bounded 1024 projector layout", () => {
    expect(css).toContain(".presentation-comparison-scene { height: min(55dvh, 505px); }");
    expect(css).toContain("grid-template-rows: auto minmax(0, 1fr)");
  });
  it("stacks result cards on mobile and defines dark/light surfaces", () => {
    expect(css).toContain(".presentation-comparison-cards { grid-template-columns: minmax(0, 1fr); }");
    expect(css).toContain("html.light .presentation-comparison-card,");
    expect(css).toContain("background: rgba(4,9,18,.48)");
  });
  it("shows the return action only for a Presentation Mode handoff and routes back safely", () => {
    expect(workspace).toContain("hasPresentationComparisonReturn(window.sessionStorage)");
    expect(workspace).toContain('<PresentationReturnBanner route="/assistant/compare"/>');
    expect(handoff).toContain("Return to Presentation");
    expect(handoff).toContain('router.push("/presentation")');
    expect(handoff).toContain("clearPresentationHandoff(window.sessionStorage)");
  });
});

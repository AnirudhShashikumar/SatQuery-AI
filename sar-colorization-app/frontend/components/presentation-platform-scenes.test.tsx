import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationAnalyticsScene } from "./presentation-analytics-scene";
import { PresentationArchitectureScene } from "./presentation-architecture-scene";
import { PresentationClosingScene } from "./presentation-closing-scene";
import { PresentationComplianceScene } from "./presentation-compliance-scene";
import { PresentationOpenPageButton, PresentationReturnBanner } from "./presentation-page-handoff";
import { hasPresentationHandoff, storePresentationHandoff } from "@/lib/presentation-handoff";
import { presentationStorageKeys } from "@/lib/presentation";
import { getAgentAnalytics, getAgentCompliance, getAgentHealth, getAgentTools } from "@/services/api";
import type { AgentHealth, AnalyticsResponse, ComplianceRequirement, ComplianceResponse, ToolDefinition } from "@/types/agent";

const push = vi.fn();
const setTheme = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push, replace: vi.fn() }) }));
vi.mock("next-themes", () => ({ useTheme: () => ({ setTheme, resolvedTheme: "dark" }) }));
vi.mock("@/services/api", async importOriginal => {
  const original = await importOriginal<typeof import("@/services/api")>();
  return { ...original, getAgentAnalytics: vi.fn(), getAgentCompliance: vi.fn(), getAgentHealth: vi.fn(), getAgentTools: vi.fn() };
});

const analytics: AnalyticsResponse = {
  generated_at: "2026-08-25T00:00:00Z",
  platform: { backend_status: "healthy", uptime_seconds: 120, python_version: "3.9", operating_system: "Darwin", architecture: "arm64", process_memory_mb: null, runtime_versions: {}, hardware_acceleration: "Apple MPS", demo_mode: true, offline_ready: true, offline_readiness_requirements: [] },
  summary: { total_executions_current_process: 3, successful_executions_current_process: 3, registered_tools: 9, available_tools: 9, mandatory_satisfied: 15, mandatory_total: 15, cache_hits_current_process: 1, cache_misses_current_process: 2, report_artifacts_generated_current_process: 4 },
  cache: { stored_results: 3, max_results: 32, ttl_seconds: 1800, hits: 1, misses: 2, hit_rate_percent: 33.3 },
  reports: { requests_generated_current_process: 1, artifacts_generated_current_process: 4, artifacts_currently_available: 4, formats: { pdf: 1, json: 1, csv: 1, zip: 1 } },
  tools: [], datasets: [], capabilities: [{ name: "Optical ingestion", status: "Available", evidence: "Validated PNG, JPEG, TIFF, and GeoTIFF ingestion." }], workflow_metrics: [],
  recent_executions: [{ request_id: "latest-request", started_at: "2026-08-25T00:00:00Z", completed_at: "2026-08-25T00:00:01Z", task: "cross_modal_analysis", input_mode: "cross_modal", primary_modality: "optical", secondary_modality: "sar", status: "success", selected_tools: ["cross_modal_optical_sar_analyzer"], duration_ms: 25, warning_count: 1, output_count: 9, cache_status: "fresh", report_generated: true, device: "cpu", selection_reason: "Compatible optical and SAR pair selected.", confidence_level: "moderate", confidence_reason: "Exactly aligned evidence.", warnings: [], trace: [{ tool: "input_validation", status: "success", duration_ms: 3, parameters: {} }, { tool: "joint_evidence_fusion", status: "success", duration_ms: 22, parameters: {} }] }],
  last_execution_trace: [{ tool: "input_validation", status: "success", duration_ms: 3, parameters: {} }],
  scientific_transparency: { metric_source: "Current backend process", history_retention: "Bounded", inference_behavior_changed: false, ai_generated_metrics: false, unavailable_value_policy: "Unavailable remains unavailable", caveats: [] },
};

function tool(id: string, displayName = id, tasks: ToolDefinition["supported_tasks"] = ["vqa"], modalities: ToolDefinition["supported_modalities"] = ["optical"]): ToolDefinition {
  return { id, display_name: displayName, supported_tasks: tasks, supported_modalities: modalities, supported_input_modes: ["single"], status: "available", remote_sensing_adapted: id === "rs_captioner", service_path: `/api/${id}`, checkpoint: null, base_architecture: null, adaptation_dataset: id === "rs_captioner" ? "RSICD" : null, model_license: null, source: null, limitations: [], required_modalities: {}, method_type: "local", evidence_outputs: ["preview"], evidence_source: "backend", supported_question_categories: [], notes: "Available." };
}
const tools: ToolDefinition[] = [
  tool("input_validator", "Input Validator"), tool("rs_captioner", "Captioner", ["captioning"]), tool("rs_vqa", "VQA"), tool("rs_grounder", "Grounder", ["grounding"]),
  tool("bitemporal_change_analyzer", "Change", ["change_description", "change_vqa"]), tool("cross_modal_optical_sar_analyzer", "Cross Modal", ["cross_modal_analysis"], ["optical", "sar"]),
  tool("pix2pix_reconstruction", "Pix2Pix", ["unsupported"], ["sar"]), tool("sarfusionformer_analysis", "SARFusionFormer", ["unsupported"], ["sar"]), tool("report_generator", "Reports", ["report_generation"]),
];
analytics.tools = tools.map(value => ({ id: value.id, display_name: value.display_name, implementation_status: value.status, lifecycle_status: "registered", device: "cpu", last_runtime_ms: 12, last_completed_at: analytics.generated_at, method_type: value.method_type, checkpoint: value.checkpoint, adaptation_dataset: value.adaptation_dataset, remote_sensing_adapted: value.remote_sensing_adapted, service_path: value.service_path }));

const requirement = (requirement: string, implementation: string, status = "available"): ComplianceRequirement => ({ requirement, implementation, status, tool_or_model: "tool", test_coverage: "focused tests", limitation: `${requirement} limitation remains declared.` });
const compliance: ComplianceResponse = { generated_at: analytics.generated_at, project: "GeoVision · SatQuery AI", mandatory_satisfied: 15, mandatory_total: 15, optional_not_implemented: [], requirements: [
  requirement("Text-guided grounding", "Local Grounding DINO", "optional_available"), requirement("Evidence", "UUID evidence products"), requirement("Confidence", "Task-specific rationale"), requirement("Observable execution trace", "Ordered trace stages"), requirement("Downloadable mission report", "Backend-authoritative PDF, JSON, CSV, and ZIP"), requirement("Remote-sensing-adapted component", "RSICD captioning"),
] };
const health: AgentHealth = { status: "healthy", module: "satquery", router: "ready", registry: "ready", specialists: {} };

let container: HTMLDivElement;
let root: Root;
async function flush() { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); }
async function render(component: React.ReactNode) { await act(async () => { root.render(component); }); await flush(); }
async function click(name: string) { const target = Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name)); expect(target).toBeTruthy(); await act(async () => { target!.dispatchEvent(new MouseEvent("click", { bubbles: true })); await Promise.resolve(); }); await flush(); }

beforeEach(() => {
  sessionStorage.clear(); push.mockReset(); setTheme.mockReset(); Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.mocked(getAgentAnalytics).mockResolvedValue(analytics); vi.mocked(getAgentCompliance).mockResolvedValue(compliance); vi.mocked(getAgentHealth).mockResolvedValue(health); vi.mocked(getAgentTools).mockResolvedValue(tools);
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.clearAllMocks(); });

describe("remaining Presentation Mode stages", () => {
  it("renders live Research Analytics values, latest execution, trace, provenance, refresh, and full-page action", async () => {
    await render(createElement(PresentationAnalyticsScene, { onContinue: vi.fn() }));
    for (const value of ["Healthy", "Apple MPS", "9/9", "15/15", "3/32", "4 generated", "3", "25 ms", "Cross Modal Analysis", "Trace stages", "Compatible optical and SAR pair selected", "Cross Modal"]) expect(container.textContent).toContain(value);
    await click("Refresh"); expect(getAgentAnalytics).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("Open Full Analytics"); expect(container.textContent).toContain("Continue");
  });

  it("renders only the requested live architecture summary and diagram preview", async () => {
    await render(createElement(PresentationArchitectureScene, { onContinue: vi.fn() }));
    for (const value of ["System overview", "Agent controller", "Registry", "Routing", "Execution trace", "Evidence flow", "Pix2Pix", "SARFusionFormer", "Grounding", "Captioning", "VQA", "Reports", "Architecture diagram preview", "Open Full Architecture"]) expect(container.textContent).toContain(value);
    expect(getAgentAnalytics).toHaveBeenCalled(); expect(getAgentHealth).toHaveBeenCalled(); expect(getAgentTools).toHaveBeenCalled(); expect(getAgentCompliance).toHaveBeenCalled();
  });

  it("renders the authoritative compliance badge, optional grounding, registry, workflows, formats, and audit fields", async () => {
    await render(createElement(PresentationComplianceScene, { onContinue: vi.fn() }));
    for (const value of ["15/15", "Mandatory", "9/9 available", "Optional Available", "Optical", "Sar", "PNG", "JPEG", "TIFF", "GeoTIFF", "PDF · JSON · CSV · ZIP", "Evidence", "Confidence", "Execution trace", "RS adaptation", "Supported workflows", "Open Full Compliance"]) expect(container.textContent).toContain(value);
  });

  it("renders a professional closing with all requested capabilities and live readiness", async () => {
    await render(createElement(PresentationClosingScene));
    for (const value of ["Questions?", "Thank you", "Problem", "Solution", "Agentic workflow", "Remote-sensing adaptation", "Judging readiness", "Single Image", "Grounding", "Captioning", "VQA", "Change", "Cross Modal", "Reports", "Analytics", "Comparison", "Architecture", "Compliance", "Presentation Mode", "Backend Healthy · 9/9 specialists · 15/15 mandatory"]) expect(container.textContent).toContain(value);
  });

  it("shows truthful loading and backend failure states without substitute values", async () => {
    vi.mocked(getAgentAnalytics).mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await render(createElement(PresentationAnalyticsScene, { onContinue: vi.fn() }));
    expect(container.textContent).toContain("Live analytics unavailable"); expect(container.textContent).toContain("Failed to fetch");
  });

  it("ignores a stale analytics completion after leaving its stage", async () => {
    let resolveAnalytics!: (value: AnalyticsResponse) => void;
    vi.mocked(getAgentAnalytics).mockReturnValue(new Promise(resolve => { resolveAnalytics = resolve; }));
    await act(async () => { root.render(createElement(PresentationAnalyticsScene, { onContinue: vi.fn() })); await Promise.resolve(); });
    await act(async () => root.unmount()); resolveAnalytics(analytics); await flush(); expect(container.textContent).toBe(""); root = createRoot(container);
  });
});

describe("shared Presentation Mode page handoff", () => {
  it("stores stage, notes, theme, fullscreen, and route before same-tab navigation", async () => {
    sessionStorage.setItem(presentationStorageKeys.notes, "true"); document.documentElement.classList.add("light");
    await render(createElement(PresentationOpenPageButton, { route: "/assistant/analytics", stageIndex: 11, children: "Open Full Analytics" }));
    await click("Open Full Analytics");
    expect(sessionStorage.getItem(presentationStorageKeys.step)).toBe("11"); expect(sessionStorage.getItem(presentationStorageKeys.notes)).toBe("true"); expect(sessionStorage.getItem(presentationStorageKeys.handoffTheme)).toBe("light"); expect(sessionStorage.getItem(presentationStorageKeys.handoffRoute)).toBe("/assistant/analytics"); expect(push).toHaveBeenCalledWith("/assistant/analytics");
    document.documentElement.classList.remove("light");
  });

  it("shows a route-scoped return action and restores Stage 12 state safely", async () => {
    storePresentationHandoff(sessionStorage, { route: "/assistant/analytics", stageIndex: 11, theme: "dark", fullscreen: false });
    await render(createElement(PresentationReturnBanner, { route: "/assistant/analytics" }));
    expect(container.textContent).toContain("Opened from Presentation Mode · Stage 12"); expect(container.textContent).toContain("presenter notes, theme, session, and fullscreen preference");
    await click("Return to Presentation"); expect(setTheme).toHaveBeenCalledWith("dark"); expect(push).toHaveBeenCalledWith("/presentation"); expect(hasPresentationHandoff(sessionStorage, "/assistant/analytics")).toBe(false); expect(sessionStorage.getItem(presentationStorageKeys.step)).toBe("11");
  });

  it("does not show the return action on ordinary full-page visits", async () => {
    await render(createElement(PresentationReturnBanner, { route: "/architecture" }));
    expect(container.textContent).not.toContain("Return to Presentation");
  });
});

describe("remaining presentation layout contract", () => {
  const css = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");
  it("bounds all four stages for a 1024px projector", () => {
    for (const name of ["analytics", "architecture", "compliance", "closing"]) expect(css).toContain(`.presentation-${name}-scene`);
    expect(css).toContain("height: min(55dvh, 505px)");
  });
  it("stacks summaries on mobile and provides dark/light surfaces", () => {
    expect(css).toContain(".presentation-closing-layout { grid-template-columns: minmax(0, 1fr); }");
    expect(css).toContain("html.light .presentation-summary-panel,");
    expect(css).toContain("background: rgba(4,9,18,.48)");
  });
});

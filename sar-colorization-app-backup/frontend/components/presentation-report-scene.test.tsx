import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationReportScene } from "./presentation-report-scene";
import { latestSatQueryRequestKey } from "@/lib/presentation-report";
import { ApiRequestError, generateAgentReport, getComparisonItem } from "@/services/api";
import type { ComparisonItem, ReportResponse } from "@/types/agent";

vi.mock("@/services/api", async importOriginal => {
  const original = await importOriginal<typeof import("@/services/api")>();
  return { ...original, generateAgentReport: vi.fn(), getComparisonItem: vi.fn() };
});

function result(status = "success", evidence = 3): ComparisonItem {
  return {
    request_id: "request-reportable", task: "cross_modal", display_name: "Optical–SAR Joint Analysis", status,
    created_at: "2026-08-24T12:30:00Z", input_mode: "cross_modal", modalities: ["optical", "sar"],
    input_previews: [{ label: "Optical", url: "/optical.png", kind: "input", width: 128, height: 128, modality: "optical" }],
    output_previews: Array.from({ length: evidence }, (_, index) => ({ label: `Evidence ${index + 1}`, url: `/evidence-${index + 1}.png`, kind: "evidence", width: 128, height: 128, modality: null })),
    answer: "The modalities agree over measured candidate-evidence regions.",
    statistics: { cross_modal: { agreement_percent: 78.491, water_likelihood_percent: 11.694 } },
    confidence: { level: "moderate", reason: "Exactly aligned deterministic evidence." },
    provenance: { method: "deterministic optical-SAR evidence fusion" },
    selected_tools: ["input_validator", "cross_modal_optical_sar_analyzer"], execution_duration_ms: 17, device: "cpu",
    warnings: status === "partial" ? ["Partial evidence only."] : [], limitations: ["Likelihood maps are not semantic ground truth."],
    report_available: false, cached: false, input_identity: {}, lineage: [],
    execution_summary: { steps: [{ tool: "pair_compatibility_check" }, { tool: "joint_evidence_fusion" }] },
  };
}

const report: ReportResponse = {
  request_id: "request-reportable", status: "success", schema_version: "1.0",
  artifacts: [
    { format: "pdf", filename: "mission.pdf", url: "/api/agent/reports/mission.pdf", size_bytes: 3200 },
    { format: "json", filename: "mission.json", url: "/api/agent/reports/mission.json", size_bytes: 1800 },
    { format: "zip", filename: "mission.zip", url: "/api/agent/reports/mission.zip", size_bytes: 6400 },
  ],
  warnings: ["Artifacts are temporary."], runtime_ms: 26,
};

let container: HTMLDivElement;
let root: Root;

async function flush() { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); }
async function renderScene(requestId?: string) {
  if (requestId) sessionStorage.setItem(latestSatQueryRequestKey, requestId);
  await act(async () => { root.render(createElement(PresentationReportScene)); });
  await flush();
}
function button(name: string) { return Array.from(container.querySelectorAll("button")).find(item => item.textContent?.includes(name)) as HTMLButtonElement | undefined; }
async function click(element: Element | undefined) { expect(element).toBeTruthy(); await act(async () => { element!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true })); await Promise.resolve(); }); }

beforeEach(() => {
  sessionStorage.clear();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.mocked(getComparisonItem).mockResolvedValue(result());
  vi.mocked(generateAgentReport).mockResolvedValue(report);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("PresentationReportScene", () => {
  it("renders a clear empty state and never loads or reruns analysis without a latest request ID", async () => {
    await renderScene();
    expect(container.textContent).toContain("No reportable presentation result");
    expect(container.textContent).toContain("never starts an analysis automatically");
    expect(getComparisonItem).not.toHaveBeenCalled();
    expect(generateAgentReport).not.toHaveBeenCalled();
  });

  it("loads the latest authoritative result with identity, task, status, input mode, tools, evidence, and timestamp", async () => {
    await renderScene("request-reportable");
    expect(getComparisonItem).toHaveBeenCalledWith("request-reportable");
    expect(container.textContent).toContain("Optical–SAR Joint Analysis");
    expect(container.textContent).toContain("Cross Modal");
    expect(container.textContent).toContain("3 products");
    expect(container.textContent).toContain("request-reportable");
    expect(container.textContent).toContain("cross_modal_optical_sar_analyzer");
    expect(container.textContent).toContain("Success");
  });

  it("blocks report generation for failed results", async () => {
    vi.mocked(getComparisonItem).mockResolvedValue(result("failed"));
    await renderScene("failed-request");
    expect(container.textContent).toContain("Failed results are not reportable");
    expect(button("Generate Report")?.disabled).toBe(true);
    expect(generateAgentReport).not.toHaveBeenCalled();
  });

  it("supports independent PDF, JSON, and Full ZIP format selection", async () => {
    await renderScene("request-reportable");
    for (const label of ["PDF", "JSON", "Full ZIP"]) {
      const format = button(label)!;
      expect(format.getAttribute("aria-pressed")).toBe("true");
      await click(format);
      expect(format.getAttribute("aria-pressed")).toBe("false");
      await click(format);
      expect(format.getAttribute("aria-pressed")).toBe("true");
    }
  });

  it("starts generation only on click and blocks duplicate generation", async () => {
    await renderScene("request-reportable");
    expect(generateAgentReport).not.toHaveBeenCalled();
    let resolveReport!: (value: ReportResponse) => void;
    vi.mocked(generateAgentReport).mockReturnValue(new Promise(resolve => { resolveReport = resolve; }));
    const generate = button("Generate Report")!;
    await act(async () => {
      generate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      generate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    expect(generateAgentReport).toHaveBeenCalledTimes(1);
    expect(generateAgentReport).toHaveBeenCalledWith("request-reportable", ["pdf", "json", "zip"]);
    expect(button("Generating")?.disabled).toBe(true);
    await act(async () => { resolveReport(report); await Promise.resolve(); });
  });

  it("renders real generated artifact links, status, runtime, sizes, and report warnings", async () => {
    await renderScene("request-reportable");
    await click(button("Generate Report"));
    await flush();
    expect(container.textContent).toContain("Success · 26 ms");
    expect(container.textContent).toContain("Artifacts are temporary");
    expect(container.querySelector('a[download="mission.pdf"]')).toBeTruthy();
    expect(container.querySelector('a[download="mission.json"]')).toBeTruthy();
    expect(container.querySelector('a[download="mission.zip"]')).toBeTruthy();
    expect(container.textContent).toContain("Download PDF");
    expect(container.textContent).toContain("Download JSON");
    expect(container.textContent).toContain("Download ZIP");
  });

  it("renders the compact authoritative report summary including all declared sections", async () => {
    await renderScene("request-reportable");
    for (const section of ["Inputs", "Query", "Answer", "Evidence", "Statistics", "Confidence", "Provenance", "Execution trace", "Limitations"]) {
      expect(container.textContent).toContain(section);
    }
    expect(container.textContent).toContain("No browser-supplied statistics");
    expect(container.textContent).toContain("2 recorded steps");
  });

  it("shows expired-result and backend-offline states distinctly", async () => {
    vi.mocked(getComparisonItem).mockRejectedValueOnce(new ApiRequestError("Expired", "COMPARISON_RESULT_EXPIRED", 410));
    await renderScene("expired-request");
    expect(container.textContent).toContain("authoritative result is unavailable or expired");

    await act(async () => root.unmount());
    root = createRoot(container);
    vi.mocked(getComparisonItem).mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await renderScene("offline-request");
    expect(container.textContent).toContain("backend is offline");
  });

  it("allows partial results and clearly discloses missing evidence", async () => {
    vi.mocked(getComparisonItem).mockResolvedValue(result("partial", 0));
    await renderScene("partial-request");
    expect(container.textContent).toContain("Partial result");
    expect(container.textContent).toContain("no stored evidence products");
    expect(container.textContent).toContain("0 evidence products");
    expect(button("Generate Report")?.disabled).toBe(false);
  });

  it("surfaces report-generation and artifact-download failures", async () => {
    vi.mocked(generateAgentReport).mockRejectedValueOnce(new ApiRequestError("Source expired", "REPORT_SOURCE_EXPIRED", 404));
    await renderScene("request-reportable");
    await click(button("Generate Report"));
    await flush();
    expect(container.textContent).toContain("authoritative result is unavailable or expired");

    vi.mocked(generateAgentReport).mockResolvedValue(report);
    await click(button("Generate Report"));
    await flush();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 410 }));
    await click(container.querySelector('a[download="mission.pdf"]') ?? undefined);
    await flush();
    expect(container.textContent).toContain("artifact is unavailable or expired");
  });

  it("drops a stale generation completion after leaving the stage", async () => {
    await renderScene("request-reportable");
    let resolveReport!: (value: ReportResponse) => void;
    vi.mocked(generateAgentReport).mockReturnValue(new Promise(resolve => { resolveReport = resolve; }));
    await click(button("Generate Report"));
    await act(async () => root.unmount());
    resolveReport(report);
    await flush();
    expect(container.textContent).toBe("");
    root = createRoot(container);
  });
});

describe("Mission Reports presentation layout contract", () => {
  const css = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");

  it("keeps summary and controls visible in a bounded 1024px projector layout", () => {
    expect(css).toContain("@media (min-width: 1024px)");
    expect(css).toContain(".presentation-report-scene { height: min(55dvh, 505px); }");
    expect(css).toContain(".presentation-report-layout { height: 100%; grid-template-rows: minmax(0, 1fr) auto; }");
  });

  it("stacks the scene and keeps the contents overview accessible on mobile", () => {
    expect(css).toContain(".presentation-report-layout { grid-template-columns: minmax(0, 1fr); }");
    expect(css).toContain(".presentation-report-overview { overflow-x: auto; }");
  });

  it("provides explicit dark and light theme surfaces", () => {
    expect(css).toContain(".presentation-report-controls,");
    expect(css).toContain("background: rgba(4,9,18,.48)");
    expect(css).toContain("html.light .presentation-report-controls,");
    expect(css).toContain("background: rgba(248,250,252,.82)");
  });
});

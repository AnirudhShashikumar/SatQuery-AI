import { describe, expect, it } from "vitest";
import { changeEngineName, presentedSpecialists, workflowLabel } from "./scientific-presentation";
import type { AgentResponse, ChangeEngine, TTPResult } from "@/types/agent";

const engine = (overrides: Partial<ChangeEngine> = {}): ChangeEngine => ({
  mode: "hybrid", primary_tool: "changerex_change_detector", supporting_tool: "deterministic_change_analyzer",
  fallback_used: false, fallback_reason: null, ...overrides,
});

const result = (changeEngine: ChangeEngine): AgentResponse => ({
  execution: { selected_tools: [changeEngine.primary_tool, changeEngine.supporting_tool ?? ""].filter(Boolean), input_mode: "bi_temporal" },
  change_engine: changeEngine,
} as AgentResponse);

describe("scientific presentation labels", () => {
  it("presents ChangerEx as the default learned specialist and deterministic analysis as support", () => {
    expect(presentedSpecialists(result(engine()))).toEqual({
      primary: "ChangerEx Change Detector",
      supporting: ["Deterministic Change Analyzer"],
      fallbackReason: null,
    });
  });

  it("shows deterministic fallback with an explicit ChangerEx availability reason", () => {
    const fallback = engine({ mode: "deterministic_fallback", primary_tool: "deterministic_change_analyzer", supporting_tool: null, fallback_used: true, fallback_reason: "checkpoint_unavailable" });
    expect(presentedSpecialists(result(fallback))).toEqual({
      primary: "Deterministic Change Analyzer",
      supporting: [],
      fallbackReason: "ChangerEx unavailable · Checkpoint Unavailable",
    });
  });

  it("shows TTP only when the response identifies TTP as selected", () => {
    const selected = engine({ mode: "ttp", primary_tool: "ttp_change_detector" });
    expect(changeEngineName(selected, { model: "TTP" } as TTPResult)).toBe("TTP");
    expect(presentedSpecialists(result(selected)).primary).toBe("TTP Change Detector");
  });

  it("uses stable workflow terminology", () => {
    expect(workflowLabel("single")).toBe("Single Image");
    expect(workflowLabel("cross_modal")).toBe("Optical + SAR");
    expect(workflowLabel("bi_temporal")).toBe("Bi-temporal");
  });
});

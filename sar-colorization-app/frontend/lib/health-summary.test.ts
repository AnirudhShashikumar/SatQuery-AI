import { describe, expect, it } from "vitest";
import { presentedHealthStatus, summarizeAgentHealth } from "./health-summary";
import type { AgentHealth } from "@/types/agent";

describe("agent health presentation", () => {
  it("treats lazy local specialists as ready on demand", () => {
    expect(presentedHealthStatus("unloaded", true)).toBe("ready");
    expect(presentedHealthStatus("ready", true)).toBe("ready");
    expect(presentedHealthStatus("failed", true)).toBe("failed");
  });

  it("derives specialist, service, unavailable, and warning counts from one payload", () => {
    const health = { status: "ready", module: "satquery_agent", router: "ready", registry: "ready", specialists: {
      changerex_change_detector: { status: "unloaded", enabled: true, device: "mps", error: null },
      remote_sensing_vqa: { status: "ready", device: "mps", error: null, warnings: ["limited"] },
      ttp_change_service: { status: "disabled", enabled: false, device: null, error: null },
    } } satisfies AgentHealth;
    expect(summarizeAgentHealth(health)).toMatchObject({
      specialists: { ready: 2, total: 2 },
      services: { ready: 0, total: 1 },
      unavailableCount: 1,
      warningCount: 1,
      failedCount: 0,
    });
  });
});

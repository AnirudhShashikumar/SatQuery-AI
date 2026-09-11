import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { benchmarkData } from "@/lib/benchmarks";
import { SpecialistBenchmarkCard } from "./specialist-benchmark-card";

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
beforeEach(() => { container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container); Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); });
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

describe("specialist benchmark card", () => {
  it("renders measured model metadata, checkpoint wrapping, latency, memory, split, and timestamp", async () => {
    const record = benchmarkData.preferred.find(item => item.specialist_id === "grounding")!;
    await act(async () => root.render(<SpecialistBenchmarkCard record={record} history={benchmarkData.historyBySpecialist.grounding}/>));
    expect(container.textContent).toContain("Grounding DINO + Grounding Specialist");
    expect(container.textContent).toContain("VRSBench grounding_v2 · validation");
    expect(container.textContent).toContain("34.83 ms mean");
    expect(container.textContent).toContain("8374 MB peak");
    expect(container.querySelector(".benchmark-wrap")).toBeTruthy();
    expect(container.querySelector(".benchmark-details > summary")?.textContent).toContain("Protocol");
  });

  it("renders a scientifically explicit unavailable state", async () => {
    const record = benchmarkData.preferred.find(item => item.specialist_id === "captioning")!;
    await act(async () => root.render(<SpecialistBenchmarkCard record={record}/>));
    expect(container.textContent).toContain("Benchmark unavailable");
    expect(container.textContent).toContain("Smoke tests are not accuracy evaluations");
  });
});

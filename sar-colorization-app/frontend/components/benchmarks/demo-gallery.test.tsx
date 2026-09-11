import { act, createElement, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { benchmarkData } from "@/lib/benchmarks";

vi.mock("next/image", () => ({ default: ({ src, alt, ...props }: { src: string; alt: string }) => createElement("img", { src, alt, ...props }) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode }) => createElement("a", { href, ...props }, children) }));
import { DemoGallery } from "./demo-gallery";

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
beforeEach(() => { container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container); Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); });
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

describe("demo gallery", () => {
  it("renders real attributed assets and differentiates expected-result classes", async () => {
    await act(async () => root.render(<DemoGallery cases={benchmarkData.demos.cases}/>));
    expect(container.querySelectorAll("article")).toHaveLength(3);
    expect(container.textContent).toContain("Curated expected behavior");
    expect(container.textContent).toContain("Benchmark annotation");
    expect(container.textContent).toContain("USGS EROS Data Center");
    expect(container.textContent).not.toContain("will detect exactly");
  });

  it("uses guided real-workflow links without injecting fake results", async () => {
    await act(async () => root.render(<DemoGallery cases={benchmarkData.demos.cases}/>));
    const hrefs = Array.from(container.querySelectorAll<HTMLAnchorElement>("a.demo-launch")).map(anchor => anchor.getAttribute("href"));
    expect(hrefs).toContain("/assistant?mode=bi_temporal");
    expect(hrefs).toContain("/assistant?intent=grounding");
    expect(container.textContent).toContain("real application pipeline");
  });
});

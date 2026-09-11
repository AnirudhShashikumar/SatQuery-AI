import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "app/globals.css"), "utf8");

describe("Bi-Temporal presentation layout contract", () => {
  it("keeps a bounded projector layout at and above 1024px", () => {
    expect(css).toContain("@media (min-width: 1024px)");
    expect(css).toContain(".presentation-change-scene { display: grid; height: min(55dvh, 505px)");
    expect(css).toContain(".presentation-change-result { flex: 1 1 auto; }");
  });

  it("stacks the approved pair and releases result scrolling on mobile", () => {
    expect(css).toContain("@media (max-width: 639px)");
    expect(css).toContain(".presentation-change-source > div.grid { grid-template-columns: minmax(0, 1fr); }");
    expect(css).toContain(".presentation-change-result { max-height: none; overflow-y: visible; }");
  });

  it("declares explicit light-theme surfaces while retaining dark defaults", () => {
    expect(css).toContain(".presentation-change-workspace { border-color: rgba(255,255,255,.08)");
    expect(css).toContain("html.light .presentation-change-workspace { border-color: #dbe3ed");
    expect(css).toContain("html.light .presentation-change-result h2 { color: #111827; }");
  });
});

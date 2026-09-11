import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ path: "/assistant", search: "", setTheme: vi.fn() }));
vi.mock("next/navigation", () => ({ usePathname: () => mocks.path, useSearchParams: () => new URLSearchParams(mocks.search) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode; [key: string]: unknown }) => createElement("a", { href, ...props }, children) }));
vi.mock("next-themes", () => ({ useTheme: () => ({ resolvedTheme: "dark", setTheme: mocks.setTheme }) }));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: { models: { agent: { available: true } } }, isLoading: false, isSuccess: true }) }));

import { isSidebarItemActive, navigationGroups, normalizeRoutePath, primaryNavigation, Sidebar } from "./sidebar";

let container: HTMLDivElement;
let root: Root;

async function renderSidebar() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => { root.render(createElement(Sidebar)); await Promise.resolve(); });
}

beforeEach(() => {
  mocks.path = "/assistant";
  mocks.search = "";
  mocks.setTheme.mockClear();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
});
afterEach(async () => { if (root) await act(async () => root.unmount()); container?.remove(); });

describe("SatQuery sidebar route safety", () => {
  it("exposes the complete grouped product navigation in the intended order", () => {
    expect(navigationGroups.map(group => group.label)).toEqual(["Assistant", "Analysis", "Workspace", "Models", "Developer", "System"]);
    expect(primaryNavigation.map(item => item.label)).toEqual([
      "New Chat", "History", "Presentation Mode",
      "Single Image", "Optical + SAR", "Bi-temporal", "Grounding", "Mission Comparison",
      "Reports", "Benchmarks", "Research Analytics", "SIH Compliance",
      "SatQuery Vision Encoder", "Pix2Pix", "SARFusionFormer", "Change Detection", "Color Corrector",
      "API", "Logs", "Settings", "About",
    ]);
  });

  it("gives every visible item a non-empty, unique destination and route contract", () => {
    const hrefs = primaryNavigation.map(item => item.href);
    expect(primaryNavigation.every(item => item.href && item.expectedHeading && item.pageFile && item.icon)).toBe(true);
    expect(new Set(hrefs).size).toBe(hrefs.length);
    for (const item of primaryNavigation) expect(existsSync(resolve(process.cwd(), item.pageFile))).toBe(true);
  });

  it("keeps restored destinations separate from accidental route reuse", () => {
    const href = (label: string) => primaryNavigation.find(item => item.label === label)?.href;
    expect(href("Change Detection")).not.toBe(href("Presentation Mode"));
    expect(href("Pix2Pix")).not.toBe(href("SARFusionFormer"));
    expect(href("Mission Comparison")).not.toBe(href("Reports"));
    expect(href("SIH Compliance")).not.toBe(href("Benchmarks"));
  });

  it("matches exact, query, prefix, nested, and trailing-slash routes safely", () => {
    const item = (label: string) => primaryNavigation.find(entry => entry.label === label)!;
    expect(normalizeRoutePath("/models/pix2pix/")).toBe("/models/pix2pix");
    expect(isSidebarItemActive("/assistant/", item("New Chat"))).toBe(true);
    expect(isSidebarItemActive("/assistant", item("New Chat"), "view=history")).toBe(false);
    expect(isSidebarItemActive("/assistant", item("History"), "view=history")).toBe(true);
    expect(isSidebarItemActive("/assistant", item("Single Image"), "mode=single")).toBe(true);
    expect(isSidebarItemActive("/assistant", item("Grounding"), "intent=grounding")).toBe(true);
    expect(isSidebarItemActive("/presentation", item("Presentation Mode"), "step=8")).toBe(true);
    expect(isSidebarItemActive("/models/change-detection/details", item("Change Detection"))).toBe(true);
    expect(isSidebarItemActive("/models/change-detection", item("Presentation Mode"))).toBe(false);
    expect(isSidebarItemActive("/models/pix2pix", item("SARFusionFormer"))).toBe(false);
  });

  it("never marks more than one item active for canonical destinations", () => {
    for (const item of primaryNavigation) {
      const [path, search = ""] = item.href.split("?");
      const active = primaryNavigation.filter(candidate => isSidebarItemActive(path, candidate, search));
      expect(active.map(candidate => candidate.label), item.href).toEqual([item.label]);
    }
  });

  it("renders accessible branding, all mobile destinations, status, and current-page state", async () => {
    await renderSidebar();
    expect(container.querySelector('nav[aria-label="SatQuery AI navigation"]')).toBeTruthy();
    expect(container.querySelector('a[aria-label="SatQuery AI home"]')?.textContent).toContain("SatQueryAI");
    expect(container.textContent).toContain("SatQuery Agent Ready");
    expect(container.querySelectorAll(".sat-nav-link")).toHaveLength(primaryNavigation.length);
    expect(Array.from(container.querySelectorAll<HTMLAnchorElement>(".sat-nav-link")).map(link => link.getAttribute("href"))).toEqual(primaryNavigation.map(item => item.href));
    expect(container.querySelector('a[aria-current="page"]')?.textContent).toContain("New Chat");
    expect(container.querySelector('button[aria-label="Open navigation"]')).toBeTruthy();
  });

  it("keeps every restored item labelled and available after desktop collapse", async () => {
    await renderSidebar();
    const collapse = container.querySelector<HTMLButtonElement>('button[aria-label="Collapse sidebar"]')!;
    await act(async () => { collapse.click(); await Promise.resolve(); });
    expect(container.querySelector("aside")?.className).toContain("sat-sidebar-collapsed");
    for (const item of primaryNavigation) {
      const link = container.querySelector<HTMLAnchorElement>(`.sat-nav-link[href="${item.href}"]`);
      expect(link?.title).toBe(item.label);
      expect(link?.dataset.tooltip).toBe(item.label);
      expect(link?.getAttribute("aria-label")).toBe(item.label);
    }
  });

  it("traps mobile focus, closes with Escape, and supports theme switching", async () => {
    await renderSidebar();
    const menu = container.querySelector<HTMLButtonElement>('button[aria-label="Open navigation"]')!;
    await act(async () => { menu.click(); await Promise.resolve(); });
    const aside = container.querySelector("aside")!;
    expect(aside.className).toContain("sat-sidebar-open");
    const focusable = Array.from(aside.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    expect(document.activeElement).toBe(focusable[0]);
    await act(async () => { aside.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true })); });
    expect(document.activeElement).toBe(focusable[focusable.length - 1]);
    await act(async () => { aside.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true })); });
    expect(document.activeElement).toBe(focusable[0]);
    await act(async () => { aside.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); await Promise.resolve(); });
    expect(aside.className).not.toContain("sat-sidebar-open");
    const theme = container.querySelector<HTMLButtonElement>('button[aria-label="Use light theme"]')!;
    await act(async () => { theme.click(); await Promise.resolve(); });
    expect(mocks.setTheme).toHaveBeenCalledWith("light");
  });

  it("contains no stale visible legacy wordmark in the presentation shell", () => {
    const source = readFileSync(resolve(process.cwd(), "components/presentation-mode.tsx"), "utf8");
    expect(source).not.toMatch(/>\s*Geo\s*</i);
    expect(source).not.toMatch(/Geo<span/i);
  });
});

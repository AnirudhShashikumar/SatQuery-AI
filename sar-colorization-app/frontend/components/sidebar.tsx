"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Activity, BarChart3, BrainCircuit, Braces, ChevronDown, FileOutput, GalleryHorizontal, GitCompareArrows, HelpCircle, History, Images,
  Layers3, Menu, MessageSquarePlus, Moon, Palette, PanelLeftClose, PanelLeftOpen, Presentation, Radar,
  ScanSearch, ScrollText, Settings2, ShieldCheck, Sparkles, Sun, WandSparkles, X, type LucideIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { summarizeAgentHealth } from "@/lib/health-summary";
import { getAgentHealth } from "@/services/api";

export type NavigationItem = {
  href: string;
  label: string;
  icon: LucideIcon;
  match?: "exact" | "prefix";
  expectedHeading: string;
  pageFile: string;
};
export type NavigationGroup = { label: string; items: readonly NavigationItem[]; collapsible?: boolean };

export const navigationGroups: readonly NavigationGroup[] = [
  { label: "Assistant", items: [
    { href: "/assistant", label: "New Chat", icon: MessageSquarePlus, expectedHeading: "Ask Earth Anything.", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant?view=history", label: "History", icon: History, expectedHeading: "Conversation history", pageFile: "app/[workspace]/page.tsx" },
    { href: "/presentation", label: "Presentation Mode", icon: Presentation, expectedHeading: "Welcome to SatQuery AI", pageFile: "app/presentation/page.tsx" },
  ] },
  { label: "Analysis", items: [
    { href: "/assistant?mode=single", label: "Single Image", icon: Images, expectedHeading: "Ask Earth Anything.", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant?mode=cross_modal", label: "Optical + SAR", icon: Layers3, expectedHeading: "Ask Earth Anything.", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant?mode=bi_temporal", label: "Bi-temporal", icon: Sparkles, expectedHeading: "Ask Earth Anything.", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant?intent=grounding", label: "Grounding", icon: ScanSearch, expectedHeading: "Ask Earth Anything.", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant/compare", label: "Mission & Sensor Comparison", icon: GitCompareArrows, expectedHeading: "Mission Comparison", pageFile: "app/assistant/compare/page.tsx" },
  ] },
  { label: "Workspace", items: [
    { href: "/reports", label: "Reports", icon: FileOutput, expectedHeading: "Reports", pageFile: "app/[workspace]/page.tsx" },
    { href: "/benchmark", label: "Benchmarks", icon: BarChart3, expectedHeading: "Benchmarks", pageFile: "app/[workspace]/page.tsx" },
    { href: "/demos", label: "Demo Gallery", icon: GalleryHorizontal, expectedHeading: "Demo Gallery", pageFile: "app/[workspace]/page.tsx" },
    { href: "/assistant/analytics", label: "Research Analytics", icon: Activity, expectedHeading: "Research Analytics", pageFile: "app/assistant/analytics/page.tsx" },
    { href: "/assistant/compliance", label: "SIH Compliance", icon: ShieldCheck, expectedHeading: "Requirement Compliance", pageFile: "app/assistant/compliance/page.tsx" },
  ] },
  { label: "Models", collapsible: true, items: [
    { href: "/models/vision-encoder", label: "SatQuery Vision Encoder", icon: BrainCircuit, match: "prefix", expectedHeading: "SatQuery Vision Encoder", pageFile: "app/models/[model]/page.tsx" },
    { href: "/models/pix2pix", label: "Pix2Pix", icon: WandSparkles, match: "prefix", expectedHeading: "Pix2Pix", pageFile: "app/models/[model]/page.tsx" },
    { href: "/models/sarfusionformer", label: "SARFusionFormer", icon: Radar, match: "prefix", expectedHeading: "SARFusionFormer", pageFile: "app/models/[model]/page.tsx" },
    { href: "/models/change-detection", label: "Change Detection", icon: ScanSearch, match: "prefix", expectedHeading: "Change Detection", pageFile: "app/models/[model]/page.tsx" },
    { href: "/models/color-corrector", label: "Color Corrector", icon: Palette, match: "prefix", expectedHeading: "Color Corrector", pageFile: "app/models/[model]/page.tsx" },
  ] },
  { label: "Developer", collapsible: true, items: [
    { href: "/api", label: "API", icon: Braces, expectedHeading: "Local API", pageFile: "app/[workspace]/page.tsx" },
    { href: "/logs", label: "Logs", icon: ScrollText, expectedHeading: "Execution logs", pageFile: "app/[workspace]/page.tsx" },
  ] },
  { label: "System", collapsible: true, items: [
    { href: "/settings", label: "Settings", icon: Settings2, expectedHeading: "AI Providers", pageFile: "app/[workspace]/page.tsx" },
    { href: "/about", label: "About", icon: HelpCircle, expectedHeading: "SatQuery AI", pageFile: "app/[workspace]/page.tsx" },
  ] },
] as const;

export const primaryNavigation: readonly NavigationItem[] = navigationGroups.flatMap(group => group.items);
const assistantRouteParameters = ["view", "mode", "intent"] as const;

export function normalizeRoutePath(path: string): string {
  if (!path || path === "/") return "/";
  return `/${path.split("/").filter(Boolean).join("/")}`;
}

export function isSidebarItemActive(path: string, hrefOrItem: string | NavigationItem, currentSearch = ""): boolean {
  const item = typeof hrefOrItem === "string" ? { href: hrefOrItem, match: "exact" as const } : hrefOrItem;
  const [rawRoute, expectedSearch = ""] = item.href.split("?");
  const route = normalizeRoutePath(rawRoute);
  const currentPath = normalizeRoutePath(path);
  const expectedParameters = new URLSearchParams(expectedSearch);
  const activeParameters = new URLSearchParams(currentSearch.replace(/^\?/, ""));

  if (expectedSearch) {
    if (currentPath !== route) return false;
    if (!Array.from(expectedParameters).every(([key, value]) => activeParameters.get(key) === value)) return false;
    return assistantRouteParameters.every(parameter => !activeParameters.has(parameter) || expectedParameters.get(parameter) === activeParameters.get(parameter));
  }
  if (route === "/assistant" && currentPath === route) return !assistantRouteParameters.some(parameter => activeParameters.has(parameter));
  return item.match === "prefix" ? currentPath === route || currentPath.startsWith(`${route}/`) : currentPath === route;
}

export function Sidebar() {
  const path = usePathname();
  const searchParameters = useSearchParams();
  const currentSearch = searchParameters.toString();
  const { resolvedTheme, setTheme } = useTheme();
  const health = useQuery({ queryKey: ["agent-health"], queryFn: () => getAgentHealth(), retry: false, refetchInterval: 30_000 });
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [themeMounted, setThemeMounted] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => setMobileOpen(false), [path, currentSearch]);
  useEffect(() => setThemeMounted(true), []);
  useEffect(() => {
    if (!mobileOpen || !sidebarRef.current) return;
    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const sidebar = sidebarRef.current;
    const focusable = () => Array.from(sidebar.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    focusable()[0]?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); setMobileOpen(false); return; }
      if (event.key !== "Tab") return;
      const elements = focusable();
      if (!elements.length) return;
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    sidebar.addEventListener("keydown", onKeyDown);
    return () => { sidebar.removeEventListener("keydown", onKeyDown); previousFocusRef.current?.focus(); };
  }, [mobileOpen]);

  const healthSnapshot = summarizeAgentHealth(health.data);
  const isOnline = health.isSuccess;
  const darkTheme = !themeMounted || resolvedTheme === "dark";

  return <>
    <header className="sat-mobile-header">
      <Link href="/" className="sat-brand" aria-label="SatQuery AI home"><BrandMark/><BrandName/></Link>
      <button type="button" className="sat-icon-button" onClick={() => setMobileOpen(open => !open)} aria-expanded={mobileOpen} aria-controls="satquery-navigation" aria-label={mobileOpen ? "Close navigation" : "Open navigation"}>{mobileOpen ? <X size={19}/> : <Menu size={19}/>}</button>
    </header>
    {mobileOpen && <button type="button" className="sat-nav-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)}/>}
    <aside ref={sidebarRef} id="satquery-navigation" aria-label="Primary navigation drawer" className={cn("sat-sidebar", mobileOpen && "sat-sidebar-open", collapsed && "sat-sidebar-collapsed")}>
      <div className="sat-sidebar-top">
        <Link href="/" className="sat-brand" aria-label="SatQuery AI home" title={collapsed ? "SatQuery AI home" : undefined}><BrandMark/><BrandName/></Link>
        <span className="sat-version">Research cloud</span>
        <button type="button" className="sat-drawer-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18}/></button>
        <button type="button" className="sat-collapse-toggle" onClick={() => setCollapsed(value => !value)} aria-expanded={!collapsed} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <PanelLeftOpen size={17}/> : <PanelLeftClose size={17}/>}</button>
      </div>
      <nav className="sat-nav" aria-label="SatQuery AI navigation">
        {navigationGroups.map(group => <SidebarNavigationGroup key={group.label} group={group} path={path} currentSearch={currentSearch}/>)}
      </nav>
      <div className="sat-sidebar-bottom">
        <div className="sat-agent-status" title={health.isSuccess ? `${healthSnapshot.specialists.ready} of ${healthSnapshot.specialists.total} specialists ready · ${healthSnapshot.services.ready} of ${healthSnapshot.services.total} services ready · ${healthSnapshot.unavailableCount} unavailable · ${healthSnapshot.warningCount} warnings` : "Checking agent health"}>
          <span className={cn("sat-agent-dot", isOnline && "sat-agent-dot-online")}/>
          <span><strong>{isOnline ? "SatQuery Agent Ready" : health.isLoading ? "Connecting to agent" : "Agent unavailable"}</strong><small>{health.isSuccess ? `${healthSnapshot.specialists.ready}/${healthSnapshot.specialists.total} specialists ready` : "Local intelligence runtime"}</small></span>
        </div>
        <button type="button" className="sat-theme-toggle" onClick={() => setTheme(darkTheme ? "light" : "dark")} aria-label={`Use ${darkTheme ? "light" : "dark"} theme`} title={collapsed ? `Use ${darkTheme ? "light" : "dark"} theme` : undefined}>{darkTheme ? <Sun size={15}/> : <Moon size={15}/>}<span>{darkTheme ? "Light interface" : "Dark interface"}</span></button>
        <p className="sat-sidebar-footnote">Multimodal remote sensing intelligence</p>
      </div>
    </aside>
  </>;
}

function BrandMark() { return <span className="sat-brand-mark" aria-hidden="true"><span/><span/><span/></span>; }
function BrandName() { return <span className="sat-brand-name"><strong>SatQuery</strong><span>AI</span></span>; }

function SidebarNavigationGroup({ group, path, currentSearch }: { group: NavigationGroup; path: string; currentSearch: string }) {
  const groupId = `sat-nav-${group.label.toLowerCase().replaceAll(" ", "-")}`;
  const hasActiveItem = group.items.some(item => isSidebarItemActive(path, item, currentSearch));
  const [open, setOpen] = useState(hasActiveItem);
  useEffect(() => { if (hasActiveItem) setOpen(true); }, [hasActiveItem]);
  const links = group.items.map(item => <SidebarLink key={`${group.label}-${item.label}`} item={item} active={isSidebarItemActive(path, item, currentSearch)}/>);
  if (!group.collapsible) return <section className="sat-nav-group" aria-labelledby={groupId}><p className="sat-nav-label" id={groupId}>{group.label}</p>{links}</section>;
  return <details className="sat-nav-group sat-nav-disclosure" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary id={groupId}><span>{group.label}</span><ChevronDown size={13}/></summary>
    <div className="sat-nav-disclosure-items">{links}</div>
  </details>;
}

function SidebarLink({ item, active }: { item: NavigationItem; active: boolean }) {
  const Icon = item.icon;
  return <Link href={item.href} title={item.label} data-tooltip={item.label} aria-label={item.label} aria-current={active ? "page" : undefined} className={cn("sat-nav-link", active && "sat-nav-link-active")}><Icon size={17}/><span>{item.label}</span>{active && <i aria-hidden="true"/>}</Link>;
}

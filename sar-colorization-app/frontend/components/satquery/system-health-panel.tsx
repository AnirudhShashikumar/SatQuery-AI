"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, ChevronDown, CircleAlert, LoaderCircle, ServerCog } from "lucide-react";
import { getAgentHealth } from "@/services/api";
import { presentedHealthStatus, summarizeAgentHealth } from "@/lib/health-summary";
import { cn } from "@/lib/utils";

const titleCase = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
const healthLabel = (name: string) => {
  const lower = name.toLowerCase();
  if (lower.includes("ttp")) return "Optional Alternate Change Service";
  if (lower === "rs_captioner") return "Remote Sensing Captioner";
  if (lower === "rs_grounder") return "Grounding DINO";
  if (lower.includes("grounding_specialist")) return "Grounding Specialist v1.1";
  if (lower.includes("vision_encoder")) return "SatQuery Vision Encoder";
  if (lower.includes("rsvqa")) return "RSVQA Specialist";
  if (lower.includes("sar_translation")) return "SAR Translation Service";
  if (lower.includes("changerex")) return "ChangerEx Change Detector";
  return titleCase(name);
};
const healthError = (name: string, error: string | null) => name.toLowerCase().includes("ttp") ? error?.replace(/\bTTP\b/gi, "alternate change service") : error;
const statusText = (name: string, status: string, smokeVerified?: boolean | null) => {
  if ((name === "rs_grounder" || name.includes("grounding_specialist")) && status !== "ready") return "Grounding specialist unavailable";
  if (name === "rs_grounder" && smokeVerified !== true) return "Grounding specialist unavailable";
  return titleCase(status);
};

export function SystemHealthPanel() {
  const health = useQuery({ queryKey: ["agent-health"], queryFn: () => getAgentHealth(), retry: false, refetchInterval: 30_000 });
  const snapshot = summarizeAgentHealth(health.data);
  const summary = health.isLoading ? "Checking specialists" : health.isError ? "Health unavailable" : `${snapshot.specialists.ready}/${snapshot.specialists.total} specialists ready`;

  return <details className="sq-health-panel">
    <summary>
      <span className={cn("sq-health-indicator", health.isSuccess && "is-ready", health.isError && "is-failed")}>{health.isLoading ? <LoaderCircle size={14} className="animate-spin"/> : health.isError ? <CircleAlert size={14}/> : <Activity size={14}/>}</span>
      <span><strong>System health</strong><small>{summary}</small></span>
      <ChevronDown size={15} aria-hidden="true"/>
    </summary>
    <div className="sq-health-content">
      {!health.isError && <dl className="sq-health-summary" aria-label="System health summary"><div><dt>Specialists ready</dt><dd>{snapshot.specialists.ready} / {snapshot.specialists.total}</dd></div><div><dt>Services ready</dt><dd>{snapshot.services.ready} / {snapshot.services.total}</dd></div><div><dt>Unavailable</dt><dd>{snapshot.unavailableCount}</dd></div><div><dt>Warnings</dt><dd>{snapshot.warningCount}</dd></div></dl>}
      {health.isError ? <p role="status">Agent health is temporarily unavailable. No readiness values are inferred.</p> : snapshot.entries.length ? <ul>{snapshot.entries.map(([name, specialist]) => {
        const status = presentedHealthStatus(specialist.status, specialist.enabled);
        return <li key={name}>
          <span className={cn("sq-health-dot", `is-${status}`)} aria-hidden="true"/>
          <span><strong>{healthLabel(name)}</strong><small>{specialist.device ? `${statusText(name, status, specialist.smoke_verified)} · ${specialist.device}` : statusText(name, status, specialist.smoke_verified)}{specialist.load_source ? ` · ${titleCase(specialist.load_source)}` : ""}{healthError(name, specialist.last_error ?? specialist.error) ? ` · ${healthError(name, specialist.last_error ?? specialist.error)}` : ""}</small></span>
          {(specialist.reuse_count ?? specialist.load_count) != null && <em>{specialist.reuse_count ? `${specialist.reuse_count} reuses` : `${specialist.load_count} loads`}</em>}
        </li>;
      })}</ul> : health.isLoading ? <p>Reading the existing agent health endpoint…</p> : <p>No specialist details were published.</p>}
      <p className="sq-health-note"><ServerCog size={13}/>Ready-on-demand specialists load on their first compatible request.</p>
    </div>
  </details>;
}

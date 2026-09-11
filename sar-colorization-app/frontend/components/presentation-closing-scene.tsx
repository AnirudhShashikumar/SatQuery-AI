"use client";

import { Check, CircleCheckBig, LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { PresentationOpenPageButton } from "@/components/presentation-page-handoff";
import { analyticsLabel } from "@/lib/analytics";
import { getAgentAnalytics } from "@/services/api";
import type { AnalyticsResponse } from "@/types/agent";

const completedCapabilities = ["Single Image", "Grounding", "Captioning", "VQA", "Change", "Cross Modal", "Reports", "Analytics", "Comparison", "Architecture", "Compliance", "Presentation Mode"];
const missionSummary = ["Problem", "Solution", "Agentic workflow", "Remote-sensing adaptation", "Evidence", "Reports", "Compliance", "Live demonstrations", "Judging readiness"];

export function PresentationClosingScene() {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    getAgentAnalytics().then(value => { if (mountedRef.current) setAnalytics(value); }).catch(() => void 0);
    return () => { mountedRef.current = false; };
  }, []);

  return <section className="presentation-closing-scene mt-4 min-h-0 flex-1" aria-label="SatQuery AI presentation closing">
    <div className="presentation-closing-layout grid min-h-0 gap-3 lg:grid-cols-[minmax(0,1.05fr)_minmax(320px,.95fr)]">
      <article className="presentation-closing-hero relative grid min-h-64 place-items-center overflow-hidden rounded-3xl border p-6 text-center"><div className="presentation-closing-halo" aria-hidden="true"/><div className="relative z-10"><CircleCheckBig className="mx-auto text-sky-300" size={42} strokeWidth={1.4}/><p className="mt-4 text-[9px] font-bold uppercase tracking-[.24em] text-sky-300">Complete guided walkthrough</p><h2 className="mt-2 text-[clamp(2.4rem,6vw,5.5rem)] font-semibold leading-none tracking-[-.05em]">Questions?</h2><p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-zinc-400">SatQuery AI connects validated Earth-observation inputs to specialist execution, inspectable evidence, reproducible reports, and transparent compliance.</p><div className="mt-4 flex flex-wrap items-center justify-center gap-2"><span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-2 text-[9px] ${analytics?.platform.backend_status === "healthy" ? "border-emerald-300/25 text-emerald-200" : "border-white/[.08] text-zinc-500"}`}>{analytics ? <Check size={11}/> : <LoaderCircle className="animate-spin" size={11}/>} {analytics ? `Backend ${analyticsLabel(analytics.platform.backend_status)} · ${analytics.summary.available_tools}/${analytics.summary.registered_tools} specialists · ${analytics.summary.mandatory_satisfied}/${analytics.summary.mandatory_total} mandatory` : "Reading live platform readiness"}</span><PresentationOpenPageButton route="/assistant" stageIndex={14}>Open SatQuery AI Assistant</PresentationOpenPageButton></div><p className="mt-5 text-xs font-medium uppercase tracking-[.2em] text-zinc-500">Thank you</p></div></article>
      <article className="presentation-summary-panel min-h-0 rounded-3xl border p-4"><p className="text-[8px] font-bold uppercase tracking-[.18em] text-zinc-500">Judging-ready platform</p><h2 className="mt-1 text-base font-semibold">Complete SatQuery AI walkthrough</h2><div className="mt-3 flex flex-wrap gap-1.5">{missionSummary.map(value => <span key={value} className="rounded-lg border border-sky-300/10 bg-sky-300/[.035] px-2 py-1 text-[8px] text-sky-100">{value}</span>)}</div><div className="mt-4 grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">{completedCapabilities.map(value => <div key={value} className="flex items-center gap-1.5 rounded-xl border border-white/[.055] bg-white/[.02] px-2 py-2 text-[8px] font-medium text-zinc-300"><span className="grid h-4 w-4 shrink-0 place-items-center rounded-full bg-emerald-300/[.10] text-emerald-300"><Check size={9}/></span>{value}</div>)}</div><p className="mt-4 text-[9px] leading-4 text-zinc-500">The presentation preserves scientific cautions: evidence products remain inspectable, model outputs are not called ground truth, and unsupported scope stays explicit.</p></article>
    </div>
  </section>;
}

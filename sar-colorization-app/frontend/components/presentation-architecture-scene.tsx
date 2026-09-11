"use client";

import { AlertTriangle, ArrowRight, Boxes, CheckCircle2, Cpu, Database, FileText, GitBranch, LoaderCircle, RefreshCw, Route, ScanSearch, Wrench } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { PresentationOpenPageButton } from "@/components/presentation-page-handoff";
import { activePathSummary, architectureLabel } from "@/lib/architecture";
import { getAgentAnalytics, getAgentCompliance, getAgentHealth, getAgentTools } from "@/services/api";
import type { AgentHealth, AnalyticsResponse, ComplianceResponse, ToolDefinition } from "@/types/agent";

type ArchitectureData = { analytics: AnalyticsResponse; health: AgentHealth; tools: ToolDefinition[]; compliance: ComplianceResponse };

export function PresentationArchitectureScene({ onContinue }: { onContinue: () => void }) {
  const [data, setData] = useState<ArchitectureData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const mountedRef = useRef(true);
  const sequenceRef = useRef(0);
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; sequenceRef.current += 1; }; }, []);
  const load = useCallback(async () => {
    const sequence = ++sequenceRef.current; setLoading(true); setError("");
    try {
      const [analytics, health, tools, compliance] = await Promise.all([getAgentAnalytics(), getAgentHealth(), getAgentTools(), getAgentCompliance()]);
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      setData({ analytics, health, tools, compliance });
    } catch (caught) {
      if (mountedRef.current && sequence === sequenceRef.current) setError(caught instanceof Error ? caught.message : "Live architecture data is unavailable.");
    } finally { if (mountedRef.current && sequence === sequenceRef.current) setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const latest = data?.analytics.recent_executions[0];
  const tool = (id: string) => data?.analytics.tools.find(item => item.id === id);
  const summaryItems = data ? [
    { label: "System overview", value: `${architectureLabel(data.analytics.platform.backend_status)} · ${data.analytics.platform.hardware_acceleration}`, icon: Boxes },
    { label: "Agent controller", value: architectureLabel(data.health.router), icon: GitBranch },
    { label: "Registry", value: `${architectureLabel(data.health.registry)} · ${data.analytics.summary.available_tools}/${data.analytics.summary.registered_tools}`, icon: Database },
    { label: "Routing", value: latest?.selection_reason ?? "No completed route in this process", icon: Route },
    { label: "Execution trace", value: latest ? `${latest.trace.length} observable stages` : "No completed trace", icon: Route },
    { label: "Evidence flow", value: latest ? `${latest.output_count} evidence products` : "No completed evidence flow", icon: ScanSearch },
    { label: "Pix2Pix", value: architectureToolStatus(tool("pix2pix_reconstruction")), icon: Cpu },
    { label: "SARFusionFormer", value: architectureToolStatus(tool("sarfusionformer_analysis")), icon: Cpu },
    { label: "Grounding", value: architectureToolStatus(tool("rs_grounder")), icon: ScanSearch },
    { label: "Captioning", value: architectureToolStatus(tool("rs_captioner")), icon: FileText },
    { label: "VQA", value: architectureToolStatus(tool("rs_vqa")), icon: Wrench },
    { label: "Reports", value: architectureToolStatus(tool("report_generator")), icon: FileText },
  ] : [];

  return <section className="presentation-architecture-scene mt-4 min-h-0 flex-1" aria-label="Architecture presentation summary">
    <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-[9px] font-bold uppercase tracking-[.18em] text-sky-300">Live architecture summary</p><div className="flex flex-wrap gap-2"><button type="button" onClick={() => void load()} disabled={loading} className="presentation-summary-secondary"><RefreshCw size={12} className={loading ? "animate-spin" : ""}/>Refresh</button><PresentationOpenPageButton route="/architecture" stageIndex={12}>Open Full Architecture</PresentationOpenPageButton><button type="button" onClick={onContinue} className="presentation-summary-secondary">Continue</button></div></div>
    {loading && !data && <ArchitectureState icon={LoaderCircle} title="Loading architecture" message="Reading the live controller, registry, execution, and compliance state." loading/>}
    {error && !data && <ArchitectureState icon={AlertTriangle} title="Architecture unavailable" message={`${error} No substitute execution path is shown.`}/>} 
    {data && <div className="presentation-architecture-layout mt-3 grid min-h-0 gap-3 lg:grid-cols-[minmax(0,1.18fr)_minmax(260px,.82fr)]">
      <article className="presentation-summary-panel min-h-0 rounded-2xl border p-3"><div className="flex items-center justify-between gap-2"><div><p className="text-[8px] font-bold uppercase tracking-[.15em] text-zinc-500">System overview</p><h2 className="mt-1 text-sm font-semibold">Inspectable platform layers</h2></div><span className="rounded-full border border-emerald-300/25 px-2 py-1 text-[8px] text-emerald-200">{data.compliance.mandatory_satisfied}/{data.compliance.mandatory_total} governed</span></div><div className="presentation-architecture-items mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-3">{summaryItems.map(({ label, value, icon: Icon }) => <div key={label} className="min-w-0 rounded-xl border border-white/[.05] bg-white/[.02] p-2"><p className="flex items-center gap-1 text-[7px] uppercase tracking-wide text-zinc-600"><Icon size={9}/>{label}</p><p className="mt-1 line-clamp-2 text-[8px] font-medium leading-3 text-zinc-300" title={value}>{value}</p></div>)}</div></article>
      <article className="presentation-summary-panel flex min-h-0 flex-col rounded-2xl border p-3"><div><p className="text-[8px] font-bold uppercase tracking-[.15em] text-zinc-500">Architecture diagram preview</p><h2 className="mt-1 text-sm font-semibold">Observable evidence path</h2></div><div className="mt-3 grid flex-1 content-center gap-1.5" aria-label="Architecture diagram preview">{[
        ["Interface", "Validated imagery and declared task"],
        ["Agent controller", data.health.router],
        ["Registry + routing", `${data.analytics.summary.available_tools} available specialists`],
        ["Specialist execution", latest?.selected_tools.join(" · ") || "Awaiting a completed workflow"],
        ["Evidence + report", latest ? `${latest.output_count} outputs · ${latest.report_generated ? "report generated" : "report not generated"}` : "Awaiting observable evidence"],
      ].map(([label, value], index, values) => <div key={label}><div className="rounded-xl border border-sky-300/12 bg-sky-300/[.035] px-3 py-2"><p className="text-[8px] font-semibold text-sky-200">{label}</p><p className="mt-0.5 line-clamp-1 text-[8px] text-zinc-500">{architectureLabel(value)}</p></div>{index < values.length - 1 && <ArrowRight className="mx-auto my-0.5 rotate-90 text-sky-400/45" size={11}/>}</div>)}</div><p className="mt-2 line-clamp-2 text-[8px] leading-3 text-zinc-500">{activePathSummary(latest)}</p></article>
    </div>}
  </section>;
}

function architectureToolStatus(tool: AnalyticsResponse["tools"][number] | undefined): string { return tool ? `${architectureLabel(tool.implementation_status)} · ${architectureLabel(tool.lifecycle_status)}` : "Not declared"; }
function ArchitectureState({ icon: Icon, title, message, loading = false }: { icon: typeof CheckCircle2; title: string; message: string; loading?: boolean }) { return <div className="presentation-summary-panel mt-3 grid min-h-56 place-items-center rounded-2xl border p-5 text-center"><div><Icon className={`mx-auto ${loading ? "animate-spin text-sky-300" : "text-rose-200"}`} size={20}/><h2 className="mt-2 text-sm font-semibold">{title}</h2><p className="mt-1 text-[10px] text-zinc-500">{message}</p></div></div>; }

"use client";

import { Activity, AlertTriangle, Clock3, Cpu, Database, FileArchive, LoaderCircle, RefreshCw, Route, ShieldCheck, Wrench } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { PresentationOpenPageButton } from "@/components/presentation-page-handoff";
import { analyticsLabel, formatAnalyticsDuration } from "@/lib/analytics";
import { getAgentAnalytics } from "@/services/api";
import type { AnalyticsResponse } from "@/types/agent";

export function PresentationAnalyticsScene({ onContinue }: { onContinue: () => void }) {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const mountedRef = useRef(true);
  const sequenceRef = useRef(0);

  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; sequenceRef.current += 1; }; }, []);
  const load = useCallback(async () => {
    const sequence = ++sequenceRef.current;
    setLoading(true); setError("");
    try {
      const response = await getAgentAnalytics();
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      setData(response);
    } catch (caught) {
      if (mountedRef.current && sequence === sequenceRef.current) setError(caught instanceof Error ? caught.message : "Research analytics are unavailable.");
    } finally {
      if (mountedRef.current && sequence === sequenceRef.current) setLoading(false);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const latest = data?.recent_executions[0];
  const provenance = latest ? latest.selected_tools.map(id => data?.tools.find(tool => tool.id === id)).filter(Boolean) : [];

  return <section className="presentation-analytics-scene mt-4 min-h-0 flex-1" aria-label="Research Analytics presentation summary">
    <SummaryHeader eyebrow="Live operational evidence" loading={loading} onRefresh={() => void load()} onContinue={onContinue} open={<PresentationOpenPageButton route="/assistant/analytics" stageIndex={11}>Open Full Analytics</PresentationOpenPageButton>}/>
    {loading && !data && <Loading label="Loading live backend analytics…"/>}
    {error && !data && <Failure message={error} retry={() => void load()}/>} 
    {data && <div className="presentation-analytics-layout mt-3 grid min-h-0 gap-3 lg:grid-cols-[minmax(0,.95fr)_minmax(0,1.05fr)]">
      <article className="presentation-summary-panel min-h-0 rounded-2xl border p-3">
        <div className="flex items-center justify-between gap-2"><div><p className="text-[8px] font-bold uppercase tracking-[.16em] text-zinc-500">Current backend process</p><h2 className="mt-1 text-sm font-semibold">Platform status</h2></div><span className={`rounded-full border px-2 py-1 text-[8px] font-semibold ${data.platform.backend_status === "healthy" ? "border-emerald-300/25 text-emerald-200" : "border-amber-300/25 text-amber-100"}`}>{analyticsLabel(data.platform.backend_status)}</span></div>
        <dl className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-2 xl:grid-cols-4">
          <AnalyticsFact icon={Cpu} label="Active device" value={data.platform.hardware_acceleration}/>
          <AnalyticsFact icon={Wrench} label="Specialists" value={`${data.summary.available_tools}/${data.summary.registered_tools}`}/>
          <AnalyticsFact icon={ShieldCheck} label="Compliance" value={`${data.summary.mandatory_satisfied}/${data.summary.mandatory_total}`}/>
          <AnalyticsFact icon={Database} label="Cache" value={`${data.cache.stored_results}/${data.cache.max_results}`}/>
          <AnalyticsFact icon={FileArchive} label="Reports" value={`${data.summary.report_artifacts_generated_current_process} generated`}/>
          <AnalyticsFact icon={Activity} label="Executions" value={String(data.summary.total_executions_current_process)}/>
          <AnalyticsFact icon={Clock3} label="Latest runtime" value={latest ? formatAnalyticsDuration(latest.duration_ms) : "Unavailable"}/>
          <AnalyticsFact icon={Route} label="Trace stages" value={latest ? String(latest.trace.length) : "0"}/>
          <AnalyticsFact icon={Cpu} label="SVE lifecycle" value={data.sve ? `${analyticsLabel(data.sve.lifecycle_status)} · ${data.sve.inference_count} runs` : "Unavailable"}/>
        </dl>
        <div className="mt-2 rounded-xl border border-white/[.06] bg-white/[.02] p-2.5"><p className="text-[8px] uppercase tracking-wide text-zinc-500">Latest workflow</p>{latest ? <><p className="mt-1 text-[10px] font-semibold text-zinc-200">{analyticsLabel(latest.task)} · {analyticsLabel(latest.status)}</p><p className="mt-1 truncate font-mono text-[8px] text-zinc-600" title={latest.request_id}>{latest.request_id}</p></> : <p className="mt-1 text-[10px] text-zinc-500">No completed workflow in this backend process.</p>}</div>
      </article>

      <article className="presentation-summary-panel flex min-h-0 flex-col rounded-2xl border p-3">
        <div><p className="text-[8px] font-bold uppercase tracking-[.16em] text-zinc-500">Latest observable execution</p><h2 className="mt-1 text-sm font-semibold">Trace and provenance</h2></div>
        {!latest ? <div className="mt-3 grid flex-1 place-items-center rounded-xl border border-white/[.06] p-4 text-center text-[10px] leading-4 text-zinc-500">Run a workflow from SatQuery Assistant to populate measured execution, trace, and provenance values.</div> : <div className="mt-2 grid min-h-0 flex-1 gap-2 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,.9fr)]">
          <ol className="presentation-analytics-trace min-h-0 space-y-1 overflow-auto rounded-xl border border-white/[.06] p-2">{latest.trace.map((step, index) => <li key={`${step.tool}-${index}`} className="flex items-center justify-between gap-2 rounded-lg bg-white/[.025] px-2 py-1.5"><span className="min-w-0 truncate text-[9px] text-zinc-300"><strong className="mr-1 text-sky-300">{index + 1}</strong>{analyticsLabel(step.tool)}</span><span className="shrink-0 text-[8px] tabular-nums text-zinc-500">{step.duration_ms} ms · {analyticsLabel(step.status)}</span></li>)}</ol>
          <div className="min-w-0 rounded-xl border border-white/[.06] p-2"><p className="text-[8px] uppercase tracking-wide text-zinc-500">Provenance summary</p><p className="mt-1.5 text-[9px] leading-4 text-zinc-400">{latest.selection_reason ?? "This specialist execution did not publish a routing rationale."}</p><div className="mt-2 flex flex-wrap gap-1">{provenance.length ? provenance.map(tool => tool && <span key={tool.id} className="rounded-lg bg-sky-400/[.08] px-2 py-1 font-mono text-[8px] text-sky-200">{tool.display_name}{tool.remote_sensing_adapted ? " · RS adapted" : ""}</span>) : <span className="text-[8px] text-zinc-600">No selected-tool provenance recorded.</span>}</div><p className="mt-2 text-[8px] leading-3 text-zinc-500">Generated {new Date(data.generated_at).toLocaleTimeString()} · values reset with the backend process.</p></div>
        </div>}
      </article>
    </div>}
  </section>;
}

function SummaryHeader({ eyebrow, loading, onRefresh, onContinue, open }: { eyebrow: string; loading: boolean; onRefresh: () => void; onContinue: () => void; open: React.ReactNode }) {
  return <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-[9px] font-bold uppercase tracking-[.18em] text-sky-300">{eyebrow}</p><div className="flex flex-wrap gap-2"><button type="button" onClick={onRefresh} disabled={loading} className="presentation-summary-secondary"><RefreshCw size={12} className={loading ? "animate-spin" : ""}/>Refresh</button>{open}<button type="button" onClick={onContinue} className="presentation-summary-secondary">Continue</button></div></div>;
}

function AnalyticsFact({ icon: Icon, label, value }: { icon: typeof Activity; label: string; value: string }) { return <div className="min-w-0 rounded-xl border border-white/[.05] bg-white/[.02] p-2"><dt className="flex items-center gap-1 text-[7px] uppercase tracking-wide text-zinc-600"><Icon size={9}/>{label}</dt><dd className="mt-1 truncate text-[9px] font-semibold text-zinc-300" title={value}>{value}</dd></div>; }
function Loading({ label }: { label: string }) { return <div className="presentation-summary-panel mt-3 grid min-h-56 place-items-center rounded-2xl border text-xs text-zinc-400"><span className="flex items-center gap-2"><LoaderCircle className="animate-spin" size={16}/>{label}</span></div>; }
function Failure({ message, retry }: { message: string; retry: () => void }) { return <div className="presentation-summary-panel mt-3 rounded-2xl border border-rose-300/20 p-5" role="alert"><AlertTriangle className="text-rose-200" size={17}/><h2 className="mt-2 text-sm font-semibold text-rose-100">Live analytics unavailable</h2><p className="mt-1 text-[10px] text-rose-100/80">{message}</p><button type="button" onClick={retry} className="mt-3 presentation-summary-secondary">Retry</button></div>; }

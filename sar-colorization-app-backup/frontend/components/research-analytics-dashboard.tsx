"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity, AlertTriangle, Boxes, CheckCircle2, Clock3, Cpu, Database,
  FileArchive, Gauge, HardDrive, History, LoaderCircle, RefreshCw, Route,
  Server, ShieldCheck, Sparkles, Wrench,
} from "lucide-react";
import { getAgentAnalytics } from "@/services/api";
import { analyticsLabel, formatAnalyticsDuration, formatUptime, isPositiveStatus, runtimeBarPercent } from "@/lib/analytics";
import { cn } from "@/lib/utils";

const dateTime = (value: string | null) => value ? new Date(value).toLocaleString() : "No completed run";
const nullable = (value: string | number | null, suffix = "") => value === null ? "Unavailable" : `${value}${suffix}`;

export function ResearchAnalyticsDashboard() {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const update = () => setVisible(document.visibilityState === "visible");
    update();
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);
  const analytics = useQuery({
    queryKey: ["agent-analytics"],
    queryFn: ({ signal }) => getAgentAnalytics(signal),
    retry: 1,
    refetchInterval: visible ? 10_000 : false,
    refetchIntervalInBackground: false,
  });

  if (analytics.isLoading) return <LoadingState/>;
  if (analytics.error || !analytics.data) return <OfflineState message={analytics.error instanceof Error ? analytics.error.message : "Research analytics are unavailable."} retry={() => analytics.refetch()}/>;
  const data = analytics.data;
  const maximumRuntime = Math.max(0, ...data.workflow_metrics.map(item => item.average_runtime_ms ?? 0));

  return <section aria-labelledby="analytics-title">
    <header className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
      <div><p className="eyebrow">Operational evidence · current process</p><h1 id="analytics-title" className="mt-3 text-4xl font-semibold sm:text-5xl">Research Analytics</h1><p className="mt-4 max-w-4xl text-sm leading-6 text-zinc-400">Live engineering telemetry, scientific provenance, workflow performance, and SIH readiness from the authoritative local backend. Unmeasured values stay unavailable.</p></div>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span className={cn("inline-flex items-center gap-2 rounded-full border px-3 py-2", data.platform.backend_status === "healthy" ? "border-emerald-300/25 bg-emerald-300/[.07] text-emerald-200" : "border-amber-300/25 bg-amber-300/[.07] text-amber-100")}><span className={cn("h-2 w-2 rounded-full", data.platform.backend_status === "healthy" ? "bg-emerald-300" : "bg-amber-300")}/>{analyticsLabel(data.platform.backend_status)}</span>
        <span className="rounded-full border border-white/[.09] px-3 py-2 text-zinc-400">Updated {new Date(data.generated_at).toLocaleTimeString()}</span>
        <button type="button" onClick={() => analytics.refetch()} disabled={analytics.isFetching} className="inline-flex items-center gap-2 rounded-full border border-sky-300/25 bg-sky-300/[.07] px-3 py-2 text-sky-200 transition hover:bg-sky-300/[.12] disabled:opacity-60"><RefreshCw size={13} className={analytics.isFetching ? "animate-spin" : ""}/>Refresh</button>
      </div>
    </header>

    <div className="mt-7 grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-live="polite">
      <Summary icon={Server} title="Backend" value={analyticsLabel(data.platform.backend_status)} note={`${data.platform.hardware_acceleration} execution`}/>
      <Summary icon={Clock3} title="Process uptime" value={formatUptime(data.platform.uptime_seconds)} note="Resets with backend"/>
      <Summary icon={Activity} title="Completed workflows" value={String(data.summary.total_executions_current_process)} note={`${data.summary.successful_executions_current_process} successful / partial`}/>
      <Summary icon={Wrench} title="Available tools" value={`${data.summary.available_tools}/${data.summary.registered_tools}`} note="Registry-declared"/>
      <Summary icon={ShieldCheck} title="SIH readiness" value={`${data.summary.mandatory_satisfied}/${data.summary.mandatory_total}`} note="Mandatory requirements met"/>
      <Summary icon={HardDrive} title="Cache hits" value={String(data.summary.cache_hits_current_process)} note={data.cache.hit_rate_percent === null ? "No cache lookups yet" : `${data.cache.hit_rate_percent}% hit rate`}/>
      <Summary icon={Gauge} title="Cache misses" value={String(data.summary.cache_misses_current_process)} note={`${data.cache.stored_results}/${data.cache.max_results} results retained`}/>
      <Summary icon={FileArchive} title="Report artifacts" value={String(data.summary.report_artifacts_generated_current_process)} note={`${data.reports.artifacts_currently_available} currently available`}/>
      {data.ttp && <Summary icon={Cpu} title="TTP service" value={analyticsLabel(data.ttp.service_status)} note={`${data.ttp.inference_count} inference · ${data.ttp.fallback_count} fallback`}/>} 
      {data.sve && <Summary icon={Sparkles} title="SVE lifecycle" value={analyticsLabel(data.sve.lifecycle_status)} note={`${data.sve.inference_count} inference · ${data.sve.model_reuse_count} reuse`}/>} 
    </div>

    {data.sve && <Panel className="mt-6" title="SatQuery Vision Encoder operations" icon={Sparkles} note="Scene-level telemetry only; no images, filenames, queries, or embeddings are retained.">
      <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Fact name="Lifecycle" value={analyticsLabel(data.sve.lifecycle_status)}/><Fact name="Loads / reuses" value={`${data.sve.model_load_count} / ${data.sve.model_reuse_count}`}/><Fact name="Inferences / failures" value={`${data.sve.inference_count} / ${data.sve.failure_count}`}/><Fact name="Checksum failures" value={String(data.sve.checksum_failure_count)}/><Fact name="CPU / MPS fallbacks" value={`${data.sve.cpu_fallback_count} / ${data.sve.mps_fallback_count}`}/><Fact name="Cache hit rate" value={nullable(data.sve.cache_hit_rate_percent, "%")}/><Fact name="Average runtime" value={formatAnalyticsDuration(data.sve.average_runtime_ms)}/><Fact name="Caption reranks · VQA disagree" value={`${data.sve.caption_reranking_count} · ${data.sve.vqa_disagreement_count}`}/></dl>
    </Panel>}

    {data.ttp && <Panel className="mt-6" title="TTP CUDA operations" icon={Cpu} note="Process-local counters and validated service health only; no image pixels, filenames, or query text are retained.">
      <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Fact name="Service status" value={analyticsLabel(data.ttp.service_status)}/><Fact name="Model loads" value={String(data.ttp.model_load_count)}/><Fact name="Model reuses" value={String(data.ttp.model_reuse_count)}/><Fact name="Inferences" value={String(data.ttp.inference_count)}/><Fact name="Failures / timeouts / OOM" value={`${data.ttp.failure_count} / ${data.ttp.timeout_count} / ${data.ttp.oom_count}`}/><Fact name="Fallbacks" value={String(data.ttp.fallback_count)}/><Fact name="Average runtime" value={formatAnalyticsDuration(data.ttp.average_runtime_ms)}/><Fact name="Average mask IoU" value={data.ttp.average_mask_iou == null ? "Unavailable" : `${data.ttp.average_mask_iou.toFixed(3)} · not accuracy`}/></dl>
    </Panel>}

    <section className="mt-9 grid gap-6 2xl:grid-cols-[1.15fr_.85fr]">
      <Panel title="Workflow performance" icon={Gauge} note="Measured completed runs only; bars compare average runtime within this process.">
        {data.workflow_metrics.every(item => item.executions === 0) ? <Empty message="No SatQuery workflow has completed since this backend started. Run an analysis to populate measured runtime and device data."/> : <div className="space-y-4">{data.workflow_metrics.map(item => <div key={item.task} className="grid gap-2 sm:grid-cols-[180px_1fr_110px] sm:items-center"><div><p className="text-sm font-medium text-zinc-200">{analyticsLabel(item.task)}</p><p className="text-[11px] text-zinc-500">{item.successful_executions}/{item.executions} successful</p></div><div className="h-2.5 overflow-hidden rounded-full bg-white/[.06]" aria-label={`${analyticsLabel(item.task)} average runtime ${formatAnalyticsDuration(item.average_runtime_ms)}`}><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-300" style={{ width: `${runtimeBarPercent(item.average_runtime_ms, maximumRuntime)}%` }}/></div><div className="text-right text-xs tabular-nums text-zinc-400">{formatAnalyticsDuration(item.average_runtime_ms)}</div></div>)}</div>}
      </Panel>
      <Panel title="Platform runtime" icon={Cpu} note="Detected in the running API process.">
        <dl className="grid gap-4 sm:grid-cols-2"><Fact name="Python" value={data.platform.python_version}/><Fact name="Operating system" value={`${data.platform.operating_system} · ${data.platform.architecture}`}/><Fact name="Acceleration" value={data.platform.hardware_acceleration}/><Fact name="Process memory" value={nullable(data.platform.process_memory_mb, " MB")}/>{Object.entries(data.platform.runtime_versions).map(([name, version]) => <Fact key={name} name={analyticsLabel(name)} value={version ?? "Unavailable"}/>)}</dl>
        <div className={cn("mt-5 rounded-xl border p-4 text-sm", data.platform.offline_ready ? "border-emerald-300/20 bg-emerald-300/[.06] text-emerald-100" : "border-amber-300/20 bg-amber-300/[.06] text-amber-100")}><p className="font-medium">Offline readiness: {data.platform.offline_ready ? "Active" : "Not fully verified"}</p>{data.platform.offline_readiness_requirements.length > 0 && <ul className="mt-2 space-y-1 text-xs leading-5 opacity-80">{data.platform.offline_readiness_requirements.map(item => <li key={item}>• {item}</li>)}</ul>}</div>
      </Panel>
    </section>

    <Panel className="mt-6" title="Model and tool operations" icon={Wrench} note="Implementation state is separate from lazy model lifecycle and real last-run data.">
      <div className="overflow-x-auto"><table className="w-full min-w-[1050px] text-left text-xs"><thead className="text-zinc-500"><tr>{["Tool", "Implementation", "Lifecycle", "Device", "Last runtime", "Last completed", "Provenance"].map(item => <th key={item} className="border-b border-white/[.08] px-3 py-3 font-medium">{item}</th>)}</tr></thead><tbody>{data.tools.map(tool => <tr key={tool.id} className="border-b border-white/[.05] align-top"><td className="px-3 py-4"><p className="font-medium text-zinc-100">{tool.display_name}</p><p className="mt-1 font-mono text-[11px] text-sky-300">{tool.id}</p></td><td className="px-3 py-4"><Status value={tool.implementation_status}/></td><td className="px-3 py-4"><Status value={tool.lifecycle_status}/></td><td className="px-3 py-4 text-zinc-300">{tool.device ?? "Not reported"}</td><td className="px-3 py-4 tabular-nums text-zinc-300">{formatAnalyticsDuration(tool.last_runtime_ms)}</td><td className="px-3 py-4 text-zinc-400">{dateTime(tool.last_completed_at)}</td><td className="max-w-[280px] px-3 py-4 leading-5 text-zinc-400">{tool.checkpoint ?? tool.method_type ?? "Deterministic / service-level"}{tool.adaptation_dataset && <span className="block text-zinc-500">{tool.adaptation_dataset}</span>}</td></tr>)}</tbody></table></div>
    </Panel>

    <section className="mt-6 grid gap-6 2xl:grid-cols-2">
      <Panel title="Dataset and model provenance" icon={Database} note="Usage labels distinguish active data, candidates, plans, undeclared sources, and model provenance.">
        <div className="space-y-3">{data.datasets.map(dataset => <article key={dataset.name} className="glass rounded-xl p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-medium text-zinc-100">{dataset.name}</h3><p className="mt-1 text-xs text-zinc-500">{dataset.used_by.length ? `Used by ${dataset.used_by.join(", ")}` : "Not used by the current runtime"}</p></div><Status value={dataset.usage_status}/></div><p className="mt-3 text-sm leading-6 text-zinc-300">{dataset.purpose}</p><p className="mt-2 text-xs leading-5 text-zinc-500">{dataset.note}</p><p className="mt-2 text-[11px] text-zinc-600">Samples: {dataset.sample_count ?? "Not asserted"}</p></article>)}</div>
      </Panel>
      <Panel title="Capability matrix" icon={Boxes} note="Unsupported and unimplemented scope remains visible.">
        <div className="space-y-2">{data.capabilities.map(capability => <div key={capability.name} className="grid gap-2 rounded-xl border border-white/[.06] p-3 sm:grid-cols-[minmax(160px,.7fr)_130px_1fr] sm:items-center"><p className="text-sm font-medium text-zinc-200">{capability.name}</p><Status value={capability.status}/><p className="text-xs leading-5 text-zinc-500">{capability.evidence}</p></div>)}</div>
      </Panel>
    </section>

    <Panel className="mt-6" title="Recent execution history" icon={History} note="Newest first · maximum 50 · safe metadata only · reset on backend restart.">
      {data.recent_executions.length === 0 ? <Empty message="No completed execution metadata is available in this backend process."/> : <div className="overflow-x-auto"><table className="w-full min-w-[1080px] text-left text-xs"><thead className="text-zinc-500"><tr>{["Completed", "Task", "Mode / modalities", "Status", "Runtime", "Outputs", "Warnings", "Cache", "Report", "Request ID"].map(item => <th key={item} className="border-b border-white/[.08] px-3 py-3 font-medium">{item}</th>)}</tr></thead><tbody>{data.recent_executions.map(item => <tr key={`${item.request_id}-${item.completed_at}`} className="border-b border-white/[.05]"><td className="whitespace-nowrap px-3 py-4 text-zinc-400">{dateTime(item.completed_at)}</td><td className="px-3 py-4 font-medium text-zinc-200">{analyticsLabel(item.task)}</td><td className="px-3 py-4 text-zinc-400">{analyticsLabel(item.input_mode)}<span className="block text-zinc-600">{[item.primary_modality, item.secondary_modality].filter(Boolean).map(value => analyticsLabel(value!)).join(" + ")}</span></td><td className="px-3 py-4"><Status value={item.status}/></td><td className="px-3 py-4 tabular-nums text-zinc-300">{formatAnalyticsDuration(item.duration_ms)}</td><td className="px-3 py-4 tabular-nums text-zinc-300">{item.output_count}</td><td className="px-3 py-4 tabular-nums text-zinc-300">{item.warning_count}</td><td className="px-3 py-4 text-zinc-400">{analyticsLabel(item.cache_status)}</td><td className="px-3 py-4 text-zinc-400">{item.report_generated ? "Generated" : "No"}</td><td className="max-w-[150px] truncate px-3 py-4 font-mono text-[11px] text-zinc-500" title={item.request_id}>{item.request_id}</td></tr>)}</tbody></table></div>}
    </Panel>

    <section className="mt-6 grid gap-6 2xl:grid-cols-[.9fr_1.1fr]">
      <Panel title="Last execution trace" icon={Route} note="Safe stage names, status, and measured duration only.">
        {data.last_execution_trace.length === 0 ? <Empty message="The trace will appear after the first completed workflow."/> : <ol className="space-y-3">{data.last_execution_trace.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-full", step.status === "success" ? "bg-emerald-300/[.10] text-emerald-200" : "bg-amber-300/[.10] text-amber-100")}>{step.status === "success" ? <CheckCircle2 size={15}/> : index + 1}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{analyticsLabel(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{formatAnalyticsDuration(step.duration_ms)}</span></li>)}</ol>}
      </Panel>
      <Panel title="Scientific transparency" icon={Sparkles} note="Contract-level safeguards for interpreting this dashboard.">
        <dl className="grid gap-4 sm:grid-cols-2"><Fact name="Metric source" value={data.scientific_transparency.metric_source}/><Fact name="History retention" value={data.scientific_transparency.history_retention}/><Fact name="Inference behavior changed" value={data.scientific_transparency.inference_behavior_changed ? "Yes" : "No"}/><Fact name="AI-generated metrics" value={data.scientific_transparency.ai_generated_metrics ? "Yes" : "No"}/><Fact name="Unavailable-value policy" value={data.scientific_transparency.unavailable_value_policy}/><Fact name="Cache policy" value={`${data.cache.stored_results}/${data.cache.max_results} results · ${Math.round(data.cache.ttl_seconds / 60)} minute TTL`}/><Fact name="Reports this process" value={`${data.reports.requests_generated_current_process} requests · ${data.reports.artifacts_generated_current_process} artifacts`}/><Fact name="Report formats" value={Object.keys(data.reports.formats).length ? Object.entries(data.reports.formats).map(([name, count]) => `${name.toUpperCase()} ${count}`).join(" · ") : "No generated artifacts"}/></dl>
        <div className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[.05] p-4"><h3 className="flex items-center gap-2 text-sm font-medium text-amber-100"><AlertTriangle size={15}/>Interpretation limits</h3><ul className="mt-3 space-y-2 text-xs leading-5 text-zinc-400">{data.scientific_transparency.caveats.map(item => <li key={item}>• {item}</li>)}</ul></div>
      </Panel>
    </section>
  </section>;
}

function LoadingState() { return <section aria-busy="true"><p className="eyebrow">Operational evidence</p><h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Research Analytics</h1><div className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{Array.from({ length: 8 }, (_, index) => <div key={index} className="panel h-32 animate-pulse bg-white/[.025]"/>)}</div><p className="mt-6 inline-flex items-center gap-2 text-sm text-zinc-400"><LoaderCircle size={16} className="animate-spin"/>Loading authoritative backend telemetry…</p></section>; }
function OfflineState({ message, retry }: { message: string; retry: () => void }) { return <section><p className="eyebrow">Operational evidence</p><h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Research Analytics</h1><div role="alert" className="panel mt-8 max-w-2xl border-rose-300/20 p-6"><AlertTriangle className="text-rose-300"/><h2 className="mt-4 text-lg font-semibold">Analytics backend unavailable</h2><p className="mt-2 text-sm leading-6 text-zinc-400">{message} No substitute or demo values are being shown.</p><button onClick={retry} className="mt-5 inline-flex items-center gap-2 rounded-xl bg-sky-400/15 px-4 py-2 text-sm text-sky-200"><RefreshCw size={15}/>Try again</button></div></section>; }
function Summary({ icon: Icon, title, value, note }: { icon: typeof Server; title: string; value: string; note: string }) { return <article className="panel p-5"><Icon size={18} className="text-sky-300"/><p className="mt-4 text-xs uppercase tracking-wide text-zinc-500">{title}</p><p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p><p className="mt-1 text-xs text-zinc-500">{note}</p></article>; }
function Panel({ title, icon: Icon, note, className, children }: { title: string; icon: typeof Server; note: string; className?: string; children: React.ReactNode }) { return <section className={cn("panel overflow-hidden", className)}><header className="border-b border-white/[.08] p-5 sm:p-6"><div className="flex items-center gap-2"><Icon size={18} className="text-sky-300"/><h2 className="text-xl font-semibold">{title}</h2></div><p className="mt-2 text-xs leading-5 text-zinc-500">{note}</p></header><div className="p-5 sm:p-6">{children}</div></section>; }
function Status({ value }: { value: string }) { const positive = isPositiveStatus(value); const negative = ["failed", "unsupported", "not implemented", "not_implemented"].includes(value.toLowerCase()); return <span className={cn("inline-flex w-fit whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px]", positive ? "border-emerald-300/25 bg-emerald-300/[.08] text-emerald-200" : negative ? "border-rose-300/25 bg-rose-300/[.08] text-rose-200" : "border-amber-300/25 bg-amber-300/[.08] text-amber-100")}>{analyticsLabel(value)}</span>; }
function Fact({ name, value }: { name: string; value: string }) { return <div><dt className="text-xs text-zinc-500">{name}</dt><dd className="mt-1 break-words text-sm leading-5 text-zinc-200">{value}</dd></div>; }
function Empty({ message }: { message: string }) { return <div className="rounded-xl border border-dashed border-white/[.10] p-6 text-center text-sm leading-6 text-zinc-500">{message}</div>; }

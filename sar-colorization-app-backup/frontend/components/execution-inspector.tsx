"use client";

import { useState } from "react";
import { Check, Clipboard, FileCheck2, ShieldAlert } from "lucide-react";
import type { AnalyticsExecution } from "@/types/agent";
import { architectureLabel, buildExecutionSummary } from "@/lib/architecture";
import { formatAnalyticsDuration } from "@/lib/analytics";
import { LiveWorkflowTimeline } from "@/components/live-workflow-timeline";

export function ExecutionInspector({ execution }: { execution: AnalyticsExecution }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(buildExecutionSummary(execution));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  const modalities = [execution.primary_modality, execution.secondary_modality].filter(Boolean).map(value => architectureLabel(value!)).join(" + ") || "Not reported";
  return <section aria-labelledby="execution-inspector-title">
    <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">Observable execution only</p><h2 id="execution-inspector-title" className="mt-2 text-2xl font-semibold">Execution Inspector</h2><p className="mt-2 font-mono text-[11px] text-zinc-600">{execution.request_id}</p></div><button type="button" onClick={copy} className="inline-flex items-center gap-2 rounded-xl border border-sky-300/25 bg-sky-300/[.07] px-3.5 py-2 text-xs text-sky-200 transition hover:bg-sky-300/[.12] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300" aria-live="polite">{copied ? <Check size={14}/> : <Clipboard size={14}/>} {copied ? "Copied" : "Copy Execution Summary"}</button></div>
    <dl className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Fact name="Task" value={architectureLabel(execution.task)}/><Fact name="Input mode" value={architectureLabel(execution.input_mode)}/><Fact name="Modalities" value={modalities}/><Fact name="Status" value={architectureLabel(execution.status)}/><Fact name="Selected tools" value={execution.selected_tools.join(" · ") || "None"}/><Fact name="Total runtime" value={formatAnalyticsDuration(execution.duration_ms)}/><Fact name="Cache" value={architectureLabel(execution.cache_status)}/><Fact name="Evidence products" value={String(execution.output_count)}/><Fact name="Warnings" value={String(execution.warning_count)}/><Fact name="Report" value={execution.report_generated ? "Generated" : "Not generated"}/><Fact name="Device" value={execution.device ?? "Not reported"}/><Fact name="Completed" value={new Date(execution.completed_at).toLocaleString()}/></dl>
    {execution.selection_reason && <div className="mt-5 rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-4"><h3 className="text-sm font-medium text-sky-100">Routing selection record</h3><p className="mt-2 text-sm leading-6 text-zinc-400">{execution.selection_reason}</p></div>}
    {execution.confidence_level && <div className="mt-3 rounded-xl border border-white/[.08] p-4"><h3 className="text-sm font-medium">Confidence disclosure · {architectureLabel(execution.confidence_level)}</h3><p className="mt-2 text-sm leading-6 text-zinc-400">{execution.confidence_reason ?? "No confidence rationale was published for this execution."}</p></div>}
    <div className="mt-3 rounded-xl border border-white/[.08] p-4"><h3 className="flex items-center gap-2 text-sm font-medium"><ShieldAlert size={15} className="text-amber-200"/>Warnings and limitations</h3>{execution.warnings.length ? <ul className="mt-2 space-y-1 text-sm leading-6 text-zinc-400">{execution.warnings.map(item => <li key={item}>• {item}</li>)}</ul> : <p className="mt-2 text-sm text-zinc-500">No warning text was recorded for this execution.</p>}</div>
    <div className="mt-7 border-t border-white/[.08] pt-7"><LiveWorkflowTimeline execution={execution}/></div>
    <details className="mt-6 rounded-xl border border-white/[.08]"><summary className="cursor-pointer px-4 py-3 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300">Ordered trace table</summary><div className="overflow-x-auto border-t border-white/[.08]"><table className="w-full min-w-[720px] text-left text-xs"><thead className="text-zinc-500"><tr><th className="px-3 py-3 font-medium">Stage</th><th className="px-3 py-3 font-medium">Tool</th><th className="px-3 py-3 font-medium">Status</th><th className="px-3 py-3 font-medium">Duration</th><th className="px-3 py-3 font-medium">Safe parameters</th></tr></thead><tbody>{execution.trace.map((step, index) => <tr key={`${step.tool}-${index}`} className="border-t border-white/[.06]"><td className="px-3 py-3 tabular-nums text-zinc-500">{index + 1}</td><td className="px-3 py-3 font-mono text-zinc-200">{step.tool}</td><td className="px-3 py-3 text-zinc-400">{architectureLabel(step.status)}</td><td className="px-3 py-3 tabular-nums text-zinc-400">{formatAnalyticsDuration(step.duration_ms)}</td><td className="max-w-[320px] px-3 py-3 font-mono text-[10px] text-zinc-500">{Object.keys(step.parameters).length ? JSON.stringify(step.parameters) : "None published"}</td></tr>)}</tbody></table></div></details>
    <p className="mt-5 inline-flex items-center gap-2 text-xs text-zinc-600"><FileCheck2 size={13}/>The copied summary excludes uploads, queries, answers, credentials, paths, hashes, and hidden reasoning.</p>
  </section>;
}

function Fact({ name, value }: { name: string; value: string }) { return <div className="rounded-xl border border-white/[.06] bg-white/[.018] p-3"><dt className="text-[11px] uppercase tracking-wide text-zinc-500">{name}</dt><dd className="mt-1.5 break-words text-sm text-zinc-200">{value}</dd></div>; }

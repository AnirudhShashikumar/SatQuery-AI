"use client";

import { useState } from "react";
import { Archive, CheckCircle2, FileJson, FileText, LoaderCircle, RefreshCw } from "lucide-react";
import { agentArtifactUrl, generateAgentReport } from "@/services/api";
import type { AgentResponse, ReportFormat } from "@/types/agent";

type Props = { result: AgentResponse; onRerun?: () => void; rerunning?: boolean; compact?: boolean; toolbar?: boolean };

const choices: Array<{ format: ReportFormat; label: string; icon: typeof FileText }> = [
  { format: "pdf", label: "PDF", icon: FileText },
  { format: "json", label: "JSON", icon: FileJson },
  { format: "zip", label: "Full ZIP", icon: Archive },
];

export function MissionReportActions({ result, onRerun, rerunning = false, compact = false, toolbar = false }: Props) {
  const [working, setWorking] = useState<ReportFormat | null>(null);
  const [message, setMessage] = useState("");
  const [failed, setFailed] = useState<ReportFormat | null>(null);
  if (result.status !== "success" && result.status !== "partial") return null;

  const download = async (format: ReportFormat) => {
    if (working) return;
    setWorking(format); setFailed(null); setMessage(`Generating ${format.toUpperCase()} mission report…`);
    try {
      const response = await generateAgentReport(result.request_id, [format]);
      const artifact = response.artifacts.find(item => item.format === format);
      if (!artifact) throw new Error("The backend did not return the requested report artifact.");
      const anchor = document.createElement("a");
      anchor.href = agentArtifactUrl(artifact.url); anchor.download = artifact.filename; document.body.appendChild(anchor); anchor.click(); anchor.remove();
      setMessage(`${artifact.filename} is ready.`);
    } catch (error) {
      setFailed(format); setMessage(error instanceof Error ? error.message : "Mission report generation failed. Please retry.");
    } finally { setWorking(null); }
  };

  const compactLike = compact || toolbar;
  const actions = <><div className={compactLike ? "flex flex-wrap gap-2" : "mt-5 flex flex-wrap gap-3"}>{choices.map(({ format, label, icon: Icon }) => <button key={format} type="button" onClick={() => download(format)} disabled={Boolean(working)} aria-label={`Download ${label} mission report`} className={toolbar ? "result-toolbar-button" : compact ? "result-primary-button" : "inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-sky-200 disabled:cursor-not-allowed disabled:opacity-50"}>{working === format ? <LoaderCircle className="animate-spin" size={toolbar ? 14 : 17}/> : <Icon size={toolbar ? 14 : 17}/>} {working === format ? "Generating…" : toolbar ? label.replace("Full ", "") : compact ? `Download ${label}` : label}</button>)}{failed && <button type="button" onClick={() => download(failed)} disabled={Boolean(working)} className="inline-flex items-center gap-2 rounded-xl border border-rose-300/25 px-4 py-3 text-sm text-rose-200"><RefreshCw size={16}/>Retry {failed.toUpperCase()}</button>}</div><p className={toolbar ? "sr-only" : compact ? "mt-2 min-h-5 text-xs text-zinc-500" : "mt-3 min-h-5 text-sm text-zinc-400"} aria-live="polite">{message}</p></>;

  if (compactLike) return <div aria-label="Mission report downloads">{actions}</div>;

  return <section className="panel mt-7 p-5 sm:p-6" aria-labelledby={`mission-report-${result.request_id}`}>
    <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">Mission export</p><h2 id={`mission-report-${result.request_id}`} className="mt-2 text-lg font-semibold">Download Report</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">Generated locally from the authoritative backend result, evidence products, provenance, confidence, and execution trace.</p></div>{result.cache?.cached && <span className="rounded-full border border-sky-300/25 bg-sky-300/[.08] px-3 py-1 text-xs text-sky-200">Cached result · {result.cache.tool_version}</span>}</div>
    {result.cache?.cached && <div className="mt-4 rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-4 text-sm text-zinc-300"><p>Originally generated {new Date(result.cache.original_generation_timestamp).toLocaleString()}; retrieved {new Date(result.cache.retrieval_timestamp).toLocaleString()}.</p><div className="mt-3 flex flex-wrap gap-2"><span className="rounded-lg bg-sky-300/10 px-3 py-2 text-xs text-sky-200"><CheckCircle2 className="mr-1.5 inline" size={14}/>Use Cached Result</span>{onRerun && <button type="button" onClick={onRerun} disabled={rerunning} className="rounded-lg border border-white/[.10] px-3 py-2 text-xs text-zinc-200 disabled:opacity-50"><RefreshCw className={`mr-1.5 inline ${rerunning ? "animate-spin" : ""}`} size={14}/>Re-run Analysis</button>}</div></div>}
    {actions}
  </section>;
}

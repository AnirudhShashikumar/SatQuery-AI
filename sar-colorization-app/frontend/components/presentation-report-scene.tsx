"use client";

import {
  AlertTriangle,
  Archive,
  CheckCircle2,
  Clock3,
  Download,
  FileJson,
  FileText,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  evidenceCount,
  formatArtifactSize,
  isReportableResult,
  latestSatQueryRequestKey,
  presentationReportError,
  presentationReportFormats,
  reportContentsOverview,
} from "@/lib/presentation-report";
import { formatPresentationLabel } from "@/lib/presentation-single-image";
import { agentArtifactUrl, generateAgentReport, getComparisonItem } from "@/services/api";
import type { ComparisonItem, ReportFormat, ReportResponse } from "@/types/agent";

const formatChoices: Array<{ format: ReportFormat; label: string; detail: string; icon: typeof FileText }> = [
  { format: "pdf", label: "PDF", detail: "Human-readable mission brief", icon: FileText },
  { format: "json", label: "JSON", detail: "Machine-readable evidence record", icon: FileJson },
  { format: "zip", label: "Full ZIP", detail: "PDF, JSON, CSV, and evidence package", icon: Archive },
];

type LoadState = "loading" | "empty" | "ready" | "error";

export function PresentationReportScene() {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [latestResult, setLatestResult] = useState<ComparisonItem | null>(null);
  const [selectedFormats, setSelectedFormats] = useState<ReportFormat[]>([...presentationReportFormats]);
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState<ReportResponse | null>(null);
  const [error, setError] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [downloading, setDownloading] = useState<ReportFormat | null>(null);
  const mountedRef = useRef(true);
  const loadSequenceRef = useRef(0);
  const generationRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      loadSequenceRef.current += 1;
    };
  }, []);

  const loadLatest = useCallback(async () => {
    const sequence = ++loadSequenceRef.current;
    const requestId = window.sessionStorage.getItem(latestSatQueryRequestKey)?.trim();
    setGenerated(null); setDownloadError(""); setError(""); setLatestResult(null);
    if (!requestId) { setLoadState("empty"); return; }
    setLoadState("loading");
    try {
      const result = await getComparisonItem(requestId);
      if (!mountedRef.current || sequence !== loadSequenceRef.current) return;
      setLatestResult(result);
      setLoadState("ready");
    } catch (caught) {
      if (mountedRef.current && sequence === loadSequenceRef.current) {
        setError(presentationReportError(caught));
        setLoadState("error");
      }
    }
  }, []);

  useEffect(() => { void loadLatest(); }, [loadLatest]);

  const toggleFormat = (format: ReportFormat) => {
    if (generating) return;
    setGenerated(null); setDownloadError("");
    setSelectedFormats(current => current.includes(format) ? current.filter(item => item !== format) : [...current, format]);
  };

  const generate = useCallback(async () => {
    if (!latestResult || !isReportableResult(latestResult) || !selectedFormats.length || generating || generationRef.current) return;
    generationRef.current = true;
    setGenerating(true); setGenerated(null); setError(""); setDownloadError("");
    try {
      const response = await generateAgentReport(latestResult.request_id, selectedFormats);
      if (!mountedRef.current) return;
      setGenerated(response);
    } catch (caught) {
      if (mountedRef.current) setError(presentationReportError(caught));
    } finally {
      generationRef.current = false;
      if (mountedRef.current) setGenerating(false);
    }
  }, [generating, latestResult, selectedFormats]);

  const downloadArtifact = async (artifact: ReportResponse["artifacts"][number]) => {
    if (downloading) return;
    setDownloading(artifact.format); setDownloadError("");
    try {
      const response = await fetch(agentArtifactUrl(artifact.url), { cache: "no-store" });
      if (!response.ok) throw new Error(response.status === 404 || response.status === 410 ? "The report artifact is unavailable or expired." : "The report artifact download failed.");
      const objectUrl = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = objectUrl; anchor.download = artifact.filename; document.body.appendChild(anchor); anchor.click(); anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    } catch (caught) {
      if (mountedRef.current) setDownloadError(presentationReportError(caught));
    } finally {
      if (mountedRef.current) setDownloading(null);
    }
  };

  const reportable = latestResult ? isReportableResult(latestResult) : false;

  return <section className="presentation-report-scene mt-4 min-h-0 flex-1" aria-label="Interactive Mission Reports demonstration">
    {loadState === "loading" && <div className="presentation-report-empty grid min-h-72 place-items-center rounded-2xl border p-6 text-center" role="status"><div><LoaderCircle className="mx-auto animate-spin text-sky-200" size={24}/><h2 className="mt-3 text-sm font-semibold">Loading the latest authoritative result</h2><p className="mt-1 text-xs text-zinc-500">No analysis or report generation has started.</p></div></div>}

    {loadState === "empty" && <div className="presentation-report-empty grid min-h-72 place-items-center rounded-2xl border p-6 text-center"><div><FileText className="mx-auto text-zinc-500" size={28}/><h2 className="mt-3 text-base font-semibold">No reportable presentation result</h2><p className="mx-auto mt-2 max-w-xl text-xs leading-5 text-zinc-500">Complete Single-Image Understanding, Bi-Temporal Change Analysis, or Optical–SAR Joint Analysis first. Stage 10 never starts an analysis automatically.</p><button type="button" onClick={() => void loadLatest()} className="mt-4 inline-flex items-center gap-2 rounded-xl border border-white/[.1] px-4 py-2 text-xs text-zinc-300"><RefreshCw size={13}/>Check Again</button></div></div>}

    {loadState === "error" && <div className="presentation-report-empty grid min-h-72 place-items-center rounded-2xl border p-6 text-center" role="alert"><div><AlertTriangle className="mx-auto text-rose-300" size={27}/><h2 className="mt-3 text-base font-semibold text-rose-100">Latest result unavailable</h2><p className="mx-auto mt-2 max-w-xl text-xs leading-5 text-rose-100/75">{error}</p><button type="button" onClick={() => void loadLatest()} className="mt-4 inline-flex items-center gap-2 rounded-xl border border-rose-300/25 px-4 py-2 text-xs text-rose-100"><RefreshCw size={13}/>Retry</button></div></div>}

    {loadState === "ready" && latestResult && <div className="presentation-report-layout grid min-h-0 gap-3 lg:grid-cols-[minmax(0,.9fr)_minmax(0,1.1fr)]">
      <article className="presentation-report-summary min-w-0 rounded-2xl border p-3 sm:p-4">
        <div className="flex flex-wrap items-start justify-between gap-2"><div><p className="text-[9px] font-bold uppercase tracking-[.17em] text-amber-200">Latest authoritative result</p><h2 className="mt-1 text-base font-semibold">{latestResult.display_name}</h2></div><span className={`rounded-full border px-2.5 py-1 text-[9px] font-semibold ${latestResult.status === "success" ? "border-emerald-300/25 text-emerald-200" : latestResult.status === "failed" ? "border-rose-300/25 text-rose-200" : "border-amber-300/25 text-amber-100"}`}>{formatPresentationLabel(latestResult.status)}</span></div>
        <dl className="mt-3 grid grid-cols-2 gap-1.5 sm:grid-cols-3">
          <ReportFact name="Task" value={formatPresentationLabel(latestResult.task)}/>
          <ReportFact name="Input mode" value={formatPresentationLabel(latestResult.input_mode)}/>
          <ReportFact name="Evidence" value={`${evidenceCount(latestResult)} product${evidenceCount(latestResult) === 1 ? "" : "s"}`}/>
          <ReportFact name="Generated" value={new Date(latestResult.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}/>
          <ReportFact name="Runtime" value={latestResult.execution_duration_ms == null ? "Unavailable" : `${latestResult.execution_duration_ms} ms`}/>
          <ReportFact name="Confidence" value={formatPresentationLabel(String(latestResult.confidence.level ?? "unavailable"))}/>
        </dl>
        <div className="mt-3"><p className="text-[8px] uppercase tracking-[.14em] text-zinc-500">Request ID</p><p className="mt-1 truncate font-mono text-[9px] text-zinc-300" title={latestResult.request_id}>{latestResult.request_id}</p></div>
        <div className="mt-3"><p className="text-[8px] uppercase tracking-[.14em] text-zinc-500">Selected tools</p><div className="mt-1.5 flex flex-wrap gap-1">{latestResult.selected_tools.length ? latestResult.selected_tools.map(tool => <span key={tool} className="rounded-lg bg-sky-400/[.08] px-2 py-1 font-mono text-[8px] text-sky-200">{tool}</span>) : <span className="text-[9px] text-zinc-500">No tools recorded</span>}</div></div>
        {latestResult.answer && <p className="mt-3 line-clamp-3 rounded-xl border border-white/[.06] bg-white/[.02] p-2 text-[10px] leading-4 text-zinc-400"><strong className="text-zinc-300">Answer:</strong> {latestResult.answer}</p>}
        {!evidenceCount(latestResult) && <p className="mt-3 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-2 text-[10px] text-amber-100">This result has no stored evidence products. The report will disclose that evidence is missing.</p>}
        {latestResult.status === "partial" && <p className="mt-3 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-2 text-[10px] text-amber-100">Partial result: generated reports retain all warnings and unavailable fields.</p>}
        {!reportable && <p className="mt-3 rounded-xl border border-rose-300/20 bg-rose-300/[.06] p-2 text-[10px] text-rose-100">Failed results are not reportable. Complete a successful, partial, or alignment-guarded workflow first.</p>}
      </article>

      <article className="presentation-report-controls min-w-0 rounded-2xl border p-3 sm:p-4">
        <div className="flex items-start justify-between gap-3"><div><p className="text-[9px] font-bold uppercase tracking-[.17em] text-sky-300">Backend-authoritative export</p><h2 className="mt-1 text-base font-semibold">Generate Mission Report</h2></div><ShieldCheck className="text-sky-200" size={20}/></div>
        <div className="mt-3 grid grid-cols-3 gap-2">{formatChoices.map(({ format, label, detail, icon: Icon }) => {
          const selected = selectedFormats.includes(format);
          return <button key={format} type="button" aria-pressed={selected} onClick={() => toggleFormat(format)} disabled={generating || !reportable} className={`presentation-report-format min-w-0 rounded-xl border p-2 text-left disabled:cursor-not-allowed disabled:opacity-40 ${selected ? "presentation-report-format-selected" : ""}`}><span className="flex items-center justify-between gap-1"><Icon size={15}/><span className={`grid h-3.5 w-3.5 place-items-center rounded-full border text-[8px] ${selected ? "border-sky-200 bg-sky-300 text-zinc-950" : "border-white/[.15]"}`}>{selected ? "✓" : ""}</span></span><span className="mt-1.5 block text-[10px] font-semibold">{label}</span><span className="mt-0.5 hidden text-[8px] leading-3 text-zinc-500 sm:block">{detail}</span></button>;
        })}</div>
        <div className="mt-3 flex flex-wrap items-center gap-2"><button type="button" onClick={() => void generate()} disabled={!reportable || generating || !selectedFormats.length} className="inline-flex h-9 items-center justify-center gap-2 rounded-xl bg-sky-300 px-4 text-xs font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-40">{generating ? <LoaderCircle className="animate-spin" size={14}/> : <FileText size={14}/>} {generating ? "Generating…" : "Generate Report"}</button>{generated && <span className="inline-flex items-center gap-1.5 text-[10px] text-emerald-300"><CheckCircle2 size={12}/>{formatPresentationLabel(generated.status)} · {generated.runtime_ms} ms</span>}</div>
        {error && <p role="alert" className="mt-2 rounded-xl border border-rose-300/20 bg-rose-300/[.06] p-2 text-[10px] text-rose-100">{error}</p>}
        {generated && <div className="mt-3"><div className="flex items-center justify-between"><h3 className="text-[9px] font-bold uppercase tracking-[.14em] text-zinc-500">Generated artifacts</h3><span className="flex items-center gap-1 text-[9px] text-zinc-500"><Clock3 size={10}/>{generated.runtime_ms} ms</span></div><div className="mt-1.5 grid gap-1.5 sm:grid-cols-3">{generated.artifacts.map(artifact => <a key={artifact.format} href={agentArtifactUrl(artifact.url)} download={artifact.filename} onClick={event => { event.preventDefault(); void downloadArtifact(artifact); }} className="flex min-w-0 items-center gap-2 rounded-xl border border-sky-300/20 px-2.5 py-2 text-[9px] text-sky-200"><Download size={12} className={downloading === artifact.format ? "animate-bounce" : ""}/><span className="min-w-0 flex-1"><strong className="block truncate">Download {artifact.format.toUpperCase()}</strong><span className="text-zinc-500">{formatArtifactSize(artifact.size_bytes)}</span></span></a>)}</div>{generated.warnings.map(warning => <p key={warning} className="mt-1 text-[8px] text-amber-100">• {warning}</p>)}</div>}
        {downloadError && <p role="alert" className="mt-2 rounded-xl border border-rose-300/20 bg-rose-300/[.06] p-2 text-[10px] text-rose-100">{downloadError}</p>}
      </article>

      <article className="presentation-report-overview min-w-0 rounded-2xl border p-3 lg:col-span-2">
        <div className="flex items-center justify-between gap-3"><div><p className="text-[9px] font-bold uppercase tracking-[.16em] text-zinc-500">Compact report summary</p><h2 className="mt-1 text-sm font-semibold">Authoritative contents overview</h2></div><span className="hidden text-[9px] text-zinc-500 sm:block">No browser-supplied statistics</span></div>
        <div className="mt-2 grid grid-cols-3 gap-1.5 sm:grid-cols-5 lg:grid-cols-9">{reportContentsOverview(latestResult).map(item => <div key={item.label} className={`rounded-xl border p-2 ${item.caution ? "border-amber-300/15 bg-amber-300/[.04]" : "border-white/[.06] bg-white/[.02]"}`}><p className="text-[8px] uppercase tracking-wide text-zinc-500">{item.label}</p><p className="mt-1 text-[9px] leading-3 text-zinc-300">{item.value}</p></div>)}</div>
      </article>
    </div>}
  </section>;
}

function ReportFact({ name, value }: { name: string; value: string }) {
  return <div className="presentation-report-fact min-w-0 rounded-xl border border-white/[.06] bg-white/[.02] p-2"><dt className="text-[8px] uppercase tracking-wide text-zinc-500">{name}</dt><dd className="mt-0.5 break-words text-[10px] font-semibold leading-4 text-zinc-200">{value}</dd></div>;
}

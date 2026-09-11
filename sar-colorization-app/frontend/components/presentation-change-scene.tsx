"use client";

import {
  AlertTriangle,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Database,
  GitCompareArrows,
  ImageOff,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Square,
} from "lucide-react";
import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import {
  approvedChangePair,
  changeDefaultDates,
  changeDefaultQuery,
  changeEvidencePreviews,
  changeResponseDisclosure,
  compatibilityLabel,
  requestFailureMessage,
  type ApprovedChangePair,
  type ChangeEvidencePreview,
} from "@/lib/presentation-change";
import { changeEngineName } from "@/lib/scientific-presentation";
import { formatPresentationLabel } from "@/lib/presentation-single-image";
import { presentationStorageKeys, type PresentationStep } from "@/lib/presentation";
import {
  ApiRequestError,
  agentPreviewUrl,
  getDemoManifest,
  getHealth,
  loadDemoFile,
  runAgentImageQuery,
} from "@/services/api";
import type { AgentResponse } from "@/types/agent";

type LoadedChangePair = ApprovedChangePair & {
  beforeFile: File;
  afterFile: File;
  beforePreviewUrl: string;
  afterPreviewUrl: string;
};

type RequestState = "idle" | "running" | "complete" | "error";

export function PresentationChangeScene({ step }: { step: PresentationStep }) {
  const configuredDates = step.default_dates ?? changeDefaultDates;
  const configuredQuery = step.default_query?.change ?? changeDefaultQuery;
  const [pair, setPair] = useState<LoadedChangePair | null>(null);
  const [beforeDate, setBeforeDate] = useState(configuredDates.before);
  const [afterDate, setAfterDate] = useState(configuredDates.after);
  const [query, setQuery] = useState(configuredQuery);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [error, setError] = useState("");
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const mountedRef = useRef(true);
  const sampleAbortRef = useRef<AbortController | null>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const previewUrlsRef = useRef<string[]>([]);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    setQuery(window.sessionStorage.getItem(presentationStorageKeys.changeQuery) || configuredQuery);
    setBeforeDate(window.sessionStorage.getItem(presentationStorageKeys.changeBeforeDate) || configuredDates.before);
    setAfterDate(window.sessionStorage.getItem(presentationStorageKeys.changeAfterDate) || configuredDates.after);
  }, [configuredDates.after, configuredDates.before, configuredQuery]);

  useEffect(() => {
    window.sessionStorage.setItem(presentationStorageKeys.changeQuery, query);
    window.sessionStorage.setItem(presentationStorageKeys.changeBeforeDate, beforeDate);
    window.sessionStorage.setItem(presentationStorageKeys.changeAfterDate, afterDate);
  }, [afterDate, beforeDate, query]);

  useEffect(() => {
    let active = true;
    const check = () => getHealth().then(() => { if (active) setBackendAvailable(true); }).catch(() => { if (active) setBackendAvailable(false); });
    check();
    const interval = window.setInterval(check, 15_000);
    return () => { active = false; window.clearInterval(interval); };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      sampleAbortRef.current?.abort();
      requestAbortRef.current?.abort();
      for (const url of previewUrlsRef.current) URL.revokeObjectURL(url);
      previewUrlsRef.current = [];
    };
  }, []);

  const loadSamples = useCallback(async () => {
    if (sampleLoading || sampleAbortRef.current || requestState === "running") return;
    const controller = new AbortController();
    sampleAbortRef.current = controller;
    setSampleLoading(true); setError(""); setResult(null); setRequestState("idle");
    try {
      const manifest = await getDemoManifest(controller.signal);
      const approved = approvedChangePair(
        manifest,
        step.demo_sample_id ?? "change_vqa",
        step.before_sample_id ?? "change-before.png",
        step.after_sample_id ?? "change-after.png",
        configuredDates,
      );
      const [beforeFile, afterFile] = await Promise.all([
        loadDemoFile(approved.before.url, approved.before.filename, approved.before.mime_type, controller.signal),
        loadDemoFile(approved.after.url, approved.after.filename, approved.after.mime_type, controller.signal),
      ]);
      if (!mountedRef.current || controller.signal.aborted) return;
      const beforePreviewUrl = URL.createObjectURL(beforeFile);
      const afterPreviewUrl = URL.createObjectURL(afterFile);
      for (const url of previewUrlsRef.current) URL.revokeObjectURL(url);
      previewUrlsRef.current = [beforePreviewUrl, afterPreviewUrl];
      setPair({ ...approved, beforeFile, afterFile, beforePreviewUrl, afterPreviewUrl });
      setBeforeDate(approved.beforeDate);
      setAfterDate(approved.afterDate);
      setBackendAvailable(true);
    } catch (caught) {
      if (!controller.signal.aborted && mountedRef.current) {
        setError(requestFailureMessage(caught));
        if (!(caught instanceof ApiRequestError)) setBackendAvailable(false);
      }
    } finally {
      if (mountedRef.current) setSampleLoading(false);
      if (sampleAbortRef.current === controller) sampleAbortRef.current = null;
    }
  }, [configuredDates, requestState, sampleLoading, step.after_sample_id, step.before_sample_id, step.demo_sample_id]);

  const execute = useCallback(async (forceRerun = false) => {
    if (!pair || !query.trim() || !beforeDate || !afterDate || requestState === "running" || requestAbortRef.current) return;
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setRequestState("running"); setError(""); setResult(null);
    try {
      await getHealth();
      if (controller.signal.aborted) return;
      const response = await runAgentImageQuery({
        query: query.trim(),
        inputMode: "bi_temporal",
        primaryModality: pair.workflow.primary_modality,
        secondaryModality: pair.workflow.secondary_modality ?? pair.workflow.primary_modality,
        primaryImage: pair.beforeFile,
        secondaryImage: pair.afterFile,
        primaryDate: beforeDate,
        secondaryDate: afterDate,
        useCache: true,
        forceRerun,
        signal: controller.signal,
      });
      if (!mountedRef.current || controller.signal.aborted) return;
      setBackendAvailable(true); setResult(response); setRequestState("complete");
      window.sessionStorage.setItem("satquery-latest-request-id", response.request_id);
      window.requestAnimationFrame(() => resultRef.current?.focus({ preventScroll: true }));
    } catch (caught) {
      if (!controller.signal.aborted && mountedRef.current) {
        setBackendAvailable(caught instanceof ApiRequestError);
        setRequestState("error");
        setError(requestFailureMessage(caught));
      }
    } finally {
      if (requestAbortRef.current === controller) requestAbortRef.current = null;
      if (mountedRef.current && controller.signal.aborted) setRequestState("idle");
    }
  }, [afterDate, beforeDate, pair, query, requestState]);

  function clearResultForEdit() {
    if (requestState === "running") return;
    setResult(null); setError(""); setRequestState("idle");
  }

  function resetScene() {
    requestAbortRef.current?.abort();
    setResult(null); setError(""); setRequestState("idle");
    setQuery(configuredQuery);
    setBeforeDate(pair?.beforeDate ?? configuredDates.before);
    setAfterDate(pair?.afterDate ?? configuredDates.after);
  }

  const compatibility = result?.change_analysis?.compatibility ?? result?.pair_compatibility ?? null;
  const canRun = Boolean(pair && query.trim() && beforeDate && afterDate && backendAvailable !== false && requestState !== "running");

  return <section className="presentation-change-scene mt-4 min-h-0 flex-1" aria-label="Interactive Bi-Temporal Change Analysis demonstration">
    <article className="presentation-change-source min-w-0 overflow-hidden rounded-2xl border">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[.08] px-3 py-2.5 sm:px-4">
        <div><p className="text-[9px] font-bold uppercase tracking-[.18em] text-amber-200">Approved Local Demo Pair</p><h2 className="mt-0.5 text-sm font-semibold">Spatially corresponding observations</h2></div>
        <div className="flex items-center gap-2"><span className={`rounded-full border px-2.5 py-1 text-[9px] font-semibold ${compatibility?.compatible ? "border-emerald-300/30 text-emerald-200" : compatibility ? "border-amber-300/30 text-amber-100" : "border-white/[.1] text-zinc-400"}`}>{compatibilityLabel(compatibility)}</span><span className={`h-2.5 w-2.5 rounded-full ${backendAvailable === false ? "bg-rose-300" : backendAvailable === true ? "bg-emerald-300" : "bg-zinc-500"}`} aria-label={backendAvailable === false ? "Backend offline" : backendAvailable === true ? "Backend connected" : "Checking backend"}/></div>
      </header>

      {!pair ? <div className="grid min-h-36 place-items-center p-4 text-center"><div><span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-sky-400/10 text-sky-200"><GitCompareArrows size={21}/></span><p className="mt-2 text-sm font-medium">Before/after pair is not loaded</p><p className="mt-1 text-[10px] leading-4 text-zinc-500">Loading retrieves two approved local files only. Analysis starts only when requested.</p><button type="button" onClick={() => void loadSamples()} disabled={sampleLoading || requestState === "running"} className="mt-2 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2 text-xs font-semibold text-zinc-950 disabled:opacity-50">{sampleLoading ? <LoaderCircle className="animate-spin" size={14}/> : <Database size={14}/>} {sampleLoading ? "Loading pair…" : "Load Approved Pair"}</button></div></div> : <div className="grid grid-cols-2 gap-px bg-white/[.06]">
        <SourceObservation label="Before" date={beforeDate} previewUrl={pair.beforePreviewUrl} filename={pair.beforeFile.name}/>
        <SourceObservation label="After" date={afterDate} previewUrl={pair.afterPreviewUrl} filename={pair.afterFile.name}/>
      </div>}
    </article>

    <article className="presentation-change-workspace mt-3 min-w-0 rounded-2xl border p-3 sm:p-4">
      <div className="grid items-end gap-2 md:grid-cols-[150px_150px_minmax(240px,1fr)_auto]">
        <DateField id="presentation-change-before-date" label="Before date" value={beforeDate} disabled={requestState === "running"} onChange={value => { setBeforeDate(value); clearResultForEdit(); }}/>
        <DateField id="presentation-change-after-date" label="After date" value={afterDate} disabled={requestState === "running"} onChange={value => { setAfterDate(value); clearResultForEdit(); }}/>
        <label htmlFor="presentation-change-query" className="min-w-0 text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Controlled question<input id="presentation-change-query" value={query} onChange={event => { setQuery(event.target.value); clearResultForEdit(); }} maxLength={2000} disabled={requestState === "running"} className="presentation-scene-query mt-1 block h-9 w-full rounded-xl border px-3 text-xs font-normal normal-case tracking-normal outline-none"/></label>
        <button type="button" onClick={() => void execute(false)} disabled={!canRun} className="inline-flex h-9 items-center justify-center gap-2 rounded-xl bg-sky-300 px-4 text-xs font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-45">{requestState === "running" ? <LoaderCircle className="animate-spin" size={14}/> : <Play size={14}/>} {requestState === "running" ? "Analyzing…" : "Run Change Analysis"}</button>
      </div>

      <div className="mt-2 flex min-h-8 flex-wrap items-center gap-2">
        {requestState === "running" && <><p role="status" className="mr-auto text-[10px] text-sky-100"><LoaderCircle className="mr-1.5 inline animate-spin" size={12}/>Validating pair, checking ChangerEx, and computing independent evidence…</p><button type="button" onClick={() => requestAbortRef.current?.abort()} className="inline-flex items-center gap-1.5 rounded-lg border border-white/[.1] px-2.5 py-1.5 text-[10px] text-zinc-300"><Square size={11}/>Cancel</button></>}
        {result && <button type="button" onClick={() => void execute(true)} disabled={requestState === "running"} className="inline-flex items-center gap-1.5 rounded-lg border border-white/[.1] px-2.5 py-1.5 text-[10px] text-zinc-300"><RefreshCw size={12}/>Re-run Analysis</button>}
        {(pair || result || error) && <button type="button" onClick={resetScene} className="ml-auto inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] text-zinc-400 hover:bg-white/[.05]"><RotateCcw size={12}/>Reset Scene</button>}
      </div>

      <div aria-live="polite" aria-atomic="true">
        {sampleLoading && <div role="status" className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-2 text-[10px] text-sky-100"><LoaderCircle className="mr-1.5 inline animate-spin" size={12}/>Loading the approved pair. No analysis request has been sent.</div>}
        {error && <div role="alert" className="rounded-xl border border-rose-300/25 bg-rose-400/[.08] p-2.5 text-xs leading-5 text-rose-200"><AlertTriangle className="mr-2 inline" size={14}/>{error}</div>}
      </div>

      {result && <PresentationChangeResult ref={resultRef} response={result}/>}
    </article>
  </section>;
}

function SourceObservation({ label, date, previewUrl, filename }: { label: string; date: string; previewUrl: string; filename: string }) {
  return <figure className="min-w-0 bg-zinc-950/45"><img src={previewUrl} alt={`${label} approved bi-temporal observation`} className="presentation-change-source-image w-full object-contain"/><figcaption className="flex items-center justify-between gap-2 border-t border-white/[.06] px-3 py-1.5"><span className="text-[10px] font-semibold text-zinc-200">{label} · {date}</span><span className="truncate text-[9px] text-zinc-500" title={filename}>{filename}</span></figcaption></figure>;
}

function DateField({ id, label, value, disabled, onChange }: { id: string; label: string; value: string; disabled: boolean; onChange: (value: string) => void }) {
  return <label htmlFor={id} className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">{label}<span className="relative mt-1 block"><CalendarDays className="pointer-events-none absolute left-2.5 top-2.5 text-zinc-500" size={13}/><input id={id} type="date" value={value} disabled={disabled} onChange={event => onChange(event.target.value)} className="presentation-scene-query h-9 w-full rounded-xl border pl-8 pr-2 text-xs font-normal tracking-normal outline-none"/></span></label>;
}

export const PresentationChangeResult = forwardRef<HTMLElement, { response: AgentResponse }>(function PresentationChangeResult({ response }, ref) {
  const change = response.change_analysis;
  const statistics = change?.statistics;
  const compatibility = change?.compatibility ?? response.pair_compatibility;
  const previews = changeEvidencePreviews(response);
  const disclosure = changeResponseDisclosure(response);
  const cacheLabel = response.cache?.cached ? "Cached Result" : "Fresh Analysis";
  const largestPercent = statistics?.regions[0]?.percentage_of_image;
  const engine = response.change_engine ?? change?.change_engine;
  const ttp = response.ttp_result ?? change?.ttp_result;
  const engineName = changeEngineName(engine, ttp);
  const comparison = response.mask_comparison ?? change?.mask_comparison;
  const semantic = response.semantic_change_summary ?? change?.semantic_change_summary;
  return <section ref={ref} tabIndex={-1} aria-labelledby="presentation-change-result-title" className="presentation-change-result min-h-0 overflow-y-auto rounded-2xl border p-3 outline-none focus-visible:ring-2 focus-visible:ring-sky-300/50">
    <div className="flex flex-wrap items-start justify-between gap-2"><div className="min-w-0 flex-1"><p className="text-[9px] font-bold uppercase tracking-[.16em] text-sky-300">{engine?.mode === "hybrid" ? `Hybrid Change Analysis · ${engineName} primary` : engine?.fallback_used ? "Deterministic fallback" : "Controlled statistics-derived answer"}</p><h2 id="presentation-change-result-title" className="mt-1 text-sm font-semibold leading-5 text-white sm:text-base sm:leading-6 lg:text-sm lg:leading-5">{response.answer || "No textual answer was produced."}</h2></div><div className="flex gap-1.5"><span className={`rounded-full border px-2 py-1 text-[9px] font-semibold ${response.cache?.cached ? "border-amber-300/25 text-amber-100" : "border-emerald-300/25 text-emerald-300"}`}>{cacheLabel}</span><span className="rounded-full border border-white/[.09] px-2 py-1 text-[9px] text-zinc-400">{change?.runtime_ms ?? response.execution.duration_ms} ms</span></div></div>

    {semantic && <div className="mt-2 rounded-xl border border-sky-300/18 bg-sky-300/[.05] p-2.5"><p className="text-[9px] font-bold uppercase tracking-[.15em] text-sky-300">What changed</p><p className="mt-1 text-xs leading-5 text-zinc-200">{semantic.expanded_answer}</p><div className="mt-2 grid grid-cols-3 gap-1.5"><ChangeFact name="Likely change" value={formatPresentationLabel(semantic.likely_change_type ?? "unknown semantic change")}/><ChangeFact name="Where" value={formatPresentationLabel(semantic.dominant_location ?? "no dominant location")}/><ChangeFact name="Evidence" value={`${formatPresentationLabel(semantic.evidence_strength)} · ${semantic.supporting_facts.length} signals`}/></div></div>}

    <div className="mt-2 grid gap-2 sm:grid-cols-[.9fr_1.1fr]">
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
        <ChangeFact name={engine?.mode === "hybrid" ? `${engineName} changed area` : "Changed pixels"} value={statistics ? `${statistics.percentage_changed.toFixed(3)}%` : "Unavailable"}/>
        <ChangeFact name={engine?.mode === "hybrid" ? `${engineName} regions` : "Regions"} value={statistics ? statistics.number_of_regions.toLocaleString() : "Unavailable"}/>
        <ChangeFact name="Largest region" value={statistics ? `${statistics.largest_connected_region.toLocaleString()} px${largestPercent === undefined ? "" : ` · ${largestPercent.toFixed(2)}%`}` : "Unavailable"}/>
        <ChangeFact name="Mask IoU" value={comparison?.iou == null ? "Unavailable" : `${comparison.iou.toFixed(3)} · not accuracy`}/>
        <ChangeFact name="Agreement" value={comparison ? `${comparison.agreement_percentage.toFixed(2)}%` : "Unavailable"}/>
        <ChangeFact name="Compatibility" value={compatibilityLabel(compatibility ?? null)}/>
      </div>
      <div><h3 className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Measured evidence products</h3>{previews.length ? <div className="mt-1 grid grid-cols-3 gap-1.5">{previews.map(preview => <ChangePreview key={preview.path} preview={preview}/>)}</div> : <div className="mt-1 rounded-xl border border-white/[.07] p-3 text-center text-[10px] text-zinc-500">No difference, mask, or overlay preview was generated.</div>}</div>
    </div>

    {disclosure && <div className={`mt-2 rounded-xl border p-2 text-[10px] leading-4 ${compatibility?.compatible ? "border-sky-300/18 bg-sky-300/[.05] text-sky-100" : "border-amber-300/22 bg-amber-300/[.06] text-amber-100"}`}>{disclosure}</div>}
    {engine?.fallback_used && <div className="mt-2 rounded-xl border border-amber-300/22 bg-amber-300/[.06] p-2 text-[10px] leading-4 text-amber-100">{engineName} was unavailable ({formatPresentationLabel(engine.fallback_reason ?? "unavailable")}). The deterministic evidence engine completed the request.</div>}
    {ttp?.status === "success" && <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4"><ChangeFact name="Model" value={ttp.model}/><ChangeFact name="Architecture" value={ttp.architecture}/><ChangeFact name="Training data" value={ttp.training_dataset}/><ChangeFact name="Checkpoint / device" value={`${ttp.checkpoint} · ${ttp.device ?? "Unavailable"}`}/><ChangeFact name="Model state" value={ttp.reused_model ? "Warm model reused" : "First loaded inference"}/><ChangeFact name={`${engineName} runtime`} value={ttp.runtime_ms == null ? "Unavailable" : `${ttp.runtime_ms} ms`}/></div>}

    <div className="mt-2">
      <div className="flex items-center justify-between"><h3 className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Execution timeline</h3><span className="flex items-center gap-1 text-[9px] text-zinc-500"><Clock3 size={10}/>{response.execution.steps.length} steps</span></div><ol className="mt-1.5 grid gap-1 sm:grid-cols-2 xl:grid-cols-4">{response.execution.steps.map((trace, index) => <li key={`${trace.tool}-${index}`} className="flex min-w-0 items-center gap-1.5 rounded-lg border border-white/[.06] px-2 py-1"><CheckCircle2 size={10} className={trace.status === "success" ? "shrink-0 text-emerald-300" : trace.status === "failed" ? "shrink-0 text-rose-300" : "shrink-0 text-amber-200"}/><span className="min-w-0 flex-1 truncate font-mono text-[8px] text-zinc-300">{trace.tool}</span><span className="text-[8px] tabular-nums text-zinc-500">{trace.duration_ms}ms</span></li>)}</ol>
    </div>

    <p className="mt-2 text-[9px] leading-4 text-zinc-500"><strong className="text-zinc-400">Confidence disclosure:</strong> {response.evidence_consistency?.label ?? change?.evidence_consistency?.label ?? response.confidence.reason}. This is not a calibrated probability. {engineName} output is not ground truth, and semantic wording is emitted only when supporting evidence passes the local claim gate.</p>
  </section>;
});

function ChangePreview({ preview }: { preview: ChangeEvidencePreview }) {
  const [failed, setFailed] = useState(false);
  return <figure className="min-w-0 overflow-hidden rounded-xl border border-white/[.08] bg-zinc-950/60">{failed ? <div className="grid aspect-[16/8] place-items-center px-2 text-center text-[9px] text-zinc-500"><ImageOff size={14}/><span>Preview unavailable or expired</span></div> : <img src={agentPreviewUrl(preview.path)} onError={() => setFailed(true)} alt={`${preview.label} from real bi-temporal change analysis`} className="aspect-[16/8] w-full object-contain"/>}<figcaption className="truncate border-t border-white/[.06] px-2 py-1 text-[9px] font-medium text-zinc-300">{preview.label}</figcaption></figure>;
}

function ChangeFact({ name, value }: { name: string; value: string }) {
  return <div className="presentation-change-fact min-w-0 rounded-xl border border-white/[.06] bg-white/[.02] p-2"><p className="text-[8px] uppercase tracking-wide text-zinc-500">{name}</p><p className="mt-0.5 break-words text-[10px] font-semibold leading-4 text-zinc-200">{value}</p></div>;
}

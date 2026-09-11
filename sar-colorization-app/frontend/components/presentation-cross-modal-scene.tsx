"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Database,
  ImageOff,
  LoaderCircle,
  Play,
  Radar,
  RefreshCw,
  RotateCcw,
  Square,
} from "lucide-react";
import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import {
  approvedCrossModalPair,
  compatibilityLabel,
  createApprovedRasterPreview,
  crossModalDefaultQuery,
  crossModalDisclosure,
  crossModalEvidencePreviews,
  crossModalFailureMessage,
  preflightCompatibilityLabel,
  type ApprovedCrossModalPair,
  type CrossModalEvidencePreview,
  type RasterPreviewMetadata,
} from "@/lib/presentation-cross-modal";
import { formatPresentationLabel } from "@/lib/presentation-single-image";
import { presentationStorageKeys, type PresentationStep } from "@/lib/presentation";
import { ApiRequestError, agentPreviewUrl, getDemoManifest, getHealth, loadDemoFile, runAgentImageQuery } from "@/services/api";
import type { AgentResponse, ImageMetadata } from "@/types/agent";

type LoadedCrossModalPair = ApprovedCrossModalPair & {
  opticalFile: File;
  sarFile: File;
  opticalPreview: string | null;
  sarPreview: string | null;
  opticalMetadata: RasterPreviewMetadata | null;
  sarMetadata: RasterPreviewMetadata | null;
  previewWarnings: string[];
};

type RequestState = "idle" | "running" | "complete" | "error";

export function PresentationCrossModalScene({ step }: { step: PresentationStep }) {
  const configuredQuery = step.default_query?.cross_modal ?? crossModalDefaultQuery;
  const [pair, setPair] = useState<LoadedCrossModalPair | null>(null);
  const [query, setQuery] = useState(configuredQuery);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [error, setError] = useState("");
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const mountedRef = useRef(true);
  const sampleAbortRef = useRef<AbortController | null>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    setQuery(window.sessionStorage.getItem(presentationStorageKeys.crossModalQuery) || configuredQuery);
  }, [configuredQuery]);

  useEffect(() => {
    window.sessionStorage.setItem(presentationStorageKeys.crossModalQuery, query);
  }, [query]);

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
    };
  }, []);

  const loadSamples = useCallback(async () => {
    if (sampleLoading || sampleAbortRef.current || requestState === "running") return;
    const controller = new AbortController();
    sampleAbortRef.current = controller;
    setSampleLoading(true); setError(""); setResult(null); setRequestState("idle");
    try {
      const manifest = await getDemoManifest(controller.signal);
      const approved = approvedCrossModalPair(
        manifest,
        step.demo_sample_id ?? "cross_modal",
        step.optical_sample_id ?? "cross-optical.tif",
        step.sar_sample_id ?? "cross-sar.tif",
      );
      const [opticalFile, sarFile] = await Promise.all([
        loadDemoFile(approved.optical.url, approved.optical.filename, approved.optical.mime_type, controller.signal),
        loadDemoFile(approved.sar.url, approved.sar.filename, approved.sar.mime_type, controller.signal),
      ]);
      if (!mountedRef.current || controller.signal.aborted) return;
      const previews = await Promise.allSettled([
        createApprovedRasterPreview(opticalFile, approved.workflow.primary_modality as "optical" | "multispectral"),
        createApprovedRasterPreview(sarFile, "sar"),
      ]);
      if (!mountedRef.current || controller.signal.aborted) return;
      const opticalPreview = previews[0].status === "fulfilled" ? previews[0].value : null;
      const sarPreview = previews[1].status === "fulfilled" ? previews[1].value : null;
      const previewWarnings = previews.flatMap(item => item.status === "rejected" ? [crossModalFailureMessage(item.reason)] : []);
      setPair({
        ...approved,
        opticalFile,
        sarFile,
        opticalPreview: opticalPreview?.url ?? null,
        sarPreview: sarPreview?.url ?? null,
        opticalMetadata: opticalPreview?.metadata ?? null,
        sarMetadata: sarPreview?.metadata ?? null,
        previewWarnings,
      });
      setBackendAvailable(true);
    } catch (caught) {
      if (!controller.signal.aborted && mountedRef.current) {
        setError(crossModalFailureMessage(caught));
        if (!(caught instanceof ApiRequestError)) setBackendAvailable(false);
      }
    } finally {
      if (mountedRef.current) setSampleLoading(false);
      if (sampleAbortRef.current === controller) sampleAbortRef.current = null;
    }
  }, [requestState, sampleLoading, step.demo_sample_id, step.optical_sample_id, step.sar_sample_id]);

  const execute = useCallback(async (forceRerun = false) => {
    if (!pair || !query.trim() || requestState === "running" || requestAbortRef.current) return;
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setRequestState("running"); setError(""); setResult(null);
    try {
      await getHealth();
      if (controller.signal.aborted) return;
      const response = await runAgentImageQuery({
        query: query.trim(),
        inputMode: "cross_modal",
        primaryModality: pair.workflow.primary_modality,
        secondaryModality: "sar",
        primaryImage: pair.opticalFile,
        secondaryImage: pair.sarFile,
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
        setError(crossModalFailureMessage(caught));
      }
    } finally {
      if (requestAbortRef.current === controller) requestAbortRef.current = null;
      if (mountedRef.current && controller.signal.aborted) setRequestState("idle");
    }
  }, [pair, query, requestState]);

  function clearResultForEdit() {
    if (requestState === "running") return;
    setResult(null); setError(""); setRequestState("idle");
  }

  function resetScene() {
    requestAbortRef.current?.abort();
    setResult(null); setError(""); setRequestState("idle"); setQuery(configuredQuery);
  }

  const compatibility = result?.pair_compatibility ?? null;
  const preflight = pair?.opticalMetadata && pair.sarMetadata
    ? preflightCompatibilityLabel(pair.opticalMetadata, pair.sarMetadata)
    : pair ? "Preflight · backend validation required" : "Compatibility pending";
  const compatibilityText = compatibility ? compatibilityLabel(compatibility) : preflight;
  const canRun = Boolean(pair && query.trim() && backendAvailable !== false && requestState !== "running");

  return <section className="presentation-cross-modal-scene mt-4 min-h-0 flex-1" aria-label="Interactive Optical and SAR Joint Analysis demonstration">
    <article className="presentation-cross-modal-source min-w-0 overflow-hidden rounded-2xl border">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[.08] px-3 py-2 sm:px-4">
        <div><p className="text-[9px] font-bold uppercase tracking-[.18em] text-amber-200">Approved Local Optical–SAR Pair</p><h2 className="mt-0.5 text-sm font-semibold">Complementary source observations</h2></div>
        <div className="flex items-center gap-2"><span className={`rounded-full border px-2.5 py-1 text-[9px] font-semibold ${compatibility?.compatible || preflight.includes("exact") ? "border-emerald-300/30 text-emerald-200" : compatibility || pair ? "border-amber-300/30 text-amber-100" : "border-white/[.1] text-zinc-400"}`}>{compatibilityText}</span><span className={`h-2.5 w-2.5 rounded-full ${backendAvailable === false ? "bg-rose-300" : backendAvailable === true ? "bg-emerald-300" : "bg-zinc-500"}`} aria-label={backendAvailable === false ? "Backend offline" : backendAvailable === true ? "Backend connected" : "Checking backend"}/></div>
      </header>

      {!pair ? <div className="grid min-h-36 place-items-center p-4 text-center"><div><span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-sky-400/10 text-sky-200"><Radar size={21}/></span><p className="mt-2 text-sm font-medium">Optical and SAR samples are not loaded</p><p className="mt-1 text-[10px] leading-4 text-zinc-500">Loading retrieves and previews two approved local files. Joint analysis starts only when requested.</p><button type="button" onClick={() => void loadSamples()} disabled={sampleLoading || requestState === "running"} className="mt-2 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2 text-xs font-semibold text-zinc-950 disabled:opacity-50">{sampleLoading ? <LoaderCircle className="animate-spin" size={14}/> : <Database size={14}/>} {sampleLoading ? "Loading pair…" : "Load Approved Pair"}</button></div></div> : <div className="grid grid-cols-2 gap-px bg-white/[.06]">
        <SourceObservation label="Optical" modality={formatPresentationLabel(pair.workflow.primary_modality)} filename={pair.opticalFile.name} previewUrl={pair.opticalPreview} localMetadata={pair.opticalMetadata} backendMetadata={result?.primary_image_metadata ?? null}/>
        <SourceObservation label="SAR" modality="Synthetic aperture radar" filename={pair.sarFile.name} previewUrl={pair.sarPreview} localMetadata={pair.sarMetadata} backendMetadata={result?.secondary_image_metadata ?? null}/>
      </div>}
    </article>

    <article className="presentation-cross-modal-workspace mt-3 min-w-0 rounded-2xl border p-3 sm:p-4">
      <div className="grid items-end gap-2 md:grid-cols-[minmax(300px,1fr)_auto]">
        <label htmlFor="presentation-cross-modal-query" className="min-w-0 text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Controlled joint-analysis question<input id="presentation-cross-modal-query" value={query} onChange={event => { setQuery(event.target.value); clearResultForEdit(); }} maxLength={2000} disabled={requestState === "running"} className="presentation-scene-query mt-1 block h-9 w-full rounded-xl border px-3 text-xs font-normal normal-case tracking-normal outline-none"/></label>
        <button type="button" onClick={() => void execute(false)} disabled={!canRun} className="inline-flex h-9 items-center justify-center gap-2 rounded-xl bg-sky-300 px-4 text-xs font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-45">{requestState === "running" ? <LoaderCircle className="animate-spin" size={14}/> : <Play size={14}/>} {requestState === "running" ? "Analyzing…" : "Run Joint Analysis"}</button>
      </div>

      <div className="mt-2 flex min-h-7 flex-wrap items-center gap-2">
        {requestState === "running" && <><p role="status" className="mr-auto text-[10px] text-sky-100"><LoaderCircle className="mr-1.5 inline animate-spin" size={12}/>Validating the pair and computing deterministic optical–SAR evidence…</p><button type="button" onClick={() => requestAbortRef.current?.abort()} className="inline-flex items-center gap-1.5 rounded-lg border border-white/[.1] px-2.5 py-1.5 text-[10px] text-zinc-300"><Square size={11}/>Cancel</button></>}
        {result && <button type="button" onClick={() => void execute(true)} disabled={requestState === "running"} className="inline-flex items-center gap-1.5 rounded-lg border border-white/[.1] px-2.5 py-1.5 text-[10px] text-zinc-300"><RefreshCw size={12}/>Re-run Analysis</button>}
        {(pair || result || error) && <button type="button" onClick={resetScene} className="ml-auto inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] text-zinc-400 hover:bg-white/[.05]"><RotateCcw size={12}/>Reset Scene</button>}
      </div>

      <div aria-live="polite" aria-atomic="true">
        {sampleLoading && <div role="status" className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-2 text-[10px] text-sky-100"><LoaderCircle className="mr-1.5 inline animate-spin" size={12}/>Loading and previewing the approved pair. No analysis request has been sent.</div>}
        {backendAvailable === false && !error && <div role="alert" className="rounded-xl border border-rose-300/25 bg-rose-400/[.08] p-2 text-[10px] text-rose-200"><AlertTriangle className="mr-1.5 inline" size={12}/>The local SatQuery backend is offline. Sample previews may load only after it restarts on port 8010.</div>}
        {pair?.previewWarnings.map(warning => <div key={warning} role="alert" className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-2 text-[10px] text-amber-100"><ImageOff className="mr-1.5 inline" size={12}/>{warning}</div>)}
        {error && <div role="alert" className="rounded-xl border border-rose-300/25 bg-rose-400/[.08] p-2.5 text-xs leading-5 text-rose-200"><AlertTriangle className="mr-2 inline" size={14}/>{error}</div>}
      </div>

      {result && <PresentationCrossModalResult ref={resultRef} response={result}/>} 
    </article>
  </section>;
}

function SourceObservation({ label, modality, filename, previewUrl, localMetadata, backendMetadata }: { label: string; modality: string; filename: string; previewUrl: string | null; localMetadata: RasterPreviewMetadata | null; backendMetadata: ImageMetadata | null }) {
  const width = backendMetadata?.width ?? localMetadata?.width;
  const height = backendMetadata?.height ?? localMetadata?.height;
  const bands = backendMetadata?.band_count ?? localMetadata?.bandCount;
  const dtype = backendMetadata?.dtype ?? localMetadata?.dtype;
  const crs = backendMetadata?.crs ?? localMetadata?.crs;
  return <figure className="min-w-0 bg-zinc-950/45">{previewUrl ? <img src={previewUrl} alt={`${label} approved cross-modal observation`} className="presentation-cross-modal-source-image w-full object-contain"/> : <div className="presentation-cross-modal-source-image grid place-items-center text-zinc-500"><ImageOff size={18}/><span className="text-[9px]">Source preview unavailable</span></div>}<figcaption className="border-t border-white/[.06] px-3 py-1.5"><div className="flex items-center justify-between gap-2"><span className="text-[10px] font-semibold text-zinc-200">{label} · {modality}</span><span className="truncate text-[9px] text-zinc-500" title={filename}>{filename}</span></div><p className="mt-0.5 truncate text-[9px] text-zinc-500">{width && height ? `${width} × ${height}` : "Dimensions unavailable"} · {bands ? `${bands} band${bands === 1 ? "" : "s"}` : "bands unavailable"} · {dtype || "dtype unavailable"} · {crs || "no CRS"}</p></figcaption></figure>;
}

export const PresentationCrossModalResult = forwardRef<HTMLElement, { response: AgentResponse }>(function PresentationCrossModalResult({ response }, ref) {
  const result = response.cross_modal_analysis;
  const statistics = result?.statistics;
  const previews = crossModalEvidencePreviews(response);
  const disclosure = crossModalDisclosure(response);
  const cacheLabel = response.cache?.cached ? "Cached Result" : "Fresh Analysis";
  const structured = result?.summary.joint_observations ?? [];
  const answer = response.vqa_details?.supported && response.answer ? response.answer : structured.join(" ") || response.answer || "No joint evidence answer was produced.";
  const metric = (value: number | null | undefined) => value === null || value === undefined ? "Unavailable" : `${value.toFixed(3)}%`;
  return <section ref={ref} tabIndex={-1} aria-labelledby="presentation-cross-modal-result-title" className="presentation-cross-modal-result min-h-0 overflow-y-auto rounded-2xl border p-3 outline-none focus-visible:ring-2 focus-visible:ring-sky-300/50">
    <div className="flex flex-wrap items-start justify-between gap-2"><div className="min-w-0 flex-1"><p className="text-[9px] font-bold uppercase tracking-[.16em] text-sky-300">Structured statistics-derived answer</p><h2 id="presentation-cross-modal-result-title" className="mt-1 text-sm font-semibold leading-5 text-white sm:text-base sm:leading-6 lg:text-sm lg:leading-5">{answer}</h2></div><div className="flex flex-wrap gap-1.5"><span className={`rounded-full border px-2 py-1 text-[9px] font-semibold ${response.cache?.cached ? "border-amber-300/25 text-amber-100" : "border-emerald-300/25 text-emerald-300"}`}>{cacheLabel}</span><span className="rounded-full border border-white/[.09] px-2 py-1 text-[9px] text-zinc-400">{compatibilityLabel(response.pair_compatibility)}</span><span className="rounded-full border border-white/[.09] px-2 py-1 text-[9px] text-zinc-400">{result?.runtime_ms ?? response.execution.duration_ms} ms</span></div></div>

    <div className="presentation-cross-modal-facts mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-6">
      <CrossModalFact name="Water likelihood" value={metric(statistics?.water_likelihood_percent)}/>
      <CrossModalFact name="Structural likelihood" value={metric(statistics?.built_up_likelihood_percent)}/>
      <CrossModalFact name="Vegetation support" value={metric(statistics?.vegetation_support_percent)}/>
      <CrossModalFact name="Agreement" value={metric(statistics?.agreement_percent)}/>
      <CrossModalFact name="Disagreement" value={metric(statistics?.disagreement_percent)}/>
      <CrossModalFact name="Valid pixels" value={metric(statistics?.valid_pixel_percent)}/>
    </div>

    {disclosure && <div className="mt-2 rounded-xl border border-amber-300/22 bg-amber-300/[.06] p-2 text-[10px] leading-4 text-amber-100">{disclosure}</div>}

    {previews.length > 0 && <div className="mt-3"><div className="flex items-center justify-between"><h3 className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Real evidence products</h3><span className="text-[9px] text-zinc-500">{previews.length} previews</span></div><div className="mt-1.5 grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">{previews.map(preview => <EvidencePreview key={`${preview.label}-${preview.path}`} preview={preview}/>)}</div></div>}

    <div className="mt-3 grid gap-2 lg:grid-cols-[.8fr_1.2fr]">
      <div className="rounded-xl border border-white/[.06] p-2"><h3 className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Connected evidence regions</h3><p className="mt-1 text-xs font-semibold text-zinc-200">{result ? `${result.regions.length.toLocaleString()} region${result.regions.length === 1 ? "" : "s"}` : "Unavailable"}</p>{result?.regions.length ? <ul className="mt-1 space-y-1">{result.regions.slice(0, 4).map(region => <li key={region.region_id} className="flex items-center justify-between gap-2 text-[9px] text-zinc-400"><span>{formatPresentationLabel(region.type)} · {region.area_pixels.toLocaleString()} px</span><span className="font-mono">[{region.bbox_pixels.join(", ")}]</span></li>)}</ul> : <p className="mt-1 text-[9px] text-zinc-500">No connected joint-evidence regions were returned.</p>}</div>
      <div><div className="flex items-center justify-between"><h3 className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Execution timeline</h3><span className="flex items-center gap-1 text-[9px] text-zinc-500"><Clock3 size={10}/>{response.execution.steps.length} steps</span></div><ol className="mt-1 grid gap-1 sm:grid-cols-2 xl:grid-cols-4">{response.execution.steps.map((trace, index) => <li key={`${trace.tool}-${index}`} className="flex min-w-0 items-center gap-1.5 rounded-lg border border-white/[.06] px-2 py-1"><CheckCircle2 size={10} className={trace.status === "success" ? "shrink-0 text-emerald-300" : trace.status === "failed" ? "shrink-0 text-rose-300" : "shrink-0 text-amber-200"}/><span className="min-w-0 flex-1 truncate font-mono text-[8px] text-zinc-300">{trace.tool}</span><span className="text-[8px] tabular-nums text-zinc-500">{trace.duration_ms}ms</span></li>)}</ol></div>
    </div>

    {result && <div className="mt-2 grid gap-2 sm:grid-cols-2"><p className="text-[9px] leading-4 text-zinc-500"><strong className="text-zinc-400">Confidence disclosure:</strong> {formatPresentationLabel(result.confidence.level)} · no calibrated score. {result.confidence.reason}</p><p className="text-[9px] leading-4 text-zinc-500"><strong className="text-zinc-400">Limitations:</strong> {result.method.limitations.join(" ")}</p></div>}
    {(result?.warnings.length || response.warnings.length) ? <details className="mt-2 rounded-xl border border-white/[.06] px-2 py-1.5 text-[9px] text-zinc-500"><summary className="cursor-pointer font-semibold text-zinc-400">Warnings and scientific cautions</summary><ul className="mt-1 space-y-1">{[...(result?.warnings ?? []), ...response.warnings].filter((item, index, all) => all.indexOf(item) === index).map(item => <li key={item}>• {item}</li>)}</ul></details> : null}
  </section>;
});

function EvidencePreview({ preview }: { preview: CrossModalEvidencePreview }) {
  const [failed, setFailed] = useState(false);
  return <figure className="min-w-0 overflow-hidden rounded-xl border border-white/[.08] bg-zinc-950/60">{failed ? <div className="grid aspect-[16/8] place-items-center px-2 text-center text-[9px] text-zinc-500"><ImageOff size={14}/><span>Preview unavailable or expired</span></div> : <img src={agentPreviewUrl(preview.path)} onError={() => setFailed(true)} alt={`${preview.label} from real optical–SAR analysis`} className="aspect-[16/8] w-full object-contain"/>}<figcaption className="truncate border-t border-white/[.06] px-2 py-1 text-[9px] font-medium text-zinc-300">{preview.label}</figcaption></figure>;
}

function CrossModalFact({ name, value }: { name: string; value: string }) {
  return <div className="presentation-cross-modal-fact min-w-0 rounded-xl border border-white/[.06] bg-white/[.02] p-2"><p className="text-[8px] uppercase tracking-wide text-zinc-500">{name}</p><p className="mt-0.5 break-words text-[10px] font-semibold leading-4 text-zinc-200">{value}</p></div>;
}

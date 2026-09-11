"use client";

import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
  FileImage,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Square,
} from "lucide-react";
import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import {
  ApiRequestError,
  agentPreviewUrl,
  getDemoManifest,
  getHealth,
  loadDemoFile,
  runAgentImageQuery,
} from "@/services/api";
import {
  approvedSingleImageWorkflow,
  formatPresentationLabel,
  responseFailureMessage,
  singleImageActionDetails,
  singleImageDefaultQueries,
  vqaEvidencePreviews,
  type EvidencePreview,
  type SingleImagePresentationAction,
} from "@/lib/presentation-single-image";
import { presentationStorageKeys, type PresentationStep } from "@/lib/presentation";
import type { AgentResponse, DemoWorkflow } from "@/types/agent";
import { RemoteSensingAdaptationPanel } from "@/components/remote-sensing-adaptation-panel";

type LoadedSample = {
  workflow: DemoWorkflow;
  file: File;
  previewUrl: string;
  width: number | null;
  height: number | null;
};

type RequestState = "idle" | "running" | "complete" | "error";

export function PresentationSingleImageScene({ step }: { step: PresentationStep }) {
  const defaultAction: SingleImagePresentationAction = "captioning";
  const [action, setAction] = useState<SingleImagePresentationAction>(defaultAction);
  const [query, setQuery] = useState(step.default_query?.captioning ?? singleImageDefaultQueries.captioning);
  const [sample, setSample] = useState<LoadedSample | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [error, setError] = useState("");
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const mountedRef = useRef(true);
  const sampleUrlRef = useRef<string | null>(null);
  const sampleAbortRef = useRef<AbortController | null>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const savedAction = window.sessionStorage.getItem(presentationStorageKeys.singleImageAction);
    const restoredAction = savedAction === "vqa" ? "vqa" : "captioning";
    const savedQuery = window.sessionStorage.getItem(presentationStorageKeys.singleImageQuery);
    setAction(restoredAction);
    setQuery(savedQuery || step.default_query?.[restoredAction] || singleImageDefaultQueries[restoredAction]);
  }, [step.default_query]);

  useEffect(() => {
    window.sessionStorage.setItem(presentationStorageKeys.singleImageAction, action);
    window.sessionStorage.setItem(presentationStorageKeys.singleImageQuery, query);
  }, [action, query]);

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
      if (sampleUrlRef.current) URL.revokeObjectURL(sampleUrlRef.current);
    };
  }, []);

  const loadSample = useCallback(async () => {
    if (sampleLoading || sampleAbortRef.current || requestState === "running") return;
    const controller = new AbortController();
    sampleAbortRef.current = controller;
    setSampleLoading(true); setError(""); setResult(null); setRequestState("idle");
    try {
      const manifest = await getDemoManifest(controller.signal);
      const workflow = approvedSingleImageWorkflow(manifest, step.demo_sample_id ?? "single_vqa");
      const descriptor = workflow.files.find(item => item.role === "primary");
      if (!descriptor) throw new Error("The approved demo manifest did not provide a primary image.");
      const file = await loadDemoFile(descriptor.url, descriptor.filename, descriptor.mime_type, controller.signal);
      const previewUrl = URL.createObjectURL(file);
      const dimensions = await decodeImageDimensions(previewUrl, controller.signal);
      if (!mountedRef.current || controller.signal.aborted) { URL.revokeObjectURL(previewUrl); return; }
      if (sampleUrlRef.current) URL.revokeObjectURL(sampleUrlRef.current);
      sampleUrlRef.current = previewUrl;
      setSample({ workflow, file, previewUrl, ...dimensions });
      setBackendAvailable(true);
    } catch (caught) {
      if (!controller.signal.aborted && mountedRef.current) {
        setError(caught instanceof Error ? caught.message : "The approved local demo sample could not be loaded.");
      }
    } finally {
      if (mountedRef.current) setSampleLoading(false);
      if (sampleAbortRef.current === controller) sampleAbortRef.current = null;
    }
  }, [requestState, sampleLoading, step.demo_sample_id]);

  const execute = useCallback(async (forceRerun = false) => {
    if (!sample || !query.trim() || requestState === "running" || requestAbortRef.current) return;
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setRequestState("running"); setError(""); setResult(null);
    try {
      await getHealth();
      if (controller.signal.aborted) return;
      const response = await runAgentImageQuery({
        query: query.trim(), inputMode: "single", primaryModality: sample.workflow.primary_modality,
        secondaryModality: null, primaryImage: sample.file, useCache: true, forceRerun, signal: controller.signal,
      });
      if (!mountedRef.current || controller.signal.aborted) return;
      setBackendAvailable(true); setResult(response); setRequestState("complete");
      window.sessionStorage.setItem("satquery-latest-request-id", response.request_id);
      window.requestAnimationFrame(() => resultRef.current?.focus({ preventScroll: true }));
    } catch (caught) {
      if (!controller.signal.aborted && mountedRef.current) {
        setBackendAvailable(caught instanceof ApiRequestError); setRequestState("error");
        setError(caught instanceof Error ? caught.message : "The local SatQuery analysis request failed.");
      }
    } finally {
      if (requestAbortRef.current === controller) requestAbortRef.current = null;
      if (mountedRef.current && controller.signal.aborted) setRequestState("idle");
    }
  }, [query, requestState, sample]);

  function selectAction(next: SingleImagePresentationAction) {
    if (requestState === "running") return;
    setAction(next);
    setQuery(step.default_query?.[next] ?? singleImageDefaultQueries[next]);
    setResult(null); setError(""); setRequestState("idle");
  }

  function resetScene() {
    requestAbortRef.current?.abort();
    setResult(null); setError(""); setRequestState("idle");
    setQuery(step.default_query?.[action] ?? singleImageDefaultQueries[action]);
  }

  const metadata = result?.primary_image_metadata ?? null;
  const canRun = Boolean(sample && query.trim() && backendAvailable !== false && requestState !== "running");

  return <section className="presentation-single-scene mt-5 grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(270px,.78fr)_minmax(430px,1.22fr)]" aria-label="Interactive Single-Image Understanding demonstration">
    <article className="presentation-scene-source min-w-0 overflow-hidden rounded-2xl border">
      <header className="flex items-center justify-between gap-3 border-b border-white/[.08] px-4 py-3">
        <div><p className="text-[9px] font-bold uppercase tracking-[.18em] text-amber-200">Approved Local Demo Sample</p><h2 className="mt-1 text-sm font-semibold">Single optical observation</h2></div>
        <span className={`h-2.5 w-2.5 rounded-full ${backendAvailable === false ? "bg-rose-300" : backendAvailable === true ? "bg-emerald-300" : "bg-zinc-500"}`} aria-label={backendAvailable === false ? "Backend offline" : backendAvailable === true ? "Backend connected" : "Checking backend"}/>
      </header>

      {!sample ? <div className="grid min-h-64 place-items-center p-5 text-center">
        <div><span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-sky-400/10 text-sky-200"><FileImage size={25}/></span><p className="mt-4 text-sm font-medium">Sample is not loaded</p><p className="mt-2 max-w-xs text-xs leading-5 text-zinc-500">Loading retrieves only the approved local file. It does not execute a model.</p><button type="button" onClick={() => void loadSample()} disabled={sampleLoading || requestState === "running"} className="mt-4 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2.5 text-xs font-semibold text-zinc-950 disabled:opacity-50">{sampleLoading ? <LoaderCircle className="animate-spin" size={15}/> : <Database size={15}/>} {sampleLoading ? "Loading sample…" : "Load Approved Sample"}</button></div>
      </div> : <div>
        <figure className="presentation-scene-image bg-zinc-950/60"><img src={sample.previewUrl} alt="Approved local optical demo sample for Single-Image Understanding" className="h-full w-full object-contain"/><figcaption className="border-t border-white/[.06] px-3 py-2 text-center text-[10px] text-amber-100">Source image · no analysis has been inferred from this preview</figcaption></figure>
        <dl className="grid grid-cols-2 gap-3 p-4 text-xs">
          <SceneFact name="Modality" value={formatPresentationLabel(sample.workflow.primary_modality)}/>
          <SceneFact name="Dimensions" value={metadata ? `${metadata.width} × ${metadata.height}` : sample.width && sample.height ? `${sample.width} × ${sample.height}` : "Preview unavailable"}/>
          <SceneFact name="Bands" value={metadata ? String(metadata.band_count) : "Inspected during analysis"}/>
          <SceneFact name="Format" value={metadata ? metadata.format.toUpperCase() : sample.file.type.replace("image/", "").toUpperCase() || "Unknown"}/>
          <SceneFact name="File" value={sample.file.name}/>
          <SceneFact name="Size" value={`${(sample.file.size / 1024).toFixed(1)} KB`}/>
        </dl>
      </div>}
    </article>

    <article className="presentation-scene-workspace min-w-0 rounded-2xl border p-4 sm:p-5">
      <div className="grid gap-2 sm:grid-cols-2" role="group" aria-label="Single-image analysis action">
        {(["captioning", "vqa"] as const).map(value => {
          const details = singleImageActionDetails[value];
          const selected = action === value;
          return <button key={value} type="button" onClick={() => selectAction(value)} disabled={requestState === "running"} aria-pressed={selected} className={`presentation-action-card rounded-xl border p-3 text-left ${selected ? "presentation-action-card-selected" : ""}`}>
            <span className="flex items-center gap-2 text-sm font-semibold">{value === "captioning" ? <Sparkles size={16}/> : <BrainCircuit size={16}/>} {details.label}</span>
            <span className="mt-1.5 block text-[10px] leading-4 text-zinc-500">{details.method}</span>
          </button>;
        })}
      </div>

      <label htmlFor="presentation-single-image-query" className="mt-3 block text-[10px] font-bold uppercase tracking-[.16em] text-zinc-500">{action === "captioning" ? "Caption request" : "Controlled question"}</label>
      <textarea id="presentation-single-image-query" value={query} onChange={event => { setQuery(event.target.value); setResult(null); setError(""); setRequestState("idle"); }} rows={2} maxLength={2000} disabled={requestState === "running"} className="presentation-scene-query mt-1.5 block w-full resize-none rounded-xl border px-3 py-2.5 text-sm leading-5 outline-none"/>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => void execute(false)} disabled={!canRun} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2.5 text-xs font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-45">{requestState === "running" ? <LoaderCircle className="animate-spin" size={15}/> : <Play size={15}/>} {requestState === "running" ? "Running real workflow…" : "Run Analysis"}</button>
        {requestState === "running" && <button type="button" onClick={() => requestAbortRef.current?.abort()} className="inline-flex items-center gap-2 rounded-xl border border-white/[.10] px-3 py-2.5 text-xs text-zinc-300"><Square size={13}/>Cancel</button>}
        {result && <button type="button" onClick={() => void execute(true)} disabled={requestState === "running"} className="inline-flex items-center gap-2 rounded-xl border border-white/[.10] px-3 py-2.5 text-xs text-zinc-300"><RefreshCw size={14}/>Re-run</button>}
        {(sample || result || error) && <button type="button" onClick={resetScene} className="ml-auto inline-flex items-center gap-2 rounded-xl px-3 py-2.5 text-xs text-zinc-400 hover:bg-white/[.05]"><RotateCcw size={14}/>Reset Scene</button>}
      </div>

      <div className="mt-3 min-h-0" aria-live="polite" aria-atomic="true">
        {sampleLoading && <ProgressNotice mode="sample"/>}
        {requestState === "running" && <ProgressNotice mode="analysis"/>}
        {error && <div role="alert" className="rounded-xl border border-rose-300/25 bg-rose-400/[.08] p-3 text-xs leading-5 text-rose-200"><AlertTriangle className="mr-2 inline" size={14}/>{error}{backendAvailable === false && <span className="mt-1 block text-rose-100">Start the local FastAPI backend on port 8010, then retry.</span>}</div>}
      </div>

      {result && <PresentationSingleImageResult ref={resultRef} result={result} action={action}/>}
    </article>
  </section>;
}

function ProgressNotice({ mode }: { mode: "sample" | "analysis" }) {
  if (mode === "sample") return <div className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-3 text-xs text-sky-100"><LoaderCircle className="mr-2 inline animate-spin" size={14}/>Loading sample from the approved local demo endpoint. No model is running.</div>;
  const stages = ["Validating image", "Routing query", "Loading or reusing specialist", "Running analysis", "Preparing evidence"];
  return <div className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-3"><p className="flex items-center gap-2 text-xs font-medium text-sky-100"><LoaderCircle className="animate-spin" size={14}/>Real backend request in progress</p><p className="mt-1 text-[10px] leading-4 text-zinc-500">Exact completion and timing are reported only when the backend trace returns.</p><div className="mt-2 flex flex-wrap gap-1.5">{stages.map(stage => <span key={stage} className="rounded-full border border-white/[.07] px-2 py-1 text-[9px] text-zinc-400">{stage}</span>)}</div></div>;
}

const PresentationSingleImageResult = forwardRef<HTMLElement, { result: AgentResponse; action: SingleImagePresentationAction }>(function PresentationSingleImageResult({ result, action }, ref) {
  const failure = responseFailureMessage(result, action);
  const cacheLabel = result.cache?.cached ? "Cached Result" : "Fresh Analysis";
  const previews = vqaEvidencePreviews(result);
  const details = result.vqa_details;
  const caption = result.caption_details;
  const evidence = details?.single_image_evidence;
  return <section ref={ref} tabIndex={-1} aria-labelledby="presentation-analysis-result-title" className="presentation-scene-result mt-4 min-h-0 overflow-y-auto rounded-2xl border p-4 outline-none focus-visible:ring-2 focus-visible:ring-sky-300/50">
    <div className="flex flex-wrap items-start justify-between gap-2"><div><p className="text-[9px] font-bold uppercase tracking-[.16em] text-sky-300">{action === "captioning" ? "Scene Description" : "Controlled VQA Answer"}</p><h2 id="presentation-analysis-result-title" className="mt-1 text-xl font-semibold leading-7 text-white">{result.answer || (failure ? "Specialist unavailable" : "No answer returned")}</h2></div><div className="flex gap-1.5"><span className={`rounded-full border px-2 py-1 text-[9px] font-semibold ${result.cache?.cached ? "border-amber-300/25 text-amber-100" : "border-emerald-300/25 text-emerald-300"}`}>{cacheLabel}</span><span className="rounded-full border border-white/[.09] px-2 py-1 text-[9px] text-zinc-400">{result.execution.duration_ms} ms</span></div></div>

    {failure && <div className="mt-3 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-3 text-xs leading-5 text-amber-100">{failure}</div>}
    {action === "vqa" && details && <p className="mt-2 text-xs text-zinc-400">Question: {details.original_question}</p>}

    <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {action === "captioning" ? <>
        <ResultFact name="Checkpoint" value={result.model?.checkpoint ?? "Unavailable"}/>
        <ResultFact name="Adaptation" value={result.model?.adaptation_dataset ?? "Unavailable"}/>
        <ResultFact name="Device" value={caption?.device ?? "Unavailable"}/>
        <ResultFact name="Model lifecycle" value={caption ? caption.model_reused ? "Reused loaded model" : `First load · ${caption.model_load_ms} ms` : "Unavailable"}/>
      </> : <>
        <ResultFact name="Question category" value={details ? formatPresentationLabel(details.question_category) : "Unavailable"}/>
        <ResultFact name="Answer source" value={details?.answer_source ?? "Unavailable"}/>
        <ResultFact name="Method" value={details?.method.name ?? "Unavailable"}/>
        <ResultFact name="Language model" value={details ? details.method.uses_language_model ? "Yes" : "No — deterministic" : "Unavailable"}/>
      </>}
    </div>

    <div className="mt-3 rounded-xl border border-white/[.07] bg-white/[.025] p-3"><p className="text-[9px] font-bold uppercase tracking-[.15em] text-zinc-500">Confidence disclosure</p><p className="mt-1 text-xs leading-5 text-zinc-300"><strong>{formatPresentationLabel(result.confidence.level)}</strong>{result.confidence.score !== null ? ` · ${result.confidence.score.toFixed(3)}` : " · no calibrated score"}. {result.confidence.reason}</p></div>

    {result.sve_result && <div className="mt-3"><RemoteSensingAdaptationPanel result={result.sve_result} compact/></div>}

    {evidence && <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4"><ResultFact name="Dominant scene" value={formatPresentationLabel(evidence.dominant_scene)}/><ResultFact name="Water support" value={`${evidence.statistics.water_support_percent.toFixed(2)}%`}/><ResultFact name="Vegetation support" value={`${evidence.statistics.vegetation_support_percent.toFixed(2)}%`}/><ResultFact name="Structural support" value={`${evidence.statistics.built_up_support_percent.toFixed(2)}%`}/></div>}

    {previews.length > 0 && <div className="mt-3"><h3 className="text-xs font-semibold">Evidence products</h3><div className="mt-2 flex gap-2 overflow-x-auto pb-1">{previews.map(preview => <EvidenceThumbnail key={preview.path} preview={preview}/>)}</div></div>}

    <div className="mt-3 grid gap-3 xl:grid-cols-[.85fr_1.15fr]">
      <div><h3 className="text-[10px] font-semibold uppercase tracking-[.14em] text-zinc-500">Limitations</h3><ul className="mt-1.5 space-y-1 text-[10px] leading-4 text-zinc-400">{(caption?.limitations ?? details?.limitations ?? result.warnings).slice(0, 3).map(item => <li key={item}>• {item}</li>)}{!(caption?.limitations ?? details?.limitations ?? result.warnings).length && <li>• No specialist limitation text was returned.</li>}</ul></div>
      <div><div className="flex items-center justify-between"><h3 className="text-[10px] font-semibold uppercase tracking-[.14em] text-zinc-500">Actual execution trace</h3><span className="flex items-center gap-1 text-[9px] text-zinc-500"><Clock3 size={11}/>{result.execution.steps.length} steps</span></div><ol className="mt-1.5 grid gap-1 sm:grid-cols-2">{result.execution.steps.slice(0, 10).map((trace, index) => <li key={`${trace.tool}-${index}`} className="flex min-w-0 items-center gap-2 rounded-lg border border-white/[.06] px-2 py-1.5"><CheckCircle2 size={11} className={trace.status === "success" ? "shrink-0 text-emerald-300" : "shrink-0 text-amber-200"}/><span className="min-w-0 flex-1 truncate font-mono text-[9px] text-zinc-300">{trace.tool}</span><span className="text-[9px] tabular-nums text-zinc-500">{trace.duration_ms}ms</span></li>)}</ol></div>
    </div>
  </section>;
});

function EvidenceThumbnail({ preview }: { preview: EvidencePreview }) {
  const [failed, setFailed] = useState(false);
  return <figure className="w-36 shrink-0 overflow-hidden rounded-xl border border-white/[.08] bg-zinc-950/60">{failed ? <div className="grid aspect-video place-items-center px-2 text-center text-[9px] text-zinc-500">Preview unavailable or expired</div> : <img src={agentPreviewUrl(preview.path)} onError={() => setFailed(true)} alt={`${preview.label} heuristic evidence preview`} className="aspect-video w-full object-contain"/>}<figcaption className="truncate border-t border-white/[.06] px-2 py-1.5 text-[9px] text-zinc-400">{preview.label}</figcaption></figure>;
}

function SceneFact({ name, value }: { name: string; value: string }) {
  return <div className="min-w-0"><dt className="text-[9px] uppercase tracking-wide text-zinc-500">{name}</dt><dd className="mt-0.5 truncate text-[11px] text-zinc-200" title={value}>{value}</dd></div>;
}

function ResultFact({ name, value }: { name: string; value: string }) {
  return <div className="min-w-0 rounded-xl border border-white/[.06] bg-white/[.02] p-2"><p className="text-[9px] uppercase tracking-wide text-zinc-500">{name}</p><p className="mt-0.5 break-words text-[10px] leading-4 text-zinc-200">{value}</p></div>;
}

function decodeImageDimensions(url: string, signal: AbortSignal): Promise<{ width: number | null; height: number | null }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const cleanup = () => { image.onload = null; image.onerror = null; signal.removeEventListener("abort", abort); };
    const abort = () => { cleanup(); reject(new DOMException("Sample loading was cancelled.", "AbortError")); };
    signal.addEventListener("abort", abort, { once: true });
    image.onload = () => { const dimensions = { width: image.naturalWidth || null, height: image.naturalHeight || null }; cleanup(); resolve(dimensions); };
    image.onerror = () => { cleanup(); resolve({ width: null, height: null }); };
    image.src = url;
  });
}

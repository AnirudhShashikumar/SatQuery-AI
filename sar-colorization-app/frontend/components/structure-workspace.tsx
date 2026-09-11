"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { LoaderCircle, ScanSearch } from "lucide-react";
import { runSARFusionFormer } from "@/services/api";
import { RemoteSensingAdaptationPanel } from "@/components/remote-sensing-adaptation-panel";
import { UploadZone } from "@/components/upload-zone";
import { ImageCard } from "@/components/image-card";
import { MetricCard } from "@/components/metric-card";
import { SectionHeader } from "@/components/section-header";
import { formatMetric } from "@/lib/utils";
import { geoVisionResults, imageAssetFromFile, imageAssetFromResult, sourceFingerprint } from "@/lib/geovision-result-store";

const sarFiles = { "application/octet-stream": [".npy"], "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"] };
const optical = { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"] };

export function StructureWorkspace() {
  const [mode, setMode] = useState<"combined" | "separate">("combined");
  const [combined, setCombined] = useState<File>(); const [vv, setVv] = useState<File>(); const [vh, setVh] = useState<File>(); const [groundTruth, setGroundTruth] = useState<File>();
  const [colorCorrect, setColorCorrect] = useState(false); const [view, setView] = useState<"enhanced" | "raw" | "corrected">("enhanced");
  const ready = mode === "combined" ? Boolean(combined) : Boolean(vv && vh);
  const inference = useMutation({
    mutationFn: () => runSARFusionFormer({ combined: mode === "combined" ? combined : undefined, vv: mode === "separate" ? vv : undefined, vh: mode === "separate" ? vh : undefined, groundTruth, applyColorCorrection: colorCorrect }),
    onSuccess: async result => {
      const storedGroundTruth = groundTruth ? await imageAssetFromFile(groundTruth) : null;
      const sourceFile = combined ?? vv;
      geoVisionResults.setSarFusionFormerResult({
        vvPreview: imageAssetFromResult(result.vv_preview, "sarfusionformer-vv-preview.png"),
        vhPreview: imageAssetFromResult(result.vh_preview, "sarfusionformer-vh-preview.png"),
        combinedPreview: imageAssetFromResult(result.sar_preview, "sarfusionformer-combined-preview.png"),
        rawOutput: imageAssetFromResult(result.raw_output, "sarfusionformer-raw.png"),
        enhancedOutput: imageAssetFromResult(result.display_output, "sarfusionformer-enhanced.png"),
        correctedOutput: result.corrected_raw_output ? imageAssetFromResult(result.corrected_raw_output, "sarfusionformer-corrected-raw.png") : null,
        selectedOutputMode: view,
        groundTruth: storedGroundTruth,
        metricsRaw: { psnr: result.metrics_raw?.psnr ?? null, ssim: result.metrics_raw?.ssim ?? null, rgbL1: result.metrics_raw?.rgb_l1 ?? null, inferenceTimeMs: result.inference_time_ms },
        metricsEnhanced: null,
        metricsCorrected: result.metrics_corrected ? { psnr: result.metrics_corrected.psnr ?? null, ssim: result.metrics_corrected.ssim ?? null, rgbL1: result.metrics_corrected.rgb_l1 ?? null, inferenceTimeMs: result.inference_time_ms } : null,
        checkpointName: result.checkpoint,
        colorCheckpointName: result.color_checkpoint,
        sourceFileName: mode === "combined" ? combined?.name ?? null : [vv?.name, vh?.name].filter(Boolean).join(" + ") || null,
        sourceHash: await sourceFingerprint(sourceFile),
        sampleId: sourceFile?.name ?? null,
        createdAt: result.created_at,
      });
      geoVisionResults.setSarComparisonMode(view);
    },
  });
  const result = inference.data; const showCorrected = view === "corrected" && result?.corrected_display_output;
  const output = showCorrected ? result?.corrected_display_output : view === "raw" ? result?.raw_output : result?.display_output;
  const metrics = showCorrected ? result?.metrics_corrected : result?.metrics_raw; const diagnostics = showCorrected ? result?.corrected_diagnostics : result?.diagnostics;
  return <><SectionHeader eyebrow="Structure reconstruction" title="SARFusionFormer">Transformer-based, structure-preserving reconstruction from raw VV/VH SAR.</SectionHeader>
    <div className="mb-8 grid gap-3 rounded-2xl border border-sky-400/15 bg-sky-400/[.055] p-5 text-sm leading-6 text-sky-100/80 sm:grid-cols-3"><span>Uses raw VV/VH SAR</span><span>Preserves terrain structure</span><span>Radiometrically faithful output</span></div>
    <section className="panel p-5 sm:p-7"><div className="mb-5 flex gap-2 rounded-xl bg-zinc-950/50 p-1 w-fit"><button onClick={() => setMode("combined")} className={`rounded-lg px-3 py-2 text-sm ${mode === "combined" ? "bg-white/10 text-white" : "text-zinc-500"}`}>Combined VV/VH</button><button onClick={() => setMode("separate")} className={`rounded-lg px-3 py-2 text-sm ${mode === "separate" ? "bg-white/10 text-white" : "text-zinc-500"}`}>Separate channels</button></div>
      <div className="grid gap-5 lg:grid-cols-2">{mode === "combined" ? <UploadZone label="Combined VV/VH array" hint=".npy · CHW or HWC" accept={{ "application/octet-stream": [".npy"] }} file={combined} onFile={setCombined}/> : <><UploadZone label="VV channel" hint=".npy, TIFF, PNG, JPEG" accept={sarFiles} file={vv} onFile={setVv}/><UploadZone label="VH channel" hint=".npy, TIFF, PNG, JPEG" accept={sarFiles} file={vh} onFile={setVh}/></>}<UploadZone label="Ground truth · optional" hint="For scientific metrics" accept={optical} file={groundTruth} onFile={setGroundTruth}/></div>
      <label className="mt-5 flex w-fit cursor-pointer items-center gap-2 text-sm text-zinc-300"><input type="checkbox" checked={colorCorrect} onChange={e => setColorCorrect(e.target.checked)} className="accent-sky-300"/>Apply optional colour correction</label><button onClick={() => inference.mutate()} disabled={!ready || inference.isPending} className="mt-6 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-5 py-3 text-sm font-semibold text-zinc-950 disabled:opacity-40">{inference.isPending ? <LoaderCircle className="animate-spin" size={17}/> : <ScanSearch size={17}/>}Generate reconstruction</button>{inference.error && <p className="mt-3 text-sm text-rose-300">{inference.error.message}</p>}</section>
    {result && <section className="mt-10"><p className="mb-4 text-sm text-zinc-400">Detected {result.channel_layout} · {result.detected_shape.join(" × ")} · VV channel 0 / VH channel 1</p><div className="mb-5 flex flex-wrap gap-2">{(["enhanced", "raw", ...(result.corrected_display_output ? ["corrected"] : [])] as const).map(tab => <button key={tab} onClick={() => { setView(tab as "enhanced" | "raw" | "corrected"); geoVisionResults.setSarComparisonMode(tab as "enhanced" | "raw" | "corrected"); }} className={`rounded-xl px-4 py-2 text-sm ${view === tab ? "bg-sky-400/15 text-sky-200" : "bg-white/[.04] text-zinc-400"}`}>{tab === "enhanced" ? "Enhanced display" : tab === "raw" ? "Raw radiometric output" : "Color corrected"}</button>)}</div><p className="mb-5 text-sm text-zinc-400">Enhanced display uses percentile contrast stretching for visualization only. Raw radiometric output is the direct network prediction.</p><div className="grid items-start gap-4 md:grid-cols-4"><ImageCard title="VV preview" image={result.vv_preview}/><ImageCard title="VH preview" image={result.vh_preview}/><ImageCard title="Combined preview" image={result.sar_preview}/><ImageCard title="Prediction" image={output} action={{ name: "sarfusionformer-raw.png", data: result.raw_output }} analysis={{ type: view === "raw" ? "sarfusionformer_raw" : view === "corrected" ? "sarfusionformer_corrected" : "sarfusionformer_enhanced", metadata: { model: "SARFusionFormer", display_mode: view }, onReport: report => geoVisionResults.setSarFusionFormerAnalysis(report) }}/></div><div className="mt-4 grid gap-3 sm:grid-cols-4"><MetricCard label="PSNR" value={formatMetric(metrics?.psnr, 2)}/><MetricCard label="SSIM" value={formatMetric(metrics?.ssim)}/><MetricCard label="RGB L1" value={formatMetric(metrics?.rgb_l1)}/><MetricCard label="Inference time" value={`${result.inference_time_ms.toFixed(0)} ms`}/></div>{result.sve_result && <div className="mt-5"><RemoteSensingAdaptationPanel result={result.sve_result}/></div>}<Link href="/comparison" className="mt-5 inline-flex rounded-xl border border-sky-300/30 px-4 py-2.5 text-sm font-medium text-sky-200 hover:bg-sky-400/[.08]">View in Model Comparison</Link>{result.warning && <p className="mt-4 rounded-xl border border-amber-400/20 bg-amber-400/[.06] p-3 text-sm text-amber-100">{result.warning}</p>}<details className="glass mt-5 rounded-2xl p-4"><summary className="cursor-pointer text-sm font-medium">Backend diagnostics</summary><pre className="mt-4 overflow-auto text-xs leading-5 text-zinc-400">{JSON.stringify(diagnostics, null, 2)}</pre></details></section>}</>;
}

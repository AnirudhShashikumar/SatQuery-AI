"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowDown, CheckCircle2, GitCompareArrows, LoaderCircle, Trash2, Upload } from "lucide-react";
import { runComparison } from "@/services/api";
import { UploadZone } from "@/components/upload-zone";
import { MetricCard } from "@/components/metric-card";
import { SectionHeader } from "@/components/section-header";
import { dataUrl, formatMetric } from "@/lib/utils";
import { geoVisionResults, imageAssetFromFile, type GeoVisionResultState, type ImageAsset, type MetricSet, useGeoVisionStore } from "@/lib/geovision-result-store";

const optical = { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"] };
type SourceMode = "latest" | "external";
type VisualMode = "side-by-side" | "slider" | "overlay" | "difference";
type AssetKey = "pix" | "sar_raw" | "sar_enhanced" | "sar_corrected" | "ground_truth";

const relativeTime = (value?: string | null) => {
  if (!value) return "Unknown time";
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60000));
  return minutes < 1 ? "Generated just now" : minutes === 1 ? "Generated 1 minute ago" : minutes < 60 ? `Generated ${minutes} minutes ago` : `Generated ${Math.round(minutes / 60)} hours ago`;
};
const sameAsset = (left?: ImageAsset | null, right?: ImageAsset | null) => Boolean(left?.url && right?.url && left.url === right.url);
const assetToFile = async (asset: ImageAsset, name: string) => {
  const response = await fetch(dataUrl(asset.url)!);
  if (!response.ok) throw new Error("A saved image is no longer available in this session.");
  return new File([await response.blob()], name, { type: asset.mimeType || "image/png" });
};

function assetFor(key: AssetKey, pix: GeoVisionResultState, sar: GeoVisionResultState, groundTruth?: ImageAsset | null) {
  if (key === "pix") return pix.pix2pix.output ?? null;
  if (key === "sar_raw") return sar.sarfusionformer.rawOutput ?? null;
  if (key === "sar_enhanced") return sar.sarfusionformer.enhancedOutput ?? null;
  if (key === "sar_corrected") return sar.sarfusionformer.correctedOutput ?? null;
  return groundTruth ?? null;
}

function DifferenceHeatmap({ left, right }: { left: ImageAsset; right: ImageAsset }) {
  const [image, setImage] = useState<string>();
  useEffect(() => {
    let cancelled = false;
    const render = async () => {
      const [first, second] = await Promise.all([loadImage(dataUrl(left.url)!), loadImage(dataUrl(right.url)!)]);
      const width = Math.min(first.naturalWidth, second.naturalWidth, 1024); const height = Math.round(width * Math.min(first.naturalHeight / first.naturalWidth, second.naturalHeight / second.naturalWidth));
      const a = document.createElement("canvas"); const b = document.createElement("canvas"); const out = document.createElement("canvas");
      [a, b, out].forEach(canvas => { canvas.width = width; canvas.height = height; });
      a.getContext("2d")?.drawImage(first, 0, 0, width, height); b.getContext("2d")?.drawImage(second, 0, 0, width, height);
      const firstPixels = a.getContext("2d")?.getImageData(0, 0, width, height); const secondPixels = b.getContext("2d")?.getImageData(0, 0, width, height); const context = out.getContext("2d");
      if (!firstPixels || !secondPixels || !context) return;
      const result = context.createImageData(width, height);
      for (let index = 0; index < result.data.length; index += 4) {
        const delta = (Math.abs(firstPixels.data[index] - secondPixels.data[index]) + Math.abs(firstPixels.data[index + 1] - secondPixels.data[index + 1]) + Math.abs(firstPixels.data[index + 2] - secondPixels.data[index + 2])) / 3;
        result.data[index] = Math.min(255, delta * 2.4); result.data[index + 1] = Math.min(255, delta * .7); result.data[index + 2] = 32; result.data[index + 3] = 255;
      }
      context.putImageData(result, 0, 0);
      if (!cancelled) setImage(out.toDataURL("image/png"));
    };
    void render().catch(() => !cancelled && setImage(undefined));
    return () => { cancelled = true; };
  }, [left, right]);
  return image ? <img src={image} alt="Absolute RGB difference heatmap" className="h-full w-full object-contain"/> : <p className="text-sm text-zinc-500">Preparing difference heatmap…</p>;
}
const loadImage = (src: string) => new Promise<HTMLImageElement>((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src; });

function SavedResultCard({ title, asset, preview, checkpoint, timestamp, metrics, outputMode, metricNotice, missing }: { title: string; asset?: ImageAsset | null; preview?: ImageAsset | null; checkpoint?: string | null; timestamp?: string | null; metrics?: MetricSet | null; outputMode?: string; metricNotice?: string; missing: ReactNode }) {
  return <article className="panel overflow-hidden"><header className="flex items-center justify-between border-b border-white/[.08] px-4 py-3"><h2 className="text-sm font-semibold">{title}</h2>{asset && <span className="rounded-full border border-sky-300/20 bg-sky-400/[.08] px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-sky-200">Latest result</span>}</header>{asset ? <><div className="aspect-video bg-zinc-950/70"><img src={dataUrl(asset.url)} alt={`${title} saved output`} className="h-full w-full object-contain"/></div><div className="space-y-2 p-4 text-xs text-zinc-500"><p>{asset.origin === "external" ? "External result" : "Current session"} · {relativeTime(timestamp)}</p><p>{asset.width && asset.height ? `${asset.width} × ${asset.height}px` : "Image dimensions not reported"}</p>{outputMode && <p>Output mode: <span className="text-zinc-300">{outputMode}</span></p>}<p className="truncate">Checkpoint: <span className="text-zinc-300">{checkpoint ?? "Not reported"}</span></p>{preview && <p className="truncate">Source preview: <span className="text-zinc-300">{preview.fileName ?? "available"}</span></p>}<div className="grid grid-cols-3 gap-2 pt-1 text-zinc-400"><span>PSNR {formatMetric(metrics?.psnr, 1)}</span><span>SSIM {formatMetric(metrics?.ssim, 2)}</span><span>L1 {formatMetric(metrics?.rgbL1, 3)}</span></div>{metricNotice && <p className="text-amber-200">{metricNotice}</p>}</div></> : <div className="grid min-h-64 place-items-center px-5 text-center text-sm text-zinc-500">{missing}</div>}</article>;
}

export function ComparisonWorkspace() {
  const store = useGeoVisionStore(state => state);
  const [sourceMode, setSourceMode] = useState<SourceMode>("latest"); const [visualMode, setVisualMode] = useState<VisualMode>("side-by-side"); const [slider, setSlider] = useState(50); const [overlay, setOverlay] = useState(50);
  const [leftKey, setLeftKey] = useState<AssetKey>("pix"); const [rightKey, setRightKey] = useState<AssetKey>("sar_raw");
  const [externalPix, setExternalPix] = useState<File>(); const [externalSar, setExternalSar] = useState<File>(); const [externalGroundTruth, setExternalGroundTruth] = useState<File>(); const [commonUpload, setCommonUpload] = useState<File>();
  const pix = store.pix2pix; const sar = store.sarfusionformer;
  const groundTruthConflict = Boolean(pix.groundTruth && sar.groundTruth && !sameAsset(pix.groundTruth, sar.groundTruth));
  const resolvedGroundTruth = store.comparison.commonGroundTruth ?? (!sar.groundTruth ? pix.groundTruth ?? null : !pix.groundTruth ? sar.groundTruth : sameAsset(pix.groundTruth, sar.groundTruth) ? pix.groundTruth : null);
  const sarMode = store.comparison.sarOutputMode ?? sar.selectedOutputMode ?? "raw";
  const sarOutput = sarMode === "enhanced" ? sar.enhancedOutput : sarMode === "corrected" ? sar.correctedOutput : sar.rawOutput;
  const sourceMatched = Boolean(pix.sourceHash && sar.sourceHash && pix.sourceHash === sar.sourceHash);
  const pixMetricsCompatible = Boolean(sourceMatched && resolvedGroundTruth && sameAsset(resolvedGroundTruth, pix.groundTruth));
  const sarMetricsCompatible = Boolean(sourceMatched && resolvedGroundTruth && sameAsset(resolvedGroundTruth, sar.groundTruth));
  const outputsHaveDifferentTimes = Boolean(pix.createdAt && sar.createdAt && Math.abs(new Date(pix.createdAt).getTime() - new Date(sar.createdAt).getTime()) > 5 * 60_000);
  const availableOptions = useMemo(() => ([
    pix.output && ["pix", "Pix2Pix"] as const,
    sar.rawOutput && ["sar_raw", "SARFusionFormer raw"] as const,
    sar.enhancedOutput && ["sar_enhanced", "SARFusionFormer enhanced"] as const,
    sar.correctedOutput && ["sar_corrected", "SARFusionFormer corrected"] as const,
    resolvedGroundTruth && ["ground_truth", "Ground truth"] as const,
  ].filter(Boolean) as [AssetKey, string][]), [pix.output, sar.rawOutput, sar.enhancedOutput, sar.correctedOutput, resolvedGroundTruth]);
  const leftAsset = assetFor(leftKey, store, store, resolvedGroundTruth); const rightAsset = assetFor(rightKey, store, store, resolvedGroundTruth);
  const compare = useMutation({
    onSuccess: result => geoVisionResults.setComparisonMetrics({ pix2pix: { psnr: result.pix2pix.psnr, ssim: result.pix2pix.ssim, rgbL1: result.pix2pix.rgb_l1 }, sarfusionformer: { psnr: result.sarfusionformer.psnr, ssim: result.sarfusionformer.ssim, rgbL1: result.sarfusionformer.rgb_l1 } }),
    mutationFn: async () => {
      if (sourceMode === "external") {
        if (!externalPix || !externalSar || !externalGroundTruth) throw new Error("Add both external outputs and one common ground truth.");
        return runComparison(externalPix, externalSar, externalGroundTruth);
      }
      if (!pix.output || !sarOutput || !resolvedGroundTruth) throw new Error("Both saved outputs and a common ground truth are required.");
      const [pixFile, sarFile, targetFile] = await Promise.all([assetToFile(pix.output, "pix2pix-output.png"), assetToFile(sarOutput, "sarfusionformer-output.png"), assetToFile(resolvedGroundTruth, "common-ground-truth.png")]);
      return runComparison(pixFile, sarFile, targetFile);
    },
  });
  const setGroundTruthFile = (file?: File) => { setCommonUpload(file); if (!file) { geoVisionResults.setComparisonGroundTruth(null); return; } void imageAssetFromFile(file, "external").then(asset => geoVisionResults.setComparisonGroundTruth(asset)); };
  const canCompareLatest = Boolean(pix.output && sarOutput && resolvedGroundTruth);
  const hasAnySaved = Boolean(pix.output || sar.rawOutput || sar.enhancedOutput || sar.correctedOutput);
  const selectSarMode = (mode: "raw" | "enhanced" | "corrected") => geoVisionResults.setSarComparisonMode(mode);
  return <><SectionHeader eyebrow="Evaluation" title="Model Comparison">Compare the latest Pix2Pix and SARFusionFormer outputs without uploading generated files again.</SectionHeader>
    <div className="mb-6 flex flex-wrap gap-2 rounded-xl bg-zinc-950/50 p-1 w-fit"><button onClick={() => setSourceMode("latest")} className={`rounded-lg px-3 py-2 text-sm ${sourceMode === "latest" ? "bg-white/10 text-white" : "text-zinc-500"}`}>Latest app results</button><button onClick={() => setSourceMode("external")} className={`rounded-lg px-3 py-2 text-sm ${sourceMode === "external" ? "bg-white/10 text-white" : "text-zinc-500"}`}>External files</button></div>
    {sourceMode === "latest" ? <><div className="grid gap-4 lg:grid-cols-3"><SavedResultCard title="Pix2Pix result" asset={pix.output} preview={pix.inputPreview} checkpoint={pix.checkpointName} timestamp={pix.createdAt} metrics={pixMetricsCompatible ? pix.metrics : null} outputMode="optical" metricNotice={pix.metrics && !pixMetricsCompatible ? "Saved metric reference is not verified against the current comparison ground truth." : undefined} missing={<><p>No Pix2Pix result saved.</p><Link href="/pix2pix" className="mt-3 text-sky-300">Run model</Link></>}/><SavedResultCard title="SARFusionFormer result" asset={sarOutput} preview={sar.combinedPreview ?? sar.vvPreview} checkpoint={sar.checkpointName} timestamp={sar.createdAt} metrics={sarMode === "enhanced" || !sarMetricsCompatible ? null : sarMode === "corrected" ? sar.metricsCorrected : sar.metricsRaw} outputMode={sarMode} metricNotice={sarMode === "enhanced" ? "Enhanced preview is display-only; scientific metrics use raw or corrected raw output." : (sar.metricsRaw && !sarMetricsCompatible ? "Saved metric reference is not verified against the current comparison ground truth." : undefined)} missing={<><p>No SARFusionFormer result saved.</p><Link href="/structure" className="mt-3 text-sky-300">Run model</Link></>}/><SavedResultCard title="Common ground truth" asset={resolvedGroundTruth} checkpoint={null} timestamp={null} metrics={null} missing="No common ground truth selected."/></div>
      {!hasAnySaved && <div className="mt-6 rounded-2xl border border-dashed border-white/[.1] p-8 text-center"><ArrowDown className="mx-auto mb-3 text-zinc-500"/><p className="text-sm text-zinc-400">No model results are available yet.</p><div className="mt-4 flex justify-center gap-3"><Link href="/pix2pix" className="rounded-xl bg-sky-300 px-4 py-2.5 text-sm font-semibold text-zinc-950">Run Pix2Pix</Link><Link href="/structure" className="rounded-xl border border-white/[.12] px-4 py-2.5 text-sm text-zinc-200">Run SARFusionFormer</Link></div></div>}
      {hasAnySaved && <><div className="mt-5 flex flex-wrap gap-2"><button onClick={() => confirm("Clear the saved Pix2Pix result?") && geoVisionResults.clearPix2PixResult()} className="inline-flex items-center gap-2 rounded-xl border border-white/[.12] px-3 py-2 text-xs text-zinc-400 hover:bg-white/[.05]"><Trash2 size={14}/>Clear Pix2Pix</button><button onClick={() => confirm("Clear the saved SARFusionFormer result?") && geoVisionResults.clearSarFusionFormerResult()} className="inline-flex items-center gap-2 rounded-xl border border-white/[.12] px-3 py-2 text-xs text-zinc-400 hover:bg-white/[.05]"><Trash2 size={14}/>Clear SARFusionFormer</button><button onClick={() => confirm("Clear comparison selections?") && geoVisionResults.clearComparison()} className="inline-flex items-center gap-2 rounded-xl border border-white/[.12] px-3 py-2 text-xs text-zinc-400 hover:bg-white/[.05]">Clear comparison</button><button onClick={() => confirm("Clear all saved session results?") && geoVisionResults.clearAllResults()} className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs text-rose-200 hover:bg-rose-400/10">Clear all session results</button></div>
        <section className="panel mt-5 p-5 sm:p-7"><div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="font-semibold">Comparison source</h2><p className="mt-1 text-sm text-zinc-500">Latest app results are retained for this browser session only.</p></div><div className="flex gap-2">{(["raw", "enhanced", "corrected"] as const).filter(mode => mode !== "corrected" || Boolean(sar.correctedOutput)).map(mode => <button key={mode} onClick={() => selectSarMode(mode)} className={`rounded-lg px-3 py-2 text-xs ${sarMode === mode ? "bg-sky-400/15 text-sky-200" : "bg-white/[.04] text-zinc-400"}`}>{mode === "enhanced" ? "Enhanced (display only)" : mode}</button>)}</div></div>
          {groundTruthConflict && !store.comparison.commonGroundTruth && <div className="mt-5 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm text-amber-100"><p>The saved model results use different ground-truth images. Select a common ground truth before calculating comparison metrics.</p><div className="mt-3 flex flex-wrap gap-2"><button onClick={() => geoVisionResults.setComparisonGroundTruth(pix.groundTruth ?? null)} className="rounded-lg border border-amber-300/30 px-3 py-2 text-xs">Use Pix2Pix ground truth</button><button onClick={() => geoVisionResults.setComparisonGroundTruth(sar.groundTruth ?? null)} className="rounded-lg border border-amber-300/30 px-3 py-2 text-xs">Use SARFusionFormer ground truth</button></div></div>}
          <div className="mt-5 grid gap-5 lg:grid-cols-3"><UploadZone label="Upload common ground truth" hint="Optional override" accept={optical} file={commonUpload} onFile={setGroundTruthFile}/><div className="lg:col-span-2 space-y-3 text-sm"><p className={sourceMatched ? "flex items-center gap-2 text-emerald-300" : "text-amber-200"}>{sourceMatched ? <><CheckCircle2 size={16}/>Matching sample verified</> : "Source match not verified"}</p><p className="text-zinc-500">{sourceMatched ? "Both source fingerprints match." : "The saved outputs may come from different source samples. Comparison is visual unless a common ground truth and matching sample identity are confirmed."}</p>{outputsHaveDifferentTimes && <p className="text-amber-200">These outputs were generated at different times. Confirm that they represent the intended samples.</p>}</div></div>
          <button onClick={() => compare.mutate()} disabled={!canCompareLatest || compare.isPending} className="mt-6 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-5 py-3 text-sm font-semibold text-zinc-950 disabled:opacity-40">{compare.isPending ? <LoaderCircle className="animate-spin" size={17}/> : <GitCompareArrows size={17}/>}Calculate comparison metrics</button>{!canCompareLatest && <p className="mt-2 text-xs text-zinc-500">Comparison metrics require both outputs and one common ground truth. Without a source match, do not claim direct scientific superiority.</p>}</section>
        <section className="panel mt-5 p-5 sm:p-7"><div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="font-semibold">Visual comparison</h2><p className="mt-1 text-sm text-zinc-500">Visual modes never modify the saved model output.</p></div><div className="flex flex-wrap gap-2">{(["side-by-side", "slider", "overlay", "difference"] as const).map(mode => <button key={mode} onClick={() => setVisualMode(mode)} className={`rounded-lg px-3 py-2 text-xs ${visualMode === mode ? "bg-sky-400/15 text-sky-200" : "bg-white/[.04] text-zinc-400"}`}>{mode}</button>)}</div></div><div className="mt-4 grid gap-3 sm:grid-cols-2"><AssetSelect label="Left image" value={leftKey} options={availableOptions} onChange={setLeftKey}/><AssetSelect label="Right image" value={rightKey} options={availableOptions} onChange={setRightKey}/></div>{leftAsset && rightAsset ? <VisualPanel left={leftAsset} right={rightAsset} mode={visualMode} slider={slider} setSlider={setSlider} overlay={overlay} setOverlay={setOverlay}/> : <p className="mt-6 text-sm text-zinc-500">Run both models to unlock visual comparison modes.</p>}</section></>}
    </> : <section className="panel p-5 sm:p-7"><details open><summary className="cursor-pointer text-sm font-semibold">Use external result files</summary><p className="mt-2 text-sm text-zinc-500">Optional testing workflow. Latest app results remain the default source.</p><div className="mt-5 grid gap-5 lg:grid-cols-3"><UploadZone label="Pix2Pix prediction" hint="Generated PNG/JPEG/TIFF" accept={optical} file={externalPix} onFile={setExternalPix}/><UploadZone label="SARFusionFormer prediction" hint="Generated PNG/JPEG/TIFF" accept={optical} file={externalSar} onFile={setExternalSar}/><UploadZone label="Common ground truth" hint="Required for comparison" accept={optical} file={externalGroundTruth} onFile={setExternalGroundTruth}/></div><button onClick={() => compare.mutate()} disabled={compare.isPending || !(externalPix && externalSar && externalGroundTruth)} className="mt-7 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-5 py-3 text-sm font-semibold text-zinc-950 disabled:opacity-40">{compare.isPending ? <LoaderCircle className="animate-spin" size={17}/> : <Upload size={17}/>}Compare external files</button></details></section>}
    {compare.error && <p className="mt-4 rounded-xl border border-rose-300/20 bg-rose-300/[.08] p-4 text-sm text-rose-100">{compare.error.message}</p>}{compare.data && <section className="mt-5 grid gap-3 sm:grid-cols-3"><MetricCard title="Pix2Pix PSNR" value={formatMetric(compare.data.pix2pix.psnr, 2)} unit="dB" source="measured"/><MetricCard title="SARFusionFormer PSNR" value={formatMetric(compare.data.sarfusionformer.psnr, 2)} unit="dB" source="measured"/><MetricCard title="SSIM · Pix / SAR" value={`${formatMetric(compare.data.pix2pix.ssim, 3)} / ${formatMetric(compare.data.sarfusionformer.ssim, 3)}`} source="measured"/></section>}</>;
}

function AssetSelect({ label, value, options, onChange }: { label: string; value: AssetKey; options: [AssetKey, string][]; onChange: (value: AssetKey) => void }) {
  const validValue = options.some(([key]) => key === value) ? value : options[0]?.[0] ?? "pix";
  return <label className="text-xs font-medium text-zinc-400"><span>{label}</span><select value={validValue} onChange={event => onChange(event.target.value as AssetKey)} className="mt-2 w-full rounded-xl border border-white/[.12] bg-zinc-950/50 px-3 py-2.5 text-sm text-zinc-200 outline-none"><option value="" disabled>No output available</option>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}

function VisualPanel({ left, right, mode, slider, setSlider, overlay, setOverlay }: { left: ImageAsset; right: ImageAsset; mode: VisualMode; slider: number; setSlider: (value: number) => void; overlay: number; setOverlay: (value: number) => void }) {
  const leftUrl = dataUrl(left.url)!; const rightUrl = dataUrl(right.url)!;
  if (mode === "side-by-side") return <div className="mt-5 grid gap-4 md:grid-cols-2"><VisualImage src={leftUrl} label="Left image"/><VisualImage src={rightUrl} label="Right image"/></div>;
  if (mode === "difference") return <div className="mt-5 grid aspect-video place-items-center overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/70"><DifferenceHeatmap left={left} right={right}/></div>;
  return <div className="mt-5"><div className="relative aspect-video overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/70"><img src={leftUrl} alt="Left comparison image" className="absolute inset-0 h-full w-full object-contain"/>{mode === "slider" ? <div className="absolute inset-0 overflow-hidden" style={{ width: `${slider}%` }}><img src={rightUrl} alt="Right comparison image" className="h-full max-w-none object-contain" style={{ width: "100vw" }}/></div> : <img src={rightUrl} alt="Right comparison image" className="absolute inset-0 h-full w-full object-contain" style={{ opacity: overlay / 100 }}/>}</div><label className="mt-3 block text-xs text-zinc-500">{mode === "slider" ? "Reveal right image" : "Right image opacity"}<input type="range" min="0" max="100" value={mode === "slider" ? slider : overlay} onChange={event => mode === "slider" ? setSlider(Number(event.target.value)) : setOverlay(Number(event.target.value))} className="mt-2 w-full accent-sky-300"/></label></div>;
}
function VisualImage({ src, label }: { src: string; label: string }) { return <article className="overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/70"><p className="border-b border-white/[.08] px-4 py-3 text-xs text-zinc-500">{label}</p><img src={src} alt={label} className="aspect-video w-full object-contain"/></article>; }

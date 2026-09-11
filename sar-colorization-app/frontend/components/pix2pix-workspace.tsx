"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { LoaderCircle, Sparkles } from "lucide-react";
import { runPix2Pix } from "@/services/api";
import { UploadZone } from "@/components/upload-zone";
import { ImageCard } from "@/components/image-card";
import { MetricCard } from "@/components/metric-card";
import { SectionHeader } from "@/components/section-header";
import { formatMetric } from "@/lib/utils";
import { geoVisionResults, imageAssetFromFile, imageAssetFromResult, sourceFingerprint, useGeoVisionStore } from "@/lib/geovision-result-store";

const images = { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"] };

export function Pix2PixWorkspace() {
  const [source, setSource] = useState<File>();
  const [groundTruth, setGroundTruth] = useState<File>();
  const [groundTruthPreview, setGroundTruthPreview] = useState<string>();
  const savedResult = useGeoVisionStore(state => state.pix2pix);
  const inference = useMutation({
    mutationFn: () => source ? runPix2Pix(source, groundTruth) : Promise.reject(new Error("Upload a SAR image first.")),
    onSuccess: async result => {
      const storedGroundTruth = groundTruth ? await imageAssetFromFile(groundTruth) : null;
      geoVisionResults.setPix2PixResult({
        inputPreview: imageAssetFromResult(result.input_preview, "pix2pix-input-preview.png"),
        output: imageAssetFromResult(result.output, "pix2pix-optical.png"),
        groundTruth: storedGroundTruth,
        metrics: { psnr: result.metrics?.psnr ?? null, ssim: result.metrics?.ssim ?? null, rgbL1: result.metrics?.rgb_l1 ?? null, inferenceTimeMs: result.inference_time_ms },
        checkpointName: result.checkpoint,
        sourceFileName: source?.name ?? null,
        sourceHash: await sourceFingerprint(source),
        sampleId: source?.name ?? null,
        createdAt: result.created_at,
      });
    },
  });
  return <><SectionHeader eyebrow="Visual reconstruction" title="Pix2Pix">Visual-quality optical reconstruction from a SAR image.</SectionHeader>
    <div className="mb-8 rounded-2xl border border-sky-400/15 bg-sky-400/[.055] p-5 text-sm leading-6 text-sky-100/80">Pix2Pix prioritizes realistic optical appearance. Generated details may be plausible rather than physically exact.</div>
    <section className="panel p-5 sm:p-7"><div className="grid gap-5 lg:grid-cols-2"><UploadZone label="SAR input" hint="PNG, JPEG, or TIFF" accept={images} file={source} onFile={setSource}/><UploadZone label="Ground truth · optional" hint="For PSNR and SSIM" accept={images} file={groundTruth} onFile={file => { setGroundTruth(file); if (!file) { setGroundTruthPreview(undefined); return; } void imageAssetFromFile(file).then(asset => setGroundTruthPreview(asset.url)); }}/></div>
      <button onClick={() => inference.mutate()} disabled={inference.isPending || !source} className="mt-7 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-5 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-sky-200 disabled:cursor-not-allowed disabled:opacity-40">{inference.isPending ? <LoaderCircle className="animate-spin" size={17}/> : <Sparkles size={17}/>}Generate reconstruction</button>{inference.error && <p className="mt-3 text-sm text-rose-300">{inference.error.message}</p>}</section>
    {inference.data && <section className="mt-10"><div className="grid items-start gap-4 md:grid-cols-3"><ImageCard title="Input SAR" image={inference.data.input_preview}/><ImageCard title="Predicted optical" image={inference.data.output} action={{ name: "pix2pix-optical.png", data: inference.data.output }} analysis={{ type: "pix2pix", metadata: { model: "Pix2Pix" }, onReport: report => geoVisionResults.setPix2PixAnalysis(report) }}/><ImageCard title="Ground truth" image={savedResult.groundTruth?.url ?? groundTruthPreview}/></div><div className="mt-4 grid gap-3 sm:grid-cols-3">{inference.data.metrics ? <><MetricCard label="PSNR" value={`${formatMetric(inference.data.metrics.psnr, 2)} dB`}/><MetricCard label="SSIM" value={formatMetric(inference.data.metrics.ssim)}/><MetricCard label="Inference time" value={`${inference.data.inference_time_ms.toFixed(0)} ms`}/></> : <MetricCard label="Inference time" value={`${inference.data.inference_time_ms.toFixed(0)} ms`} detail="Add ground truth to calculate metrics."/>}</div><Link href="/comparison" className="mt-5 inline-flex rounded-xl border border-sky-300/30 px-4 py-2.5 text-sm font-medium text-sky-200 hover:bg-sky-400/[.08]">View in Model Comparison</Link></section>}</>;
}

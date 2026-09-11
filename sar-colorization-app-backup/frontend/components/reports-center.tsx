"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Download, FileArchive, FileJson, FileSpreadsheet, FileText, ImageDown, LoaderCircle } from "lucide-react";
import { useGeoVisionStore, type GeoVisionResultState, type ImageAsset } from "@/lib/geovision-result-store";
import { createMetadataJson, createMetricsCsv, createPdf, createZip, downloadBlob, resolveImageAssetToBlob, safeTimestamp, type ExportImage, type ExportModel, type ExportReport, type ReportSource } from "@/lib/report-export";
import { dataUrl, formatMetric } from "@/lib/utils";

const disclaimer = "Pix2Pix may generate plausible synthetic detail. SARFusionFormer enhanced display is visualization-only and must not be used for scientific metrics. Generated outputs are reconstructions, not exact ground truth.";
const label = { pix2pix: "Pix2Pix", sarfusionformer: "SARFusionFormer", comparison: "Model Comparison", combined: "Combined Report" } as const;

const image = (asset: ImageAsset | null | undefined, name: string, title: string): ExportImage[] => asset ? [{ asset, fileName: name, label: title }] : [];

export function ReportsCenter() {
  const state = useGeoVisionStore(snapshot => snapshot);
  const [source, setSource] = useState<ReportSource>("combined");
  const [sarMode, setSarMode] = useState<"raw" | "enhanced" | "corrected">(state.comparison.sarOutputMode ?? "raw");
  const [working, setWorking] = useState<"pdf" | "csv" | "json" | "images" | "zip" | null>(null);
  const [message, setMessage] = useState("");
  const pixAvailable = Boolean(state.pix2pix.output); const sarAvailable = Boolean(state.sarfusionformer.rawOutput || state.sarfusionformer.enhancedOutput); const both = pixAvailable && sarAvailable;
  const available = { pix2pix: pixAvailable, sarfusionformer: sarAvailable, comparison: both, combined: both } satisfies Record<ReportSource, boolean>;
  const activeSource = available[source] ? source : both ? "combined" : pixAvailable ? "pix2pix" : sarAvailable ? "sarfusionformer" : "combined";
  const sarOutput = sarMode === "enhanced" ? state.sarfusionformer.enhancedOutput : sarMode === "corrected" ? state.sarfusionformer.correctedOutput : state.sarfusionformer.rawOutput;
  const report = useMemo<ExportReport>(() => buildReport(state, activeSource, sarMode, sarOutput), [state, activeSource, sarMode, sarOutput]);
  const hasData = report.models.length > 0;
  const exportName = `SatQuery_${label[activeSource].replace(/\s+/g, "_")}_Report_${safeTimestamp(new Date(report.generatedAt)).slice(0, 10)}`;

  const runExport = async (kind: NonNullable<typeof working>) => {
    if (!hasData || working) return;
    setWorking(kind); setMessage(kind === "pdf" ? "Generating PDF…" : kind === "zip" ? "Collecting data…" : "Preparing download…");
    try {
      if (kind === "pdf") { downloadBlob(await createPdf(report), `${exportName}.pdf`); setMessage("PDF report downloaded."); }
      if (kind === "csv") { downloadBlob(createMetricsCsv(report), `SatQuery_Metrics_${safeTimestamp()}.csv`); setMessage("Metrics CSV downloaded."); }
      if (kind === "json") { downloadBlob(createMetadataJson(report), `${exportName}.json`); setMessage("Metadata JSON downloaded."); }
      if (kind === "images") {
        let count = 0;
        for (const model of report.models) for (const item of model.images) { try { downloadBlob(await resolveImageAssetToBlob(item.asset), `${model.name.toLowerCase().replace(/\s+/g, "-")}-${item.fileName}`); count += 1; } catch { /* Continue with remaining session images. */ } }
        if (!count) throw new Error("No report images could be resolved for download.");
        setMessage(`${count} image${count === 1 ? "" : "s"} downloaded.`);
      }
      if (kind === "zip") { setMessage("Generating report files…"); const [pdf, csv, metadata] = await Promise.all([createPdf(report), Promise.resolve(createMetricsCsv(report)), Promise.resolve(createMetadataJson(report))]); setMessage("Compressing report package…"); downloadBlob(await createZip(report, pdf, csv, metadata), `SatQuery_Report_Package_${safeTimestamp()}.zip`); setMessage("ZIP package downloaded."); }
    } catch (error) { setMessage(error instanceof Error ? error.message : "Report export failed. Please try again."); }
    finally { setWorking(null); }
  };

  return <section className="report-page"><header className="report-header"><div><p className="eyebrow">Decision-ready intelligence</p><h1>Reports</h1><p>Create reproducible PDF, JSON, CSV, imagery, and complete evidence packages from the latest SatQuery results.</p></div><span><Download size={16}/> Enterprise export</span></header>
    {!hasData ? <EmptyReport/> : <><div className="mt-7 flex flex-wrap items-end gap-4"><label className="text-sm text-zinc-400"><span>Report source</span><select value={activeSource} onChange={event => setSource(event.target.value as ReportSource)} className="ml-3 rounded-xl border border-white/[.12] bg-zinc-950/50 px-3 py-2 text-sm text-zinc-200 outline-none">{(Object.keys(label) as ReportSource[]).map(key => <option key={key} value={key} disabled={!available[key]}>{label[key]}{available[key] ? "" : " · unavailable"}</option>)}</select></label>{(activeSource === "sarfusionformer" || activeSource === "comparison" || activeSource === "combined") && <label className="text-sm text-zinc-400"><span>SAR output</span><select value={sarMode} onChange={event => setSarMode(event.target.value as typeof sarMode)} className="ml-3 rounded-xl border border-white/[.12] bg-zinc-950/50 px-3 py-2 text-sm text-zinc-200 outline-none"><option value="raw">Raw</option><option value="enhanced">Enhanced</option><option value="corrected" disabled={!state.sarfusionformer.correctedOutput}>Color-corrected</option></select></label>}</div>
      <div className="mt-5 flex flex-wrap gap-3">{([ ["pdf", "Download PDF", FileText], ["csv", "Download Metrics CSV", FileSpreadsheet], ["json", "Download JSON", FileJson], ["images", "Download Images", ImageDown], ["zip", "Download ZIP", FileArchive] ] as const).map(([kind, text, Icon]) => <button key={kind} onClick={() => runExport(kind)} disabled={Boolean(working)} title={working ? "An export is already in progress." : undefined} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-3 text-sm font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-40">{working === kind ? <LoaderCircle className="animate-spin" size={17}/> : <Icon size={17}/>} {working === kind ? (kind === "pdf" ? "Generating PDF…" : kind === "zip" ? "Generating ZIP…" : "Preparing…") : text}</button>)}</div>
      <p className="mt-3 min-h-5 text-sm text-zinc-400" aria-live="polite">{message}</p><ReportPreview report={report} source={activeSource} sarMode={sarMode}/></>}
  </section>;
}

function EmptyReport() { return <article className="empty-report"><span><FileText size={23}/></span><p className="eyebrow">SatQuery report workspace</p><h2>No report data available</h2><p>Run an analysis or reconstruction workflow first. SatQuery will automatically assemble the latest results, evidence, provenance, and scientific disclosures here.</p><div><Link href="/assistant">Open Assistant</Link><Link href="/pix2pix">Run Pix2Pix</Link><Link href="/structure">Run SARFusionFormer</Link></div><ul aria-label="Available export formats">{([[FileText,"PDF report"],[FileJson,"Structured JSON"],[FileSpreadsheet,"Metrics CSV"],[FileArchive,"Evidence ZIP"]] as const).map(([FormatIcon, text]) => <li key={text}><FormatIcon size={16}/>{text}<small>Ready after analysis</small></li>)}</ul></article>; }

function ReportPreview({ report, source, sarMode }: { report: ExportReport; source: ReportSource; sarMode: string }) { return <article className="panel mt-7 max-w-5xl overflow-hidden"><header className="border-b border-white/[.08] p-6 sm:p-8"><p className="eyebrow">SatQuery report preview</p><h2 className="mt-2 text-2xl font-semibold">{label[source]}</h2><p className="mt-2 text-sm text-zinc-500">Report ID: {report.id} · Generated {new Date(report.generatedAt).toLocaleString()} · v{report.version}</p></header><div className="space-y-8 p-6 sm:p-8">{report.models.map(model => <section key={model.name}><h3 className="text-lg font-semibold">{model.name}</h3><p className="mt-2 text-sm text-zinc-500">Checkpoint: {model.checkpoint ?? "Not reported"} · Input: {model.inputFile ?? "Not reported"} · Output: {model.outputMode ?? "default"}</p>{model.outputMode === "enhanced" && <p className="mt-2 text-sm text-amber-200">Display-enhanced visualization — excluded from scientific metrics.</p>}<div className="mt-4 grid gap-3 sm:grid-cols-4"><Metric name="PSNR" value={formatMetric(model.metrics?.psnr, 2)} unit="dB"/><Metric name="SSIM" value={formatMetric(model.metrics?.ssim, 3)}/><Metric name="RGB L1" value={formatMetric(model.metrics?.rgbL1, 3)}/><Metric name="Inference" value={formatMetric(model.metrics?.inferenceTimeMs, 0)} unit="ms"/></div><div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{model.images.map(item => <figure key={`${model.name}-${item.fileName}`} className="overflow-hidden rounded-xl border border-white/[.08] bg-zinc-950/50"><img src={dataUrl(item.asset.url)} alt={item.label} className="aspect-video w-full object-contain"/><figcaption className="px-3 py-2 text-xs text-zinc-500">{item.label}</figcaption></figure>)}</div></section>)}<section className="border-t border-white/[.08] pt-6">{report.models.some(model => model.aiAnalysis) && <div className="mb-6"><h3 className="font-semibold">AI-assisted interpretation</h3>{report.models.filter(model => model.aiAnalysis).map(model => <p key={model.name} className="mt-2 text-sm leading-6 text-zinc-400"><span className="font-medium text-zinc-200">{model.name}: </span>{String(model.aiAnalysis?.executive_summary ?? "Analysis available in the export package.")}</p>)}</div>}<h3 className="font-semibold">Scientific notes</h3><p className="mt-2 text-sm leading-6 text-zinc-400">{report.scientificDisclaimer}</p></section></div><footer className="border-t border-white/[.08] px-6 py-4 text-xs text-zinc-500">Generated by SatQuery AI · {report.generatedAt} · {report.id} · SAR mode: {sarMode}</footer></article>; }
function Metric({ name, value, unit }: { name: string; value: string; unit?: string }) { return <div className="glass rounded-xl p-3"><p className="text-xs text-zinc-500">{name}</p><p className="mt-1 text-lg font-semibold">{value} {unit}</p></div>; }

function buildReport(state: GeoVisionResultState, source: ReportSource, sarMode: "raw" | "enhanced" | "corrected", selectedSarOutput?: ImageAsset | null): ExportReport {
  const models: ExportModel[] = [];
  const includePix = source === "pix2pix" || source === "comparison" || source === "combined";
  const includeSar = source === "sarfusionformer" || source === "comparison" || source === "combined";
  if (includePix && state.pix2pix.output) models.push({ name: "Pix2Pix", checkpoint: state.pix2pix.checkpointName, inputFile: state.pix2pix.sourceFileName, outputMode: "optical", metrics: source === "comparison" ? state.comparison.metrics?.pix2pix ?? null : state.pix2pix.metrics, createdAt: state.pix2pix.createdAt, aiAnalysis: state.pix2pix.aiAnalysis, images: [...image(state.pix2pix.inputPreview, "input_sar.png", "Input SAR"), ...image(state.pix2pix.output, "pix2pix_prediction.png", "Pix2Pix prediction"), ...image(state.pix2pix.groundTruth, "ground_truth.png", "Ground truth")] });
  if (includeSar && selectedSarOutput) models.push({ name: "SARFusionFormer", checkpoint: state.sarfusionformer.checkpointName, colorCheckpoint: state.sarfusionformer.colorCheckpointName, inputFile: state.sarfusionformer.sourceFileName, outputMode: sarMode, metrics: source === "comparison" ? state.comparison.metrics?.sarfusionformer ?? null : sarMode === "raw" ? state.sarfusionformer.metricsRaw : sarMode === "corrected" ? state.sarfusionformer.metricsCorrected : null, createdAt: state.sarfusionformer.createdAt, aiAnalysis: state.sarfusionformer.aiAnalysis, images: [...image(state.sarfusionformer.vvPreview, "vv_preview.png", "VV preview"), ...image(state.sarfusionformer.vhPreview, "vh_preview.png", "VH preview"), ...image(state.sarfusionformer.combinedPreview, "combined_sar.png", "Combined SAR"), ...image(state.sarfusionformer.rawOutput, "sarfusionformer_raw.png", "Raw output"), ...image(state.sarfusionformer.enhancedOutput, "sarfusionformer_enhanced.png", "Enhanced output"), ...image(state.sarfusionformer.correctedOutput, "sarfusionformer_corrected.png", "Color-corrected output"), ...image(state.comparison.commonGroundTruth ?? state.sarfusionformer.groundTruth, "ground_truth.png", "Ground truth")] });
  const generatedAt = new Date().toISOString();
  return { id: `SQ-${safeTimestamp(new Date(generatedAt)).replace(/[_-]/g, "").slice(0, 14)}`, generatedAt, source, version: "1.0", models, scientificDisclaimer: disclaimer };
}

"use client";

import { dataUrl } from "@/lib/utils";
import type { ImageAsset, MetricSet } from "@/lib/geovision-result-store";

export type ReportSource = "pix2pix" | "sarfusionformer" | "comparison" | "combined";
export type ExportImage = { label: string; fileName: string; asset: ImageAsset };
export type ExportModel = {
  name: string;
  checkpoint?: string | null;
  colorCheckpoint?: string | null;
  inputFile?: string | null;
  outputMode?: string | null;
  metrics?: MetricSet | null;
  createdAt?: string | null;
  images: ExportImage[];
  aiAnalysis?: Record<string, unknown> | null;
};
export type ExportReport = {
  id: string;
  generatedAt: string;
  source: ReportSource;
  version: string;
  models: ExportModel[];
  scientificDisclaimer: string;
};

export async function resolveImageAssetToBlob(asset: ImageAsset): Promise<Blob> {
  const response = await fetch(dataUrl(asset.url)!);
  if (!response.ok) throw new Error(`Could not load ${asset.fileName ?? "image"} for export.`);
  return response.blob();
}

const toDataUrl = (blob: Blob) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader();
  reader.onerror = () => reject(reader.error ?? new Error("Could not encode image for PDF."));
  reader.onload = () => resolve(String(reader.result));
  reader.readAsDataURL(blob);
});

export function safeTimestamp(value = new Date()): string {
  return value.toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
}

export function csvEscape(value: unknown): string {
  const text = value == null ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function createMetricsCsv(report: ExportReport): Blob {
  const header = ["report_id", "generated_at", "model", "checkpoint", "input_file", "output_mode", "psnr", "ssim", "rgb_l1", "inference_time_ms", "image_width", "image_height", "ground_truth_available", "ai_analysis_available"];
  const rows = report.models.map(model => {
    const groundTruth = model.images.find(image => image.fileName === "ground_truth.png")?.asset;
    const prediction = model.images.find(image => /prediction|raw\.png|corrected/.test(image.fileName))?.asset;
    return [report.id, report.generatedAt, model.name, model.checkpoint, model.inputFile, model.outputMode, model.metrics?.psnr, model.metrics?.ssim, model.metrics?.rgbL1, model.metrics?.inferenceTimeMs, prediction?.width ?? groundTruth?.width, prediction?.height ?? groundTruth?.height, Boolean(groundTruth), Boolean(model.aiAnalysis)];
  });
  return new Blob([[header, ...rows].map(row => row.map(csvEscape).join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
}

export function createMetadataJson(report: ExportReport): Blob {
  const models = report.models.map(model => ({
    name: model.name, checkpoint: model.checkpoint ?? null, color_checkpoint: model.colorCheckpoint ?? null,
    input_file: model.inputFile ?? null, output_mode: model.outputMode ?? null, metrics: model.metrics ?? null,
    created_at: model.createdAt ?? null, images: model.images.map(image => ({ label: image.label, file_name: image.fileName, width: image.asset.width ?? null, height: image.asset.height ?? null, mime_type: image.asset.mimeType ?? null, origin: image.asset.origin })),
    ai_analysis_available: Boolean(model.aiAnalysis),
  }));
  const payload = { report: { id: report.id, generated_at: report.generatedAt, source: report.source, version: report.version }, pix2pix: models.find(model => model.name === "Pix2Pix") ?? null, sarfusionformer: models.find(model => model.name === "SARFusionFormer") ?? null, comparison: report.source === "comparison" ? { included: true } : null, ai_analysis: report.models.map(model => ({ model: model.name, available: Boolean(model.aiAnalysis) })), scientific_disclaimer: report.scientificDisclaimer };
  return new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
}

export async function createPdf(report: ExportReport): Promise<Blob> {
  const { jsPDF } = await import("jspdf");
  const document = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = document.internal.pageSize.getWidth();
  const pageHeight = document.internal.pageSize.getHeight();
  let y = 54;
  const write = (text: string, size = 10, bold = false) => { document.setFont("helvetica", bold ? "bold" : "normal"); document.setFontSize(size); const lines = document.splitTextToSize(text, pageWidth - 72); document.text(lines, 36, y); y += lines.length * (size + 4); };
  const page = () => { document.addPage(); y = 54; };
  write("SatQuery AI", 18, true); write(`${report.source === "combined" ? "Combined" : report.source === "comparison" ? "Comparison" : report.models[0]?.name ?? ""} Report`, 16, true); write(`Report ID: ${report.id}`); write(`Generated: ${report.generatedAt}`); y += 10;
  for (const model of report.models) {
    if (y > pageHeight - 180) page();
    write(model.name, 14, true); write(`Checkpoint: ${model.checkpoint ?? "Not reported"}`); write(`Input: ${model.inputFile ?? "Not reported"} · Output mode: ${model.outputMode ?? "default"}`); write(`PSNR: ${model.metrics?.psnr ?? "—"} · SSIM: ${model.metrics?.ssim ?? "—"} · RGB L1: ${model.metrics?.rgbL1 ?? "—"} · Inference: ${model.metrics?.inferenceTimeMs ?? "—"} ms`);
    for (const image of model.images.slice(0, 6)) {
      try {
        const source = await toDataUrl(await resolveImageAssetToBlob(image.asset));
        if (y > pageHeight - 180) page();
        write(image.label, 9, true);
        document.addImage(source, "PNG", 36, y, 150, 112); y += 122;
      } catch { write(`${image.label}: unavailable for PDF export.`); }
    }
    if (model.aiAnalysis) write(`AI-assisted interpretation: ${String(model.aiAnalysis.executive_summary ?? "Available in the analysis export.")}`);
    y += 8;
  }
  if (y > pageHeight - 120) page();
  write("Scientific notes", 13, true); write(report.scientificDisclaimer); write("Generated by SatQuery AI", 9);
  const pages = document.getNumberOfPages();
  for (let index = 1; index <= pages; index += 1) { document.setPage(index); document.setFontSize(8); document.text(`SatQuery AI · ${report.id} · Page ${index} of ${pages}`, 36, pageHeight - 24); }
  return document.output("blob");
}

export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

export async function createZip(report: ExportReport, pdf: Blob, csv: Blob, metadata: Blob): Promise<Blob> {
  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  zip.file("report/report.pdf", pdf); zip.file("report/metrics.csv", csv); zip.file("report/metadata.json", metadata);
  const analysis: Record<string, unknown> = {};
  for (const model of report.models) {
    if (model.aiAnalysis) analysis[model.name] = model.aiAnalysis;
    for (const image of model.images) {
      try { zip.file(`images/${model.name.toLowerCase().replace(/\s+/g, "-")}-${image.fileName}`, await resolveImageAssetToBlob(image.asset)); }
      catch { /* Omit images that cannot be resolved because their remote source blocked export. */ }
    }
  }
  if (Object.keys(analysis).length) { zip.file("analysis/ai_analysis.json", JSON.stringify(analysis, null, 2)); zip.file("analysis/ai_analysis.md", Object.entries(analysis).map(([model, value]) => `# ${model}\n\n${JSON.stringify(value, null, 2)}`).join("\n\n")); }
  zip.file("readme/README.txt", `SatQuery AI report package\n\nModels: ${report.models.map(model => model.name).join(", ")}\nMetrics: PSNR (higher is better), SSIM (higher is better), RGB L1 (lower is better).\n\nScientific limitations:\n${report.scientificDisclaimer}\n`);
  return zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
}

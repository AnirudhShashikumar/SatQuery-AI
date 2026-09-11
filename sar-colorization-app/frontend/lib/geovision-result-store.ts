"use client";

import { useCallback, useRef, useSyncExternalStore } from "react";
import type { ImageAnalysisReport } from "@/types/api";

export type AssetOrigin = "current-session" | "external";
export type ImageAsset = {
  url: string;
  fileName?: string;
  width?: number;
  height?: number;
  mimeType?: string;
  origin: AssetOrigin;
};

export type MetricSet = {
  psnr?: number | null;
  ssim?: number | null;
  rgbL1?: number | null;
  inferenceTimeMs?: number | null;
};

export type Pix2PixResultState = {
  inputPreview?: ImageAsset | null;
  output?: ImageAsset | null;
  groundTruth?: ImageAsset | null;
  metrics?: MetricSet | null;
  checkpointName?: string | null;
  sourceFileName?: string | null;
  sourceHash?: string | null;
  sampleId?: string | null;
  createdAt?: string | null;
  aiAnalysis?: ImageAnalysisReport | null;
};

export type SarFusionFormerResultState = {
  vvPreview?: ImageAsset | null;
  vhPreview?: ImageAsset | null;
  combinedPreview?: ImageAsset | null;
  rawOutput?: ImageAsset | null;
  enhancedOutput?: ImageAsset | null;
  correctedOutput?: ImageAsset | null;
  selectedOutputMode?: "raw" | "enhanced" | "corrected";
  groundTruth?: ImageAsset | null;
  metricsRaw?: MetricSet | null;
  metricsEnhanced?: MetricSet | null;
  metricsCorrected?: MetricSet | null;
  checkpointName?: string | null;
  colorCheckpointName?: string | null;
  sourceFileName?: string | null;
  sourceHash?: string | null;
  sampleId?: string | null;
  createdAt?: string | null;
  aiAnalysis?: ImageAnalysisReport | null;
};

export type ComparisonMetrics = { pix2pix: MetricSet; sarfusionformer: MetricSet };

export type ComparisonState = {
  commonGroundTruth?: ImageAsset | null;
  sarOutputMode?: "raw" | "enhanced" | "corrected";
  metrics?: ComparisonMetrics | null;
};

export type GeoVisionResultState = {
  pix2pix: Pix2PixResultState;
  sarfusionformer: SarFusionFormerResultState;
  comparison: ComparisonState;
};

const STORAGE_KEY = "geovision-result-store-v1";
const emptyState: GeoVisionResultState = {
  pix2pix: {},
  sarfusionformer: {},
  comparison: { sarOutputMode: "raw" },
};

let state: GeoVisionResultState = emptyState;
let hydrated = false;
const listeners = new Set<() => void>();

const emit = () => listeners.forEach(listener => listener());
const persist = () => {
  if (typeof window === "undefined") return;
  try { window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, state })); }
  catch { /* Preserve the in-memory result if session storage reaches its browser quota. */ }
};
const update = (next: GeoVisionResultState) => { state = next; persist(); emit(); };

export const geoVisionResults = {
  hydrate() {
    if (hydrated || typeof window === "undefined") return;
    hydrated = true;
    try {
      const saved = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "null") as { version?: number; state?: GeoVisionResultState } | null;
      if (saved?.version === 1 && saved.state) state = saved.state;
    } catch { window.sessionStorage.removeItem(STORAGE_KEY); }
    emit();
  },
  subscribe(listener: () => void) { listeners.add(listener); return () => listeners.delete(listener); },
  getState: () => state,
  setPix2PixResult(result: Pix2PixResultState) { update({ ...state, pix2pix: result }); },
  setSarFusionFormerResult(result: SarFusionFormerResultState) { update({ ...state, sarfusionformer: result }); },
  setPix2PixAnalysis(aiAnalysis: ImageAnalysisReport) { update({ ...state, pix2pix: { ...state.pix2pix, aiAnalysis } }); },
  setSarFusionFormerAnalysis(aiAnalysis: ImageAnalysisReport) { update({ ...state, sarfusionformer: { ...state.sarfusionformer, aiAnalysis } }); },
  setComparisonGroundTruth(image: ImageAsset | null) { update({ ...state, comparison: { ...state.comparison, commonGroundTruth: image } }); },
  setSarComparisonMode(mode: "raw" | "enhanced" | "corrected") { update({ ...state, comparison: { ...state.comparison, sarOutputMode: mode } }); },
  setComparisonMetrics(metrics: ComparisonMetrics) { update({ ...state, comparison: { ...state.comparison, metrics } }); },
  clearPix2PixResult() { update({ ...state, pix2pix: {} }); },
  clearSarFusionFormerResult() { update({ ...state, sarfusionformer: {} }); },
  clearComparison() { update({ ...state, comparison: { sarOutputMode: "raw" } }); },
  clearAllResults() { update(emptyState); },
};

export function useGeoVisionStore<T>(selector: (snapshot: GeoVisionResultState) => T): T {
  const selectorRef = useRef(selector);
  selectorRef.current = selector;
  const getSnapshot = useCallback(() => selectorRef.current(state), []);
  const getServerSnapshot = useCallback(() => selectorRef.current(emptyState), []);
  return useSyncExternalStore(geoVisionResults.subscribe, getSnapshot, getServerSnapshot);
}

export async function imageAssetFromFile(file: File, origin: AssetOrigin = "current-session"): Promise<ImageAsset> {
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read image file."));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
  const dimensions = await new Promise<{ width: number; height: number } | null>(resolve => { const image = new Image(); image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight }); image.onerror = () => resolve(null); image.src = url; });
  return { url, fileName: file.name, mimeType: file.type || "image/png", origin, ...(dimensions ?? {}) };
}

export const imageAssetFromResult = (url: string, fileName: string): ImageAsset => ({ url, fileName, mimeType: "image/png", origin: "current-session" });
export async function sourceFingerprint(file?: File): Promise<string | null> {
  if (!file) return null;
  if (!globalThis.crypto?.subtle) return `${file.name}:${file.size}:${file.lastModified}`;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
}

import type { AgentResponse, DemoManifest, DemoWorkflow, PairCompatibility } from "@/types/agent";

export const crossModalDefaultQuery = "Use the optical and SAR images together to identify built-up and water-covered regions.";

export type ApprovedCrossModalPair = {
  workflow: DemoWorkflow;
  optical: DemoWorkflow["files"][number];
  sar: DemoWorkflow["files"][number];
};

export type RasterPreviewMetadata = {
  width: number;
  height: number;
  bandCount: number;
  dtype: string;
  crs: string | null;
  transformKey: string | null;
  bounds: [number, number, number, number] | null;
};

export type CrossModalEvidencePreview = { label: string; path: string };

export function approvedCrossModalPair(
  manifest: DemoManifest,
  workflowId: string,
  opticalSampleId: string,
  sarSampleId: string,
): ApprovedCrossModalPair {
  if (!manifest.enabled) throw new Error("Local demo mode is disabled. Start the backend with SATQUERY_DEMO_MODE=true to use the approved optical–SAR pair.");
  const workflow = manifest.workflows.find(item => item.id === workflowId);
  if (!workflow || workflow.input_mode !== "cross_modal") throw new Error("The approved optical–SAR demo pair is unavailable from the local manifest.");
  if (!(["optical", "multispectral"].includes(workflow.primary_modality) && workflow.secondary_modality === "sar")) {
    throw new Error("The approved demo workflow does not declare a valid optical–SAR modality pair.");
  }
  const optical = workflow.files.find(item => item.role === "primary" && item.filename === opticalSampleId);
  const sar = workflow.files.find(item => item.role === "secondary" && item.filename === sarSampleId);
  if (!optical) throw new Error("The approved optical sample is missing from the local manifest.");
  if (!sar) throw new Error("The approved SAR sample is missing from the local manifest.");
  return { workflow, optical, sar };
}

export function crossModalEvidencePreviews(response: AgentResponse): CrossModalEvidencePreview[] {
  const previews = response.cross_modal_analysis?.previews;
  if (!previews) return [];
  const candidates = [
    ["Optical evidence", previews.optical_evidence],
    ["SAR evidence", previews.sar_evidence],
    ["Joint evidence", previews.joint_evidence],
    ["Water likelihood", previews.water_likelihood],
    ["Structural likelihood", previews.built_up_likelihood],
    ["Vegetation support", previews.vegetation_support],
    ["Agreement", previews.agreement],
    ["Disagreement", previews.disagreement],
    ["Joint overlay", previews.joint_overlay],
  ] as const;
  return candidates.flatMap(([label, path]) => path ? [{ label, path }] : []);
}

export function compatibilityLabel(compatibility: PairCompatibility | null): string {
  if (!compatibility) return "Compatibility pending";
  if (!compatibility.compatible || compatibility.alignment_level === "incompatible") return "Incompatible pair";
  if (compatibility.alignment_level === "exact") return "Compatible · exact alignment";
  if (compatibility.alignment_level === "geospatial_overlap") return "Alignment required";
  return "Compatible · visual only";
}

export function preflightCompatibilityLabel(optical: RasterPreviewMetadata, sar: RasterPreviewMetadata): string {
  if (optical.width !== sar.width || optical.height !== sar.height) return "Preflight · incompatible dimensions";
  if (!optical.crs || !sar.crs || !optical.transformKey || !sar.transformKey) return "Preflight · visual only";
  if (optical.crs !== sar.crs) return "Preflight · incompatible CRS";
  if (optical.transformKey === sar.transformKey) return "Preflight · exact candidate";
  return "Preflight · alignment required";
}

export function crossModalDisclosure(response: AgentResponse): string | null {
  const result = response.cross_modal_analysis;
  const compatibility = response.pair_compatibility;
  if (response.status === "alignment_required" || result?.status === "alignment_required" || compatibility?.alignment_level === "geospatial_overlap") {
    return "Alignment is required. SatQuery did not resize, register, reproject, resample, or fuse either image.";
  }
  if (result?.status === "failed" || compatibility?.alignment_level === "incompatible" || compatibility && !compatibility.compatible) {
    return "This optical–SAR pair is incompatible. Fusion stopped and no joint statistics were fabricated.";
  }
  if (result?.status === "partial" && !result.statistics || compatibility?.alignment_level === "visual_only") {
    return "Visual-only comparison: the source previews remain available, but no pixel-level fusion or joint statistics were computed.";
  }
  if (!result) return response.answer || response.warnings[0] || "The backend did not return an optical–SAR analysis result.";
  return null;
}

export function crossModalFailureMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : "The local SatQuery optical–SAR request failed.";
  if (/demo mode|demo workflow|demo pair|optical sample|sar sample|source preview|preview failed/i.test(message)) return message;
  if (/fetch|network|backend|failed to connect/i.test(message)) return "The local SatQuery backend is offline. Start it on port 8010 and retry.";
  if (/expired|not found|404/i.test(message)) return "The result or preview has expired. Re-run analysis to generate a current evidence set.";
  return message;
}

type TiffEntry = { type: number; count: number; valueOffset: number; entryOffset: number };

const typeSize: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8 };

function tiffValues(view: DataView, little: boolean, entry: TiffEntry): number[] {
  const size = typeSize[entry.type];
  if (!size) return [];
  const start = size * entry.count <= 4 ? entry.entryOffset + 8 : entry.valueOffset;
  const values: number[] = [];
  for (let index = 0; index < entry.count; index += 1) {
    const offset = start + index * size;
    if (entry.type === 1 || entry.type === 2) values.push(view.getUint8(offset));
    else if (entry.type === 3) values.push(view.getUint16(offset, little));
    else if (entry.type === 4) values.push(view.getUint32(offset, little));
    else if (entry.type === 11) values.push(view.getFloat32(offset, little));
    else if (entry.type === 12) values.push(view.getFloat64(offset, little));
  }
  return values;
}

function percentile(values: number[], fraction: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * fraction)))] ?? 0;
}

export async function createApprovedRasterPreview(file: File, modality: "optical" | "multispectral" | "sar"): Promise<{ url: string; metadata: RasterPreviewMetadata }> {
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);
  if (view.byteLength < 16) throw new Error(`The approved ${modality === "sar" ? "SAR" : "optical"} sample is empty or truncated.`);
  const marker = String.fromCharCode(view.getUint8(0), view.getUint8(1));
  const little = marker === "II";
  if (!little && marker !== "MM") throw new Error("The approved source preview is not a TIFF raster.");
  if (view.getUint16(2, little) !== 42) throw new Error("The approved source uses an unsupported TIFF variant.");

  const ifdOffset = view.getUint32(4, little);
  const entryCount = view.getUint16(ifdOffset, little);
  const entries = new Map<number, TiffEntry>();
  for (let index = 0; index < entryCount; index += 1) {
    const entryOffset = ifdOffset + 2 + index * 12;
    const tag = view.getUint16(entryOffset, little);
    entries.set(tag, {
      type: view.getUint16(entryOffset + 2, little),
      count: view.getUint32(entryOffset + 4, little),
      valueOffset: view.getUint32(entryOffset + 8, little),
      entryOffset,
    });
  }
  const values = (tag: number) => {
    const entry = entries.get(tag);
    return entry ? tiffValues(view, little, entry) : [];
  };
  const width = values(256)[0] ?? 0;
  const height = values(257)[0] ?? 0;
  const bits = values(258);
  const compression = values(259)[0] ?? 1;
  const samples = values(277)[0] ?? 1;
  const sampleFormat = values(339)[0] ?? 1;
  const offsets = values(273);
  const byteCounts = values(279);
  if (!width || !height || !offsets.length || compression !== 1 || ![8, 32].includes(bits[0] ?? 0)) {
    throw new Error("This approved TIFF cannot be previewed locally. Analysis has not started; verify the source or reload the sample.");
  }

  const bytes = new Uint8Array(buffer);
  const raster = new Uint8Array(byteCounts.reduce((sum, count) => sum + count, 0));
  let rasterOffset = 0;
  offsets.forEach((offset, index) => {
    const count = byteCounts[index] ?? 0;
    raster.set(bytes.subarray(offset, offset + count), rasterOffset);
    rasterOffset += count;
  });

  const pixels = new Uint8ClampedArray(width * height * 4);
  if ((bits[0] ?? 0) === 8 && sampleFormat === 1) {
    for (let index = 0; index < width * height; index += 1) {
      const source = index * samples;
      const target = index * 4;
      pixels[target] = raster[source] ?? 0;
      pixels[target + 1] = samples > 1 ? raster[source + 1] ?? pixels[target] : pixels[target];
      pixels[target + 2] = samples > 2 ? raster[source + 2] ?? pixels[target] : pixels[target];
      pixels[target + 3] = 255;
    }
  } else if ((bits[0] ?? 0) === 32 && sampleFormat === 3) {
    const rasterView = new DataView(raster.buffer, raster.byteOffset, raster.byteLength);
    const raw: number[] = [];
    for (let index = 0; index < width * height; index += 1) raw.push(rasterView.getFloat32(index * samples * 4, little));
    const finite = raw.filter(Number.isFinite);
    const low = percentile(finite, 0.02);
    const high = percentile(finite, 0.98);
    const range = high > low ? high - low : 1;
    raw.forEach((value, index) => {
      const intensity = Number.isFinite(value) ? Math.round(Math.min(1, Math.max(0, (value - low) / range)) * 255) : 0;
      pixels[index * 4] = intensity;
      pixels[index * 4 + 1] = intensity;
      pixels[index * 4 + 2] = intensity;
      pixels[index * 4 + 3] = 255;
    });
  } else {
    throw new Error("The approved TIFF sample has an unsupported pixel type for local preview.");
  }

  const scale = values(33550);
  const tiepoint = values(33922);
  const geoKeys = values(34735);
  let epsg: number | null = null;
  if (geoKeys.length >= 4) {
    const keyCount = geoKeys[3] ?? 0;
    for (let index = 0; index < keyCount; index += 1) {
      const offset = 4 + index * 4;
      const key = geoKeys[offset];
      if ((key === 2048 || key === 3072) && geoKeys[offset + 1] === 0) epsg = geoKeys[offset + 3] ?? null;
    }
  }
  const transformKey = scale.length >= 2 && tiepoint.length >= 6
    ? [scale[0], scale[1], tiepoint[3], tiepoint[4]].map(value => Number(value).toPrecision(12)).join(":")
    : null;
  const bounds: [number, number, number, number] | null = scale.length >= 2 && tiepoint.length >= 6
    ? [tiepoint[3], tiepoint[4] - height * scale[1], tiepoint[3] + width * scale[0], tiepoint[4]]
    : null;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("The browser could not create a source preview. The raster remains unchanged and analysis has not started.");
  context.putImageData(new ImageData(pixels, width, height), 0, 0);
  return {
    url: canvas.toDataURL("image/png"),
    metadata: {
      width,
      height,
      bandCount: samples,
      dtype: sampleFormat === 3 ? `float${bits[0]}` : `uint${bits[0]}`,
      crs: epsg ? `EPSG:${epsg}` : null,
      transformKey,
      bounds,
    },
  };
}

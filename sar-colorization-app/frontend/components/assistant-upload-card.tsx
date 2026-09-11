"use client";

import { CheckCircle2, FileImage, LoaderCircle, RefreshCw, Satellite, X } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { agentPreviewUrl } from "@/services/api";
import type { ImageMetadata } from "@/types/agent";
import { cn } from "@/lib/utils";

const accept = {
  "image/tiff": [".tif", ".tiff"],
  "image/png": [".png"],
  "image/jpeg": [".jpg", ".jpeg"],
};
const maxSize = 100 * 1024 * 1024;

function value(value: boolean | null) {
  return value === null ? "Unavailable" : value ? "Yes" : "No";
}

export function AssistantUploadCard({
  label,
  hint,
  file,
  metadata,
  busy,
  onFile,
  onError,
}: {
  label: string;
  hint: string;
  file?: File;
  metadata?: ImageMetadata | null;
  busy: boolean;
  onFile: (file?: File) => void;
  onError: (message: string) => void;
}) {
  const dropzone = useDropzone({
    accept,
    maxFiles: 1,
    maxSize,
    disabled: busy,
    onDropAccepted: ([next]) => onFile(next),
    onDropRejected: rejection => onError(rejection[0]?.errors[0]?.message ?? "Select a supported image up to 100 MB."),
  });
  const preview = agentPreviewUrl(metadata?.preview_url);
  return <article className={cn("assistant-upload-card", file && "assistant-upload-ready")}>
    <header className="assistant-upload-header">
      <div><p className="assistant-upload-role">Satellite imagery</p><h3>{label}</h3><p>{hint}</p></div>
      {file && !busy && <div className="assistant-upload-actions"><button type="button" onClick={dropzone.open} aria-label={`Replace ${label}`}><RefreshCw size={15}/><span>Replace</span></button><button type="button" onClick={() => onFile()} aria-label={`Remove ${label}`}><X size={16}/><span className="sr-only">Remove</span></button></div>}
    </header>
    <input {...dropzone.getInputProps()} />
    {!file ? <button type="button" {...dropzone.getRootProps()} className={cn("assistant-dropzone group", dropzone.isDragActive && "assistant-dropzone-active")}>
      <span className="assistant-upload-icon"><Satellite size={22}/></span>
      <span className="assistant-dropzone-title">{dropzone.isDragActive ? "Drop satellite imagery here" : "Drop satellite imagery here"}</span>
      <span className="assistant-dropzone-formats">GeoTIFF · TIFF · PNG · JPEG · maximum 100 MB</span>
      <span className="assistant-select-observation">Select observation</span>
      <span className="assistant-format-guidance">GeoTIFF recommended when geospatial metadata matters.</span>
    </button> : <div>
      <div className="assistant-upload-file">
        {preview ? <img src={preview} alt={`${label} display preview`} className="assistant-upload-thumb"/> : <span className="assistant-upload-file-icon">{busy ? <LoaderCircle size={19} className="animate-spin"/> : <FileImage size={19}/>}</span>}
        <div><p>{file.name}</p><span>{(file.size / 1024 / 1024).toFixed(2)} MB · {busy ? "Preparing metadata…" : metadata ? "Input verified" : "Ready for local upload"}</span></div>
        {!busy && <CheckCircle2 size={17} className="shrink-0 text-emerald-300" aria-label="File ready"/>}
      </div>
      {metadata && <div className="border-t border-white/[.08]">
        <div className="assistant-upload-summary"><span><CheckCircle2 size={13}/>Compatible input</span><span>{metadata.is_georeferenced ? "Georeferenced" : "No georeference"}</span><span>{(metadata.effective_modality ?? metadata.auto_detected_modality ?? "unknown").replaceAll("_", " ")}</span></div>
        <p className="assistant-upload-core-facts">{metadata.format.toUpperCase()} · {metadata.width} × {metadata.height} · {(metadata.auto_detected_modality ?? "unknown").replaceAll("_", " ")}</p>
        <details className="assistant-upload-metadata"><summary>Scientific metadata</summary><dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
            <div><dt className="text-zinc-500">Dimensions</dt><dd className="mt-1 text-zinc-200">{metadata.width} × {metadata.height}</dd></div>
            <div><dt className="text-zinc-500">Bands</dt><dd className="mt-1 text-zinc-200">{metadata.band_count}</dd></div>
            {metadata.available_band_names && metadata.available_band_names.length > 0 && <div className="col-span-2"><dt className="text-zinc-500">Available bands</dt><dd className="mt-1 text-zinc-200">{metadata.available_band_names.join(", ")}</dd></div>}
            {metadata.selected_visual_bands && metadata.selected_visual_bands.length > 0 && <div className="col-span-2"><dt className="text-zinc-500">Visual preview mapping</dt><dd className="mt-1 text-zinc-200">{metadata.selected_visual_bands.join(" / ")}</dd></div>}
            {metadata.band_selection_reason && <div className="col-span-2"><dt className="text-zinc-500">Band-selection provenance</dt><dd className="mt-1 leading-5 text-zinc-200">{metadata.band_selection_reason}</dd></div>}
            <div><dt className="text-zinc-500">Data type</dt><dd className="mt-1 text-zinc-200">{metadata.dtype}</dd></div>
            <div><dt className="text-zinc-500">Format</dt><dd className="mt-1 text-zinc-200">{metadata.format.toUpperCase()}</dd></div>
            <div><dt className="text-zinc-500">Representation</dt><dd className="mt-1 text-zinc-200">{(metadata.representation ?? "unknown representation").replaceAll("_", " ")}</dd></div>
            <div><dt className="text-zinc-500">Detected modality</dt><dd className="mt-1 text-zinc-200">{(metadata.auto_detected_modality ?? "unknown").replaceAll("_", " ")} · {metadata.auto_detection_confidence ?? "unavailable"}</dd></div>
            {metadata.auto_detection_reason && <div className="col-span-2"><dt className="text-zinc-500">Detection reason</dt><dd className="mt-1 leading-5 text-zinc-200">{metadata.auto_detection_reason}</dd></div>}
            <div className="col-span-2"><dt className="text-zinc-500">CRS</dt><dd className="mt-1 break-all text-zinc-200">{metadata.crs ?? "Unavailable"}</dd></div>
            <div><dt className="text-zinc-500">Georeferenced</dt><dd className="mt-1 text-zinc-200">{value(metadata.is_georeferenced)}</dd></div>
            <div><dt className="text-zinc-500">NoData</dt><dd className="mt-1 text-zinc-200">{metadata.nodata ?? "Unavailable"}</dd></div>
            {metadata.bounds && <div className="col-span-2"><dt className="text-zinc-500">Bounds</dt><dd className="mt-1 font-mono text-[11px] leading-5 text-zinc-200">L {metadata.bounds.left} · B {metadata.bounds.bottom} · R {metadata.bounds.right} · T {metadata.bounds.top}</dd></div>}
            {metadata.color_interpretation.length > 0 && <div className="col-span-2"><dt className="text-zinc-500">Color interpretation</dt><dd className="mt-1 text-zinc-200">{metadata.color_interpretation.join(", ")}</dd></div>}
        </dl></details>
        {preview && <p className="assistant-preview-disclosure">Display preview only — not a scientific product.</p>}
        {metadata.warnings.length > 0 && <ul className="space-y-1 border-t border-white/[.08] px-4 py-3 text-xs leading-5 text-amber-100">{metadata.warnings.map(warning => <li key={warning}>• {warning}</li>)}</ul>}
      </div>}
    </div>}
  </article>;
}

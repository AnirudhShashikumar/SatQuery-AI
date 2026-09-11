"use client";

import { CheckCircle2, FileImage, FileUp, LoaderCircle, X } from "lucide-react";
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
    <header className="flex items-start justify-between gap-3 border-b border-white/[.08] px-4 py-3">
      <div><h3 className="text-sm font-semibold">{label}</h3><p className="mt-1 text-xs text-zinc-500">{hint}</p></div>
      {file && !busy && <button type="button" onClick={() => onFile()} aria-label={`Remove ${label}`} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white"><X size={16}/></button>}
    </header>
    {!file ? <button type="button" {...dropzone.getRootProps()} className={cn("assistant-dropzone group", dropzone.isDragActive && "assistant-dropzone-active")}>
      <input {...dropzone.getInputProps()} />
      <span className="assistant-upload-icon"><FileUp size={23}/></span>
      <span className="text-sm font-medium">{dropzone.isDragActive ? "Drop image here" : "Select File or drag and drop"}</span>
      <span className="mt-1 text-xs text-zinc-500">GeoTIFF, TIFF, PNG, JPEG · maximum 100 MB</span>
    </button> : <div>
      <div className="flex items-center gap-3 p-4">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-sky-400/10 text-sky-200">{busy ? <LoaderCircle size={19} className="animate-spin"/> : <FileImage size={19}/>}</span>
        <div className="min-w-0 flex-1"><p className="truncate text-sm font-medium">{file.name}</p><p className="mt-1 text-xs text-zinc-500">{(file.size / 1024 / 1024).toFixed(2)} MB · {busy ? "Uploading and preparing metadata…" : metadata ? "Backend inspection complete" : "Ready for local upload"}</p></div>
        {!busy && <CheckCircle2 size={17} className="shrink-0 text-emerald-300" aria-label="File ready"/>}
      </div>
      {metadata && <div className="border-t border-white/[.08]">
        {preview && <figure className="bg-zinc-950/60"><img src={preview} alt={`${label} display preview`} className="aspect-video w-full object-contain"/><figcaption className="border-t border-white/[.06] px-3 py-2 text-center text-[11px] text-amber-200">Display preview only — not a scientific product.</figcaption></figure>}
        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 p-4 text-xs">
          <div><dt className="text-zinc-500">Dimensions</dt><dd className="mt-1 text-zinc-200">{metadata.width} × {metadata.height}</dd></div>
          <div><dt className="text-zinc-500">Bands</dt><dd className="mt-1 text-zinc-200">{metadata.band_count}</dd></div>
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
        </dl>
        {metadata.warnings.length > 0 && <ul className="space-y-1 border-t border-white/[.08] px-4 py-3 text-xs leading-5 text-amber-100">{metadata.warnings.map(warning => <li key={warning}>• {warning}</li>)}</ul>}
      </div>}
    </div>}
  </article>;
}

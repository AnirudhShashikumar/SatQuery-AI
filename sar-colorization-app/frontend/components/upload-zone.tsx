"use client";

import { FileUp, X } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";

export function UploadZone({ label, hint, accept, file, onFile }: {
  label: string; hint: string; accept: Record<string, string[]>; file?: File; onFile: (file?: File) => void;
}) {
  const dropzone = useDropzone({ accept, maxFiles: 1, onDrop: ([next]) => onFile(next) });
  return <div className="space-y-2"><div className="flex items-baseline justify-between"><label className="text-sm font-medium">{label}</label><span className="text-xs text-zinc-500">{hint}</span></div>
    {file ? <div className="glass flex items-center justify-between rounded-2xl px-4 py-3"><div className="min-w-0"><p className="truncate text-sm">{file.name}</p><p className="text-xs text-zinc-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p></div><button onClick={() => onFile()} aria-label={`Remove ${label}`} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white"><X size={16}/></button></div> :
    <button {...dropzone.getRootProps()} className={cn("group flex min-h-32 w-full flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-white/[.018] px-5 text-center transition hover:border-sky-400/50 hover:bg-sky-400/[.04]", dropzone.isDragActive && "border-sky-400 bg-sky-400/[.08]")}><input {...dropzone.getInputProps()} /><FileUp className="mb-3 text-sky-300 transition group-hover:-translate-y-0.5" size={24}/><span className="text-sm font-medium">Drop file here, or browse</span><span className="mt-1 text-xs text-zinc-500">{hint}</span></button>}
  </div>;
}

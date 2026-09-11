"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { Download, Expand } from "lucide-react";
import { ImageAnalysis } from "@/components/image-analysis";
import { FullscreenImageViewer } from "@/components/viewer/fullscreen-image-viewer";
import { ImageViewer } from "@/components/viewer/image-viewer";
import { dataUrl } from "@/lib/utils";

type Analysis = { type: string; metadata?: Record<string, unknown>; onReport?: (report: import("@/types/api").ImageAnalysisReport) => void };
type Action = { name: string; data: string };

export function ImageCard({ title, image, action, analysis }: { title: string; image?: string | null; action?: Action; analysis?: Analysis }) {
  const src = useMemo(() => dataUrl(image), [image]);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const closeFullscreen = useCallback(() => { setFullscreen(false); requestAnimationFrame(() => triggerRef.current?.focus()); }, []);
  return <article className="panel overflow-hidden transition duration-200 hover:-translate-y-0.5 hover:border-sky-300/20"><header className="flex items-center justify-between px-4 py-3"><h3 className="text-sm font-medium">{title}</h3>{src && <div className="flex gap-1">{analysis && <ImageAnalysis image={src} title={title} analysisType={analysis.type} metadata={analysis.metadata} onReport={analysis.onReport}/>}<button ref={triggerRef} onClick={() => setFullscreen(true)} aria-label={`Open ${title} fullscreen viewer`} className="rounded-md p-1.5 text-zinc-400 hover:bg-white/10 hover:text-white"><Expand size={15}/></button>{action && <a href={src} download={action.name} className="rounded-md p-1.5 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Download image"><Download size={15}/></a>}</div>}</header><ImageViewer src={src} alt={title}/><FullscreenImageViewer open={fullscreen} src={src} title={title} downloadName={action?.name} onClose={closeFullscreen}/></article>;
}

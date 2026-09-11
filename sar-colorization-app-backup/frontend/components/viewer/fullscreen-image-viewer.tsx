"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, Minus, Plus, RotateCcw, X } from "lucide-react";
import { useImageTransform } from "@/hooks/use-image-transform";

type Props = { open: boolean; src?: string | null; title: string; downloadName?: string; onClose: () => void };

export function FullscreenImageViewer({ open, src, title, downloadName, onClose }: Props) {
  const [mounted, setMounted] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const { transform, zoomIn, zoomOut, reset, panHandlers } = useImageTransform();
  useEffect(() => { setMounted(true); return () => setMounted(false); }, []);
  useEffect(() => { if (!open) reset(); }, [open, reset]);
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", escape);
    requestAnimationFrame(() => closeRef.current?.focus());
    return () => { document.body.style.overflow = previous; window.removeEventListener("keydown", escape); };
  }, [onClose, open]);
  const onKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "+" || event.key === "=") { event.preventDefault(); zoomIn(); }
    if (event.key === "-") { event.preventDefault(); zoomOut(); }
    if (event.key === "0") { event.preventDefault(); reset(); }
  }, [reset, zoomIn, zoomOut]);
  if (!mounted || !open || !src) return null;
  return createPortal(<div className="fixed inset-0 z-[70] bg-black/90" role="dialog" aria-modal="true" aria-label={`${title} fullscreen viewer`}><div {...panHandlers} tabIndex={0} onKeyDown={onKeyDown} className="relative flex h-dvh w-dvw items-center justify-center overflow-hidden outline-none [contain:layout_paint_style]" style={{ touchAction: "none", cursor: transform.scale > 1 ? "grab" : "default" }}>
    <img src={src} alt={title} draggable={false} decoding="async" className="max-h-full max-w-full select-none object-contain transition-transform duration-150 ease-out [backface-visibility:hidden] [-webkit-backface-visibility:hidden] [-webkit-user-drag:none]" style={{ transform: `translate3d(${transform.translateX}px, ${transform.translateY}px, 0) scale(${transform.scale})`, transformOrigin: "center center", willChange: "transform" }}/>
    <div onPointerDown={event => event.stopPropagation()} className="absolute right-4 top-4 z-10 flex items-center gap-1 rounded-xl border border-white/10 bg-zinc-950/90 p-1"><span className="px-1 text-xs tabular-nums text-zinc-400" aria-live="polite">{Math.round(transform.scale * 100)}%</span><button onClick={zoomIn} className="rounded-lg p-2 hover:bg-white/10" aria-label="Zoom in"><Plus size={16}/></button><button onClick={zoomOut} className="rounded-lg p-2 hover:bg-white/10" aria-label="Zoom out"><Minus size={16}/></button><button onClick={reset} className="rounded-lg p-2 hover:bg-white/10" aria-label="Reset zoom"><RotateCcw size={16}/></button>{downloadName && <a href={src} download={downloadName} className="rounded-lg p-2 hover:bg-white/10" aria-label="Download image"><Download size={16}/></a>}<button ref={closeRef} onClick={onClose} className="rounded-lg p-2 hover:bg-white/10" aria-label="Close fullscreen viewer"><X size={16}/></button></div>
  </div></div>, document.body);
}

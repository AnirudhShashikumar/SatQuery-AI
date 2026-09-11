"use client";

import { useCallback } from "react";
import { useImageTransform } from "@/hooks/use-image-transform";

type Props = { src?: string | null; alt: string };

export function ImageViewer({ src, alt }: Props) {
  const { transform, zoomIn, panHandlers } = useImageTransform();
  const onWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    if (event.deltaY >= 0) return;
    event.preventDefault(); zoomIn();
  }, [zoomIn]);
  return <div {...panHandlers} onWheel={onWheel} onDoubleClick={zoomIn} className="relative flex aspect-square min-h-0 items-center justify-center overflow-hidden bg-zinc-950/70 [contain:layout_paint_style]" style={{ touchAction: "none", cursor: transform.scale > 1 ? "grab" : "default" }}>
    {src ? <img src={src} alt={alt} draggable={false} decoding="async" className="max-h-full max-w-full select-none object-contain transition-transform duration-150 ease-out [backface-visibility:hidden] [-webkit-backface-visibility:hidden] [-webkit-user-drag:none]" style={{ transform: `translate3d(${transform.translateX}px, ${transform.translateY}px, 0) scale(${transform.scale})`, transformOrigin: "center center", willChange: "transform" }}/> : <span className="text-sm text-zinc-600">Awaiting ground truth</span>}
  </div>;
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Expand, Minus, Plus, RotateCcw, X } from "lucide-react";
import type { ComparabilityResult, ComparisonPreview } from "@/types/agent";
import { agentPreviewUrl } from "@/services/api";
import { dimensionsMatch } from "@/lib/mission-comparison";

type ImageChoice = ComparisonPreview & { requestId: string; resultName: string };
type Mode = "side-by-side" | "swipe" | "overlay" | "difference";

const loadImage = (src: string) => new Promise<HTMLImageElement>((resolve, reject) => { const image = new Image(); image.crossOrigin = "anonymous"; image.onload = () => resolve(image); image.onerror = reject; image.src = src; });

function DifferenceImage({ left, right }: { left: ImageChoice; right: ImageChoice }) {
  const [src, setSrc] = useState<string>();
  useEffect(() => {
    let cancelled = false;
    Promise.all([loadImage(agentPreviewUrl(left.url)!), loadImage(agentPreviewUrl(right.url)!)]).then(([a, b]) => {
      if (cancelled || a.naturalWidth !== b.naturalWidth || a.naturalHeight !== b.naturalHeight) return;
      const canvas = document.createElement("canvas"); canvas.width = a.naturalWidth; canvas.height = a.naturalHeight;
      const first = document.createElement("canvas"); first.width = canvas.width; first.height = canvas.height;
      const second = document.createElement("canvas"); second.width = canvas.width; second.height = canvas.height;
      first.getContext("2d")?.drawImage(a, 0, 0); second.getContext("2d")?.drawImage(b, 0, 0);
      const firstPixels = first.getContext("2d")?.getImageData(0, 0, canvas.width, canvas.height); const secondPixels = second.getContext("2d")?.getImageData(0, 0, canvas.width, canvas.height); const context = canvas.getContext("2d");
      if (!firstPixels || !secondPixels || !context) return;
      const output = context.createImageData(canvas.width, canvas.height);
      for (let index = 0; index < output.data.length; index += 4) { const delta = (Math.abs(firstPixels.data[index] - secondPixels.data[index]) + Math.abs(firstPixels.data[index + 1] - secondPixels.data[index + 1]) + Math.abs(firstPixels.data[index + 2] - secondPixels.data[index + 2])) / 3; output.data[index] = Math.min(255, delta * 2.5); output.data[index + 1] = Math.min(255, delta * .8); output.data[index + 2] = 35; output.data[index + 3] = 255; }
      context.putImageData(output, 0, 0); if (!cancelled) setSrc(canvas.toDataURL("image/png"));
    }).catch(() => !cancelled && setSrc(undefined));
    return () => { cancelled = true; };
  }, [left, right]);
  return src ? <img src={src} alt="Scientifically permitted absolute pixel difference view" draggable={false} className="max-h-full max-w-full object-contain"/> : <p className="text-sm text-zinc-500">Preparing valid difference view…</p>;
}

export function MissionImageComparisonViewer({ images, assessments }: { images: ImageChoice[]; assessments: ComparabilityResult[] }) {
  const [mode, setMode] = useState<Mode>("side-by-side"); const [swipe, setSwipe] = useState(50); const [opacity, setOpacity] = useState(50); const [scale, setScale] = useState(1); const [position, setPosition] = useState({ x: 0, y: 0 }); const [fullscreen, setFullscreen] = useState(false); const pointer = useRef<{ x: number; y: number; id: number } | undefined>(undefined);
  const pair = images.length === 2 ? assessments.find(value => [value.left_request_id, value.right_request_id].every(id => images.some(image => image.requestId === id))) : undefined;
  const exactDimensions = images.length === 2 && dimensionsMatch(images[0], images[1]);
  const overlayAllowed = Boolean(pair?.overlay_allowed && exactDimensions); const differenceAllowed = Boolean(pair?.difference_allowed && exactDimensions);
  useEffect(() => { if ((mode === "overlay" && !overlayAllowed) || (mode === "difference" && !differenceAllowed) || (mode === "swipe" && images.length !== 2)) setMode("side-by-side"); }, [differenceAllowed, images.length, mode, overlayAllowed]);
  const transform = `translate3d(${position.x}px,${position.y}px,0) scale(${scale})`;
  const reset = () => { setScale(1); setPosition({ x: 0, y: 0 }); };
  const zoom = (amount: number) => setScale(value => Math.min(8, Math.max(1, Number((value + amount).toFixed(2)))));
  const panHandlers = { onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => { if (scale <= 1) return; event.currentTarget.setPointerCapture(event.pointerId); pointer.current = { x: event.clientX, y: event.clientY, id: event.pointerId }; }, onPointerMove: (event: React.PointerEvent<HTMLDivElement>) => { const active = pointer.current; if (!active || active.id !== event.pointerId) return; const dx = event.clientX - active.x; const dy = event.clientY - active.y; active.x = event.clientX; active.y = event.clientY; setPosition(value => ({ x: value.x + dx, y: value.y + dy })); }, onPointerUp: () => { pointer.current = undefined; }, onPointerCancel: () => { pointer.current = undefined; } };
  const content = useMemo(() => {
    const image = (value: ImageChoice, className = "") => <img key={value.requestId} src={agentPreviewUrl(value.url)} alt={`${value.resultName}: ${value.label}`} draggable={false} decoding="async" className={`max-h-full max-w-full select-none object-contain ${className}`} style={{ transform, transformOrigin: "center", willChange: "transform" }}/>;
    if (mode === "side-by-side") return <div className={`grid h-full w-full gap-2 ${images.length > 2 ? "grid-cols-2" : "sm:grid-cols-2"}`}>{images.map(value => <figure key={value.requestId} className="relative grid min-h-52 place-items-center overflow-hidden rounded-xl bg-black/35"><figcaption className="absolute left-2 top-2 z-10 rounded-md bg-black/70 px-2 py-1 text-xs text-white">{value.resultName}</figcaption>{image(value)}</figure>)}</div>;
    if (images.length !== 2) return null;
    if (mode === "swipe") return <div className="relative h-full w-full overflow-hidden rounded-xl bg-black/35">{image(images[0], "absolute inset-0 h-full w-full") }<div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - swipe}% 0 0)` }}>{image(images[1], "absolute inset-0 h-full w-full")}</div><span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-xs">{images[0].resultName}</span><span className="absolute bottom-2 right-2 rounded bg-black/70 px-2 py-1 text-xs">{images[1].resultName}</span></div>;
    if (mode === "overlay") return <div className="relative h-full w-full overflow-hidden rounded-xl bg-black/35">{image(images[0], "absolute inset-0 h-full w-full")}<div className="absolute inset-0" style={{ opacity: opacity / 100 }}>{image(images[1], "absolute inset-0 h-full w-full")}</div></div>;
    return <div className="grid h-full place-items-center overflow-hidden rounded-xl bg-black/35" style={{ transform, transformOrigin: "center" }}><DifferenceImage left={images[0]} right={images[1]}/></div>;
  }, [images, mode, opacity, swipe, transform]);
  if (!images.length) return <section className="panel p-5 text-sm text-zinc-500">Selected results do not contain image evidence for visual comparison.</section>;
  return <section className={fullscreen ? "fixed inset-0 z-[90] flex flex-col bg-zinc-950 p-4" : "panel p-4 sm:p-5"} aria-label="Image comparison viewer"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-semibold">Image comparison viewer</h3><p className="mt-1 text-xs text-zinc-500">Zoom and pan are synchronized across visible images.</p></div><div className="flex flex-wrap gap-1">{(["side-by-side", "swipe", "overlay", "difference"] as Mode[]).map(value => { const disabled = (value === "swipe" && images.length !== 2) || (value === "overlay" && !overlayAllowed) || (value === "difference" && !differenceAllowed); return <button key={value} type="button" disabled={disabled} title={disabled ? value === "swipe" ? "Swipe requires exactly two images." : "Requires a direct comparison and matching dimensions." : undefined} onClick={() => setMode(value)} className={`rounded-lg px-3 py-2 text-xs ${mode === value ? "bg-sky-400/15 text-sky-200" : "bg-white/[.04] text-zinc-400 disabled:opacity-35"}`}>{value.replace("-", " ")}</button>; })}</div></div>
    {(mode === "swipe" || mode === "overlay") && <label className="mt-3 flex items-center gap-3 text-xs text-zinc-400"><span>{mode === "swipe" ? "Swipe position" : "Overlay opacity"}</span><input aria-label={mode === "swipe" ? "Swipe position" : "Overlay opacity"} className="w-56 accent-sky-300" type="range" min="0" max="100" value={mode === "swipe" ? swipe : opacity} onChange={event => mode === "swipe" ? setSwipe(Number(event.target.value)) : setOpacity(Number(event.target.value))}/></label>}
    {!overlayAllowed && images.length === 2 && <p className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[.05] px-3 py-2 text-xs text-amber-100">Overlay is disabled unless the backend marks the pair directly comparable and both output dimensions match.</p>}
    <div {...panHandlers} onWheel={event => { event.preventDefault(); zoom(event.deltaY < 0 ? .25 : -.25); }} tabIndex={0} onKeyDown={event => { if (event.key === "+" || event.key === "=") zoom(.25); if (event.key === "-") zoom(-.25); if (event.key === "0") reset(); }} className={`mt-4 min-h-[360px] flex-1 overflow-hidden outline-none ${fullscreen ? "h-[calc(100dvh-9rem)]" : "h-[min(62vh,640px)]"}`} style={{ touchAction: "none", cursor: scale > 1 ? "grab" : "default" }}>{content}</div>
    <div className="mt-3 flex justify-end gap-1"><span className="px-2 py-2 text-xs tabular-nums text-zinc-500">{Math.round(scale * 100)}%</span><button onClick={() => zoom(.25)} aria-label="Zoom in" className="rounded-lg p-2 hover:bg-white/10"><Plus size={16}/></button><button onClick={() => zoom(-.25)} aria-label="Zoom out" className="rounded-lg p-2 hover:bg-white/10"><Minus size={16}/></button><button onClick={reset} aria-label="Fit and reset images" className="rounded-lg p-2 hover:bg-white/10"><RotateCcw size={16}/></button><button onClick={() => setFullscreen(value => !value)} aria-label={fullscreen ? "Exit full screen" : "Open full screen"} className="rounded-lg p-2 hover:bg-white/10">{fullscreen ? <X size={16}/> : <Expand size={16}/>}</button></div>
  </section>;
}

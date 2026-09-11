"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type ViewTransform = { scale: number; translateX: number; translateY: number };
export const MIN_SCALE = 1;
export const MAX_SCALE = 8;
export const ZOOM_STEP = 0.25;

export const clampScale = (value: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));
export const nextScale = (scale: number, amount: number) => Number(clampScale(scale + amount).toFixed(2));

export function useImageTransform() {
  const [transform, setTransformState] = useState<ViewTransform>({ scale: MIN_SCALE, translateX: 0, translateY: 0 });
  const transformRef = useRef(transform);
  const pointer = useRef<{ x: number; y: number; id: number } | null>(null);
  const frame = useRef<number | null>(null);
  const pending = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => () => { if (frame.current !== null) cancelAnimationFrame(frame.current); }, []);

  const setTransform = useCallback((updater: (current: ViewTransform) => ViewTransform) => {
    setTransformState(current => {
      const next = updater(current);
      transformRef.current = next;
      return next;
    });
  }, []);
  const zoomBy = useCallback((amount: number) => setTransform(current => ({ ...current, scale: nextScale(current.scale, amount) })), [setTransform]);
  const zoomIn = useCallback(() => zoomBy(ZOOM_STEP), [zoomBy]);
  const zoomOut = useCallback(() => zoomBy(-ZOOM_STEP), [zoomBy]);
  const reset = useCallback(() => setTransform(() => ({ scale: MIN_SCALE, translateX: 0, translateY: 0 })), [setTransform]);

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (transformRef.current.scale <= MIN_SCALE) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pointer.current = { x: event.clientX, y: event.clientY, id: event.pointerId };
  }, []);
  const onPointerMove = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const active = pointer.current;
    if (!active || active.id !== event.pointerId) return;
    pending.current = { x: event.clientX - active.x, y: event.clientY - active.y };
    active.x = event.clientX; active.y = event.clientY;
    if (frame.current !== null) return;
    frame.current = requestAnimationFrame(() => {
      const delta = pending.current; frame.current = null; pending.current = null;
      if (!delta) return;
      setTransform(current => ({ ...current, translateX: current.translateX + delta.x, translateY: current.translateY + delta.y }));
    });
  }, [setTransform]);
  const finishPan = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (pointer.current?.id !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    pointer.current = null;
  }, []);

  return { transform, zoomIn, zoomOut, reset, panHandlers: { onPointerDown, onPointerMove, onPointerUp: finishPan, onPointerCancel: finishPan } };
}

"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

function recoveryFor(message: string) {
  const lower = message.toLowerCase();
  if (lower.includes("format") || lower.includes("file type") || lower.includes("supported image")) return "Choose a GeoTIFF, TIFF, PNG, or JPEG and try again.";
  if (lower.includes("align") || lower.includes("compatible")) return "Replace one observation with an aligned image of the same area, or use Single Image analysis.";
  if (lower.includes("cancel")) return "Your inputs remain available. Send the request again when ready.";
  if (lower.includes("backend") || lower.includes("unavailable") || lower.includes("fetch")) return "Your inputs remain in this browser. Confirm the local SatQuery backend is running, then retry.";
  if (lower.includes("translation") || lower.includes("pix2pix")) return "Native SAR analysis may still be available. Retry or ask a SAR-native question.";
  return "Review the highlighted input requirements, keep the current observations, and retry when ready.";
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <section role="alert" className="sq-error-state">
    <span><AlertTriangle size={19}/></span>
    <div><p className="eyebrow">Analysis needs attention</p><h2>SatQuery could not complete this request</h2><p>{message}</p><small>{recoveryFor(message)}</small></div>
    {onRetry && <button type="button" onClick={onRetry}><RefreshCw size={14}/>Retry</button>}
  </section>;
}

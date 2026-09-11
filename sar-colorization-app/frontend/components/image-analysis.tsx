"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Brain, CheckCircle2, LoaderCircle, RotateCcw, Settings2, X } from "lucide-react";
import { analyzeImage, ApiRequestError, getProviderSettings } from "@/services/api";
import { AIProviderModal } from "@/components/ai-provider-modal";
import type { ImageAnalysisReport } from "@/types/api";

type Props = {
  image: string;
  analysisType: string;
  title: string;
  metadata?: Record<string, unknown>;
  onReport?: (report: ImageAnalysisReport) => void;
};

type Phase = "idle" | "checking_provider" | "preparing_image" | "uploading" | "analyzing" | "success" | "error";
type ModalState = { open: boolean; image: string | null; analysisType: string | null; modelName: string | null };

const sections = [
  ["Terrain and land cover", "terrain"],
  ["Structural features", "structural_and_human_features"],
  ["Vegetation and water", "vegetation_and_water"],
  ["Reconstruction quality", "image_quality"],
  ["Possible artifacts", "possible_artifacts"],
  ["Scientific observations", "notes"],
  ["Confidence and limitations", "limitations"],
  ["Recommended actions", "recommended_actions"],
] as const;

const errorDetails: Record<string, { title: string; message: string; action: string }> = {
  PROVIDER_NOT_CONFIGURED: { title: "Gemini is not connected", message: "Connect Google Gemini before using AI Analysis.", action: "Open AI Provider Settings" },
  INVALID_API_KEY: { title: "Invalid Gemini API key", message: "The saved Gemini API key is invalid or no longer authorized.", action: "Replace or test the key" },
  UNSUPPORTED_MODEL: { title: "Gemini model unavailable", message: "The selected Gemini model is unavailable. Choose a supported model in Settings.", action: "Open AI Provider Settings" },
  MODEL_PERMISSION_DENIED: { title: "Model access denied", message: "This API key does not have access to the selected Gemini model.", action: "Choose another model or API key" },
  QUOTA_EXCEEDED: { title: "Gemini quota exceeded", message: "Your Gemini quota has been reached. Try again later or use another key.", action: "Retry later" },
  RATE_LIMITED: { title: "Gemini is rate limited", message: "Gemini is temporarily rate-limited. Wait briefly and retry.", action: "Retry" },
  TIMEOUT: { title: "Gemini timed out", message: "Gemini did not respond in time. Your reconstruction result is unaffected.", action: "Retry" },
  NETWORK_ERROR: { title: "Network unavailable", message: "SatQuery AI could not reach Google Gemini. Check your connection and retry.", action: "Retry" },
  GOOGLE_API_UNAVAILABLE: { title: "Gemini is temporarily unavailable", message: "Google Gemini is temporarily unavailable. Your reconstruction result is unaffected.", action: "Retry" },
  UNKNOWN_PROVIDER_ERROR: { title: "Gemini provider error", message: "Gemini returned an unexpected provider error. Your reconstruction result is unaffected.", action: "Retry" },
  INVALID_IMAGE: { title: "Invalid analysis image", message: "The generated image could not be prepared for analysis.", action: "Retry" },
  IMAGE_TOO_LARGE: { title: "Image is too large", message: "The generated image exceeds the AI Analysis upload limit.", action: "Use a smaller image" },
  BACKEND_UNAVAILABLE: { title: "AI backend unavailable", message: "SatQuery AI could not reach the AI Analysis backend.", action: "Retry" },
  MALFORMED_RESPONSE: { title: "Gemini response could not be read", message: "Gemini returned an incomplete analysis. Your reconstruction result is unchanged.", action: "Retry" },
};

function normalizedError(error: unknown) {
  const code = error instanceof ApiRequestError ? error.code : "BACKEND_UNAVAILABLE";
  return { code, ...(errorDetails[code] ?? { title: "AI Analysis failed", message: error instanceof Error ? error.message : "An unexpected AI Analysis error occurred.", action: "Retry" }) };
}

export async function imageAssetToBlob(source: string): Promise<Blob> {
  const response = await fetch(source);
  if (!response.ok) throw new ApiRequestError("The selected image is no longer available for analysis.", "INVALID_IMAGE", response.status);
  const sourceBlob = await response.blob();
  if (!sourceBlob.size) throw new ApiRequestError("The selected image is empty.", "INVALID_IMAGE", 400);
  if (["image/png", "image/jpeg", "image/webp"].includes(sourceBlob.type)) return sourceBlob;
  const bitmap = await createImageBitmap(sourceBlob).catch(() => null);
  if (!bitmap) throw new ApiRequestError("The selected image format cannot be converted for analysis.", "INVALID_IMAGE", 400);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
  bitmap.close();
  const png = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
  if (!png?.size) throw new ApiRequestError("The selected image could not be prepared for analysis.", "INVALID_IMAGE", 400);
  return png;
}

export function ImageAnalysis({ image, analysisType, title, metadata, onReport }: Props) {
  const queryClient = useQueryClient();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [mounted, setMounted] = useState(false);
  const [modal, setModal] = useState<ModalState>({ open: false, image: null, analysisType: null, modelName: null });
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const [configurationMessage, setConfigurationMessage] = useState("");
  const provider = useQuery({ queryKey: ["ai-provider-status"], queryFn: getProviderSettings, enabled: false, staleTime: 10_000 });
  const analysis = useMutation({
    mutationFn: async (input: { image: string; analysisType: string; modelName: string; }) => {
      setPhase("preparing_image");
      const blob = await imageAssetToBlob(input.image);
      if (!blob.size) throw new ApiRequestError("The generated image is empty.", "INVALID_IMAGE", 400);
      setPhase("uploading");
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      setPhase("analyzing");
      return analyzeImage({
        image: blob, analysisType: input.analysisType, modelName: input.modelName,
        checkpointName: typeof metadata?.checkpoint === "string" ? metadata.checkpoint : undefined,
        displayMode: typeof metadata?.display_mode === "string" ? metadata.display_mode : undefined,
        metrics: metadata?.metrics as Record<string, unknown> | undefined,
        metadata,
      });
    },
    onSuccess: result => { setPhase("success"); onReport?.(result.report); },
    onError: () => setPhase("error"),
  });

  useEffect(() => { setMounted(true); return () => setMounted(false); }, []);
  useEffect(() => {
    if (!modal.open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [modal.open]);
  useEffect(() => {
    if (!["checking_provider", "preparing_image", "uploading", "analyzing"].includes(phase)) { setElapsed(0); return; }
    const started = Date.now(); const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 250);
    return () => window.clearInterval(timer);
  }, [phase]);

  const close = () => { if (analysis.isPending) return; setModal(current => ({ ...current, open: false })); setPhase("idle"); requestAnimationFrame(() => triggerRef.current?.focus()); };
  const startRequest = (snapshot: ModalState, modelName: string) => {
    if (!snapshot.image || !snapshot.analysisType || analysis.isPending) return;
    setModal({ ...snapshot, modelName }); analysis.mutate({ image: snapshot.image, analysisType: snapshot.analysisType, modelName });
  };
  const beginAnalysis = async () => {
    if (analysis.isPending) return;
    const snapshot: ModalState = { open: true, image, analysisType, modelName: null };
    setModal(snapshot); setPhase("checking_provider");
    try {
      const status = (await provider.refetch()).data;
      if (!status?.configured || !status.supported || status.connection_status !== "connected" || !status.model) {
        setModal({ open: false, image: null, analysisType: null, modelName: null }); setPhase("idle");
        setConfigurationMessage(status?.connection_status === "invalid" ? "Your Gemini connection needs attention. Replace or retest the key to continue." : "AI analysis requires an API key.");
        setConfigurationOpen(true); return;
      }
      startRequest(snapshot, status.model);
    } catch { setPhase("error"); }
  };
  const retry = () => { if (modal.image && modal.analysisType && modal.modelName) { analysis.reset(); startRequest(modal, modal.modelName); } };
  const continueAfterConfiguration = async () => {
    setConfigurationOpen(false); queryClient.invalidateQueries({ queryKey: ["ai-provider-status"] });
    setConfigurationMessage(""); await beginAnalysis();
  };
  const result = analysis.data; const report = result?.report; const error = analysis.error ? normalizedError(analysis.error) : null;
  const statusText: Record<Phase, string> = {
    idle: "", checking_provider: "Checking Gemini connection…", preparing_image: "Preparing reconstruction image…", uploading: "Uploading image securely…", analyzing: "Analyzing terrain, structure, land cover, and reconstruction quality…", success: "Analysis complete.", error: "Analysis could not be completed.",
  };

  const dialog = modal.open ? <div className="fixed inset-0 z-[100]" role="presentation"><div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onMouseDown={close}/><section role="dialog" aria-modal="true" aria-labelledby="ai-analysis-title" aria-describedby="ai-analysis-description" className="absolute left-1/2 top-1/2 flex w-[min(720px,calc(100vw-32px))] max-h-[88vh] min-h-[420px] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-2xl border border-white/10 bg-zinc-950 shadow-2xl"><header className="flex shrink-0 items-start justify-between border-b border-white/[.08] px-5 py-4 sm:px-6"><div><p className="text-xs uppercase tracking-[.16em] text-sky-300">AI Image Analysis</p><h2 id="ai-analysis-title" className="mt-1 text-lg font-semibold">{title}</h2><p id="ai-analysis-description" className="mt-1 text-xs text-zinc-500">Qualitative review only. Reconstruction and metrics remain unchanged.</p></div><button onClick={close} disabled={analysis.isPending} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white disabled:opacity-40" aria-label="Close AI Analysis"><X size={18}/></button></header><div className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-6" aria-live="polite">{["checking_provider", "preparing_image", "uploading", "analyzing"].includes(phase) && <div className="flex min-h-56 flex-col items-center justify-center text-center"><LoaderCircle className="animate-spin text-sky-300" size={28}/><p className="mt-4 text-sm text-zinc-200">{statusText[phase]}</p><p className="mt-2 text-xs text-zinc-500">{elapsed}s elapsed</p></div>}{phase === "success" && report && <div className="space-y-6 text-sm leading-6 text-zinc-300"><div className="rounded-xl border border-sky-400/15 bg-sky-400/[.06] p-4"><p className="text-xs font-semibold uppercase tracking-[.12em] text-sky-300">Executive summary · {report.confidence} confidence{result.cached ? " · Previously analyzed" : ""}</p><p className="mt-2">{report.executive_summary}</p></div><div className="grid gap-5 sm:grid-cols-2">{sections.map(([label, key]) => <section key={key}><h3 className="font-medium text-zinc-100">{label}</h3><ul className="mt-2 space-y-1 text-zinc-400">{report[key].map(item => <li key={item}>• {item}</li>)}</ul></section>)}</div><div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-amber-50/90"><span className="font-medium">Scientific caution: </span>{report.disclaimer}</div></div>}{phase === "error" && error && <div className="min-h-56 rounded-xl border border-rose-400/20 bg-rose-400/[.08] p-5"><div className="flex gap-3"><AlertTriangle className="shrink-0 text-rose-300" size={20}/><div><h3 className="font-semibold text-rose-100">{error.title}</h3><p className="mt-2 text-sm leading-6 text-rose-100/85">{error.message}</p><p className="mt-3 text-xs text-rose-200/70">Recommended action: {error.action}</p></div></div></div>}{process.env.NEXT_PUBLIC_GEOVISION_DEBUG === "true" && result?.debug && <details className="mt-5 rounded-xl border border-white/[.08] bg-white/[.025] p-4 text-xs text-zinc-400"><summary className="cursor-pointer font-medium text-zinc-200">Developer diagnostics</summary><pre className="mt-3 overflow-auto whitespace-pre-wrap">{JSON.stringify(result.debug, null, 2)}</pre></details>}</div>{phase === "error" && <footer className="flex shrink-0 flex-wrap justify-end gap-2 border-t border-white/[.08] px-5 py-4 sm:px-6"><button onClick={() => { setModal(current => ({ ...current, open: false })); setConfigurationOpen(true); }} className="inline-flex items-center gap-2 rounded-xl border border-white/[.12] px-3 py-2 text-sm text-zinc-200 hover:bg-white/[.05]"><Settings2 size={15}/>Settings</button><button onClick={retry} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-3 py-2 text-sm font-semibold text-zinc-950 hover:bg-sky-200"><RotateCcw size={15}/>Retry</button></footer>}{phase === "success" && <footer className="flex shrink-0 justify-end border-t border-white/[.08] px-5 py-4 sm:px-6"><button onClick={close} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-3 py-2 text-sm font-semibold text-zinc-950 hover:bg-sky-200"><CheckCircle2 size={15}/>Done</button></footer>}</section></div> : null;

  return <><button ref={triggerRef} onClick={beginAnalysis} disabled={analysis.isPending} aria-label="Analyze image" className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-sky-200 hover:bg-sky-300/10 disabled:cursor-not-allowed disabled:opacity-50"><Brain size={15}/>AI Analysis</button>{mounted && dialog ? createPortal(dialog, document.body) : null}<AIProviderModal open={configurationOpen} onClose={() => setConfigurationOpen(false)} onSaved={continueAfterConfiguration} notice={configurationMessage}/></>;
}

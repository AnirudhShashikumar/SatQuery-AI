"use client";

import { ArrowLeft, ExternalLink } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { clearPresentationHandoff, hasPresentationHandoff, presentationRuntimeState, readPresentationHandoff, storePresentationHandoff } from "@/lib/presentation-handoff";

export function PresentationOpenPageButton({ route, stageIndex, children }: { route: string; stageIndex: number; children: React.ReactNode }) {
  const router = useRouter();
  const open = () => {
    const runtime = presentationRuntimeState(document);
    storePresentationHandoff(window.sessionStorage, { route, stageIndex, ...runtime });
    router.push(route);
  };
  return <button type="button" onClick={open} className="presentation-summary-primary"><ExternalLink size={12}/>{children}</button>;
}

export function PresentationReturnBanner({ route }: { route: string }) {
  const router = useRouter();
  const { setTheme } = useTheme();
  const [visible, setVisible] = useState(false);

  useEffect(() => { setVisible(hasPresentationHandoff(window.sessionStorage, route)); }, [route]);
  if (!visible) return null;
  const stageNumber = (readPresentationHandoff(window.sessionStorage)?.stageIndex ?? 0) + 1;

  const returnToPresentation = async () => {
    const handoff = readPresentationHandoff(window.sessionStorage);
    if (!handoff || handoff.route !== route) { setVisible(false); return; }
    if (handoff.theme !== "system") setTheme(handoff.theme);
    if (handoff.fullscreen && !document.fullscreenElement && document.documentElement.requestFullscreen) {
      try { await document.documentElement.requestFullscreen(); }
      catch { /* Return safely even when browser fullscreen permission is unavailable. */ }
    }
    clearPresentationHandoff(window.sessionStorage);
    setVisible(false);
    router.push("/presentation");
  };

  return <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-sky-300/20 bg-sky-300/[.06] p-4"><div><p className="text-sm font-semibold text-sky-100">Opened from Presentation Mode · Stage {stageNumber}</p><p className="mt-1 text-xs text-zinc-400">Your stage, presenter notes, theme, session, and fullscreen preference are preserved.</p></div><button type="button" onClick={() => void returnToPresentation()} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2 text-sm font-semibold text-zinc-950"><ArrowLeft size={15}/>Return to Presentation</button></div>;
}

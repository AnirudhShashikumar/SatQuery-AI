"use client";

import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Boxes,
  CircleAlert,
  CircleCheckBig,
  Database,
  Expand,
  Eye,
  EyeOff,
  FileText,
  GitCompareArrows,
  Globe2,
  Layers3,
  Maximize2,
  Minimize2,
  Moon,
  Orbit,
  Radar,
  Route,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useCallback, useEffect, useRef, useState } from "react";
import { PresentationSingleImageScene } from "@/components/presentation-single-image-scene";
import { PresentationChangeScene } from "@/components/presentation-change-scene";
import { PresentationCrossModalScene } from "@/components/presentation-cross-modal-scene";
import { PresentationReportScene } from "@/components/presentation-report-scene";
import { PresentationComparisonScene } from "@/components/presentation-comparison-scene";
import { PresentationAnalyticsScene } from "@/components/presentation-analytics-scene";
import { PresentationArchitectureScene } from "@/components/presentation-architecture-scene";
import { PresentationComplianceScene } from "@/components/presentation-compliance-scene";
import { PresentationClosingScene } from "@/components/presentation-closing-scene";
import {
  clearPresentationState,
  isPresentationTypingTarget,
  presentationCommandForKey,
  presentationSteps,
  presentationStorageKeys,
  readPresentationReturnRoute,
  readPresentationState,
  type PresentationCommand,
  type PresentationIcon,
  type PresentationStep,
} from "@/lib/presentation";

const iconMap: Record<PresentationIcon, typeof Orbit> = {
  welcome: Globe2,
  problem: CircleAlert,
  solution: Sparkles,
  inputs: Database,
  workflow: Route,
  understanding: Eye,
  grounding: ScanSearch,
  change: Layers3,
  joint: Radar,
  reports: FileText,
  comparison: GitCompareArrows,
  analytics: Activity,
  architecture: Boxes,
  compliance: ShieldCheck,
  closing: CircleCheckBig,
};

const HISTORY_GUARD = "__satqueryPresentationGuard";

export function PresentationMode() {
  const router = useRouter();
  const { resolvedTheme, setTheme } = useTheme();
  const shellRef = useRef<HTMLDivElement>(null);
  const leavingRef = useRef(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [notesVisible, setNotesVisible] = useState(false);
  const [restored, setRestored] = useState(false);
  const [themeReady, setThemeReady] = useState(false);
  const [fullscreenSupported, setFullscreenSupported] = useState(true);
  const [fullscreenActive, setFullscreenActive] = useState(false);
  const [fullscreenMessage, setFullscreenMessage] = useState("");

  const step: PresentationStep = presentationSteps[stepIndex];
  const Icon = iconMap[step.icon ?? "welcome"];
  const interactiveSingleImageScene = step.scene_type === "single_image_understanding";
  const interactiveChangeScene = step.scene_type === "bi_temporal_change";
  const interactiveCrossModalScene = step.scene_type === "cross_modal_analysis";
  const interactiveReportScene = step.scene_type === "mission_report";
  const interactiveComparisonScene = step.scene_type === "mission_comparison";
  const interactiveAnalyticsScene = step.scene_type === "research_analytics";
  const interactiveArchitectureScene = step.scene_type === "architecture_summary";
  const interactiveComplianceScene = step.scene_type === "compliance_summary";
  const interactiveClosingScene = step.scene_type === "closing_summary";
  const interactiveScene = interactiveSingleImageScene || interactiveChangeScene || interactiveCrossModalScene || interactiveReportScene || interactiveComparisonScene || interactiveAnalyticsScene || interactiveArchitectureScene || interactiveComplianceScene || interactiveClosingScene;

  useEffect(() => {
    const saved = readPresentationState(window.sessionStorage);
    setStepIndex(saved.step);
    setNotesVisible(saved.notesVisible);
    setRestored(true);
    setThemeReady(true);
  }, []);

  useEffect(() => {
    if (!restored) return;
    window.sessionStorage.setItem(presentationStorageKeys.step, String(stepIndex));
  }, [restored, stepIndex]);

  useEffect(() => {
    if (!restored) return;
    window.sessionStorage.setItem(presentationStorageKeys.notes, String(notesVisible));
  }, [notesVisible, restored]);

  useEffect(() => {
    const supported = Boolean(document.fullscreenEnabled && shellRef.current?.requestFullscreen);
    const syncFullscreen = () => setFullscreenActive(Boolean(document.fullscreenElement));
    setFullscreenSupported(supported);
    syncFullscreen();
    document.addEventListener("fullscreenchange", syncFullscreen);
    return () => document.removeEventListener("fullscreenchange", syncFullscreen);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    if (!fullscreenSupported || !shellRef.current?.requestFullscreen) {
      setFullscreenMessage("Browser fullscreen is unavailable. The presentation still fills the current window.");
      return;
    }
    setFullscreenMessage("");
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await shellRef.current.requestFullscreen();
      shellRef.current?.focus({ preventScroll: true });
    } catch {
      setFullscreenMessage("Fullscreen could not be opened. Continue in the current window or check browser permissions.");
    }
  }, [fullscreenSupported]);

  const exitPresentation = useCallback(async () => {
    if (leavingRef.current) return;
    leavingRef.current = true;
    const destination = readPresentationReturnRoute(window.sessionStorage);
    clearPresentationState(window.sessionStorage);
    if (document.fullscreenElement) {
      try { await document.exitFullscreen(); }
      catch { /* The route remains safely exitable even if the browser owns fullscreen state. */ }
    }

    const finish = () => router.replace(destination);
    if (window.history.state?.[HISTORY_GUARD]) {
      let finished = false;
      const finishOnce = () => {
        if (finished) return;
        finished = true;
        finish();
      };
      window.addEventListener("popstate", finishOnce, { once: true });
      window.history.back();
      window.setTimeout(finishOnce, 180);
      return;
    }
    finish();
  }, [router]);

  const confirmExit = useCallback(() => {
    if (window.confirm("Exit Presentation Mode and return to your previous SatQuery page?")) void exitPresentation();
  }, [exitPresentation]);

  const runCommand = useCallback((command: PresentationCommand) => {
    if (command === "next") setStepIndex(current => Math.min(current + 1, presentationSteps.length - 1));
    if (command === "previous") setStepIndex(current => Math.max(current - 1, 0));
    if (command === "first") setStepIndex(0);
    if (command === "last") setStepIndex(presentationSteps.length - 1);
    if (command === "notes") setNotesVisible(current => !current);
    if (command === "fullscreen") void toggleFullscreen();
    if (command === "exit") {
      if (document.fullscreenElement) void document.exitFullscreen().catch(() => setFullscreenMessage("Use the browser control to leave fullscreen."));
      else confirmExit();
    }
  }, [confirmExit, toggleFullscreen]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isPresentationTypingTarget(event.target)) return;
      const command = presentationCommandForKey(event.key);
      if (!command) return;
      event.preventDefault();
      runCommand(command);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [runCommand]);

  useEffect(() => {
    if (!restored) return;
    const presentationUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    window.history.pushState({ ...window.history.state, [HISTORY_GUARD]: true }, "", presentationUrl);

    const onPopState = () => {
      if (leavingRef.current) return;
      if (window.confirm("Leave Presentation Mode and return to your previous SatQuery page?")) {
        void exitPresentation();
      } else {
        window.history.pushState({ ...window.history.state, [HISTORY_GUARD]: true }, "", presentationUrl);
      }
    };
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      if (leavingRef.current) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("popstate", onPopState);
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [exitPresentation, restored]);

  return <div ref={shellRef} tabIndex={-1} className="presentation-shell min-h-dvh overflow-x-hidden text-zinc-100 outline-none">
    <header className="presentation-header mx-auto flex w-full max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6 lg:px-10" aria-label="Presentation controls">
      <div className="flex min-w-0 items-center gap-3">
        <span className="presentation-logo grid h-10 w-10 shrink-0 place-items-center rounded-xl" aria-hidden="true"><Orbit size={21}/></span>
        <div className="min-w-0"><p className="text-[9px] font-bold uppercase tracking-[.2em] text-sky-300">Guided product demo</p><p className="truncate text-base font-bold tracking-tight">SatQuery <span className="text-sky-400">AI</span></p></div>
      </div>

      <div className="order-3 w-full text-center sm:order-none sm:w-auto">
        <p className="text-xs font-semibold tabular-nums text-zinc-300">Step {step.number} of {presentationSteps.length}</p>
        <p className="mt-0.5 max-w-72 truncate text-[10px] uppercase tracking-[.16em] text-zinc-500">{step.title}</p>
      </div>

      <div className="flex items-center gap-1.5">
        <button type="button" onClick={() => setNotesVisible(value => !value)} aria-pressed={notesVisible} className="presentation-control" aria-label={notesVisible ? "Hide presenter notes" : "Show presenter notes"}>{notesVisible ? <EyeOff size={16}/> : <Eye size={16}/>}<span className="hidden xl:inline">Notes</span></button>
        <button type="button" onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")} className="presentation-icon-control" aria-label={themeReady ? `Use ${resolvedTheme === "dark" ? "light" : "dark"} theme` : "Toggle presentation theme"}>{themeReady && resolvedTheme === "light" ? <Moon size={16}/> : <Sun size={16}/>}</button>
        <button type="button" onClick={() => void toggleFullscreen()} aria-pressed={fullscreenActive} className="presentation-control" aria-label={fullscreenActive ? "Exit browser fullscreen" : "Enter browser fullscreen"}>{fullscreenActive ? <Minimize2 size={16}/> : <Maximize2 size={16}/>}<span className="hidden xl:inline">{fullscreenActive ? "Window" : "Fullscreen"}</span></button>
        <button type="button" onClick={() => void exitPresentation()} className="presentation-control presentation-exit" aria-label="Exit Presentation Mode"><X size={16}/><span className="hidden sm:inline">Exit</span></button>
      </div>
    </header>

    <div className="presentation-progress-track" role="progressbar" aria-label="Presentation progress" aria-valuemin={1} aria-valuemax={presentationSteps.length} aria-valuenow={step.number} aria-valuetext={`Step ${step.number} of ${presentationSteps.length}: ${step.title}`}>
      <span style={{ width: `${(step.number / presentationSteps.length) * 100}%` }}/>
    </div>

    {(!fullscreenSupported || fullscreenMessage) && <div className="presentation-status mx-4 mt-3 flex max-w-2xl shrink-0 self-center items-center gap-2 rounded-xl border px-4 py-2 text-xs" role="status"><Expand size={14} className="shrink-0"/>{fullscreenMessage || "Browser fullscreen is unavailable. Presentation Mode will remain in this window."}</div>}

    <main className={`presentation-main mx-auto grid w-full max-w-[1540px] gap-4 px-4 py-4 sm:px-6 sm:py-6 lg:px-10 ${notesVisible ? "presentation-main-with-notes" : ""}`} aria-labelledby="presentation-stage-title">
      <article className="presentation-stage relative min-w-0 overflow-hidden rounded-[1.75rem] border p-5 sm:p-8 lg:p-10">
        <div className="presentation-stage-glow" aria-hidden="true"/>
        <div className="relative z-10 flex h-full min-h-0 flex-col">
          <p className="text-[10px] font-bold uppercase tracking-[.24em] text-sky-300">Stage {String(step.number).padStart(2, "0")}</p>
          <h1 id="presentation-stage-title" className={`mt-3 max-w-5xl font-semibold leading-[.96] tracking-[-.045em] text-white ${interactiveScene ? "text-[clamp(1.8rem,3.8vw,3.6rem)]" : "text-[clamp(2rem,5.2vw,5.25rem)]"}`}>{step.title}</h1>
          <p className={`max-w-4xl leading-relaxed text-zinc-400 ${interactiveScene ? "mt-2 text-sm sm:text-base" : "mt-4 text-[clamp(.95rem,1.55vw,1.35rem)]"}`}>{step.subtitle}</p>

          {interactiveSingleImageScene ? <PresentationSingleImageScene step={step}/> : interactiveChangeScene ? <PresentationChangeScene step={step}/> : interactiveCrossModalScene ? <PresentationCrossModalScene step={step}/> : interactiveReportScene ? <PresentationReportScene/> : interactiveComparisonScene ? <PresentationComparisonScene onContinue={() => runCommand("next")}/> : interactiveAnalyticsScene ? <PresentationAnalyticsScene onContinue={() => runCommand("next")}/> : interactiveArchitectureScene ? <PresentationArchitectureScene onContinue={() => runCommand("next")}/> : interactiveComplianceScene ? <PresentationComplianceScene onContinue={() => runCommand("next")}/> : interactiveClosingScene ? <PresentationClosingScene/> : <div className="presentation-stage-content mt-6 grid min-h-0 flex-1 items-center gap-5 lg:grid-cols-[minmax(280px,.8fr)_minmax(360px,1.2fr)] lg:gap-8">
            <div className="presentation-visual relative mx-auto grid aspect-square w-[min(44vw,30vh,310px)] min-w-48 place-items-center rounded-full" aria-label={`${step.title} visual placeholder`}>
              <span className="presentation-orbit presentation-orbit-one" aria-hidden="true"/>
              <span className="presentation-orbit presentation-orbit-two" aria-hidden="true"/>
              <span className="presentation-visual-core grid h-24 w-24 place-items-center rounded-[2rem] sm:h-28 sm:w-28"><Icon size={44} strokeWidth={1.45}/></span>
            </div>

            <ul className="grid gap-3" aria-label={`${step.title} talking points`}>
              {step.points.map((point, index) => <li key={point} className="presentation-point flex items-start gap-4 rounded-2xl border px-4 py-3.5 sm:px-5 sm:py-4">
                <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-sky-400/10 text-xs font-bold tabular-nums text-sky-300">{String(index + 1).padStart(2, "0")}</span>
                <span className="text-sm font-medium leading-6 text-zinc-200 sm:text-base">{point}</span>
              </li>)}
            </ul>
          </div>}
        </div>
      </article>

      {notesVisible && step.presenterNote && <aside className="presentation-notes min-w-0 rounded-[1.5rem] border p-4 sm:p-5" aria-labelledby="presenter-notes-title">
        <div className="flex items-center justify-between gap-3"><div><p className="text-[9px] font-bold uppercase tracking-[.2em] text-amber-200">Presenter only</p><h2 id="presenter-notes-title" className="mt-1 text-base font-semibold">Speaker notes</h2></div><button type="button" onClick={() => setNotesVisible(false)} className="presentation-icon-control" aria-label="Dismiss presenter notes"><X size={15}/></button></div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
          <NoteSection label="Say" text={step.presenterNote.say}/>
          <NoteSection label="Show" text={step.presenterNote.show}/>
          <NoteSection label="Watch for" text={step.presenterNote.watchFor}/>
        </div>
      </aside>}
    </main>

    <footer className="presentation-footer mx-auto flex w-full max-w-[1600px] items-center justify-between gap-3 px-4 py-3 sm:px-6 lg:px-10">
      <button type="button" onClick={() => runCommand("previous")} disabled={stepIndex === 0} className="presentation-nav-button"><ArrowLeft size={17}/><span>Previous</span></button>
      <div className="hidden min-w-0 flex-1 items-center justify-center gap-1.5 px-4 md:flex" aria-hidden="true">
        {presentationSteps.map((item, index) => <span key={item.number} className={`h-1.5 rounded-full transition-[width,background-color] ${index === stepIndex ? "w-8 bg-sky-300" : index < stepIndex ? "w-3 bg-sky-400/45" : "w-3 bg-white/10"}`}/>) }
      </div>
      <button type="button" onClick={() => runCommand("next")} disabled={stepIndex === presentationSteps.length - 1} className="presentation-nav-button presentation-nav-primary"><span>Next</span><ArrowRight size={17}/></button>
    </footer>

    <p className="sr-only">Keyboard shortcuts: Right Arrow or Space for next, Left Arrow or Backspace for previous, Home for first, End for last, F for fullscreen, N for notes, and Escape to leave fullscreen or confirm exit.</p>
    <p className="sr-only" aria-live="polite" aria-atomic="true">Step {step.number} of {presentationSteps.length}: {step.title}</p>
  </div>;
}

function NoteSection({ label, text }: { label: string; text: string }) {
  return <section className="rounded-xl border border-white/[.07] bg-white/[.025] p-3"><h3 className="text-[10px] font-bold uppercase tracking-[.16em] text-amber-200">{label}</h3><p className="mt-2 text-xs leading-5 text-zinc-400">{text}</p></section>;
}

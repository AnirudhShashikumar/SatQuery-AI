"use client";

import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2, CircleDot, Clock3, MinusCircle, XCircle } from "lucide-react";
import type { AnalyticsExecution } from "@/types/agent";
import { architectureLabel } from "@/lib/architecture";
import { formatAnalyticsDuration } from "@/lib/analytics";
import { cn } from "@/lib/utils";

const tones: Record<string, string> = {
  success: "border-emerald-300/25 bg-emerald-300/[.06]",
  failed: "border-rose-300/25 bg-rose-300/[.06]",
  skipped: "border-white/[.06] bg-white/[.018] opacity-65",
  not_implemented: "border-amber-300/20 bg-amber-300/[.05]",
  running: "border-sky-300/40 bg-sky-300/[.08]",
  pending: "border-sky-300/30 bg-sky-300/[.05]",
};

function StageIcon({ status }: { status: string }) {
  if (status === "success") return <CheckCircle2 size={15} className="text-emerald-300"/>;
  if (status === "failed") return <XCircle size={15} className="text-rose-300"/>;
  if (status === "skipped" || status === "not_implemented") return <MinusCircle size={15} className="text-zinc-500"/>;
  return <CircleDot size={15} className="text-sky-300"/>;
}

export function LiveWorkflowTimeline({ execution }: { execution: AnalyticsExecution }) {
  const reducedMotion = useReducedMotion();
  return <section aria-labelledby="live-pipeline-title">
    <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="eyebrow">Exact observable stages</p><h3 id="live-pipeline-title" className="mt-2 text-xl font-semibold">Live Pipeline Timeline</h3></div><span className="inline-flex items-center gap-1.5 text-xs text-zinc-500"><Clock3 size={13}/>{formatAnalyticsDuration(execution.duration_ms)} total</span></div>
    <ol className="relative mt-6 space-y-3 before:absolute before:bottom-5 before:left-[18px] before:top-5 before:w-px before:bg-gradient-to-b before:from-sky-400/50 before:via-white/10 before:to-transparent">
      {execution.trace.map((step, index) => {
        const parameters = Object.entries(step.parameters);
        return <motion.li key={`${execution.request_id}-${step.tool}-${index}`} initial={reducedMotion ? false : { opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: .22, delay: reducedMotion ? 0 : Math.min(index * .055, .45) }} className="relative grid gap-3 pl-12 sm:grid-cols-[1fr_auto] sm:items-start">
          <span className={cn("absolute left-1.5 top-3 z-10 grid h-7 w-7 place-items-center rounded-full border bg-zinc-950", step.status === "success" ? "border-emerald-300/35" : step.status === "failed" ? "border-rose-300/35" : "border-white/10")}><StageIcon status={step.status}/></span>
          <div className={cn("rounded-xl border p-3.5", tones[step.status] ?? "border-white/[.08] bg-white/[.025]")}><div className="flex flex-wrap items-center gap-2"><span className="font-mono text-xs text-zinc-200">{step.tool}</span><span className="rounded-full border border-white/[.08] px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-500">{architectureLabel(step.status)}</span></div>{parameters.length > 0 && <dl className="mt-3 flex flex-wrap gap-2">{parameters.map(([name, value]) => <div key={name} className="rounded-lg bg-black/10 px-2.5 py-1.5 text-[11px]"><dt className="inline text-zinc-500">{architectureLabel(name)}: </dt><dd className="inline text-zinc-300">{Array.isArray(value) ? value.join(", ") : String(value)}</dd></div>)}</dl>}</div>
          <span className="pt-3 text-right text-xs tabular-nums text-zinc-500">{formatAnalyticsDuration(step.duration_ms)}</span>
        </motion.li>;
      })}
    </ol>
  </section>;
}

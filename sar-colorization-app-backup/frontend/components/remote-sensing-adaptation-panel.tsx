import { AlertTriangle, CheckCircle2, Cpu, LoaderCircle, Satellite } from "lucide-react";
import type { SVEResult } from "@/types/agent";

type Props = { result?: SVEResult | null; loading?: boolean; compact?: boolean };

const label = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());

export function RemoteSensingAdaptationPanel({ result, loading = false, compact = false }: Props) {
  const status = loading ? "loading" : result?.status ?? "unavailable";
  const ready = Boolean(result?.available && status === "success");
  const statusText = loading ? "Loading" : status === "success" ? "Ready" : label(status);
  const StatusIcon = loading ? LoaderCircle : ready ? CheckCircle2 : AlertTriangle;
  return <section
    tabIndex={0}
    aria-labelledby="remote-sensing-adaptation-title"
    aria-busy={loading}
    className={`rounded-2xl border border-sky-300/15 bg-sky-300/[.035] p-4 outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60 ${compact ? "text-xs" : ""}`}
  >
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="flex min-w-0 items-start gap-3"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-sky-300/10 text-sky-200" aria-hidden="true"><Satellite size={18}/></span><div><p className="text-[9px] font-bold uppercase tracking-[.18em] text-sky-300">Remote-Sensing Adaptation</p><h3 id="remote-sensing-adaptation-title" className="mt-1 font-semibold text-zinc-100">SatQuery Vision Encoder v1</h3><p className="mt-1 text-[11px] text-zinc-500">OpenCLIP ViT-L/14 · Adapted on BigEarthNet.txt</p></div></div>
      <div className="flex flex-wrap items-center gap-2"><span className="rounded-full border border-sky-300/25 px-2.5 py-1 text-[9px] font-semibold text-sky-200">Remote-sensing adapted</span><span role="status" className="inline-flex items-center gap-1.5 rounded-full border border-white/[.09] px-2.5 py-1 text-[9px] text-zinc-300"><StatusIcon size={11} className={loading ? "animate-spin" : ""}/>{statusText}</span></div>
    </div>

    {ready ? <>
      <dl className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Fact name="Device" value={result?.device ?? "Unavailable"}/>
        <Fact name="Runtime" value={result?.runtime_ms == null ? "Unavailable" : `${result.runtime_ms.toLocaleString()} ms`}/>
        <Fact name="Caption consistency" value={result?.caption_consistency ? result.caption_consistency.score.toFixed(3) : "Not requested"}/>
        <Fact name="VQA consistency" value={result?.vqa_consistency ? label(result.vqa_consistency.state) : "Not requested"}/>
      </dl>
      {result?.scene_priors.length ? <div className="mt-3"><p className="text-[9px] font-bold uppercase tracking-[.14em] text-zinc-500">Similarity-based scene priors · not probabilities</p><ul className="mt-2 flex flex-wrap gap-2" aria-label="Top scene priors">{result.scene_priors.map(prior => <li key={prior.label} className="rounded-full border border-white/[.08] bg-white/[.025] px-2.5 py-1 text-[10px] text-zinc-300"><span>{prior.label}</span> <span className="font-mono text-sky-200">{prior.similarity.toFixed(3)}</span></li>)}</ul></div> : null}
      {result?.semantic_comparison && <p className="mt-3 text-[11px] leading-5 text-zinc-400"><strong className="text-zinc-200">{result.semantic_comparison.label}:</strong> {result.semantic_comparison.similarity == null ? label(result.semantic_comparison.status) : result.semantic_comparison.similarity.toFixed(3)}. {result.semantic_comparison.disclaimer}</p>}
      {result?.fallback && <p className="mt-3 flex items-start gap-2 text-[11px] leading-5 text-amber-100"><Cpu size={13} className="mt-1 shrink-0"/>{result.fallback}</p>}
    </> : <p className="mt-3 text-[11px] leading-5 text-amber-100">{loading ? "The verified adapter and exact OpenCLIP backbone are loading. Existing specialist execution remains independent." : result?.warning ?? "Scene-level embedding evidence is unavailable for this result; existing specialist output remains unchanged."}</p>}

    <details className="mt-3"><summary className="cursor-pointer text-[10px] font-semibold text-zinc-400">Limitations and scientific disclosure</summary><p className="mt-2 text-[10px] leading-5 text-zinc-500">{result?.disclaimer ?? "Scene-level embedding evidence does not produce segmentation, grounding, calibrated probabilities, or ground truth."}</p>{result?.limitations.length ? <ul className="mt-1 space-y-1 text-[10px] leading-4 text-zinc-500">{result.limitations.slice(0, compact ? 3 : 6).map(item => <li key={item}>• {item}</li>)}</ul> : null}</details>
  </section>;
}

function Fact({ name, value }: { name: string; value: string }) {
  return <div className="rounded-xl border border-white/[.06] bg-white/[.02] p-2"><dt className="text-[9px] uppercase tracking-wide text-zinc-500">{name}</dt><dd className="mt-1 text-[11px] text-zinc-200">{value}</dd></div>;
}

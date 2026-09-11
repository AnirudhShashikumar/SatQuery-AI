"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, CheckCircle2, Clock3, Database, FlaskConical, Gauge, Layers3, LoaderCircle, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { MetricCard } from "@/components/metric-card";
import { getBenchmark } from "@/services/api";
import type { BenchmarkData, BenchmarkMetrics } from "@/types/api";

type DataMode = "live" | "saved" | "demo";
type ModelMode = "pix2pix" | "sarfusionformer" | "side-by-side";

const demoMetrics: Record<"pix2pix" | "sarfusionformer", BenchmarkMetrics> = {
  pix2pix: { psnr: 24.8, ssim: 0.84, rgb_l1: 0.071, inference_time_ms: 318, model_size_mb: null, gpu_memory_mb: null },
  sarfusionformer: { psnr: 23.9, ssim: 0.81, rgb_l1: 0.083, inference_time_ms: 340, model_size_mb: null, gpu_memory_mb: null },
};

const labels = { pix2pix: "Pix2Pix", sarfusionformer: "SARFusionFormer" };
const metricRows: { key: keyof BenchmarkMetrics; label: string; unit?: string; higher?: boolean }[] = [
  { key: "psnr", label: "PSNR", unit: "dB", higher: true },
  { key: "ssim", label: "SSIM", higher: true },
  { key: "rgb_l1", label: "RGB L1", higher: false },
  { key: "inference_time_ms", label: "Inference Time", unit: "ms", higher: false },
  { key: "model_size_mb", label: "Model Size", unit: "MB", higher: false },
];

function format(value: number | null, digits = 2) { return value === null ? "Not measured" : Number.isInteger(value) ? String(value) : value.toFixed(digits); }
function sourceFor(mode: DataMode, value: number | null, derived = false) { return value === null ? undefined : derived ? "derived" : mode === "demo" ? "demo" : "measured" as const; }
function best(a: number | null, b: number | null, higher: boolean | undefined, side: "a" | "b") { if (a === null || b === null || a === b) return false; return side === "a" ? (higher ? a > b : a < b) : (higher ? b > a : b < a); }

export function BenchmarkView() {
  const [dataMode, setDataMode] = useState<DataMode>("demo");
  const [modelMode, setModelMode] = useState<ModelMode>("pix2pix");
  const live = useQuery({ queryKey: ["benchmark"], queryFn: getBenchmark, retry: false, refetchInterval: 15_000 });
  const data: BenchmarkData | null = live.data ?? null;
  const availableLive = Boolean(data?.models.pix2pix.sample || data?.models.sarfusionformer.sample);
  const metricsFor = (model: "pix2pix" | "sarfusionformer") => dataMode === "demo" ? demoMetrics[model] : data?.models[model].metrics ?? null;
  const selected = modelMode === "sarfusionformer" ? "sarfusionformer" : "pix2pix";
  const metrics = metricsFor(selected);
  const showNoData = dataMode === "saved" || (dataMode === "live" && !availableLive);
  return <section className="benchmark-page"><header className="benchmark-header"><div><p className="eyebrow">Evaluation intelligence</p><h1>Benchmarks</h1><p>Measured quality, coverage, and latency across vision-language reasoning and reconstruction workloads.</p></div><span><ShieldCheck size={15}/> Evidence-backed metrics</span></header><VqaBenchmarkSnapshot/>
    <section className="benchmark-model-lab"><div><p className="eyebrow">Reconstruction lab</p><h2>Model evaluation</h2><p>Inspect live, saved, or illustrative reconstruction metrics separately from the measured VQA snapshot.</p></div><div className="mt-7 flex flex-wrap gap-2">{([ ["live", "Live Results", Database], ["saved", "Saved Benchmark", Layers3], ["demo", "Demo Preview", FlaskConical] ] as const).map(([mode, label, Icon]) => <button key={mode} onClick={() => setDataMode(mode)} className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm transition ${dataMode === mode ? "bg-sky-400/15 text-sky-200" : "bg-white/[.04] text-zinc-400 hover:bg-white/[.08]"}`}><Icon size={16}/>{label}</button>)}</div><div className="mt-3 flex flex-wrap gap-2">{([ ["pix2pix", "Pix2Pix"], ["sarfusionformer", "SARFusionFormer"], ["side-by-side", "Side-by-side"] ] as const).map(([mode, label]) => <button key={mode} onClick={() => setModelMode(mode)} className={`rounded-lg px-3 py-1.5 text-xs transition ${modelMode === mode ? "bg-white/[.10] text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}>{label}</button>)}</div>{dataMode === "demo" && <p className="mt-5 rounded-2xl border border-amber-300/20 bg-amber-300/[.07] p-4 text-sm leading-6 text-amber-100">Preview values are illustrative and must not be used as reported experimental results.</p>}{showNoData && <p className="mt-5 rounded-2xl border border-sky-300/15 bg-sky-300/[.06] p-4 text-sm leading-6 text-sky-100/80">{dataMode === "saved" ? "No saved benchmark is available in this browser." : "No measured current sample is available yet. Run an inference with ground truth or compare two outputs to populate live metrics."}</p>}{live.isLoading && <p className="mt-5 inline-flex items-center gap-2 text-sm text-zinc-400"><LoaderCircle className="animate-spin" size={16}/>Loading benchmark metadata…</p>}{live.error && <p className="mt-5 text-sm text-rose-300">{live.error.message}</p>}
    {modelMode !== "side-by-side" && <ModelCards model={selected} metrics={metrics} mode={dataMode} modelSize={data?.models[selected].metrics.model_size_mb ?? null}/>} {modelMode === "side-by-side" && <ComparisonTable pix={metricsFor("pix2pix")} sar={metricsFor("sarfusionformer")} mode={dataMode}/>}<Leaderboard pix={metricsFor("pix2pix")} sar={metricsFor("sarfusionformer")} mode={dataMode}/></section></section>;
}

const questionTypes = [
  { label: "Comparison", value: 40, samples: "6 / 15" },
  { label: "Count", value: 40, samples: "6 / 15" },
  { label: "Presence", value: 33.33, samples: "5 / 15" },
  { label: "Rural / urban", value: 40, samples: "2 / 5" },
] as const;

function VqaBenchmarkSnapshot() {
  return <section className="vqa-benchmark" aria-labelledby="vqa-benchmark-title">
    <div className="benchmark-datasets">
      <article className="dataset-card dataset-card-active"><div><span>Measured · 50 samples</span><CheckCircle2 size={16}/></div><h2 id="vqa-benchmark-title">RSVQA-LR</h2><p>Remote-sensing visual question answering smoke benchmark</p><dl><div><dt>Accuracy</dt><dd>38.0%</dd></div><div><dt>Coverage</dt><dd>70.0%</dd></div><div><dt>Mean latency</dt><dd>1.06 s</dd></div></dl></article>
      <article className="dataset-card"><div><span>Evaluation ready</span><Activity size={16}/></div><h2>VRSBench</h2><p>Comprehensive remote-sensing vision-language evaluation</p><dl><div><dt>Accuracy</dt><dd>Not run</dd></div><div><dt>Coverage</dt><dd>Not run</dd></div><div><dt>Latency</dt><dd>Not run</dd></div></dl></article>
      <article className="dataset-card"><div><span>Evaluation ready</span><Activity size={16}/></div><h2>CDVQA</h2><p>Change-detection visual question answering evaluation</p><dl><div><dt>Accuracy</dt><dd>Not run</dd></div><div><dt>Coverage</dt><dd>Not run</dd></div><div><dt>Latency</dt><dd>Not run</dd></div></dl></article>
    </div>
    <div className="benchmark-overview">
      <div className="benchmark-kpis"><article><Gauge size={17}/><span><small>Overall accuracy</small><strong>38.00%</strong><em>19 of 50 correct</em></span></article><article><CheckCircle2 size={17}/><span><small>Answer coverage</small><strong>70.00%</strong><em>35 answered · 15 null</em></span></article><article><Clock3 size={17}/><span><small>P95 latency</small><strong>2.23 s</strong><em>1.06 s average</em></span></article></div>
      <article className="benchmark-chart"><header><div><span>Accuracy by task</span><small>RSVQA-LR smoke · measured</small></div><BarChart3 size={18}/></header><div>{questionTypes.map(row => <div className="benchmark-bar" key={row.label}><span>{row.label}</span><div><i style={{ width: `${row.value}%` }}/></div><strong>{row.value.toFixed(row.value % 1 ? 2 : 0)}%</strong><small>{row.samples}</small></div>)}</div></article>
    </div>
    <p className="benchmark-disclosure">Snapshot generated from the completed 50-question local smoke run. VRSBench and CDVQA are intentionally marked unmeasured; no synthetic scores are shown.</p>
  </section>;
}

function ModelCards({ model, metrics, mode, modelSize }: { model: "pix2pix" | "sarfusionformer"; metrics: BenchmarkMetrics | null; mode: DataMode; modelSize: number | null }) {
  const metric = metrics ?? { psnr: null, ssim: null, rgb_l1: null, inference_time_ms: null, model_size_mb: null, gpu_memory_mb: null };
  const size = mode === "demo" ? modelSize : metric.model_size_mb;
  return <><div className="mt-7 flex items-center gap-2"><BarChart3 size={17} className="text-sky-300"/><h2 className="font-semibold">{labels[model]} metrics</h2><span className="text-xs text-zinc-500">{mode === "live" && metrics ? "Measured on current sample" : mode === "demo" ? "Illustrative preview" : "Not measured"}</span></div><div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><MetricCard title="PSNR" value={format(metric.psnr, 1)} unit={metric.psnr === null ? undefined : "dB"} trend="higher" source={sourceFor(mode, metric.psnr)}/><MetricCard title="SSIM" value={format(metric.ssim)} trend="higher" source={sourceFor(mode, metric.ssim)}/><MetricCard title="RGB L1" value={format(metric.rgb_l1, 3)} trend="lower" source={sourceFor(mode, metric.rgb_l1)}/><MetricCard title="Inference Time" value={format(metric.inference_time_ms, 0)} unit={metric.inference_time_ms === null ? undefined : "ms"} trend="lower" source={sourceFor(mode, metric.inference_time_ms)}/><MetricCard title="GPU Memory" value={format(metric.gpu_memory_mb, 1)} unit={metric.gpu_memory_mb === null ? undefined : "MB"} description={metric.gpu_memory_mb === null ? "Not measured by backend" : "Measured on this device"} source={sourceFor(mode, metric.gpu_memory_mb)}/><MetricCard title="Model Size" value={format(size, 2)} unit={size === null ? undefined : "MB"} description={size === null ? "Checkpoint unavailable" : "Deployment footprint"} source={sourceFor(mode, size, true)}/><MetricCard title="Visual Quality" value="Not measured" description="Qualitative evaluation required"/><MetricCard title="Structural Fidelity" value="Not measured" description="Qualitative evaluation required"/></div></>;
}

function ComparisonTable({ pix, sar, mode }: { pix: BenchmarkMetrics | null; sar: BenchmarkMetrics | null; mode: DataMode }) {
  const left = pix ?? { psnr: null, ssim: null, rgb_l1: null, inference_time_ms: null, model_size_mb: null, gpu_memory_mb: null }; const right = sar ?? left;
  return <div className="panel mt-7 overflow-x-auto"><table className="w-full min-w-[620px] text-left text-sm"><caption className="px-5 pt-5 text-left text-base font-semibold">Side-by-side metrics <span className="ml-2 text-xs font-normal text-zinc-500">{mode === "demo" ? "Illustrative preview" : "Measured where available"}</span></caption><thead className="text-xs uppercase tracking-wider text-zinc-500"><tr><th className="px-5 py-4">Metric</th><th className="px-5 py-4">Pix2Pix</th><th className="px-5 py-4">SARFusionFormer</th></tr></thead><tbody>{metricRows.map(row => { const a = left[row.key] as number | null; const b = right[row.key] as number | null; return <tr key={row.key} className="border-t border-white/[.08]"><th className="px-5 py-4 font-medium text-zinc-200">{row.label}</th><td className={`px-5 py-4 tabular-nums ${best(a, b, row.higher, "a") ? "text-emerald-300" : "text-zinc-300"}`}>{format(a, row.key === "rgb_l1" ? 3 : 2)}{a !== null && row.unit ? ` ${row.unit}` : ""}</td><td className={`px-5 py-4 tabular-nums ${best(a, b, row.higher, "b") ? "text-emerald-300" : "text-zinc-300"}`}>{format(b, row.key === "rgb_l1" ? 3 : 2)}{b !== null && row.unit ? ` ${row.unit}` : ""}</td></tr>; })}</tbody></table></div>;
}

function Leaderboard({ pix, sar, mode }: { pix: BenchmarkMetrics | null; sar: BenchmarkMetrics | null; mode: DataMode }) {
  const notes = mode === "live" && pix && sar ? ["Best PSNR on this sample", "Best SSIM on this sample", "Fastest on this device", "Lower RGB error on this sample"] : ["Metrics appear when a common measured sample is available."];
  return <article className="panel mt-5 overflow-x-auto p-5 sm:p-6"><h2 className="font-semibold">Model roles and measured metrics</h2><p className="mt-1 text-sm text-zinc-500">No arbitrary ranks or overall scores are assigned.</p><table className="mt-5 w-full min-w-[800px] text-left text-sm"><thead className="text-xs uppercase tracking-wider text-zinc-500"><tr><th className="pb-3">Model</th><th className="pb-3">Visual Role</th><th className="pb-3">Structural Role</th><th className="pb-3">PSNR</th><th className="pb-3">SSIM</th><th className="pb-3">RGB L1</th><th className="pb-3">Inference</th></tr></thead><tbody className="text-zinc-300"><tr className="border-t border-white/[.08]"><td className="py-4 font-medium">Pix2Pix</td><td>Visual realism</td><td>Moderate</td><td>{format(pix?.psnr ?? null, 1)}</td><td>{format(pix?.ssim ?? null)}</td><td>{format(pix?.rgb_l1 ?? null, 3)}</td><td>{format(pix?.inference_time_ms ?? null, 0)}{pix?.inference_time_ms !== null && pix?.inference_time_ms !== undefined ? " ms" : ""}</td></tr><tr className="border-t border-white/[.08]"><td className="py-4 font-medium">SARFusionFormer</td><td>Structure preservation</td><td>Strong</td><td>{format(sar?.psnr ?? null, 1)}</td><td>{format(sar?.ssim ?? null)}</td><td>{format(sar?.rgb_l1 ?? null, 3)}</td><td>{format(sar?.inference_time_ms ?? null, 0)}{sar?.inference_time_ms !== null && sar?.inference_time_ms !== undefined ? " ms" : ""}</td></tr></tbody></table><div className="mt-5 flex flex-wrap gap-2">{notes.map(note => <span key={note} className="rounded-full border border-white/[.08] bg-white/[.03] px-3 py-1.5 text-xs text-zinc-400">{note}</span>)}</div></article>;
}

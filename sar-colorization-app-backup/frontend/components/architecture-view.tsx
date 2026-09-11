"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Braces, CheckCircle2, Cpu, Eye, GitBranch, Layers3, LoaderCircle, Network, RefreshCw, ScanLine, SlidersHorizontal, Sparkles, SplitSquareVertical } from "lucide-react";
import { AgenticArchitectureVisualizer } from "@/components/agentic-architecture-visualizer";
import { getAgentAnalytics, getAgentCompliance, getAgentHealth, getAgentTools } from "@/services/api";
import { cn } from "@/lib/utils";
import { DEFAULT_ARCHITECTURE_TAB } from "@/lib/architecture";

type ArchitectureTab = "agentic" | "pix2pix" | "sarfusionformer";
type ModelStage = { name: string; purpose: string; verified: string; operations: string; notes: string; icon: keyof typeof icons };

const pixStages: ModelStage[] = [
  { name: "SAR Input", purpose: "Accept the repository's existing three-channel SAR image representation.", verified: "Backend model construction uses c_in=3.", operations: "Decode RGB-compatible input and prepare a batched tensor.", notes: "This differs from SARFusionFormer's separate VV/VH path.", icon: "scan" },
  { name: "Resize & Normalize", purpose: "Match the deployed generator input preparation.", verified: "Bicubic resize to 256 × 256; [0,1] values are scaled to [-1,1].", operations: "Resize, tensor conversion, and deterministic range normalization.", notes: "The transformation is implemented in the existing Pix2Pix inference path.", icon: "sliders" },
  { name: "U-Net Encoder", purpose: "Extract a hierarchy of spatial features.", verified: "Eight repository-defined downsampling blocks; the first omits normalization.", operations: "Convolutional downsampling from the input to the deepest feature representation.", notes: "Each encoder activation is retained for its matching decoder skip connection.", icon: "layers" },
  { name: "Skip Connections", purpose: "Carry spatial detail from encoder stages into the decoder.", verified: "Repository code concatenates seven matching encoder and decoder activations.", operations: "Channel-wise feature concatenation at matching scales.", notes: "This is the defining U-Net path used by the deployed generator.", icon: "network" },
  { name: "U-Net Decoder", purpose: "Reconstruct the optical-style image from deep and skip features.", verified: "Eight repository-defined upsampling blocks; the first three use dropout.", operations: "Transposed-convolution upsampling and skip-feature fusion.", notes: "The deployed model uses the generator only.", icon: "split" },
  { name: "RGB Head", purpose: "Map the final decoder features to three output channels.", verified: "A 3 × 3 convolution maps 64 channels to 3, followed by Tanh.", operations: "RGB regression in the normalized generator range.", notes: "The backend rescales generator output from [-1,1] to [0,1].", icon: "braces" },
  { name: "PatchGAN Discriminator", purpose: "Provide local adversarial supervision during training.", verified: "The repository default is a three-layer 70 × 70 PatchGAN.", operations: "Scores concatenated SAR/optical patches during conditional-GAN training.", notes: "Training only: backend inference constructs Pix2Pix with is_train=False, so the discriminator is not executed.", icon: "network" },
  { name: "Reconstruction Output", purpose: "Return a displayable optical reconstruction.", verified: "The backend clamps and encodes the generated RGB output.", operations: "Convert the real model output to a PNG-compatible representation.", notes: "Visual plausibility does not imply physical or radiometric equivalence.", icon: "eye" },
];

const sarStages: ModelStage[] = [
  { name: "VV / VH Input", purpose: "Accept co-registered dual-polarization SAR observations.", verified: "SARFusionFormer validates a B × 2 × H × W input tensor.", operations: "Validate numeric VV/VH channels and matching spatial dimensions.", notes: "Channel 0 is VV and channel 1 is VH in the deployed preparation path.", icon: "scan" },
  { name: "Preprocessing", purpose: "Reproduce the deployed training-compatible input representation.", verified: "Invalid values become zero; each channel uses 1st/99th percentile normalization and 256 × 256 resize.", operations: "Finite-value handling, independent channel normalization, and resize.", notes: "No log/dB, gamma, global, histogram, or [-1,1] input normalization is applied.", icon: "sliders" },
  { name: "Multi-Scale Encoder", purpose: "Extract fine, mid, and coarse spatial features.", verified: "Repository code uses a convolutional stem, residual blocks, GroupNorm, GELU, and two stride-2 transitions.", operations: "Produce full-, half-, and quarter-scale feature maps.", notes: "The three scales are passed to feature fusion and decoder context paths.", icon: "layers" },
  { name: "Feature Fusion", purpose: "Align and combine all three encoder scales.", verified: "Fine and mid features are downsampled to the coarse scale, concatenated, projected, and refined.", operations: "Cross-scale channel alignment and residual fusion.", notes: "This connects local convolutional evidence to the transformer bottleneck.", icon: "network" },
  { name: "Swin Bottleneck", purpose: "Model longer-range structure with shifted-window attention.", verified: "The deployed backend constructs 4 blocks, 6 attention heads, and window size 8.", operations: "Alternating regular and shifted window attention with MLP residual updates.", notes: "These values come directly from backend model construction, not inferred checkpoint statistics.", icon: "cpu" },
  { name: "Dual Lab Decoder", purpose: "Recover spatial detail and predict luminance and chroma.", verified: "Two upsampling stages use cross-attention to mid/fine SAR features; separate heads predict one L and two ab channels.", operations: "Cross-attention, residual refinement, luminance prediction, and chroma prediction.", notes: "The decoder also exposes auxiliary outputs internally; the primary result is normalized Lab.", icon: "split" },
  { name: "Lab → RGB", purpose: "Convert the normalized scientific model output to clipped sRGB.", verified: "The repository implements normalized CIE Lab conversion followed by [0,1] clamping.", operations: "Lab denormalization, XYZ conversion, sRGB transfer function, and clamp.", notes: "The network output is Lab; RGB is a deterministic conversion.", icon: "braces" },
  { name: "Optional Color Correction", purpose: "Apply a separately controlled display refinement after raw prediction.", verified: "A small residual 1 × 1 convolutional network is loaded independently and disabled unless requested.", operations: "Bounded per-pixel residual RGB adjustment.", notes: "It does not replace the raw output or alter raw model metrics.", icon: "sparkles" },
  { name: "RGB Reconstruction", purpose: "Return raw radiometric and visualization-ready products.", verified: "Raw model output is preserved; enhanced display uses a separate global 2nd/98th percentile stretch.", operations: "PNG encoding for raw, enhanced, and optional corrected views.", notes: "Display enhancement is visualization-only and never changes measured raw metrics.", icon: "eye" },
];

const icons = { scan: ScanLine, sliders: SlidersHorizontal, layers: Layers3, network: Network, cpu: Cpu, split: SplitSquareVertical, braces: Braces, sparkles: Sparkles, eye: Eye };

export function ArchitectureView() {
  const [tab, setTab] = useState<ArchitectureTab>(DEFAULT_ARCHITECTURE_TAB);
  const analytics = useQuery({ queryKey: ["agent-analytics", "architecture"], queryFn: ({ signal }) => getAgentAnalytics(signal), retry: 1, refetchInterval: 10_000, refetchIntervalInBackground: false });
  const tools = useQuery({ queryKey: ["agent-tools", "architecture"], queryFn: getAgentTools, retry: 1, staleTime: 30_000 });
  const health = useQuery({ queryKey: ["agent-health", "architecture"], queryFn: ({ signal }) => getAgentHealth(signal), retry: 1, refetchInterval: 30_000, refetchIntervalInBackground: false });
  const compliance = useQuery({ queryKey: ["agent-compliance", "architecture"], queryFn: getAgentCompliance, retry: 1, staleTime: 30_000 });
  const reducedMotion = useReducedMotion();
  const liveReady = analytics.data && tools.data && health.data && compliance.data;
  const error = analytics.error ?? tools.error ?? health.error ?? compliance.error;
  const onTabsKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const order: ArchitectureTab[] = ["agentic", "pix2pix", "sarfusionformer"];
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const next = order[(order.indexOf(tab) + offset + order.length) % order.length];
    setTab(next);
    document.getElementById(`architecture-tab-${next}`)?.focus();
  };

  return <section><header className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between"><div><p className="eyebrow">Live orchestration and verified model structure</p><h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Agentic Workflow Visualizer</h1><p className="mt-4 max-w-4xl text-sm leading-6 text-zinc-400">Follow a real request from validated imagery through deterministic routing, specialist evidence, scientific governance, and mission output.</p></div>{analytics.data && <div className="flex items-center gap-2 rounded-full border border-white/[.08] px-3 py-2 text-xs text-zinc-500"><span className={cn("h-2 w-2 rounded-full", analytics.data.platform.backend_status === "healthy" ? "bg-emerald-300" : "bg-amber-300")}/>{analytics.data.platform.backend_status} · {analytics.isFetching ? "refreshing" : "live"}<button type="button" onClick={() => analytics.refetch()} className="ml-1 rounded-full p-1 text-sky-300 hover:bg-sky-300/10" aria-label="Refresh architecture data"><RefreshCw size={13} className={analytics.isFetching ? "animate-spin" : ""}/></button></div>}</header>
    <div role="tablist" aria-label="Architecture views" onKeyDown={onTabsKeyDown} className="mt-7 flex flex-wrap gap-2">{([["agentic", "Agentic System", GitBranch], ["pix2pix", "Pix2Pix", Eye], ["sarfusionformer", "SARFusionFormer", Cpu]] as const).map(([id, label, Icon]) => <button id={`architecture-tab-${id}`} key={id} role="tab" aria-selected={tab === id} aria-controls={`architecture-panel-${id}`} tabIndex={tab === id ? 0 : -1} onClick={() => setTab(id)} className={cn("inline-flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300", tab === id ? "border-sky-300/35 bg-sky-300/[.10] text-sky-200" : "border-white/[.07] bg-white/[.025] text-zinc-400 hover:bg-white/[.05]")}><Icon size={15}/>{label}</button>)}</div>
    <AnimatePresence mode="wait"><motion.div key={tab} id={`architecture-panel-${tab}`} role="tabpanel" aria-labelledby={`architecture-tab-${tab}`} initial={reducedMotion ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={reducedMotion ? undefined : { opacity: 0, y: -5 }} transition={{ duration: .2 }} className="mt-7">
      {tab === "agentic" && <>{!liveReady && !error && <Loading/>}{error && <ErrorState message={error instanceof Error ? error.message : "Live architecture data is unavailable."}/>} {liveReady && <AgenticArchitectureVisualizer analytics={analytics.data} tools={tools.data} health={health.data} compliance={compliance.data} refreshing={analytics.isFetching}/>}</>}
      {tab === "pix2pix" && <ModelArchitectureExplorer name="Pix2Pix" subtitle="Verified generator path with training-only discriminator clearly separated." stages={pixStages}/>}
      {tab === "sarfusionformer" && <ModelArchitectureExplorer name="SARFusionFormer" subtitle="Verified VV/VH preprocessing, multi-scale fusion, shifted-window attention, and Lab reconstruction." stages={sarStages}/>}
    </motion.div></AnimatePresence>
  </section>;
}

function ModelArchitectureExplorer({ name, subtitle, stages }: { name: string; subtitle: string; stages: ModelStage[] }) {
  const [selected, setSelected] = useState(0);
  const stage = stages[selected];
  const Icon = icons[stage.icon];
  return <section><div><p className="eyebrow">Repository-verified architecture</p><h2 className="mt-2 text-2xl font-semibold">{name}</h2><p className="mt-2 text-sm leading-6 text-zinc-500">{subtitle} No parameter counts or tensor dimensions are asserted unless present in the code.</p></div><div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{stages.map((item, index) => { const StageIcon = icons[item.icon]; const active = selected === index; return <button key={item.name} type="button" aria-pressed={active} onClick={() => setSelected(index)} className={cn("panel p-4 text-left transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300", active ? "border-sky-300/45 bg-sky-300/[.07]" : "hover:border-sky-300/20")}><div className="flex items-center justify-between"><span className="eyebrow">Stage {index + 1}</span><StageIcon size={16} className={active ? "text-sky-300" : "text-zinc-600"}/></div><h3 className="mt-2 text-sm font-semibold">{item.name}</h3><p className="mt-2 text-xs leading-5 text-zinc-500">{item.purpose}</p></button>; })}</div><article className="panel mt-5 p-5 sm:p-6" aria-live="polite"><div className="flex items-start justify-between gap-4"><div><p className="eyebrow">{name} · Stage {selected + 1}</p><h3 className="mt-2 text-xl font-semibold">{stage.name}</h3></div><span className="grid h-10 w-10 place-items-center rounded-xl bg-sky-300/[.08] text-sky-300"><Icon size={18}/></span></div><p className="mt-4 text-sm leading-6 text-zinc-400">{stage.purpose}</p><div className="mt-5 grid gap-4 lg:grid-cols-3"><Detail title="Verified from code" text={stage.verified} icon={CheckCircle2}/><Detail title="Operations" text={stage.operations} icon={Braces}/><Detail title="Scientific note" text={stage.notes} icon={AlertTriangle}/></div></article></section>;
}

function Detail({ title, text, icon: Icon }: { title: string; text: string; icon: typeof CheckCircle2 }) { return <div className="rounded-xl border border-white/[.07] bg-white/[.018] p-4"><h4 className="flex items-center gap-2 text-sm font-medium"><Icon size={14} className="text-sky-300"/>{title}</h4><p className="mt-2 text-sm leading-6 text-zinc-400">{text}</p></div>; }
function Loading() { return <div className="panel p-8 text-center" aria-busy="true"><LoaderCircle className="mx-auto animate-spin text-sky-300"/><p className="mt-3 text-sm text-zinc-500">Loading live registry, health, compliance, and execution data…</p></div>; }
function ErrorState({ message }: { message: string }) { return <div role="alert" className="panel border-rose-300/20 p-6"><AlertTriangle className="text-rose-300"/><h2 className="mt-3 text-lg font-semibold">Live architecture data unavailable</h2><p className="mt-2 text-sm text-zinc-400">{message} No substitute execution path is being shown.</p></div>; }

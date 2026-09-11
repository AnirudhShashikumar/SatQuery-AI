"use client";

import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowDown, Binary, BrainCircuit, CheckCircle2, CircleDot, FileOutput, GitBranch, Layers3, LockKeyhole, MinusCircle, ScanSearch, ShieldCheck, Upload, XCircle } from "lucide-react";
import type { AgentHealth, AnalyticsExecution, AnalyticsResponse, ComplianceResponse, ToolDefinition } from "@/types/agent";
import { activePathSummary, architectureNodeState, statusDescription, type ArchitectureNode, type PathState } from "@/lib/architecture";
import { cn } from "@/lib/utils";
import { ExecutionInspector } from "@/components/execution-inspector";
import { SpecialistInspector } from "@/components/specialist-inspector";

type Layer = { id: string; label: string; description: string; icon: typeof Upload; nodes: ArchitectureNode[] };

const layers: Layer[] = [
  { id: "input", label: "Input Layer", description: "Accepted imagery and pairing modes", icon: Upload, nodes: [
    { id: "single-optical", label: "Single Optical", detail: "Optical or RGB-like multispectral", matchModes: ["single"], matchModalities: ["optical", "multispectral"] },
    { id: "single-sar", label: "Single SAR", detail: "Accepted for validated SAR workflows", matchModes: ["single"], matchModalities: ["sar"] },
    { id: "optical-sar", label: "Optical + SAR Pair", detail: "Cross-modal exactly aligned pair", matchModes: ["cross_modal"] },
    { id: "bitemporal", label: "Bi-Temporal Pair", detail: "Before and after imagery", matchModes: ["bi_temporal"] },
    { id: "formats", label: "Supported Formats", detail: "GeoTIFF · TIFF · PNG · JPEG" },
  ]},
  { id: "validation", label: "Validation Layer", description: "No silent registration or modality coercion", icon: ShieldCheck, nodes: [
    { id: "file-validation", label: "File Validation", detail: "Signature, size, and decode checks", matchTrace: ["upload", "file_type_validation", "input_validator"] },
    { id: "metadata", label: "Metadata Extraction", detail: "Raster properties and safe previews", matchTrace: ["metadata", "metadata_extraction"] },
    { id: "modality", label: "Modality Validation", detail: "Task-aware modality constraints", matchTrace: ["modality_validation", "input_validator"] },
    { id: "compatibility", label: "Pair Compatibility", detail: "Dimensions, CRS, transform, overlap", matchTrace: ["pair_validation", "pair_compatibility"] },
    { id: "alignment", label: "Alignment Policy", detail: "Returns alignment_required when necessary", matchTrace: ["pair_validation", "pair_compatibility"] },
  ]},
  { id: "controller", label: "Agent Controller", description: "Deterministic, input-aware orchestration", icon: BrainCircuit, nodes: [
    { id: "normalizer", label: "Query Normalizer", detail: "Bounded query preparation", matchTrace: ["query_normalization"] },
    { id: "classifier", label: "Task Classifier", detail: "Supported intent taxonomy", matchTrace: ["query_routing", "question_classification"] },
    { id: "router", label: "Input-Aware Router", detail: "Task + mode + modality decision", matchTrace: ["query_routing"] },
    { id: "registry", label: "Tool Registry Selection", detail: "Declared compatible specialist", matchTrace: ["specialist_selection"] },
    { id: "parameters", label: "Permitted Parameters", detail: "Safe bounded execution controls", matchTrace: ["specialist_selection"] },
  ]},
  { id: "specialists", label: "Specialist Layer", description: "Only the selected implementation executes", icon: ScanSearch, nodes: [
    { id: "captioner", label: "RS Captioner", detail: "RSICD-adapted optical BLIP", matchTools: ["rs_captioner"] },
    { id: "vqa", label: "Controlled VQA", detail: "Deterministic optical evidence", matchTools: ["rs_vqa"] },
    { id: "grounder", label: "Text-Guided Grounder", detail: "Grounding DINO boxes", matchTools: ["rs_grounder"] },
    { id: "ttp-change", label: "Alternate Learned Detector", detail: "Optional remote learned change mask", matchTools: ["ttp_change_detector"], matchTrace: ["ttp_inference"] },
    { id: "changerex-change", label: "ChangerEx Change Detector", detail: "Local learned change mask · default", matchTools: ["changerex_change_detector"], matchTrace: ["changerex_inference"] },
    { id: "change", label: "Deterministic Change Analyzer", detail: "Independent difference, mask, and fallback", matchTools: ["bitemporal_change_analyzer", "deterministic_change_analyzer"], matchTrace: ["deterministic_analysis"] },
    { id: "cross", label: "Optical–SAR Analyzer", detail: "Deterministic evidence fusion", matchTools: ["cross_modal_optical_sar_analyzer"] },
    { id: "pix", label: "Pix2Pix Reconstruction", detail: "Independent reconstruction endpoint", matchTools: ["pix2pix_reconstruction"] },
    { id: "sar", label: "SARFusionFormer", detail: "Independent VV/VH endpoint", matchTools: ["sarfusionformer_analysis"] },
    { id: "report", label: "Report Generator", detail: "Backend-authoritative artifacts", matchTools: ["report_generator"], report: true },
  ]},
  { id: "evidence", label: "Evidence Fusion", description: "Observable products, not hidden reasoning", icon: Layers3, nodes: [
    { id: "captions", label: "Captions", detail: "Model-generated optical description", matchTasks: ["captioning"] },
    { id: "answers", label: "Controlled Answers", detail: "Computed evidence and templates", matchTasks: ["vqa", "change_vqa", "change_description", "cross_modal_analysis"] },
    { id: "boxes", label: "Bounding Boxes", detail: "Model-produced detections", matchTasks: ["grounding"] },
    { id: "masks", label: "Change Masks", detail: "Deterministic binary products", matchTrace: ["thresholding", "morphology"] },
    { id: "agreement", label: "Agreement Maps", detail: "Optical–SAR support maps", matchTasks: ["cross_modal_analysis"], evidence: true },
    { id: "regions", label: "Regions", detail: "Connected evidence components", matchTrace: ["connected_components", "region_extraction"] },
    { id: "statistics", label: "Statistics", detail: "Measured task-specific values", evidence: true },
  ]},
  { id: "governance", label: "Governance Layer", description: "Scientific disclosure and traceability", icon: LockKeyhole, nodes: [
    { id: "confidence", label: "Confidence Disclosure", detail: "No invented calibrated score", confidence: true },
    { id: "limitations", label: "Warnings & Limitations", detail: "Explicit scientific scope", warnings: true },
    { id: "provenance", label: "Model Provenance", detail: "Checkpoint and adaptation status", matchTools: ["rs_captioner", "rs_grounder"] },
    { id: "trace", label: "Execution Trace", detail: "Ordered tools, status, and timing", traceRecord: true },
    { id: "compliance", label: "Compliance Mapping", detail: "Mandatory SIH readiness matrix" },
  ]},
  { id: "output", label: "Output Layer", description: "Interactive response and temporary mission artifacts", icon: FileOutput, nodes: [
    { id: "interactive", label: "Interactive Result", detail: "Typed response and evidence", anyExecution: true },
    { id: "pdf", label: "PDF", detail: "Paginated mission report", report: true },
    { id: "json", label: "JSON", detail: "Versioned typed document", report: true },
    { id: "csv", label: "CSV", detail: "Statistics and regions", report: true },
    { id: "zip", label: "ZIP", detail: "Complete evidence package", report: true },
  ]},
];

export function AgenticArchitectureVisualizer({ analytics, tools, health, compliance, refreshing }: { analytics: AnalyticsResponse; tools: ToolDefinition[]; health: AgentHealth; compliance: ComplianceResponse; refreshing: boolean }) {
  const execution = analytics.recent_executions[0];
  const reducedMotion = useReducedMotion();
  return <div className="space-y-8"><section aria-labelledby="system-architecture-title"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Upload → evidence-backed output</p><h2 id="system-architecture-title" className="mt-2 text-2xl font-semibold">System Architecture</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-500">Grouped layers keep the control flow readable. Highlighting is derived only from the newest completed backend execution.</p></div><div className="flex flex-wrap gap-2 text-[11px]"><Legend state="completed"/><Legend state="failed"/><Legend state="skipped"/><Legend state="neutral"/></div></div>
      <p className="sr-only" aria-live="polite">{activePathSummary(execution)}</p>
      {!execution && <div className="panel mt-6 border-dashed p-6 text-center"><GitBranch className="mx-auto text-zinc-500"/><p className="mt-3 text-sm text-zinc-400">Run a workflow in SatQuery Assistant to visualize its execution path.</p><Link href="/assistant" className="mt-4 inline-flex rounded-xl bg-sky-300 px-4 py-2 text-sm font-medium text-zinc-950">Open Assistant</Link></div>}
      {execution && <div className="panel mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 p-4 text-xs"><span className="inline-flex items-center gap-2 font-medium text-zinc-200"><CircleDot size={14} className="text-sky-300"/>Latest real path</span><span className="text-zinc-500">{execution.task.replaceAll("_", " ")}</span><span className="text-zinc-500">{execution.duration_ms} ms</span><span className="text-zinc-500">{execution.cache_status.replaceAll("_", " ")}</span><span className="text-zinc-500">{execution.output_count} evidence products</span><span className="ml-auto text-zinc-600">{refreshing ? "Refreshing…" : `Completed ${new Date(execution.completed_at).toLocaleString()}`}</span></div>}
      <div className="mt-5 space-y-3" aria-label="SatQuery AI agentic system architecture diagram">{layers.map((layer, layerIndex) => <div key={layer.id}><LayerView layer={layer} execution={execution} executionKey={execution?.request_id} reducedMotion={Boolean(reducedMotion)}/>{layerIndex < layers.length - 1 && <div className="flex h-9 items-center justify-center" aria-hidden="true"><ArrowDown size={17} className="text-sky-400/45"/></div>}</div>)}</div>
    </section>

    <section className="panel p-5 sm:p-6"><ExecutionArea execution={execution}/></section>
    <section className="panel p-5 sm:p-6"><SpecialistInspector tools={tools} analytics={analytics} health={health}/></section>
    <TrustPanel execution={execution} tools={tools} compliance={compliance}/>
  </div>;
}

function LayerView({ layer, execution, executionKey, reducedMotion }: { layer: Layer; execution?: AnalyticsExecution; executionKey?: string; reducedMotion: boolean }) {
  const Icon = layer.icon;
  return <section className="panel overflow-hidden"><header className="flex flex-col gap-3 border-b border-white/[.08] p-4 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-sky-300/[.07] text-sky-300"><Icon size={17}/></span><div><h3 className="text-sm font-semibold">{layer.label}</h3><p className="mt-0.5 text-xs text-zinc-500">{layer.description}</p></div></div><span className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">{layer.id}</span></header><div className="grid gap-2 p-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5">{layer.nodes.map((node, index) => { const state = architectureNodeState(node, execution); return <motion.article key={`${executionKey ?? "empty"}-${node.id}`} initial={reducedMotion || state === "neutral" ? false : { opacity: .35, scale: .98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .28, delay: reducedMotion ? 0 : Math.min(index * .035, .18) }} title={statusDescription(state)} className={cn("architecture-node rounded-xl border p-3", stateClass[state])}><div className="flex items-start justify-between gap-2"><p className="text-xs font-semibold text-zinc-200">{node.label}</p><StateIcon state={state}/></div><p className="mt-1.5 text-[11px] leading-4 text-zinc-500">{node.detail}</p><span className="mt-2 block text-[9px] uppercase tracking-wide text-zinc-600">{statusDescription(state)}</span></motion.article>; })}</div></section>;
}

function ExecutionArea({ execution }: { execution?: AnalyticsExecution }) { return execution ? <ExecutionInspector execution={execution}/> : <div className="py-8 text-center"><Binary className="mx-auto text-zinc-600"/><h2 className="mt-3 text-xl font-semibold">Live Workflow</h2><p className="mt-2 text-sm text-zinc-500">No execution trace is available in this backend process.</p><Link href="/assistant" className="mt-4 inline-flex rounded-xl border border-sky-300/25 px-4 py-2 text-sm text-sky-200">Open Assistant</Link></div>; }

function TrustPanel({ execution, tools, compliance }: { execution?: AnalyticsExecution; tools: ToolDefinition[]; compliance: ComplianceResponse }) {
  const selected = execution ? tools.filter(tool => execution.selected_tools.some(id => id === tool.id || ["deterministic_change_analysis", "deterministic_change_analyzer"].includes(id) && tool.id === "bitemporal_change_analyzer")) : [];
  return <section className="panel overflow-hidden" aria-labelledby="audit-title"><header className="border-b border-white/[.08] p-5 sm:p-6"><div className="flex items-center gap-2"><ShieldCheck size={19} className="text-sky-300"/><h2 id="audit-title" className="text-2xl font-semibold">Why This Result Is Auditable</h2></div><p className="mt-2 text-sm text-zinc-500">{compliance.mandatory_satisfied}/{compliance.mandatory_total} mandatory requirements are declared available by the backend compliance matrix.</p></header><div className="grid gap-6 p-5 sm:p-6 xl:grid-cols-2"><div><h3 className="text-sm font-semibold">Latest selection and evidence</h3>{execution ? <dl className="mt-4 space-y-3 text-sm"><TrustFact name="Selected implementation" value={selected.map(item => item.display_name).join(" · ") || execution.selected_tools.join(" · ")}/><TrustFact name="Routing record" value={execution.selection_reason ?? "This direct specialist endpoint did not publish a query-routing stage."}/><TrustFact name="Remote-sensing adaptation" value={selected.map(item => `${item.display_name}: ${item.remote_sensing_adapted ? "yes" : "no"}`).join(" · ") || "Not declared for this execution"}/><TrustFact name="Evidence produced" value={`${execution.output_count} observable product(s)`}/><TrustFact name="Confidence interpretation" value={execution.confidence_reason ?? "No confidence rationale applies to this deterministic output."}/><TrustFact name="Report traceability" value={execution.report_generated ? "A report was generated for this request." : "No report has been generated for this request."}/></dl> : <p className="mt-3 text-sm text-zinc-500">Run an Assistant workflow to populate selection-specific audit facts.</p>}</div><div><h3 className="text-sm font-semibold">Scientific safeguards</h3><ul className="mt-4 space-y-3 text-sm leading-6 text-zinc-400"><li>• Generic hosted VLMs are not used for mandatory analysis.</li><li>• Model scores are not automatically treated as calibrated probabilities.</li><li>• Incompatible image pairs are not silently aligned.</li><li>• Heuristic masks are labelled as heuristic.</li><li>• Model-generated detections are not called ground truth.</li></ul></div></div></section>;
}

const stateClass: Record<PathState, string> = { active: "architecture-node-active border-sky-300/50 bg-sky-300/[.09]", completed: "architecture-node-completed border-emerald-300/25 bg-emerald-300/[.055]", failed: "border-rose-300/35 bg-rose-300/[.065]", skipped: "border-white/[.06] bg-white/[.015] opacity-60", neutral: "border-white/[.07] bg-white/[.018]" };
function StateIcon({ state }: { state: PathState }) { if (state === "completed") return <CheckCircle2 size={13} className="text-emerald-300" aria-label="Completed"/>; if (state === "failed") return <XCircle size={13} className="text-rose-300" aria-label="Failed"/>; if (state === "skipped") return <MinusCircle size={13} className="text-zinc-500" aria-label="Skipped"/>; if (state === "active") return <CircleDot size={13} className="text-sky-300" aria-label="Active"/>; return <span className="h-2 w-2 rounded-full bg-zinc-700" aria-label="Not involved"/>; }
function Legend({ state }: { state: PathState }) { return <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[.07] px-2.5 py-1 text-zinc-500"><StateIcon state={state}/>{statusDescription(state).replace("Involved and ", "")}</span>; }
function TrustFact({ name, value }: { name: string; value: string }) { return <div><dt className="text-xs text-zinc-500">{name}</dt><dd className="mt-1 leading-6 text-zinc-300">{value}</dd></div>; }

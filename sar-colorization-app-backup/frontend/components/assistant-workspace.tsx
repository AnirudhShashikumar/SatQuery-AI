"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { AlertTriangle, ArrowLeftRight, ArrowUp, Check, CheckCircle2, Clock3, FileQuestion, ImagePlus, Images, LoaderCircle, MessageCircle, Paperclip, Plus, RadioTower, Route, ScanSearch, Sparkles, XCircle } from "lucide-react";
import { AssistantUploadCard } from "@/components/assistant-upload-card";
import { AssistantResultExperience } from "@/components/assistant-result-experience";
import { agentPreviewUrl, getDemoManifest, getHealth, inspectAgentImage, loadDemoFile, runAgentImageQuery } from "@/services/api";
import type { AgentResponse, ChangeAnalysisResponse, DemoManifest, DemoWorkflow, ImageInspectionResponse, ImageModality, InputMode, Modality, PairCompatibility } from "@/types/agent";
import { cn } from "@/lib/utils";
import { swapTemporalDates } from "@/lib/temporal-inputs";
import { readRecentConversations, rememberConversation, type RecentConversation } from "@/lib/recent-conversations";

const modes: Array<{ id: InputMode; label: string; description: string }> = [
  { id: "single", label: "Single Image", description: "Optical, multispectral, or SAR" },
  { id: "cross_modal", label: "Optical + SAR Pair", description: "Validate complementary observations" },
  { id: "bi_temporal", label: "Bi-Temporal Analysis", description: "Compute visual and structural change products" },
];

const singleExamples = [
  "Highlight the water body.",
  "Locate the buildings.",
  "Show the roads.",
  "Mark the aircraft.",
  "Find the stadium.",
  "Highlight vegetation.",
  "What is the dominant land-cover type?",
  "Is a water body visible?",
  "Are buildings present?",
  "Is vegetation visible?",
  "Does this scene contain agricultural land?",
  "Is this mainly urban or rural?",
  "Describe the land cover and major objects visible in this image.",
];
const sarPreviewExamples = [
  "Highlight probable water regions.",
  "Describe the major radar-backscatter patterns.",
  "Show low-backscatter candidate regions.",
  "Assess whether this image is suitable for SAR analysis.",
];
const scientificSarExamples = [
  "Highlight probable water regions.",
  "Compare VV and VH responses.",
  "Estimate broad built-up regions.",
  "Generate an optical-like visualization.",
  "Describe the dominant scattering patterns.",
];
const temporalExamples = [
  "What changed between these dates?",
  "How much of the image changed?",
  "Where did the largest change occur?",
];
const crossExamples = [
  "Where do both modalities agree?",
  "What percentage appears water-like?",
  "Are structurally complex regions present?",
];

const allModalities: Array<{ value: Modality; label: string }> = [
  { value: "optical", label: "Optical" },
  { value: "multispectral", label: "Multispectral" },
  { value: "sar", label: "SAR" },
  { value: "unknown", label: "Unknown" },
];

const detailedModalities: Array<{ value: ImageModality; label: string }> = [
  { value: "auto", label: "Auto Detect" },
  { value: "optical_rgb", label: "Optical RGB" },
  { value: "optical_grayscale", label: "Optical Grayscale" },
  { value: "panchromatic", label: "Panchromatic" },
  { value: "sar_preview", label: "SAR Preview" },
  { value: "sar_vv", label: "SAR VV" },
  { value: "sar_vh", label: "SAR VH" },
  { value: "sar_vv_vh", label: "SAR VV/VH" },
  { value: "multispectral", label: "Multispectral" },
  { value: "unknown", label: "Other / Unknown" },
];

function coarseFromDetailed(value: ImageModality, fallback: Modality): Modality {
  if (["sar_preview", "sar_vv", "sar_vh", "sar_vv_vh"].includes(value)) return "sar";
  if (value === "multispectral") return "multispectral";
  if (["optical_rgb", "optical_grayscale", "panchromatic"].includes(value)) return "optical";
  if (value === "unknown") return "unknown";
  return fallback;
}

function label(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function fact(value: boolean | null) {
  return value === null ? "Unavailable" : value ? "Yes" : "No";
}

const executionCopy: Record<InputMode, string[]> = {
  single: ["Validating observation", "Detecting modality", "Selecting a specialist", "Reading scene evidence", "Checking confidence", "Composing answer"],
  cross_modal: ["Validating observation pair", "Detecting modalities", "Selecting specialists", "Reconciling optical and SAR evidence", "Checking agreement", "Composing answer"],
  bi_temporal: ["Validating temporal pair", "Confirming alignment", "Selecting change specialists", "Computing change evidence", "Checking supporting signals", "Composing answer"],
};

function activeSpecialists(mode: InputMode, query: string) {
  if (mode === "cross_modal") return ["SatQuery Vision Encoder", "Cross-modal Analyzer"];
  if (mode === "bi_temporal") return ["TTP Change Detector", "Deterministic Change Analyzer"];
  if (/\b(highlight|locate|show|mark|find)\b/i.test(query)) return ["SatQuery Vision Encoder", "Grounding DINO"];
  if (/\?|\b(what|where|is|are|does|how)\b/i.test(query)) return ["SatQuery Vision Encoder", "Remote Sensing VQA"];
  return ["SatQuery Vision Encoder", "Remote Sensing Captioner"];
}

function AssistantExecutionProgress({ mode, query, stage, onCancel }: { mode: InputMode; query: string; stage: number; onCancel: () => void }) {
  const steps = executionCopy[mode];
  const specialists = activeSpecialists(mode, query);
  return <section className="assistant-execution" aria-live="polite" aria-label="Live analysis progress">
    <header><div><span className="assistant-thinking-dot"/><p className="eyebrow">Live execution</p><h3>SatQuery is thinking</h3></div><button type="button" onClick={onCancel}>Cancel request</button></header>
    <div className="assistant-execution-grid">
      <ol className="assistant-stage-list">{steps.map((step, index) => <li key={step} className={cn(index < stage && "is-complete", index === stage && "is-current")}><span>{index < stage ? <Check size={13}/> : index + 1}</span><p>{step}</p>{index === stage && <i aria-hidden="true"/>}</li>)}</ol>
      <div className="assistant-specialists"><p>Active specialists</p>{specialists.map((specialist, index) => <div key={specialist} className={cn(stage >= index + 2 && "is-active")}><span/><strong>{specialist}</strong><small>{stage >= index + 2 ? "Active" : "Queued"}</small></div>)}</div>
    </div>
  </section>;
}

function CompatibilityCard({ compatibility }: { compatibility: PairCompatibility }) {
  const content = {
    exact: { title: "Exact Alignment", text: "Pixel grids and geospatial metadata match.", tone: "emerald" },
    geospatial_overlap: { title: "Geospatial Overlap", text: "Images overlap geographically but require alignment before joint analysis.", tone: "amber" },
    visual_only: { title: "Visual Comparison Only", text: "Georeferencing is unavailable. The pair cannot be treated as co-registered.", tone: "amber" },
    incompatible: { title: "Incompatible", text: "The pair cannot be used for this workflow.", tone: "rose" },
  }[compatibility.alignment_level];
  return <section className="panel mt-7 overflow-hidden">
    <header className="flex flex-wrap items-start justify-between gap-3 border-b border-white/[.08] p-5">
      <div><p className="eyebrow">Pair compatibility</p><h2 className="mt-2 text-xl font-semibold">{content.title}</h2><p className="mt-2 text-sm leading-6 text-zinc-400">{content.text}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", content.tone === "emerald" ? "border-emerald-300/25 bg-emerald-400/10 text-emerald-300" : content.tone === "rose" ? "border-rose-300/25 bg-rose-400/10 text-rose-200" : "border-amber-300/25 bg-amber-400/10 text-amber-100")}>{compatibility.compatible ? "Compatible" : "Incompatible"}</span>
    </header>
    <div className="grid gap-4 p-5 sm:grid-cols-2 xl:grid-cols-3">
      <CompatibilityFact name="Same dimensions" value={fact(compatibility.same_dimensions)}/>
      <CompatibilityFact name="Same CRS" value={fact(compatibility.same_crs)}/>
      <CompatibilityFact name="Same transform" value={fact(compatibility.same_transform)}/>
      <CompatibilityFact name="Bounds overlap" value={fact(compatibility.bounds_overlap)}/>
      <CompatibilityFact name="Overlap ratio" value={compatibility.overlap_ratio === null ? "Unavailable" : `${(compatibility.overlap_ratio * 100).toFixed(1)}%`}/>
      <CompatibilityFact name="Resampling required" value={fact(compatibility.resampling_required)}/>
    </div>
    {(compatibility.warnings.length > 0 || compatibility.errors.length > 0) && <div className="border-t border-white/[.08] p-5 text-sm leading-6"><ul className="space-y-2">{compatibility.errors.map(error => <li key={error} className="flex gap-2 text-rose-200"><XCircle size={15} className="mt-1 shrink-0"/>{error}</li>)}{compatibility.warnings.map(warning => <li key={warning} className="flex gap-2 text-amber-100"><AlertTriangle size={15} className="mt-1 shrink-0"/>{warning}</li>)}</ul></div>}
  </section>;
}

function CompatibilityFact({ name, value }: { name: string; value: string }) {
  return <div className="glass rounded-xl p-3"><p className="text-xs text-zinc-500">{name}</p><p className="mt-1 break-words text-sm font-medium text-zinc-200">{value}</p></div>;
}

function ResultPanel({ result }: { result: AgentResponse }) {
  const routedOnly = result.status === "not_implemented";
  const metadata = result.primary_image_metadata;
  const successful = result.status === "success" || result.status === "partial";
  return <section className="panel mt-7 overflow-hidden" aria-live="polite">
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[.08] p-5 sm:p-6">
      <div><p className="eyebrow">SatQuery specialist result</p><h2 className="mt-2 text-xl font-semibold">{result.answer ? "Scene Description" : label(result.task)}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">{result.execution.selection_reason}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", result.status === "failed" ? "border-rose-300/25 bg-rose-400/10 text-rose-200" : successful ? "border-emerald-300/25 bg-emerald-400/10 text-emerald-300" : "border-amber-300/25 bg-amber-400/10 text-amber-100")}>{label(result.status)}</span>
    </header>
    <div className="grid gap-6 p-5 sm:p-6 xl:grid-cols-[1fr_.9fr]">
      <div className="space-y-5">
        {result.answer && <div className="rounded-2xl border border-sky-300/20 bg-sky-300/[.06] p-5"><div className="flex flex-wrap items-center justify-between gap-3"><h3 className="text-sm font-semibold text-sky-100">Model observation</h3>{result.caption_details && <span className="rounded-full border border-sky-300/20 px-2.5 py-1 text-xs text-sky-200">{label(result.caption_details.modality)}</span>}</div><p className="mt-3 text-lg leading-8 text-zinc-100">{result.answer}</p></div>}
        {routedOnly && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100"><strong className="font-semibold">Workflow routed successfully.</strong> The selected specialist does not support this modality or workflow yet.</div>}
        {result.status === "failed" && <div className="rounded-xl border border-rose-300/20 bg-rose-300/[.06] p-4 text-sm text-rose-200">Validation or the selected local specialist failed safely. Review the warnings and execution trace below.</div>}
        {result.model && result.caption_details && <div><h3 className="text-sm font-semibold">Model provenance</h3><div className="mt-3 grid gap-3 sm:grid-cols-2"><CompatibilityFact name="Specialist" value={result.model.tool_id}/><CompatibilityFact name="Checkpoint" value={result.model.checkpoint}/><CompatibilityFact name="Base architecture" value={result.model.base_architecture}/><CompatibilityFact name="Adaptation dataset" value={result.model.adaptation_dataset}/><CompatibilityFact name="Remote-sensing adapted" value={result.model.remote_sensing_adapted ? "Yes" : "No"}/><CompatibilityFact name="License" value={result.model.license ?? "Unavailable"}/><CompatibilityFact name="Device" value={result.caption_details.device}/><CompatibilityFact name="Caption runtime" value={`${result.caption_details.runtime_ms} ms`}/><CompatibilityFact name="Model lifecycle" value={result.caption_details.model_reused ? "Reused loaded model" : `Loaded in ${result.caption_details.model_load_ms} ms`}/></div></div>}
        {metadata && result.caption_details && <div><h3 className="text-sm font-semibold">Input facts</h3><div className="mt-3 grid gap-3 sm:grid-cols-2"><CompatibilityFact name="Modality" value={label(result.caption_details.modality)}/><CompatibilityFact name="Dimensions" value={`${metadata.width} × ${metadata.height}`}/><CompatibilityFact name="Bands used" value={result.caption_details.bands_used.join(", ")}/><CompatibilityFact name="Georeferenced" value={metadata.is_georeferenced ? "Yes" : "No"}/><CompatibilityFact name="Representation" value={result.caption_details.image_representation}/></div></div>}
        <div><h3 className="text-sm font-semibold">Selected tools</h3><div className="mt-3 flex flex-wrap gap-2">{result.execution.selected_tools.map(tool => <span key={tool} className="rounded-lg border border-sky-300/15 bg-sky-300/[.05] px-2.5 py-1.5 font-mono text-xs text-sky-200">{tool}</span>)}</div></div>
        <div><h3 className="text-sm font-semibold">Confidence</h3><p className="mt-2 text-sm text-zinc-400"><span className="font-medium text-zinc-200">{label(result.confidence.level)}{result.confidence.score !== null ? ` · ${(result.confidence.score * 100).toFixed(1)}%` : " · No calibrated score"}:</span> {result.confidence.reason}</p></div>
        {result.caption_details && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-4"><h3 className="flex items-center gap-2 text-sm font-semibold text-amber-100"><AlertTriangle size={15}/>Limitations</h3><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.caption_details.limitations.map(item => <li key={item}>• {item}</li>)}</ul></div>}
        {result.warnings.length > 0 && <div><h3 className="flex items-center gap-2 text-sm font-semibold"><AlertTriangle size={15} className="text-amber-200"/>Warnings</h3><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.warnings.map(warning => <li key={warning}>• {warning}</li>)}</ul></div>}
      </div>
      <div>
        <div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold"><Route size={16} className="text-sky-200"/>Execution trace</h3><span className="flex items-center gap-1.5 text-xs text-zinc-500"><Clock3 size={13}/>{result.execution.duration_ms} ms</span></div>
        <ol className="mt-4 space-y-3">{result.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-7 w-7 place-items-center rounded-full", step.status === "success" ? "bg-emerald-400/10 text-emerald-300" : step.status === "failed" ? "bg-rose-400/10 text-rose-300" : "bg-amber-400/10 text-amber-200")}>{step.status === "success" ? <CheckCircle2 size={15}/> : <span className="text-xs font-semibold">{index + 1}</span>}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{label(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{step.duration_ms} ms</span></li>)}</ol>
        <p className="mt-4 break-all text-[11px] text-zinc-600">Request ID: {result.request_id}</p>
      </div>
    </div>
  </section>;
}

export function GroundingResultPanel({ response }: { response: AgentResponse }) {
  const result = response.grounding_result!;
  const acceptedDetections = result.detections ?? result.accepted_detections ?? [];
  const hasAcceptedDetections = acceptedDetections.length > 0;
  const rejectedCandidates = result.rejected_candidates ?? [];
  const highestRejected = rejectedCandidates.reduce<(typeof rejectedCandidates)[number] | null>((highest, candidate) => highest === null || (candidate.score ?? -1) > (highest.score ?? -1) ? candidate : highest, null);
  const rejectionReasons = Array.from(new Set(rejectedCandidates.flatMap(candidate => candidate.rejection_reasons)));
  const [showAnnotations, setShowAnnotations] = useState(hasAcceptedDetections);
  const originalPreview = response.primary_image_metadata?.preview_url ?? null;
  const visiblePreview = hasAcceptedDetections && showAnnotations ? result.annotated_preview_url : originalPreview;
  return <section className="panel mt-7 overflow-hidden" aria-live="polite">
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[.08] p-5 sm:p-6">
      <div><p className="eyebrow">Grounding DINO result</p><h2 className="mt-2 text-xl font-semibold">{hasAcceptedDetections ? "Text-Guided Grounding" : "No reliable localization found"}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">{hasAcceptedDetections ? <>Candidate regions for “{result.target_phrase}”. {response.answer}</> : "Grounding DINO found a possible scene-level match, but it did not meet the operational reliability gates for precise localization."}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", hasAcceptedDetections ? "border-sky-300/25 bg-sky-300/[.08] text-sky-200" : "border-amber-300/25 bg-amber-300/[.08] text-amber-100")}>{hasAcceptedDetections ? `${acceptedDetections.length} detection${acceptedDetections.length === 1 ? "" : "s"}` : `0 accepted · ${result.rejected_candidate_count ?? rejectedCandidates.length} rejected`}</span>
    </header>
    <div className="space-y-6 p-5 sm:p-6">
      {!hasAcceptedDetections && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-4"><p className="text-sm font-semibold text-amber-100">No reliable localized region</p><p className="mt-2 text-sm leading-6 text-zinc-400">{result.empty_result_explanation ?? response.answer}</p></div>}
      <div className="grid gap-6 xl:grid-cols-[1.15fr_.85fr]">
        <div>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3"><h3 className="text-sm font-semibold">{hasAcceptedDetections ? "Grounding preview" : "Original source preview"}</h3>{hasAcceptedDetections && <button type="button" onClick={() => setShowAnnotations(value => !value)} disabled={!result.annotated_preview_url || !originalPreview} className="rounded-lg border border-white/[.10] px-3 py-1.5 text-xs text-zinc-300 hover:bg-white/[.05] disabled:opacity-40">{showAnnotations ? "Hide labels and scores" : "Show labels and scores"}</button>}</div>
          {visiblePreview ? <img src={agentPreviewUrl(visiblePreview)} alt={hasAcceptedDetections && showAnnotations ? `Grounding DINO boxes for ${result.target_phrase}` : "Uploaded source preview without rejected annotations"} className="max-h-[620px] w-full rounded-xl border border-white/[.08] bg-black/20 object-contain"/> : <div className="grid min-h-64 place-items-center rounded-xl border border-dashed border-white/[.12] text-sm text-zinc-500">Source preview unavailable. No replacement region was fabricated.</div>}
          <p className="mt-3 text-xs leading-5 text-amber-100">{hasAcceptedDetections ? "Model-produced Grounding DINO boxes and alignment scores. They are not ground truth or calibrated scientific probabilities." : "Rejected candidates are retained for audit but are not drawn as accepted visual evidence."}</p>
        </div>
        <div className="space-y-5">
          <div><h3 className="text-sm font-semibold">Model and runtime</h3><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2"><CompatibilityFact name="Checkpoint" value={result.model.checkpoint}/><CompatibilityFact name="Architecture" value={result.model.base_architecture}/><CompatibilityFact name="Device" value={result.device}/><CompatibilityFact name="Runtime" value={result.runtime_ms + " ms"}/><CompatibilityFact name="Model lifecycle" value={result.model_reused ? "Reused loaded model" : "Loaded in " + result.model_load_ms + " ms"}/><CompatibilityFact name="Remote-sensing adapted" value="No — zero-shot"/></div></div>
          <div><h3 className="text-sm font-semibold">Input preparation</h3><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2"><CompatibilityFact name="Original query" value={result.original_query}/><CompatibilityFact name="Extracted target" value={result.target_phrase}/><CompatibilityFact name="Modality" value={label(result.input.modality)}/><CompatibilityFact name="Source size" value={result.input.original_width + " × " + result.input.original_height}/><CompatibilityFact name="Model input" value={result.input.model_input_width + " × " + result.input.model_input_height}/><CompatibilityFact name="Representation" value={result.input.representation}/><CompatibilityFact name="Bands" value={result.input.bands_used.join(", ")}/><CompatibilityFact name="Masks" value="Not connected"/></div></div>
          {hasAcceptedDetections ? <div className="rounded-xl border border-white/[.08] p-4"><h3 className="text-sm font-semibold">Confidence</h3><p className="mt-2 text-sm leading-6 text-zinc-400"><span className="font-medium text-zinc-200">{label(result.confidence.level)}{result.confidence.score !== null ? ` · ${result.confidence.score.toFixed(3)}` : " · no score"}.</span> {result.confidence.reason}</p></div> : <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-4"><h3 className="text-sm font-semibold text-amber-100">Operational gate outcome</h3><dl className="mt-3 space-y-2 text-sm"><div><dt className="text-xs text-zinc-500">Requested target</dt><dd className="text-zinc-200">{result.target_phrase}</dd></div><div><dt className="text-xs text-zinc-500">Rejected candidates</dt><dd className="text-zinc-200">{result.rejected_candidate_count ?? rejectedCandidates.length}</dd></div><div><dt className="text-xs text-zinc-500">Highest rejected alignment score</dt><dd className="font-mono text-zinc-200">{highestRejected?.score?.toFixed(4) ?? "Unavailable"}</dd></div><div><dt className="text-xs text-zinc-500">Rejected box area ratio</dt><dd className="font-mono text-zinc-200">{highestRejected?.box_area_ratio === null || highestRejected?.box_area_ratio === undefined ? "Unavailable" : `${(highestRejected.box_area_ratio * 100).toFixed(2)}%`}</dd></div></dl><p className="mt-3 text-xs font-medium text-zinc-300">Rejection reasons</p><ul className="mt-1 space-y-1 text-xs text-zinc-400">{rejectionReasons.length ? rejectionReasons.map(reason => <li key={reason}>• {label(reason)}</li>) : <li>• No candidate exceeded the processor threshold.</li>}</ul><p className="mt-3 text-xs leading-5 text-zinc-400"><strong className="text-zinc-300">Zero-shot limitation:</strong> the checkpoint is not remote-sensing adapted and a scene-level match is not verified localization.</p><p className="mt-2 text-xs leading-5 text-amber-100">{result.operational_threshold_disclaimer}</p></div>}
        </div>
      </div>
      <div><h3 className="text-sm font-semibold">Accepted detections</h3>{acceptedDetections.length ? <div className="mt-3 overflow-x-auto rounded-xl border border-white/[.08]"><table className="w-full min-w-[760px] text-left text-xs"><thead className="bg-white/[.04] text-zinc-400"><tr><th className="px-4 py-3">Label</th><th className="px-4 py-3">Alignment score</th><th className="px-4 py-3">Pixel box [x₁, y₁, x₂, y₂]</th><th className="px-4 py-3">World box / CRS</th><th className="px-4 py-3">Mask</th></tr></thead><tbody>{acceptedDetections.map((detection, index) => <tr key={`${detection.label}-${index}`} className="border-t border-white/[.06] text-zinc-300"><td className="px-4 py-3">{detection.label}</td><td className="px-4 py-3 font-mono">{detection.score.toFixed(4)}</td><td className="px-4 py-3 font-mono">[{detection.bbox_pixels.join(", ")}]</td><td className="px-4 py-3 font-mono">{detection.bbox_world ? `[${detection.bbox_world.map(value => Number(value).toFixed(3)).join(", ")}] · ${detection.crs}` : "Unavailable (not georeferenced)"}</td><td className="px-4 py-3">Not connected</td></tr>)}</tbody></table></div> : <div className="mt-3 rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-4 text-sm text-amber-100">Zero accepted detections. Rejected candidates remain in audit metadata and were not drawn.</div>}</div>
      <div className="grid gap-5 xl:grid-cols-2"><div className="rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-4"><h3 className="flex items-center gap-2 text-sm font-semibold text-amber-100"><AlertTriangle size={15}/>Limitations</h3><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{(result.limitations ?? []).map(item => <li key={item}>• {item}</li>)}</ul></div><div><div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold"><Route size={16} className="text-sky-200"/>Execution trace</h3><span className="text-xs text-zinc-500">{response.execution.duration_ms} ms</span></div><ol className="mt-4 space-y-3">{response.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-7 w-7 place-items-center rounded-full", step.status === "success" ? "bg-emerald-400/10 text-emerald-300" : step.status === "failed" ? "bg-rose-400/10 text-rose-300" : "bg-amber-400/10 text-amber-200")}>{step.status === "success" ? <CheckCircle2 size={15}/> : index + 1}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{label(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{step.duration_ms} ms</span></li>)}</ol></div></div>
    </div>
  </section>;
}

function ChangeResultPanel({ result }: { result: ChangeAnalysisResponse }) {
  const hybrid = result.change_engine?.mode === "hybrid" || result.change_engine?.mode === "ttp";
  const previewItems = hybrid ? [
    ["Before", result.previews.before], ["After", result.previews.after], ["Difference", result.previews.difference],
    ["TTP Mask", result.previews.ttp_mask], ["Deterministic Mask", result.previews.deterministic_mask],
    ["Agreement", result.previews.agreement], ["Disagreement", result.previews.disagreement], ["TTP Overlay", result.previews.ttp_overlay],
  ] as const : [
    ["Before", result.previews.before],
    ["After", result.previews.after],
    ["Difference", result.previews.difference],
    ["Binary mask", result.previews.mask],
    ["Overlay", result.previews.overlay],
  ] as const;
  const statistics = result.statistics;
  return <section className="panel mt-7 overflow-hidden" aria-live="polite">
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[.08] p-5 sm:p-6">
      <div><p className="eyebrow">{hybrid ? "Hybrid Change Analysis" : result.change_engine?.fallback_used ? "Deterministic fallback" : "Deterministic change products"}</p><h2 className="mt-2 text-xl font-semibold">Bi-Temporal Analysis</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">{hybrid ? "Primary learned mask: TTP · Supporting evidence: Deterministic analyzer · Automatic fallback: Enabled." : "The deterministic analyzer completed this result. Causes and semantic change types are not inferred."}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", result.status === "success" ? "border-emerald-300/25 bg-emerald-400/10 text-emerald-300" : result.status === "failed" ? "border-rose-300/25 bg-rose-400/10 text-rose-200" : "border-amber-300/25 bg-amber-400/10 text-amber-100")}>{label(result.status)}</span>
    </header>
    <div className="space-y-7 p-5 sm:p-6">
      {result.status === "alignment_required" && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100">The pair needs explicit alignment, reprojection, or resampling. SatQuery did not modify either image and did not compute a difference map.</div>}
      {result.change_engine?.fallback_used && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100">TTP did not complete ({label(result.change_engine.fallback_reason ?? "unavailable")}). The independent deterministic analyzer completed the request.</div>}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {previewItems.map(([name, path]) => path && <figure key={name} className="overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/60"><figcaption className="border-b border-white/[.08] px-4 py-3 text-sm font-medium text-zinc-200">{name}</figcaption><img src={agentPreviewUrl(path)} alt={`${name} change-analysis product`} className="aspect-video w-full object-contain"/></figure>)}
      </div>
      {statistics && <div><h3 className="text-sm font-semibold">Statistics</h3><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-6">{hybrid ? <><CompatibilityFact name="TTP changed area" value={`${statistics.percentage_changed.toFixed(3)}%`}/><CompatibilityFact name="Deterministic changed area" value={result.deterministic_statistics ? `${result.deterministic_statistics.percentage_changed.toFixed(3)}%` : "Unavailable"}/><CompatibilityFact name="Mask IoU" value={result.mask_comparison?.iou?.toFixed(3) ?? "Unavailable"}/><CompatibilityFact name="Agreement" value={result.mask_comparison ? `${result.mask_comparison.agreement_percentage.toFixed(3)}%` : "Unavailable"}/><CompatibilityFact name="Disagreement" value={result.mask_comparison ? `${result.mask_comparison.disagreement_percentage.toFixed(3)}%` : "Unavailable"}/><CompatibilityFact name="TTP runtime" value={result.ttp_result?.runtime_ms == null ? "Unavailable" : `${result.ttp_result.runtime_ms} ms`}/></> : <><CompatibilityFact name="Analysis grid" value={`${statistics.analysis_width} × ${statistics.analysis_height}`}/><CompatibilityFact name="Total pixels" value={statistics.total_pixels.toLocaleString()}/><CompatibilityFact name="Changed pixels" value={statistics.changed_pixels.toLocaleString()}/><CompatibilityFact name="Percentage changed" value={`${statistics.percentage_changed.toFixed(3)}%`}/><CompatibilityFact name="Connected regions" value={statistics.number_of_regions.toLocaleString()}/><CompatibilityFact name="Largest region" value={`${statistics.largest_connected_region.toLocaleString()} px`}/></>}</div><p className="mt-3 text-xs text-zinc-500">{hybrid ? "Mask agreement is evidence consistency, not ground-truth accuracy. TTP output is not ground truth." : `Normalized threshold: ${statistics.normalized_threshold.toFixed(4)}.`}</p></div>}
      {result.warnings.length > 0 && <div><h3 className="flex items-center gap-2 text-sm font-semibold"><AlertTriangle size={15} className="text-amber-200"/>Warnings</h3><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.warnings.map(warning => <li key={warning}>• {warning}</li>)}</ul></div>}
      <div className="grid gap-6 xl:grid-cols-[1fr_.9fr]"><div><h3 className="text-sm font-semibold">Dates and method</h3><div className="mt-3 grid gap-3 sm:grid-cols-2"><CompatibilityFact name="Before" value={result.before_date}/><CompatibilityFact name="After" value={result.after_date}/><CompatibilityFact name="Runtime" value={`${result.runtime_ms} ms`}/><CompatibilityFact name="Registration performed" value="No"/></div></div><div><div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold"><Route size={16} className="text-sky-200"/>Execution trace</h3><span className="flex items-center gap-1.5 text-xs text-zinc-500"><Clock3 size={13}/>{result.execution.duration_ms} ms</span></div><ol className="mt-4 space-y-3">{result.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-7 w-7 place-items-center rounded-full", step.status === "success" ? "bg-emerald-400/10 text-emerald-300" : step.status === "failed" ? "bg-rose-400/10 text-rose-300" : "bg-amber-400/10 text-amber-200")}>{step.status === "success" ? <CheckCircle2 size={15}/> : <span className="text-xs font-semibold">{index + 1}</span>}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{label(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{step.duration_ms} ms</span></li>)}</ol><p className="mt-4 break-all text-[11px] text-zinc-600">Request ID: {result.request_id}</p></div></div>
    </div>
  </section>;
}

function CrossModalResultPanel({ response }: { response: AgentResponse }) {
  const result = response.cross_modal_analysis!;
  const statistics = result.statistics;
  const previews = [
    ["Optical Evidence", result.previews.optical_evidence],
    ["SAR Evidence", result.previews.sar_evidence],
    ["Joint Evidence", result.previews.joint_evidence],
    ["Agreement", result.previews.agreement],
    ["Disagreement", result.previews.disagreement],
    ["Joint Overlay", result.previews.joint_overlay],
  ] as const;
  const metric = (value: number | null) => value === null ? "Unavailable" : `${value.toFixed(3)}%`;
  return <section className="panel mt-7 overflow-hidden" aria-live="polite">
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[.08] p-5 sm:p-6">
      <div><p className="eyebrow">Optical–SAR joint analysis</p><h2 className="mt-2 text-xl font-semibold">Deterministic Evidence Fusion</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">{response.answer ?? "No pixel-level joint summary was computed."}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", result.status === "success" ? "border-emerald-300/25 bg-emerald-400/10 text-emerald-300" : result.status === "failed" ? "border-rose-300/25 bg-rose-400/10 text-rose-200" : "border-amber-300/25 bg-amber-400/10 text-amber-100")}>{label(result.status)}</span>
    </header>
    <div className="space-y-8 p-5 sm:p-6">
      {response.vqa_details && <div className="grid gap-3 sm:grid-cols-3"><CompatibilityFact name="Original question" value={response.vqa_details.original_question}/><CompatibilityFact name="Question category" value={label(response.vqa_details.question_category)}/><CompatibilityFact name="Answer source" value={response.vqa_details.answer_source}/></div>}
      {result.status === "partial" && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100">This pair supports visual inspection only or contains limited evidence. Review the compatibility and limitations before interpreting it.</div>}
      {result.status === "alignment_required" && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100">Explicit alignment or reprojection is required. SatQuery did not resize, register, or fuse this pair.</div>}
      <div className="grid gap-5 lg:grid-cols-3">
        <ObservationCard title="Optical observations" items={result.summary.optical_observations}/>
        <ObservationCard title="SAR observations" items={result.summary.sar_observations}/>
        <ObservationCard title="Joint inference" items={result.summary.joint_observations}/>
      </div>
      {previews.some(([, path]) => path) && <div><h3 className="text-sm font-semibold">Evidence products</h3><div className="mt-3 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{previews.map(([name, path]) => path && <figure key={name} className="overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/60"><figcaption className="border-b border-white/[.08] px-4 py-3 text-sm font-medium text-zinc-200">{name}</figcaption><img src={agentPreviewUrl(path)} alt={`${name} optical-SAR evidence product`} className="aspect-video w-full object-contain"/></figure>)}</div></div>}
      {statistics && <div><h3 className="text-sm font-semibold">Measured candidate evidence</h3><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-6"><CompatibilityFact name="Water likelihood" value={metric(statistics.water_likelihood_percent)}/><CompatibilityFact name="Structural likelihood" value={metric(statistics.built_up_likelihood_percent)}/><CompatibilityFact name="Vegetation support" value={metric(statistics.vegetation_support_percent)}/><CompatibilityFact name="Agreement" value={metric(statistics.agreement_percent)}/><CompatibilityFact name="Disagreement" value={metric(statistics.disagreement_percent)}/><CompatibilityFact name="Valid pixels" value={metric(statistics.valid_pixel_percent)}/></div><p className="mt-3 text-xs text-zinc-500">Analysis grid: {statistics.analysis_width} × {statistics.analysis_height} · Candidate percentages are deterministic support measures, not model probabilities.</p></div>}
      {result.regions.length > 0 && <div><h3 className="text-sm font-semibold">Connected evidence regions</h3><div className="mt-3 overflow-x-auto rounded-xl border border-white/[.08]"><table className="w-full min-w-[760px] text-left text-xs"><thead className="bg-white/[.03] text-zinc-500"><tr><th className="px-4 py-3 font-medium">Type</th><th className="px-4 py-3 font-medium">Area</th><th className="px-4 py-3 font-medium">Pixel bounding box</th><th className="px-4 py-3 font-medium">Modality support</th></tr></thead><tbody>{result.regions.map(region => <tr key={region.region_id} className="border-t border-white/[.06] text-zinc-300"><td className="px-4 py-3">{label(region.type)}</td><td className="px-4 py-3">{region.area_pixels.toLocaleString()} px · {region.area_percent.toFixed(3)}%</td><td className="px-4 py-3 font-mono">[{region.bbox_pixels.join(", ")}]</td><td className="px-4 py-3">Optical {region.support.optical ? "yes" : "no"} · SAR {region.support.sar ? "yes" : "no"}</td></tr>)}</tbody></table></div></div>}
      <div className="grid gap-5 xl:grid-cols-2"><div className="space-y-5"><div className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-4"><h3 className="text-sm font-semibold text-sky-100">Methodology</h3><p className="mt-2 text-sm text-zinc-300">{result.method.name} · v{result.method.version} · Uses trained model: {result.method.uses_trained_model ? "yes" : "no"}</p><h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-zinc-500">Assumptions</h4><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.method.assumptions.map(item => <li key={item}>• {item}</li>)}</ul><h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-zinc-500">Limitations</h4><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.method.limitations.map(item => <li key={item}>• {item}</li>)}</ul></div><div className="rounded-xl border border-white/[.08] p-4"><h3 className="text-sm font-semibold">Confidence</h3><p className="mt-2 text-sm leading-6 text-zinc-400"><span className="font-medium text-zinc-200">{label(result.confidence.level)} · No calibrated score.</span> {result.confidence.reason}</p></div>{(result.optical_preparation || result.sar_preparation) && <details className="rounded-xl border border-white/[.08] p-4"><summary className="cursor-pointer text-sm font-semibold">Preparation record</summary><div className="mt-4 space-y-4 text-xs leading-5 text-zinc-400">{result.optical_preparation && <div><p className="font-medium text-zinc-200">Optical</p><p>{result.optical_preparation.stretch_method}</p><p>{result.optical_preparation.resize_status}</p></div>}{result.sar_preparation && <div><p className="font-medium text-zinc-200">SAR</p><p>{result.sar_preparation.stretch_method}</p><p>{result.sar_preparation.log_transform}</p><p>{result.sar_preparation.resize_status}</p></div>}</div></details>}</div><div><div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold"><Route size={16} className="text-sky-200"/>Execution trace</h3><span className="flex items-center gap-1.5 text-xs text-zinc-500"><Clock3 size={13}/>{response.execution.duration_ms} ms</span></div><ol className="mt-4 space-y-3">{response.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-7 w-7 place-items-center rounded-full", step.status === "success" ? "bg-emerald-400/10 text-emerald-300" : step.status === "failed" ? "bg-rose-400/10 text-rose-300" : "bg-amber-400/10 text-amber-200")}>{step.status === "success" ? <CheckCircle2 size={15}/> : <span className="text-xs font-semibold">{index + 1}</span>}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{label(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{step.duration_ms} ms</span></li>)}</ol></div></div>
      {result.warnings.length > 0 && <div><h3 className="flex items-center gap-2 text-sm font-semibold"><AlertTriangle size={15} className="text-amber-200"/>Warnings</h3><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{result.warnings.map(warning => <li key={warning}>• {warning}</li>)}</ul></div>}
    </div>
  </section>;
}

function ObservationCard({ title, items }: { title: string; items: string[] }) {
  return <article className="rounded-xl border border-white/[.08] bg-white/[.025] p-4"><h3 className="text-sm font-semibold">{title}</h3>{items.length > 0 ? <ul className="mt-3 space-y-2 text-sm leading-6 text-zinc-400">{items.map(item => <li key={item}>• {item}</li>)}</ul> : <p className="mt-3 text-sm text-zinc-500">Unavailable for this pair.</p>}</article>;
}

function ControlledVQAResultPanel({ response }: { response: AgentResponse }) {
  const details = response.vqa_details!;
  const evidence = details.single_image_evidence;
  const previewItems = evidence ? [
    ["Water support", evidence.previews.water_support],
    ["Vegetation support", evidence.previews.vegetation_support],
    ["Structural support", evidence.previews.built_up_support],
    ["Agriculture support", evidence.previews.agriculture_support],
    ["Combined evidence overlay", evidence.previews.combined_overlay],
  ] as const : [];
  const stats = evidence?.statistics;
  return <section className="panel mt-7 overflow-hidden" aria-live="polite">
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[.08] p-5 sm:p-6">
      <div><p className="eyebrow">Controlled evidence-grounded VQA</p><h2 className="mt-2 text-xl font-semibold">{response.answer ?? "Unsupported question"}</h2><p className="mt-3 text-sm leading-6 text-zinc-400">Question: {details.original_question}</p></div>
      <span className={cn("rounded-full border px-3 py-1 text-xs font-medium", details.supported ? "border-emerald-300/25 bg-emerald-400/10 text-emerald-300" : "border-amber-300/25 bg-amber-400/10 text-amber-100")}>{details.supported ? label(details.question_category) : "Unsupported"}</span>
    </header>
    <div className="space-y-7 p-5 sm:p-6">
      <div className="grid gap-3 sm:grid-cols-3"><CompatibilityFact name="Question category" value={label(details.question_category)}/><CompatibilityFact name="Target concept" value={details.target_concept ? label(details.target_concept) : "Unavailable"}/><CompatibilityFact name="Answer source" value={details.answer_source}/></div>
      {!details.supported && <div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-sm leading-6 text-amber-100">This question is outside the controlled taxonomy. No unsupported answer or semantic mask was fabricated.</div>}
      {stats && <div><h3 className="text-sm font-semibold">Measured heuristic support</h3><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-7"><CompatibilityFact name="Water" value={`${stats.water_support_percent.toFixed(3)}%`}/><CompatibilityFact name="Vegetation" value={`${stats.vegetation_support_percent.toFixed(3)}%`}/><CompatibilityFact name="Structural" value={`${stats.built_up_support_percent.toFixed(3)}%`}/><CompatibilityFact name="Barren" value={`${stats.barren_support_percent.toFixed(3)}%`}/><CompatibilityFact name="Agriculture" value={`${stats.agriculture_support_percent.toFixed(3)}%`}/><CompatibilityFact name="Edge density" value={`${stats.edge_density_percent.toFixed(3)}%`}/><CompatibilityFact name="Valid pixels" value={`${stats.valid_pixel_percent.toFixed(3)}%`}/></div><p className="mt-3 text-xs text-zinc-500">Dominant scene: {label(evidence!.dominant_scene)} · These values are support measures, not class probabilities.</p></div>}
      {previewItems.length > 0 && <div><h3 className="text-sm font-semibold">Heuristic evidence previews</h3><div className="mt-3 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{previewItems.map(([name, path]) => path && <figure key={name} className="overflow-hidden rounded-2xl border border-white/[.08] bg-zinc-950/60"><figcaption className="border-b border-white/[.08] px-4 py-3 text-sm font-medium text-zinc-200">{name}</figcaption><img src={agentPreviewUrl(path)} alt={`${name} heuristic evidence`} className="aspect-video w-full object-contain"/></figure>)}</div></div>}
      {evidence && evidence.regions.length > 0 && <div><h3 className="text-sm font-semibold">Major evidence regions</h3><div className="mt-3 overflow-x-auto rounded-xl border border-white/[.08]"><table className="w-full min-w-[650px] text-left text-xs"><thead className="bg-white/[.03] text-zinc-500"><tr><th className="px-4 py-3">Type</th><th className="px-4 py-3">Area</th><th className="px-4 py-3">Pixel bounding box</th></tr></thead><tbody>{evidence.regions.map(region => <tr key={region.region_id} className="border-t border-white/[.06] text-zinc-300"><td className="px-4 py-3">{label(region.type)}</td><td className="px-4 py-3">{region.area_pixels.toLocaleString()} px · {region.area_percent.toFixed(3)}%</td><td className="px-4 py-3 font-mono">[{region.bbox_pixels.join(", ")}]</td></tr>)}</tbody></table></div></div>}
      <div className="grid gap-5 xl:grid-cols-2"><div className="space-y-5"><div className="rounded-xl border border-sky-300/15 bg-sky-300/[.05] p-4"><h3 className="text-sm font-semibold text-sky-100">Methodology</h3><p className="mt-2 text-sm text-zinc-300">{details.method.name} · {details.method.method_type} · Language model: no</p><h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-zinc-500">Limitations</h4><ul className="mt-2 space-y-2 text-sm leading-6 text-zinc-400">{details.limitations.map(item => <li key={item}>• {item}</li>)}</ul></div><div className="rounded-xl border border-white/[.08] p-4"><h3 className="text-sm font-semibold">Confidence</h3><p className="mt-2 text-sm leading-6 text-zinc-400"><span className="font-medium text-zinc-200">{label(details.confidence.level)} · No calibrated score.</span> {details.confidence.reason}</p></div></div>{evidence ? <div><div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold"><Route size={16} className="text-sky-200"/>Execution trace</h3><span className="text-xs text-zinc-500">{response.execution.duration_ms} ms</span></div><ol className="mt-4 space-y-3">{response.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`} className="glass flex items-center gap-3 rounded-xl p-3"><span className={cn("grid h-7 w-7 place-items-center rounded-full", step.status === "success" ? "bg-emerald-400/10 text-emerald-300" : step.status === "failed" ? "bg-rose-400/10 text-rose-300" : "bg-amber-400/10 text-amber-200")}>{step.status === "success" ? <CheckCircle2 size={15}/> : index + 1}</span><div className="min-w-0 flex-1"><p className="truncate font-mono text-xs text-zinc-200">{step.tool}</p><p className="mt-1 text-[11px] text-zinc-500">{label(step.status)}</p></div><span className="text-xs tabular-nums text-zinc-500">{step.duration_ms} ms</span></li>)}</ol></div> : <div><h3 className="text-sm font-semibold">Statistics used</h3><div className="mt-3 space-y-2">{Object.entries(details.statistics_used).map(([name, value]) => <div key={name} className="glass rounded-xl p-3"><p className="text-xs text-zinc-500">{label(name)}</p><p className="mt-1 break-words text-sm text-zinc-200">{typeof value === "object" ? JSON.stringify(value) : String(value)}</p></div>)}</div></div>}</div>
    </div>
  </section>;
}

export function AssistantWorkspace() {
  const searchParameters = useSearchParams();
  const historyView = searchParameters.get("view") === "history";
  const [mode, setMode] = useState<InputMode>("single");
  const [uploadsOpen, setUploadsOpen] = useState(false);
  const [secondObservation, setSecondObservation] = useState(false);
  const [primaryModality, setPrimaryModality] = useState<Modality>("optical");
  const [secondaryModality, setSecondaryModality] = useState<Modality>("sar");
  const [primaryFile, setPrimaryFile] = useState<File>();
  const [primaryImageModality, setPrimaryImageModality] = useState<ImageModality>("auto");
  const [primaryInspection, setPrimaryInspection] = useState<ImageInspectionResponse>();
  const [secondaryInspection, setSecondaryInspection] = useState<ImageInspectionResponse>();
  const [inspecting, setInspecting] = useState(false);
  const [secondaryFile, setSecondaryFile] = useState<File>();
  const [temporalDates, setTemporalDates] = useState({ primary: "", secondary: "" });
  const primaryDate = temporalDates.primary;
  const secondaryDate = temporalDates.secondary;
  const [query, setQuery] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [promptNotice, setPromptNotice] = useState("");
  const [executionStage, setExecutionStage] = useState(0);
  const [result, setResult] = useState<AgentResponse>();
  const [changeResult, setChangeResult] = useState<ChangeAnalysisResponse>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [useCache, setUseCache] = useState(true);
  const [demo, setDemo] = useState<DemoManifest>();
  const [demoLoading, setDemoLoading] = useState("");
  const [demoBadge, setDemoBadge] = useState("");
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [conversationHistory, setConversationHistory] = useState<RecentConversation[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const inspectionSequence = useRef(0);
  const secondaryInspectionSequence = useRef(0);

  useEffect(() => {
    let active = true;
    const checkBackend = () => getHealth().then(() => { if (active) setBackendAvailable(true); }).catch(() => { if (active) setBackendAvailable(false); });
    checkBackend();
    getDemoManifest().then(setDemo).catch(() => setDemo(undefined));
    const interval = window.setInterval(checkBackend, 15_000);
    return () => { active = false; window.clearInterval(interval); };
  }, []);

  useEffect(() => {
    if (historyView) setConversationHistory(readRecentConversations(window.localStorage));
  }, [historyView]);

  useEffect(() => {
    if (!loading) { setExecutionStage(0); return; }
    const timer = window.setInterval(() => setExecutionStage(stage => Math.min(stage + 1, 5)), 850);
    return () => window.clearInterval(timer);
  }, [loading]);

  useEffect(() => {
    const requestedMode = searchParameters.get("mode") as InputMode | null;
    const prompt = searchParameters.get("prompt");
    const groundingIntent = searchParameters.get("intent") === "grounding";
    const attachIntent = searchParameters.get("attach") === "1";
    if (requestedMode && ["single", "cross_modal", "bi_temporal"].includes(requestedMode)) {
      setMode(requestedMode);
      setUploadsOpen(true);
      setSecondObservation(requestedMode !== "single");
      setSecondaryModality(requestedMode === "cross_modal" ? "sar" : "optical");
      if (!prompt) setQuery(requestedMode === "cross_modal" ? crossExamples[0] : requestedMode === "bi_temporal" ? temporalExamples[0] : singleExamples[0]);
    }
    if (groundingIntent && !prompt) { setQuery("Highlight every visible water body."); setUploadsOpen(true); }
    if (attachIntent) setUploadsOpen(true);
    if (prompt) setQuery(prompt);
  }, [searchParameters]);

  function resetInspection() { setResult(undefined); setChangeResult(undefined); setError(""); }
  function selectMode(next: InputMode) {
    setMode(next);
    setUploadsOpen(true);
    setSecondObservation(next !== "single");
    setPrimaryImageModality("auto");
    setPrimaryInspection(undefined);
    setPrimaryModality("optical");
    setSecondaryModality(next === "cross_modal" ? "sar" : "optical");
    if (!query.trim()) setQuery(next === "cross_modal" ? crossExamples[0] : next === "bi_temporal" ? temporalExamples[0] : singleExamples[0]);
    if (next === "single") { setSecondaryFile(undefined); setSecondaryInspection(undefined); }
    if (next !== "bi_temporal") setTemporalDates({ primary: "", secondary: "" });
    setDemoBadge("");
    resetInspection();
  }
  async function updatePrimary(file?: File) {
    const sequence = ++inspectionSequence.current;
    setPrimaryFile(file); setPrimaryInspection(undefined); setPrimaryImageModality("auto"); resetInspection();
    if (!file) return;
    setInspecting(true);
    try {
      const inspected = await inspectAgentImage(file);
      if (inspectionSequence.current !== sequence) return;
      setPrimaryInspection(inspected);
      setPrimaryImageModality("auto");
      setBackendAvailable(true);
    } catch (caught) {
      if (inspectionSequence.current === sequence) setError(caught instanceof Error ? caught.message : "Image inspection failed.");
    } finally {
      if (inspectionSequence.current === sequence) setInspecting(false);
    }
  }
  async function updateSecondary(file?: File) {
    const sequence = ++secondaryInspectionSequence.current;
    setSecondaryFile(file); setSecondaryInspection(undefined); resetInspection();
    if (!file) return;
    const filenamePair = `${primaryFile?.name ?? ""} ${file.name}`.toLowerCase();
    setMode(/\b(sar|radar|vv|vh)\b/.test(filenamePair) ? "cross_modal" : "bi_temporal");
    setInspecting(true);
    try {
      const inspected = await inspectAgentImage(file);
      if (secondaryInspectionSequence.current !== sequence) return;
      setSecondaryInspection(inspected);
      setBackendAvailable(true);
    } catch (caught) {
      if (secondaryInspectionSequence.current === sequence) setError(caught instanceof Error ? caught.message : "Second-image inspection failed.");
    } finally {
      if (secondaryInspectionSequence.current === sequence) setInspecting(false);
    }
  }
  function changePrimaryModality(next: Modality) {
    setPrimaryModality(next);
    if (mode === "bi_temporal") setSecondaryModality(next);
    resetInspection();
  }
  function swapTemporalInputs() {
    const currentFile = primaryFile; const currentModality = primaryModality;
    setPrimaryFile(secondaryFile); setSecondaryFile(currentFile);
    setTemporalDates(swapTemporalDates);
    setPrimaryModality(secondaryModality); setSecondaryModality(currentModality);
    resetInspection();
  }

  useEffect(() => {
    if (!secondObservation || !primaryFile || !secondaryFile || !primaryInspection || !secondaryInspection) return;
    const primaryDetected = coarseFromDetailed(primaryInspection.metadata.auto_detected_modality ?? "unknown", primaryModality);
    const secondaryDetected = coarseFromDetailed(secondaryInspection.metadata.auto_detected_modality ?? "unknown", secondaryModality);
    const detectedMode: InputMode = [primaryDetected, secondaryDetected].includes("sar") && [primaryDetected, secondaryDetected].some(value => value === "optical" || value === "multispectral") ? "cross_modal" : "bi_temporal";
    setMode(detectedMode);
    setPrimaryModality(primaryDetected);
    setSecondaryModality(detectedMode === "cross_modal" ? secondaryDetected : primaryDetected);
  }, [primaryFile, primaryInspection, primaryModality, secondObservation, secondaryFile, secondaryInspection, secondaryModality]);

  const pairMode = secondObservation || mode !== "single";
  const modalityValid = mode !== "cross_modal" || ([primaryModality, secondaryModality].includes("sar") && [primaryModality, secondaryModality].some(value => value === "optical" || value === "multispectral"));
  const datesValid = mode !== "bi_temporal" || Boolean(primaryDate && secondaryDate && primaryDate < secondaryDate);
  const pairIncompatible = result?.pair_compatibility?.compatible === false || changeResult?.compatibility.compatible === false;
  const needsConfirmation = mode === "single" && Boolean(primaryInspection?.requires_modality_confirmation) && ["auto", "unknown"].includes(primaryImageModality);
  const canAnalyze = Boolean(primaryFile && (mode === "bi_temporal" || query.trim()) && (!pairMode || secondaryFile) && modalityValid && datesValid && !pairIncompatible && !needsConfirmation && !loading && !inspecting && backendAvailable !== false);
  const canSendPrompt = Boolean(query.trim() && !loading && backendAvailable !== false);

  async function executeAnalysis(forceRerun = false) {
    if (!primaryFile || !canAnalyze) return;
    const controller = new AbortController(); abortRef.current = controller;
    setLoading(true); setError(""); setPromptNotice(""); setLastQuestion(query.trim()); setResult(undefined); setChangeResult(undefined);
    rememberConversation(window.localStorage, query, mode);
    try {
      const effectiveCoarse = mode === "single" ? coarseFromDetailed(primaryImageModality, primaryModality) : primaryModality;
      const response = await runAgentImageQuery({ query: query.trim(), inputMode: mode, primaryModality: effectiveCoarse, primaryImageModality: mode === "single" ? primaryImageModality : "auto", secondaryModality: pairMode ? secondaryModality : null, primaryImage: primaryFile, secondaryImage: pairMode ? secondaryFile : undefined, primaryDate: mode === "bi_temporal" ? primaryDate : undefined, secondaryDate: mode === "bi_temporal" ? secondaryDate : undefined, useCache, forceRerun, signal: controller.signal });
      setBackendAvailable(true);
      setResult(response);
      setChangeResult(response.change_analysis ?? undefined);
      window.sessionStorage.setItem("satquery-latest-request-id", response.request_id);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") setError("The local upload request was cancelled.");
      else { setBackendAvailable(false); setError(caught instanceof Error ? caught.message : "The SatQuery backend is unavailable."); }
    } finally {
      abortRef.current = null; setLoading(false);
    }
  }

  async function analyze(event: FormEvent) {
    event.preventDefault();
    if (!primaryFile) { setUploadsOpen(true); setPromptNotice("Attach at least one satellite observation so SatQuery can ground this question in visual evidence."); return; }
    if (pairMode && !secondaryFile) { setUploadsOpen(true); setPromptNotice("Add the second observation, or switch the detected workflow to Single Image."); return; }
    if (!canAnalyze) { setPromptNotice("Review the detected workflow and highlighted input requirements before sending this question."); return; }
    await executeAnalysis(false);
  }

  async function loadDemo(workflow: DemoWorkflow) {
    if (demoLoading || loading) return;
    setDemoLoading(workflow.id); setError(""); resetInspection();
    try {
      const loaded = await Promise.all(workflow.files.map(async item => ({ role: item.role, file: await loadDemoFile(item.url, item.filename, item.mime_type) })));
      setMode(workflow.input_mode); setPrimaryModality(workflow.primary_modality); setSecondaryModality(workflow.secondary_modality ?? "optical");
      setUploadsOpen(true); setSecondObservation(workflow.input_mode !== "single");
      setPrimaryFile(loaded.find(item => item.role === "primary")?.file); setSecondaryFile(loaded.find(item => item.role === "secondary")?.file);
      setTemporalDates({ primary: workflow.primary_date ?? "", secondary: workflow.secondary_date ?? "" }); setQuery(workflow.query); setDemoBadge(workflow.title);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "The approved local demo sample could not be loaded."); }
    finally { setDemoLoading(""); }
  }

  function selectPreset(next: "describe" | "question" | "change" | "cross") {
    const nextMode: InputMode = next === "change" ? "bi_temporal" : next === "cross" ? "cross_modal" : "single";
    selectMode(nextMode);
    setQuery(next === "describe" ? "Describe the land cover and major objects visible in this image." : next === "question" ? "What is the dominant land-cover type?" : next === "change" ? temporalExamples[0] : crossExamples[0]);
  }

  const primaryLabel = mode === "bi_temporal" ? "Earlier-date image" : mode === "cross_modal" ? "Optical / multispectral image" : "Primary image";
  const secondaryLabel = mode === "bi_temporal" ? "Later-date image" : "SAR image";
  const primaryOptions = mode === "cross_modal" ? allModalities.filter(item => item.value === "optical" || item.value === "multispectral") : mode === "bi_temporal" ? allModalities.filter(item => item.value !== "unknown") : allModalities;
  const promptModality = primaryImageModality === "auto" ? primaryInspection?.metadata.auto_detected_modality : primaryImageModality;
  const examples = mode === "cross_modal" ? crossExamples : mode === "bi_temporal" ? temporalExamples : promptModality === "sar_preview" ? sarPreviewExamples : ["sar_vv", "sar_vh", "sar_vv_vh"].includes(promptModality ?? "") ? scientificSarExamples : singleExamples.filter(item => item !== "Mark the aircraft.");

  if (historyView) return <section className="assistant-workspace assistant-history">
    <header><div><p className="eyebrow">Assistant workspace</p><h1>Conversation history</h1><p>Return to recent questions and continue exploring the same line of inquiry.</p></div><Link href="/assistant"><Plus size={15}/>New conversation</Link></header>
    {conversationHistory.length > 0 ? <div className="assistant-history-list">{conversationHistory.map(item => <Link key={item.id} href={`/assistant?prompt=${encodeURIComponent(item.prompt)}`}><span><MessageCircle size={16}/></span><div><strong>{item.prompt}</strong><small>{item.mode ? label(item.mode) : "Earth intelligence"} · {new Date(item.createdAt).toLocaleString()}</small></div><ArrowUp size={15}/></Link>)}</div> : <div className="assistant-history-empty"><span><MessageCircle size={22}/></span><h2>No conversations yet</h2><p>Ask your first question about a satellite observation. SatQuery will keep it here for a quick return.</p><Link href="/assistant">Ask Earth Anything <ArrowUp size={14}/></Link></div>}
  </section>;

  return <section className="assistant-workspace">
    <header className="assistant-hero">
      <div className="assistant-hero-orb" aria-hidden="true"><span/><span/></div>
      <div className="assistant-hero-copy"><p className="eyebrow">SatQuery spatial reasoning</p><h1>Ask Earth Anything.</h1><p>Upload remote-sensing observations and ask a natural-language question. SatQuery selects the right specialist, grounds the evidence, and keeps its reasoning traceable.</p></div>
      <span className={cn("assistant-backend-status", backendAvailable === false ? "assistant-backend-offline" : backendAvailable === true ? "assistant-backend-online" : "")}><RadioTower size={14}/>{backendAvailable === null ? "Checking agent" : backendAvailable ? "SatQuery Agent Ready" : "Agent unavailable"}</span>
    </header>

    {backendAvailable === false && <div role="alert" className="mt-5 rounded-xl border border-rose-300/20 bg-rose-300/[.06] p-4 text-sm leading-6 text-rose-100">The local SatQuery backend is unavailable. Start FastAPI on port 8010, then wait for the automatic connection check or reload this page.</div>}

    <section className="assistant-conversation" aria-label="Current conversation">
      <article className="assistant-message assistant-message-agent"><span><MessageCircle size={17}/></span><div><small>SatQuery</small><h2>What would you like to understand?</h2><p>Start with a question, then attach the observation SatQuery should examine. A second image automatically opens cross-modal or change analysis.</p></div></article>
      {lastQuestion && <article className="assistant-message assistant-message-user"><div><small>You</small><p>{lastQuestion}</p></div></article>}
    </section>

    <div className="assistant-presets" aria-label="Assistant workflow presets">{([
      ["describe", "Describe Scene", ImagePlus], ["question", "Ask a Question", FileQuestion], ["change", "Analyze Changes", ScanSearch], ["cross", "Analyze Optical + SAR", Images],
    ] as const).map(([id, text, Icon]) => <button key={id} type="button" onClick={() => selectPreset(id)}><Icon size={16}/>{text}</button>)}</div>

    {demo?.enabled && <section className="mt-7 rounded-2xl border border-amber-300/20 bg-amber-300/[.06] p-5"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="eyebrow text-amber-200">Local demo mode</p><h2 className="mt-2 font-semibold">Approved offline samples</h2><p className="mt-2 text-sm leading-6 text-zinc-400">Files are served from the local backend and run through the real analysis workflows. No answers are hardcoded.</p></div><span className="rounded-full border border-amber-300/25 px-3 py-1 text-xs text-amber-100">Demo samples</span></div><div className="mt-4 flex flex-wrap gap-2">{demo.workflows.map(workflow => <button key={workflow.id} type="button" onClick={() => loadDemo(workflow)} disabled={Boolean(demoLoading || loading)} title={workflow.description} className="rounded-xl border border-amber-300/20 px-3 py-2 text-xs text-amber-100 hover:bg-amber-300/10 disabled:opacity-50">{demoLoading === workflow.id ? <LoaderCircle className="mr-1.5 inline animate-spin" size={14}/> : null}Load Demo Sample · {workflow.title}</button>)}</div></section>}
    {demoBadge && <div className="mt-5 inline-flex rounded-full border border-amber-300/25 bg-amber-300/[.08] px-3 py-1 text-xs font-medium text-amber-100">Demo Sample · {demoBadge}</div>}

    <form onSubmit={analyze} className="assistant-composer">
      <div className="assistant-composer-heading"><span><Sparkles size={15}/>Message SatQuery</span><small>Evidence-first · local processing</small></div>
      <div className="assistant-question"><label>Ask about Earth<textarea value={query} onChange={event => { setQuery(event.target.value); setPromptNotice(""); resetInspection(); }} rows={3} maxLength={2000} placeholder="Describe this satellite image..."/></label><div className="assistant-question-footer"><span>{query.length.toLocaleString()} / 2,000</span><button type="submit" disabled={!canSendPrompt} aria-label="Send question to SatQuery">{loading ? <Sparkles size={17} className="assistant-pulse-icon"/> : <ArrowUp size={17}/>}</button></div></div>
      <div className="assistant-composer-tools"><button type="button" className={cn("assistant-attachment-toggle", uploadsOpen && "is-open")} onClick={() => setUploadsOpen(value => !value)} aria-expanded={uploadsOpen}><Paperclip size={15}/>{primaryFile ? "Review attachments" : "Attach imagery"}{primaryFile && <CheckCircle2 size={14}/>}</button><label><input type="checkbox" checked={useCache} onChange={event => setUseCache(event.target.checked)}/>Reuse matching analysis</label></div>
      {promptNotice && <p role="status" className="assistant-prompt-notice"><Paperclip size={15}/>{promptNotice}</p>}
      <div className="assistant-examples"><p>Try asking</p><div>{examples.slice(0, 6).map(example => <button type="button" key={example} onClick={() => { setQuery(example); setPromptNotice(""); resetInspection(); }}>{example}</button>)}</div></div>

      {uploadsOpen && <section className="assistant-attachments" aria-label="Attached observations">
        <header><div><p className="eyebrow">Visual context</p><h2>Attached observations</h2></div>{primaryFile && <span className="assistant-detected-workflow"><CheckCircle2 size={14}/>Detected workflow <strong>{modes.find(item => item.id === mode)?.label}</strong></span>}</header>
        <div className={cn("grid gap-4", pairMode && "xl:grid-cols-2")}>
          <AssistantUploadCard label={primaryLabel} hint={mode === "cross_modal" ? "Optical or multispectral observation" : "GeoTIFF, TIFF, PNG, or JPEG"} file={primaryFile} metadata={changeResult?.before_metadata ?? result?.primary_image_metadata ?? primaryInspection?.metadata} busy={loading || inspecting} onFile={updatePrimary} onError={setError}/>
          {pairMode && <AssistantUploadCard label={secondaryLabel} hint={mode === "cross_modal" ? "SAR observation" : "Same area, later date"} file={secondaryFile} metadata={changeResult?.after_metadata ?? result?.secondary_image_metadata ?? secondaryInspection?.metadata} busy={loading || inspecting} onFile={updateSecondary} onError={setError}/>} 
        </div>
        {!pairMode && <button type="button" className="assistant-add-observation" onClick={() => { setSecondObservation(true); setMode("bi_temporal"); }}><Plus size={15}/>Add a second observation</button>}

        <details className="assistant-input-details"><summary>Override detected workflow</summary><fieldset className="assistant-mode-picker"><legend>Analysis workflow</legend><div>{modes.map(item => <button type="button" key={item.id} onClick={() => selectMode(item.id)} className={cn(mode === item.id && "assistant-mode-active")} aria-pressed={mode === item.id}><span>{item.label}</span><small>{item.description}</small></button>)}</div></fieldset></details>
        <details className="assistant-input-details"><summary>Review detected modalities</summary><div className={cn("grid gap-4 pt-4", pairMode && "md:grid-cols-2")}>
          {mode === "single" ? <label className="text-sm font-medium">Primary modality<select value={primaryImageModality} onChange={event => { setPrimaryImageModality(event.target.value as ImageModality); resetInspection(); }} className="mt-2 block w-full rounded-xl border border-white/[.10] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none focus:border-sky-300/60">{detailedModalities.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label> : <label className="text-sm font-medium">Primary modality<select value={primaryModality} onChange={event => changePrimaryModality(event.target.value as Modality)} className="mt-2 block w-full rounded-xl border border-white/[.10] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none focus:border-sky-300/60">{primaryOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>}
          {pairMode && <label className="text-sm font-medium">Secondary modality<select value={secondaryModality} disabled={mode === "cross_modal" || mode === "bi_temporal"} onChange={event => { setSecondaryModality(event.target.value as Modality); resetInspection(); }} className="mt-2 block w-full rounded-xl border border-white/[.10] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none disabled:cursor-not-allowed disabled:opacity-70">{allModalities.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>}
        </div></details>

        {mode === "single" && primaryInspection && <section className={cn("rounded-xl border p-4 text-sm", needsConfirmation ? "border-amber-300/25 bg-amber-300/[.06]" : "border-sky-300/20 bg-sky-300/[.05]")} aria-label="Modality inspection">
          <div className="flex flex-wrap items-center justify-between gap-3"><strong className={needsConfirmation ? "text-amber-100" : "text-sky-100"}>Remote-sensing representation inspection</strong><span className="rounded-full border border-white/20 px-2.5 py-1 text-xs">{label(primaryInspection.metadata.representation ?? "unknown_representation")}</span></div>
          <dl className="mt-3 grid gap-3 text-xs sm:grid-cols-3"><div><dt className="text-zinc-500">Detected modality</dt><dd className="mt-1 text-zinc-200">{label(primaryInspection.metadata.auto_detected_modality ?? "unknown")}</dd></div><div><dt className="text-zinc-500">Detection confidence</dt><dd className="mt-1 text-zinc-200">{label(primaryInspection.metadata.auto_detection_confidence ?? "unavailable")}</dd></div><div><dt className="text-zinc-500">Effective modality</dt><dd className="mt-1 text-zinc-200">{label(primaryImageModality === "auto" ? (primaryInspection.metadata.effective_modality ?? "unknown") : primaryImageModality)}</dd></div></dl>
          <p className="mt-3 leading-6 text-zinc-400">{primaryInspection.metadata.auto_detection_reason}</p>
          {needsConfirmation && <p className="mt-2 font-medium leading-6 text-amber-100">This single-band display cannot be verified automatically. Confirm whether it is SAR, optical grayscale, panchromatic, or another representation.</p>}
          {primaryImageModality === "sar_preview" && <p className="mt-2 inline-flex rounded-full border border-amber-300/25 bg-amber-300/[.08] px-3 py-1 text-xs font-semibold text-amber-100">Display Preview — Qualitative Analysis Only</p>}
        </section>}

        {mode === "bi_temporal" && <div className="panel grid items-end gap-4 p-5 md:grid-cols-[1fr_1fr_auto]"><label className="text-sm font-medium">Earlier date<input type="date" value={primaryDate} onChange={event => { setTemporalDates(current => ({ ...current, primary: event.target.value })); resetInspection(); }} className="mt-2 block w-full rounded-xl border border-white/[.10] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none"/></label><label className="text-sm font-medium">Later date<input type="date" value={secondaryDate} onChange={event => { setTemporalDates(current => ({ ...current, secondary: event.target.value })); resetInspection(); }} className="mt-2 block w-full rounded-xl border border-white/[.10] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none"/></label><button type="button" onClick={swapTemporalInputs} className="flex items-center justify-center gap-2 rounded-xl border border-white/[.10] px-4 py-3 text-sm text-zinc-300 hover:bg-white/[.05]"><ArrowLeftRight size={16}/>Swap</button></div>}
        {mode === "bi_temporal" && <p className="assistant-evidence-note"><strong>Hybrid change analysis.</strong> TTP supplies the primary learned mask; the deterministic analyzer supplies independent evidence and fallback.</p>}
      </section>}

      {!datesValid && mode === "bi_temporal" && <p className="text-sm text-amber-200">Choose distinct dates with the earlier date first.</p>}
      {!modalityValid && <p className="text-sm text-rose-200">Cross-modal analysis requires one optical or multispectral image and one SAR image.</p>}
      {needsConfirmation && <p className="text-sm text-amber-200">Confirm the modality before analysis. No specialist will execute while this grayscale display remains ambiguous.</p>}
      {pairIncompatible && <p className="text-sm text-rose-200">Analyze is disabled until the incompatible pair is replaced.</p>}
      {loading && <AssistantExecutionProgress mode={mode} query={query} stage={executionStage} onCancel={() => abortRef.current?.abort()}/>} 
    </form>
    {error && <div role="alert" className="mt-7 rounded-xl border border-rose-300/25 bg-rose-400/[.08] p-4 text-sm text-rose-200">{error}</div>}
    {result && <AssistantResultExperience result={result} changeResult={changeResult} query={query} onRerun={() => executeAnalysis(true)} rerunning={loading}>
      {(changeResult?.compatibility ?? result.pair_compatibility) && <CompatibilityCard compatibility={(changeResult?.compatibility ?? result.pair_compatibility)!}/>} 
      {result.vqa_details && !result.cross_modal_analysis && <ControlledVQAResultPanel response={result}/>} 
      {changeResult && <ChangeResultPanel result={changeResult}/>} 
      {result.grounding_result && <GroundingResultPanel response={result}/>} 
      {result.cross_modal_analysis ? <CrossModalResultPanel response={result}/> : !result.vqa_details && !result.grounding_result && <ResultPanel result={result}/>} 
    </AssistantResultExperience>}
  </section>;
}

"use client";

import Link from "next/link";
import { type ReactNode, useMemo, useState } from "react";
import {
  Activity, Check, CheckCircle2, ChevronDown, ChevronUp, Columns2,
  Expand, Eye, FileSearch, Layers3, Minus, Plus, RotateCcw, Route, ShieldCheck,
  SlidersHorizontal, Sparkles, X,
} from "lucide-react";
import { MissionReportActions } from "@/components/mission-report-actions";
import { RemoteSensingAdaptationPanel } from "@/components/remote-sensing-adaptation-panel";
import { agentPreviewUrl } from "@/services/api";
import type { AgentResponse, ChangeAnalysisResponse } from "@/types/agent";
import { cn } from "@/lib/utils";

type Props = {
  result: AgentResponse;
  changeResult?: ChangeAnalysisResponse;
  query: string;
  onRerun?: () => void;
  rerunning?: boolean;
  children: ReactNode;
};

type EvidenceProduct = {
  label: string;
  url: string;
  description: string;
  contribution: "Primary" | "Supporting";
};

type Metric = { label: string; value: string; note?: string };
type Region = { id: string; type: string; areaPixels: number; areaPercent: number; bounds: string };

const titleCase = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
const percent = (value: number | null | undefined, digits = 2) => value == null ? "Unavailable" : `${value.toFixed(digits)}%`;

function addEvidence(items: EvidenceProduct[], label: string, url: string | null | undefined, description: string, contribution: EvidenceProduct["contribution"] = "Supporting") {
  if (url && !items.some(item => item.url === url)) items.push({ label, url, description, contribution });
}

function evidenceProducts(result: AgentResponse, change?: ChangeAnalysisResponse): EvidenceProduct[] {
  const items: EvidenceProduct[] = [];
  const vqa = result.vqa_details?.single_image_evidence;
  const cross = result.cross_modal_analysis;
  const grounding = result.grounding_result;
  const sarProducts = result.sar_water_analysis?.evidence_products ?? result.sar_scene_analysis?.evidence_products ?? [];

  addEvidence(items, change ? "Before" : cross ? "Optical source" : "Original image", change?.previews.before ?? cross?.previews.optical ?? result.primary_image_metadata?.preview_url, "Authoritative source-image preview used by the workflow.", "Primary");
  addEvidence(items, change ? "After" : cross ? "SAR source" : "Secondary image", change?.previews.after ?? cross?.previews.sar ?? result.secondary_image_metadata?.preview_url, "Second authoritative observation supplied to the workflow.", "Primary");

  if (grounding?.annotated_preview_url) addEvidence(items, "Grounding overlay", grounding.annotated_preview_url, "Accepted model-produced localization boxes over the source image.", "Primary");
  sarProducts.forEach(product => addEvidence(items, product.label, product.reference, product.description, product.type.includes("overlay") || product.type.includes("mask") ? "Primary" : "Supporting"));
  if (vqa) {
    addEvidence(items, "Combined evidence overlay", vqa.previews.combined_overlay, "Combined deterministic support layers used by the controlled answer.", "Primary");
    const target = result.vqa_details?.target_concept;
    const maps: Array<readonly [string, string | null, string]> = [
      ["Water support", vqa.previews.water_support, "Dark, homogeneous candidate regions used as water-support evidence."],
      ["Vegetation support", vqa.previews.vegetation_support, "Visible-spectrum vegetation-support evidence."],
      ["Built-up support", vqa.previews.built_up_support, "Texture and edge-based structural-support evidence."],
      ["Agriculture support", vqa.previews.agriculture_support, "Candidate agricultural-pattern support evidence."],
    ];
    maps.sort((a, b) => Number(b[0].toLowerCase().includes(target ?? "__none__")) - Number(a[0].toLowerCase().includes(target ?? "__none__")));
    maps.forEach(([name, url, description], index) => addEvidence(items, name, url, description, index === 0 && Boolean(target) ? "Primary" : "Supporting"));
  }
  if (change) {
    addEvidence(items, "Difference heatmap", change.previews.difference, "Normalized absolute pixel-difference evidence.");
    if (change.change_engine?.mode === "hybrid" || change.change_engine?.mode === "ttp") {
      addEvidence(items, "TTP learned mask", change.previews.ttp_mask ?? change.previews.mask, "Primary TTP model-generated binary change prediction; not ground truth.", "Primary");
      addEvidence(items, "Deterministic mask", change.previews.deterministic_mask, "Independent normalized-difference supporting evidence.");
      addEvidence(items, "Agreement map", change.previews.agreement, "Pixels where the learned and deterministic masks agree; not an accuracy map.");
      addEvidence(items, "Disagreement map", change.previews.disagreement, "Pixels where the learned and deterministic masks differ.");
      addEvidence(items, "TTP overlay", change.previews.ttp_overlay ?? change.previews.overlay, "TTP learned change mask overlaid on the later source observation.", "Primary");
    } else {
      addEvidence(items, "Deterministic change mask", change.previews.deterministic_mask ?? change.previews.mask, "Thresholded and morphologically cleaned deterministic change mask.", "Primary");
      addEvidence(items, "Deterministic overlay", change.previews.deterministic_overlay ?? change.previews.overlay, "Deterministic change evidence overlaid on the later observation.", "Primary");
    }
  }
  if (cross) {
    addEvidence(items, "Joint evidence overlay", cross.previews.joint_overlay, "Optical and SAR candidate evidence combined on the analysis grid.", "Primary");
    addEvidence(items, "Joint evidence", cross.previews.joint_evidence, "Shared candidate support across both modalities.", "Primary");
    addEvidence(items, "Agreement", cross.previews.agreement, "Pixels where optical and SAR evidence agree.");
    addEvidence(items, "Disagreement", cross.previews.disagreement, "Pixels where the two modality-specific evidence layers differ.");
    addEvidence(items, "Water likelihood", cross.previews.water_likelihood, "Deterministic water-likelihood support evidence.");
    addEvidence(items, "Built-up likelihood", cross.previews.built_up_likelihood, "Deterministic structural-likelihood support evidence.");
    addEvidence(items, "Vegetation support", cross.previews.vegetation_support, "Visible-spectrum vegetation-support evidence.");
  }
  return items;
}

function metrics(result: AgentResponse, change?: ChangeAnalysisResponse): Metric[] {
  const output: Metric[] = [];
  const vqa = result.vqa_details?.single_image_evidence;
  const cross = result.cross_modal_analysis;
  const grounding = result.grounding_result;
  const sarWater = result.sar_water_analysis;
  const sarScene = result.sar_scene_analysis;
  const evidenceCount = evidenceProducts(result, change).length;
  if (change?.statistics) {
    const stats = change.statistics;
    if (change.change_engine?.mode === "hybrid" || change.change_engine?.mode === "ttp") {
      output.push(
        { label: "TTP changed area", value: percent(change.ttp_result?.changed_percentage, 3), note: "Primary learned mask" },
        { label: "Deterministic changed area", value: percent(change.deterministic_statistics?.percentage_changed, 3), note: "Independent support" },
        { label: "Mask IoU", value: change.mask_comparison?.iou == null ? "Unavailable" : change.mask_comparison.iou.toFixed(3), note: "Evidence overlap, not accuracy" },
        { label: "Agreement", value: percent(change.mask_comparison?.agreement_percentage, 3) },
        { label: "Disagreement", value: percent(change.mask_comparison?.disagreement_percentage, 3) },
        { label: "TTP regions", value: (change.ttp_result?.region_count ?? stats.number_of_regions).toLocaleString() },
        { label: "TTP runtime", value: change.ttp_result?.runtime_ms == null ? "Unavailable" : `${change.ttp_result.runtime_ms.toLocaleString()} ms` },
      );
    } else output.push(
      { label: "Changed pixels", value: percent(stats.percentage_changed, 3), note: `${stats.changed_pixels.toLocaleString()} of ${stats.total_pixels.toLocaleString()}` },
      { label: "Largest region", value: `${stats.largest_connected_region.toLocaleString()} px` },
      { label: "Connected regions", value: stats.number_of_regions.toLocaleString() },
      { label: "Analysis grid", value: `${stats.analysis_width} × ${stats.analysis_height}` },
    );
  } else if (vqa) {
    const stats = vqa.statistics;
    const target = result.vqa_details?.target_concept;
    const values: Array<readonly [string, number, string]> = [
      ["Water support", stats.water_support_percent, "water"],
      ["Vegetation support", stats.vegetation_support_percent, "vegetation"],
      ["Built-up support", stats.built_up_support_percent, "building"],
      ["Agriculture support", stats.agriculture_support_percent, "agriculture"],
      ["Edge density", stats.edge_density_percent, "edge"],
      ["Valid pixels", stats.valid_pixel_percent, "valid"],
    ];
    values.sort((a, b) => Number((target ?? "").includes(b[2])) - Number((target ?? "").includes(a[2])));
    values.slice(0, 5).forEach(([name, value]) => output.push({ label: name, value: percent(value, 3) }));
  } else if (cross?.statistics) {
    output.push(
      { label: "Water likelihood", value: percent(cross.statistics.water_likelihood_percent) },
      { label: "Structural likelihood", value: percent(cross.statistics.built_up_likelihood_percent) },
      { label: "Agreement", value: percent(cross.statistics.agreement_percent) },
      { label: "Valid pixels", value: percent(cross.statistics.valid_pixel_percent) },
      { label: "Connected regions", value: cross.regions.length.toLocaleString() },
    );
  } else if (sarWater) {
    output.push(
      { label: "Candidate image area", value: percent(sarWater.image_area_percent, 3), note: `${sarWater.candidate_pixels.toLocaleString()} pixels` },
      ...(sarWater.geographic_area_square_meters != null ? [{ label: "Estimated geographic area", value: `${sarWater.geographic_area_square_meters.toLocaleString()} m²`, note: sarWater.geographic_area_method ?? "Projected raster estimate" }] : []),
      { label: "Connected regions", value: sarWater.regions.length.toLocaleString() },
      { label: "Heuristic reliability", value: sarWater.heuristic_reliability == null ? "Unavailable" : `${(sarWater.heuristic_reliability * 100).toFixed(1)}%`, note: "Not model confidence" },
      { label: "Input quality", value: sarWater.input_quality_score == null ? "Unavailable" : `${(sarWater.input_quality_score * 100).toFixed(1)}%` },
      { label: "Model confidence", value: "Not available", note: "Heuristic method" },
    );
  } else if (sarScene) {
    output.push(
      { label: "Valid pixels", value: percent(sarScene.valid_pixel_percent) },
      { label: "Low returns", value: percent(sarScene.low_backscatter_percent) },
      { label: "Mid returns", value: percent(sarScene.mid_backscatter_percent) },
      { label: "High returns", value: percent(sarScene.high_backscatter_percent) },
      { label: "Texture index", value: sarScene.texture_index.toFixed(3), note: "Deterministic, not semantic" },
    );
  } else if (grounding) {
    const acceptedDetections = grounding.detections ?? grounding.accepted_detections ?? [];
    const rejectedCandidates = grounding.rejected_candidates ?? [];
    const candidates = [...acceptedDetections.map(item => item.score), ...rejectedCandidates.map(item => item.score).filter((value): value is number => value != null)];
    output.push(
      { label: "Accepted regions", value: (grounding.accepted_detection_count ?? acceptedDetections.length).toLocaleString() },
      { label: "Rejected candidates", value: (grounding.rejected_candidate_count ?? rejectedCandidates.length).toLocaleString() },
      { label: "Top alignment score", value: candidates.length ? Math.max(...candidates).toFixed(4) : "Unavailable", note: "Operational score, not probability" },
      { label: "Model runtime", value: grounding.runtime_ms == null ? "Unavailable" : `${grounding.runtime_ms.toLocaleString()} ms` },
    );
  } else if (result.primary_image_metadata) {
    output.push(
      { label: "Dimensions", value: `${result.primary_image_metadata.width} × ${result.primary_image_metadata.height}` },
      { label: "Bands", value: result.primary_image_metadata.band_count.toLocaleString() },
      { label: "Georeferenced", value: result.primary_image_metadata.is_georeferenced ? "Yes" : "No" },
    );
  }
  output.push(
    { label: "Runtime", value: `${result.execution.duration_ms.toLocaleString()} ms` },
    { label: "Evidence products", value: evidenceCount.toLocaleString() },
  );
  return output.slice(0, 7);
}

function regions(result: AgentResponse, change?: ChangeAnalysisResponse): Region[] {
  if (change?.statistics) return change.statistics.regions.map(region => ({ id: String(region.region_id), type: "Change", areaPixels: region.area_pixels, areaPercent: region.percentage_of_image, bounds: `(${region.bounding_box.left}, ${region.bounding_box.top}) → (${region.bounding_box.right}, ${region.bounding_box.bottom})` }));
  if (result.vqa_details?.single_image_evidence) return result.vqa_details.single_image_evidence.regions.map(region => ({ id: region.region_id, type: titleCase(region.type), areaPixels: region.area_pixels, areaPercent: region.area_percent, bounds: `[${region.bbox_pixels.join(", ")}]` }));
  if (result.cross_modal_analysis) return result.cross_modal_analysis.regions.map(region => ({ id: region.region_id, type: titleCase(region.type), areaPixels: region.area_pixels, areaPercent: region.area_percent, bounds: `[${region.bbox_pixels.join(", ")}]` }));
  if (result.grounding_result) return (result.grounding_result.detections ?? result.grounding_result.accepted_detections ?? []).map((detection, index) => ({ id: String(index + 1), type: detection.label, areaPixels: detection.quality?.box_area ?? 0, areaPercent: (detection.quality?.box_area_ratio ?? 0) * 100, bounds: `[${detection.bbox_pixels.join(", ")}]` }));
  if (result.sar_water_analysis) return result.sar_water_analysis.regions.map(region => ({ id: String(region.region_id), type: "Probable water candidate", areaPixels: region.area_pixels, areaPercent: region.image_area_percent, bounds: `[${region.bounding_box.join(", ")}]` }));
  return [];
}

function limitationBadges(result: AgentResponse): string[] {
  const values = [
    ...result.warnings,
    ...(result.vqa_details?.limitations ?? []),
    ...(result.grounding_result?.limitations ?? []),
    ...(result.cross_modal_analysis?.method.limitations ?? []),
    ...(result.cross_modal_analysis?.warnings ?? []),
    ...(result.sar_water_analysis?.limitations ?? []),
    ...(result.sar_water_analysis?.warnings ?? []),
    ...(result.sar_scene_analysis?.limitations ?? []),
    ...(result.sar_scene_analysis?.warnings ?? []),
    ...(result.ttp_result?.limitations ?? []),
    ...(result.ttp_result?.warnings ?? []),
    ...(result.sve_result?.limitations ?? result.cross_modal_analysis?.sve_result?.limitations ?? []),
  ];
  if (result.primary_image_metadata && !result.primary_image_metadata.is_georeferenced) values.unshift("No georeference");
  if (result.confidence.level !== "high") values.unshift(`${titleCase(result.confidence.level)} confidence`);
  return Array.from(new Set(values.filter(Boolean)));
}

function timelineLabel(tool: string) {
  const lower = tool.toLowerCase();
  if (lower.includes("upload") || lower.includes("ingest")) return "Upload";
  if (lower.includes("valid")) return "Validation";
  if (lower.includes("metadata")) return "Metadata";
  if (lower.includes("rout") || lower.includes("classif")) return "Routing";
  if (lower.includes("evidence")) return "Evidence";
  if (lower.includes("answer") || lower.includes("vqa") || lower.includes("caption")) return "Answer";
  if (lower.includes("report")) return "Report";
  return titleCase(tool.replace(/^rs_/, ""));
}

function EvidenceWorkbench({ products }: { products: EvidenceProduct[] }) {
  const [selected, setSelected] = useState(Math.min(1, Math.max(0, products.length - 1)));
  const [mode, setMode] = useState<"layer" | "overlay" | "split">(products.length > 1 ? "overlay" : "layer");
  const [opacity, setOpacity] = useState(62);
  const [zoom, setZoom] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const base = products[0];
  const active = products[selected] ?? base;
  const visible = showMore ? products : products.slice(0, 4);
  const transform = `scale(${zoom})`;
  if (!base) return <EmptyEvidence/>;

  const viewer = <div className="relative grid min-h-[320px] flex-1 place-items-center overflow-hidden rounded-2xl bg-slate-950/80 sm:min-h-[430px]" aria-label="Evidence image viewer">
    {mode === "layer" || products.length === 1 ? <img src={agentPreviewUrl(active.url)} alt={active.label} className="max-h-full max-w-full object-contain transition-transform" style={{ transform }}/> : <>
      <img src={agentPreviewUrl(base.url)} alt={base.label} className="absolute inset-0 h-full w-full object-contain transition-transform" style={{ transform }}/>
      <div className="absolute inset-0 overflow-hidden" style={mode === "overlay" ? { opacity: opacity / 100 } : { clipPath: `inset(0 ${100 - opacity}% 0 0)` }}><img src={agentPreviewUrl(active.url)} alt={active.label} className="absolute inset-0 h-full w-full object-contain transition-transform" style={{ transform }}/></div>
      {mode === "split" && <span className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_0_1px_rgba(0,0,0,.6)]" style={{ left: `${opacity}%` }}/>} 
    </>}
    <span className="absolute left-3 top-3 rounded-lg bg-black/70 px-2.5 py-1 text-xs font-medium text-white">{active.label}</span>
  </div>;

  return <section className={fullscreen ? "fixed inset-0 z-[100] flex flex-col bg-zinc-950 p-4 sm:p-6" : "result-evidence-card"} aria-labelledby="evidence-viewer-title">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h3 id="evidence-viewer-title" className="text-lg font-semibold">Evidence viewer</h3><p className="mt-1 text-sm text-zinc-500">{active.description}</p></div>
      <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Evidence viewer controls">
        {products.length > 1 && <><button type="button" onClick={() => setMode("layer")} aria-pressed={mode === "layer"} className={cn("result-icon-button", mode === "layer" && "result-icon-button-active")}><Layers3 size={16}/><span>Layer</span></button><button type="button" onClick={() => setMode("overlay")} aria-pressed={mode === "overlay"} className={cn("result-icon-button", mode === "overlay" && "result-icon-button-active")}><SlidersHorizontal size={16}/><span>Opacity</span></button><button type="button" onClick={() => setMode("split")} aria-pressed={mode === "split"} className={cn("result-icon-button", mode === "split" && "result-icon-button-active")}><Columns2 size={16}/><span>Split</span></button></>}
        <button type="button" onClick={() => setZoom(value => Math.min(4, value + .25))} aria-label="Zoom in" className="result-square-button"><Plus size={16}/></button>
        <button type="button" onClick={() => setZoom(value => Math.max(1, value - .25))} aria-label="Zoom out" className="result-square-button"><Minus size={16}/></button>
        <button type="button" onClick={() => { setZoom(1); setOpacity(62); }} aria-label="Fit image and reset" className="result-square-button"><RotateCcw size={16}/></button>
        <button type="button" onClick={() => setFullscreen(value => !value)} aria-label={fullscreen ? "Exit fullscreen evidence viewer" : "Open fullscreen evidence viewer"} className="result-square-button">{fullscreen ? <X size={16}/> : <Expand size={16}/>}</button>
      </div>
    </div>
    {mode !== "layer" && products.length > 1 && <label className="mt-3 flex max-w-md items-center gap-3 text-xs text-zinc-500"><span>{mode === "split" ? "Split position" : "Evidence opacity"}</span><input aria-label={mode === "split" ? "Split position" : "Evidence opacity"} type="range" min="0" max="100" value={opacity} onChange={event => setOpacity(Number(event.target.value))} className="min-w-0 flex-1 accent-sky-500"/><span className="w-9 tabular-nums">{opacity}%</span></label>}
    <div className="mt-4 flex min-h-0 flex-1 flex-col">{viewer}</div>
    <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4" role="tablist" aria-label="Change evidence products">{visible.map(item => <button type="button" role="tab" key={`${item.label}-${item.url}`} onClick={() => setSelected(products.indexOf(item))} aria-selected={active.url === item.url} className={cn("rounded-xl border p-3 text-left transition", active.url === item.url ? "border-sky-400/50 bg-sky-400/[.08]" : "border-white/[.08] bg-white/[.02] hover:border-sky-400/25")}><span className="flex items-center justify-between gap-2 text-sm font-medium"><span>{item.label}</span>{active.url === item.url && <Check size={14} className="text-sky-300"/>}</span><span className={cn("mt-2 inline-flex rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide", item.contribution === "Primary" ? "bg-emerald-400/10 text-emerald-300" : "bg-white/[.05] text-zinc-500")}>{item.contribution}</span></button>)}</div>
    {products.length > 4 && <button type="button" onClick={() => setShowMore(value => !value)} className="mt-3 inline-flex items-center gap-2 self-start text-sm font-medium text-sky-300">{showMore ? <ChevronUp size={15}/> : <ChevronDown size={15}/>} {showMore ? "Show primary evidence only" : `View More Evidence (${products.length - 4})`}</button>}
  </section>;
}

function EmptyEvidence() {
  return <div className="result-empty-state"><Eye size={28}/><div><h3 className="font-semibold text-zinc-200">No visual evidence available</h3><p className="mt-1 text-sm text-zinc-500">This result did not publish an image preview. Review the audit record for the backend-provided reason and next action.</p></div></div>;
}

function RegionDigest({ values }: { values: Region[] }) {
  const [showAll, setShowAll] = useState(false);
  if (!values.length) return <div className="result-empty-state mt-4"><FileSearch size={24}/><div><h3 className="font-semibold text-zinc-200">No connected regions reported</h3><p className="mt-1 text-sm text-zinc-500">The authoritative result contains no region table for this workflow. No regions are inferred in the browser.</p></div></div>;
  const largest = values.reduce((current, item) => item.areaPixels > current.areaPixels ? item : current, values[0]);
  const average = values.reduce((sum, item) => sum + item.areaPercent, 0) / values.length;
  const displayed = showAll ? values : values.slice(0, 5);
  return <div className="mt-5">
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><MetricCard metric={{ label: "Largest region", value: percent(largest.areaPercent, 3), note: `${largest.areaPixels.toLocaleString()} px` }}/><MetricCard metric={{ label: "Total connected regions", value: values.length.toLocaleString() }}/><MetricCard metric={{ label: "Average area", value: percent(average, 3) }}/><MetricCard metric={{ label: "Suppressed small regions", value: "Not reported", note: "No browser assumption" }}/></div>
    <div className="mt-4 overflow-hidden rounded-2xl border border-white/[.08]"><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="bg-white/[.035] text-zinc-500"><tr><th className="px-4 py-3 font-medium">Region</th><th className="px-4 py-3 font-medium">Type</th><th className="px-4 py-3 font-medium">Area</th><th className="px-4 py-3 font-medium">Pixel bounds</th></tr></thead><tbody>{displayed.map(item => <tr key={`${item.type}-${item.id}`} className="border-t border-white/[.06]"><td className="px-4 py-3 text-zinc-400">{item.id}</td><td className="px-4 py-3 font-medium text-zinc-200">{item.type}</td><td className="px-4 py-3 text-zinc-400">{item.areaPixels.toLocaleString()} px · {percent(item.areaPercent, 3)}</td><td className="px-4 py-3 font-mono text-zinc-500">{item.bounds}</td></tr>)}</tbody></table></div></div>
    {values.length > 5 && <button type="button" onClick={() => setShowAll(value => !value)} aria-expanded={showAll} className="mt-3 inline-flex items-center gap-2 text-sm font-medium text-sky-300">{showAll ? <ChevronUp size={15}/> : <ChevronDown size={15}/>} {showAll ? "Show Top 5 Regions" : `Show All Regions (${values.length})`}</button>}
  </div>;
}

function MetricCard({ metric }: { metric: Metric }) {
  return <article className="result-metric-card"><p className="text-xs font-medium text-zinc-500">{metric.label}</p><p className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">{metric.value}</p>{metric.note && <p className="mt-1 text-[11px] leading-4 text-zinc-500">{metric.note}</p>}</article>;
}

export function AssistantResultExperience({ result, changeResult, query, onRerun, rerunning, children }: Props) {
  const [view, setView] = useState<"judge" | "research">("judge");
  const [auditOpen, setAuditOpen] = useState(false);
  const products = useMemo(() => evidenceProducts(result, changeResult), [changeResult, result]);
  const keyMetrics = useMemo(() => metrics(result, changeResult), [changeResult, result]);
  const regionValues = useMemo(() => regions(result, changeResult), [changeResult, result]);
  const limitations = useMemo(() => limitationBadges(result), [result]);
  const successful = result.status === "success" || result.status === "partial";
  const specialist = result.execution.selected_tools[result.execution.selected_tools.length - 1] ?? "No specialist selected";
  const answer = result.answer ?? (changeResult?.status === "success" ? "Deterministic change products generated." : "No direct answer was published.");
  const reasons = Array.from(new Set([
    ...result.evidence.map(item => item.description || item.label),
    result.execution.selection_reason,
    result.confidence.reason,
  ].filter((value): value is string => Boolean(value)))).slice(0, 6);
  const engine = result.change_engine ?? changeResult?.change_engine;
  const ttp = result.ttp_result ?? changeResult?.ttp_result;
  const consistency = result.evidence_consistency ?? changeResult?.evidence_consistency;
  const sve = result.sve_result ?? changeResult?.sve_result ?? result.cross_modal_analysis?.sve_result;

  return <section className="result-experience mt-10" aria-labelledby={`result-title-${result.request_id}`}>
    <div className="result-sticky-header">
      <div className="min-w-0 flex-1"><p className="eyebrow">Executive summary</p><p className="mt-1 truncate text-sm font-medium text-zinc-200" title={query}>{query}</p></div>
      <div className="flex flex-wrap items-center justify-end gap-2"><span className={cn("result-status-badge", result.status === "failed" ? "result-status-error" : successful ? "result-status-success" : "result-status-warning")}>{successful && <CheckCircle2 size={13}/>} {titleCase(result.result_status || result.status)}</span><span className="hidden text-xs tabular-nums text-zinc-500 lg:inline">{titleCase(result.confidence.level)} · {result.execution.duration_ms.toLocaleString()} ms</span><MissionReportActions result={result} onRerun={onRerun} rerunning={rerunning} toolbar/><Link href="/assistant/compare" className="result-secondary-button">Compare</Link></div>
    </div>

    <div className="mt-4 flex justify-end"><div className="result-view-toggle" role="group" aria-label="Result detail level"><button type="button" onClick={() => setView("judge")} aria-pressed={view === "judge"} className={cn(view === "judge" && "result-view-active")}><Sparkles size={14}/>Judge View</button><button type="button" onClick={() => setView("research")} aria-pressed={view === "research"} className={cn(view === "research" && "result-view-active")}><FileSearch size={14}/>Research View</button></div></div>

    <article className="result-hero mt-4">
      <div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><span className="result-kicker">Direct answer</span><span className={cn("result-status-badge", result.status === "failed" ? "result-status-error" : successful ? "result-status-success" : "result-status-warning")}>{titleCase(result.result_status || result.status)}</span>{result.primary_image_metadata?.representation === "display_preview" && result.primary_image_metadata.effective_modality === "sar_preview" && <span className="result-limitation-badge">Display Preview — Qualitative Analysis Only</span>}</div><h2 id={`result-title-${result.request_id}`} className="mt-5 max-w-5xl text-2xl font-semibold leading-tight tracking-tight text-zinc-50 sm:text-3xl lg:text-4xl">{answer}</h2><p className="mt-4 max-w-4xl text-sm leading-6 text-zinc-400"><span className="font-semibold text-zinc-200">Query:</span> {query}</p></div>
      <dl className="result-hero-facts"><div><dt>{result.sar_water_analysis ? "Heuristic reliability" : "Confidence"}</dt><dd>{result.sar_water_analysis?.heuristic_reliability != null ? `${(result.sar_water_analysis.heuristic_reliability * 100).toFixed(1)}% · not model confidence` : `${titleCase(result.confidence.level)}${result.confidence.score !== null ? ` · ${(result.confidence.score * 100).toFixed(1)}%` : ""}`}</dd></div><div><dt>Runtime</dt><dd>{result.execution.duration_ms.toLocaleString()} ms</dd></div><div><dt>Specialist</dt><dd>{titleCase(specialist)}</dd></div><div><dt>Task</dt><dd>{titleCase(result.task)}</dd></div><div><dt>Effective modality</dt><dd>{titleCase(result.primary_image_metadata?.effective_modality ?? "unknown")}</dd></div><div><dt>Representation</dt><dd>{titleCase(result.primary_image_metadata?.representation ?? "unknown")}</dd></div></dl>
      <p className="border-t border-white/[.07] pt-5 text-[11px] leading-5 text-zinc-500">PDF, JSON, and Full ZIP actions stay visible in the sticky result header and are generated from the authoritative backend record. Compare opens stored-result comparison.</p>
    </article>

    {sve && <div className="mt-4"><RemoteSensingAdaptationPanel result={sve}/></div>}

    {engine && <section className="result-section" aria-labelledby={`engine-${result.request_id}`}>
      <div className="result-section-heading"><div><p className="result-section-number">01</p><h2 id={`engine-${result.request_id}`}>Change Engine & Provenance</h2></div><p>Learned and deterministic evidence remain visibly source-labelled.</p></div>
      <div className="grid gap-4 lg:grid-cols-[.85fr_1.15fr]">
        <article className="result-explain-card"><h3><Layers3 size={18}/>{engine.mode === "hybrid" ? "Hybrid Change Analysis" : engine.mode === "ttp" ? "TTP Change Analysis" : engine.fallback_used ? "Deterministic Fallback" : "Deterministic Change Analysis"}</h3><dl className="mt-4 grid gap-3 text-sm"><div><dt className="text-zinc-500">Primary learned mask</dt><dd className="mt-1 text-zinc-200">{engine.primary_tool === "ttp_change_detector" ? "TTP" : "Unavailable"}</dd></div><div><dt className="text-zinc-500">Supporting evidence</dt><dd className="mt-1 text-zinc-200">{engine.supporting_tool ? "Deterministic analyzer" : "Deterministic analyzer is primary"}</dd></div><div><dt className="text-zinc-500">Automatic fallback</dt><dd className="mt-1 text-zinc-200">Enabled{engine.fallback_used ? ` · used (${titleCase(engine.fallback_reason ?? "unspecified")})` : " · not used"}</dd></div></dl></article>
        <article className="result-explain-card"><h3><ShieldCheck size={18}/>TTP provenance</h3>{ttp?.status === "success" ? <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2"><div><dt className="text-zinc-500">Model</dt><dd className="mt-1 text-zinc-200">{ttp.model}</dd></div><div><dt className="text-zinc-500">Architecture</dt><dd className="mt-1 text-zinc-200">{ttp.architecture}</dd></div><div><dt className="text-zinc-500">Training dataset</dt><dd className="mt-1 text-zinc-200">{ttp.training_dataset}</dd></div><div><dt className="text-zinc-500">Checkpoint</dt><dd className="mt-1 text-zinc-200">{ttp.checkpoint}</dd></div><div><dt className="text-zinc-500">Runtime device</dt><dd className="mt-1 text-zinc-200">{ttp.device ?? "Unavailable"}</dd></div><div><dt className="text-zinc-500">Model reuse</dt><dd className="mt-1 text-zinc-200">{ttp.reused_model ? "Warm model reused" : "Loaded for first inference"}</dd></div></dl> : <p className="mt-4 text-sm leading-6 text-amber-100">TTP evidence is unavailable for this result. The deterministic fallback reason is disclosed above.</p>}{consistency && <p className="mt-4 text-xs leading-5 text-zinc-400"><strong className="text-zinc-200">{consistency.label}.</strong> {consistency.disclosure}</p>}</article>
      </div>
    </section>}

    <section className="result-section" aria-labelledby={`evidence-${result.request_id}`}><div className="result-section-heading"><div><p className="result-section-number">02</p><h2 id={`evidence-${result.request_id}`}>Evidence</h2></div><p>Source imagery and the most relevant backend-produced evidence first.</p></div><EvidenceWorkbench products={products}/></section>

    <section className="result-section" aria-labelledby={`statistics-${result.request_id}`}><div className="result-section-heading"><div><p className="result-section-number">03</p><h2 id={`statistics-${result.request_id}`}>Key Statistics</h2></div><p>Measured values remain distinct from calibrated probabilities.</p></div><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-7">{keyMetrics.map(metric => <MetricCard key={metric.label} metric={metric}/>)}</div>{view === "research" && <RegionDigest values={regionValues}/>}</section>

    <section className="result-section" aria-labelledby={`why-${result.request_id}`}><div className="result-section-heading"><div><p className="result-section-number">04</p><h2 id={`why-${result.request_id}`}>Why This Answer?</h2></div><p>Published evidence, routing rationale, confidence, and limitations.</p></div><div className="grid gap-4 lg:grid-cols-[1.15fr_.85fr]"><article className="result-explain-card"><h3><ShieldCheck size={18}/>Backend-published rationale</h3>{reasons.length ? <ul>{reasons.map(reason => <li key={reason}><CheckCircle2 size={16}/><span>{reason}</span></li>)}</ul> : <p className="mt-4 text-sm text-zinc-500">No explanatory evidence text was published. The browser does not invent a rationale.</p>}</article><article className="result-explain-card"><h3><Activity size={18}/>Limitations</h3>{limitations.length ? <div className="mt-4 flex flex-wrap gap-2">{limitations.slice(0, view === "judge" ? 8 : limitations.length).map(item => <span key={item} className="result-limitation-badge">{item}</span>)}</div> : <p className="mt-4 text-sm text-zinc-500">No limitation text was published for this result.</p>}</article></div>
      <article className="result-decision-card"><div><span>Query</span><strong>{query}</strong></div><i>→</i><div><span>Task classification</span><strong>{titleCase(result.task)}</strong></div><i>→</i><div><span>Selected specialist</span><strong>{titleCase(specialist)}</strong></div><i>→</i><div><span>Reason</span><strong>{result.execution.selection_reason}</strong></div><i>→</i><div><span>Execution</span><strong>{titleCase(result.status)}</strong></div></article>
    </section>

    <section className="result-section" aria-labelledby={`timeline-${result.request_id}`}><div className="result-section-heading"><div><p className="result-section-number">05</p><h2 id={`timeline-${result.request_id}`}>Execution Timeline</h2></div><p>Observable backend steps with their published runtime.</p></div><ol className="result-timeline">{result.execution.steps.map((step, index) => <li key={`${step.tool}-${index}`}><span className={cn(step.status === "success" ? "result-timeline-success" : step.status === "failed" ? "result-timeline-error" : "result-timeline-warning")}>{step.status === "success" ? <Check size={15}/> : index + 1}</span><div><strong>{timelineLabel(step.tool)}</strong><small>{step.duration_ms.toLocaleString()} ms</small></div></li>)}</ol></section>

    {view === "research" && <section className="result-section" aria-labelledby={`audit-${result.request_id}`}><button type="button" onClick={() => setAuditOpen(value => !value)} aria-expanded={auditOpen} aria-controls={`audit-content-${result.request_id}`} className="result-audit-toggle"><span><span className="result-section-number">AUDIT</span><strong id={`audit-${result.request_id}`}>Complete scientific record</strong><small>Metadata, provenance, parameters, confidence, warnings, full region tables, and specialist output.</small></span>{auditOpen ? <ChevronUp size={20}/> : <ChevronDown size={20}/>}</button>{auditOpen && <div id={`audit-content-${result.request_id}`} className="result-audit-content"><p className="mb-4 flex items-center gap-2 text-xs text-zinc-500"><Route size={14}/>Request ID: <span className="break-all font-mono">{result.request_id}</span></p>{children}</div>}</section>}
  </section>;
}

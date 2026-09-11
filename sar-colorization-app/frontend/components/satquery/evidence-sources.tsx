"use client";

import { CheckCircle2, CircleAlert, Database, RotateCcw } from "lucide-react";
import type { AgentResponse, ChangeAnalysisResponse } from "@/types/agent";
import { specialistLabel } from "@/lib/scientific-presentation";
import { cn } from "@/lib/utils";
import { authoritativeTranslatedEvidence } from "@/lib/evidence-lifecycle";

type SourceStatus = "used" | "available" | "skipped" | "fallback" | "failed";
type Source = { id: string; label: string; role: string; model: string; status: SourceStatus; disclosure?: string };

const titleCase = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());

function sourceLabel(tool: string) {
  const lower = tool.toLowerCase();
  if (lower.includes("ground")) return "Grounding DINO";
  if (lower.includes("rsvqa") || lower.includes("vqa")) return "Visual question answering";
  if (lower.includes("caption")) return "Remote-sensing captioner";
  if (lower.includes("sve") || lower.includes("vision_encoder")) return "SatQuery Vision Encoder";
  if (lower.includes("changerex")) return "ChangerEx Change Detector";
  if (lower.includes("ttp")) return "TTP change detector";
  if (lower.includes("deterministic") && lower.includes("change")) return "Deterministic Change Analyzer";
  if (lower.includes("change")) return "Change specialist";
  if (lower.includes("cross")) return "Cross-modal analyzer";
  if (lower.includes("sar") && lower.includes("water")) return "Native SAR analysis";
  return titleCase(tool.replace(/^rs_/, ""));
}

export function EvidenceSources({ result, changeResult }: { result: AgentResponse; changeResult?: ChangeAnalysisResponse }) {
  const byId = new Map<string, Source>();
  for (const tool of result.execution.selected_tools) {
    if (["upload", "validator", "router", "metadata"].some(value => tool.toLowerCase().includes(value))) continue;
    const label = sourceLabel(tool);
    byId.set(label, { id: label, label, role: "Agent-selected workflow source", model: specialistLabel(tool), status: "used" });
  }
  for (const step of result.execution.steps.filter(item => item.status === "failed")) {
    if (step.tool.startsWith("sar_translation_") || step.tool.startsWith("translated_optical_") || step.tool === "translation_semantic_comparison") continue;
    const label = sourceLabel(step.tool);
    byId.set(label, { id: label, label, role: "Attempted workflow source", model: titleCase(step.tool), status: "failed" });
  }
  if (result.sve_result) byId.set("SatQuery Vision Encoder", { id: "sve", label: "SatQuery Vision Encoder", role: "Remote-sensing scene evidence", model: result.sve_result.model, status: result.sve_result.status === "success" || result.sve_result.status === "ready" ? "used" : result.sve_result.fallback ? "fallback" : "failed", disclosure: result.sve_result.disclaimer });
  if (result.vqa_details) {
    const label = result.task === "change_vqa" || result.task === "change_description" ? "Controlled change answer" : "Visual question answering";
    byId.set(label, { id: "vqa", label, role: "Primary answer source", model: titleCase(result.vqa_details.answer_source), status: "used", disclosure: result.vqa_details.supported ? undefined : "Question family unsupported" });
  }
  for (const role of ["optical", "sar"] as const) {
    if (result.cross_modal_analysis?.evidence_products?.some(product => product.evidence_type === "native_evidence" && product.source_role === role)) {
      byId.set(`native-${role}`, { id: `native-${role}`, label: role === "sar" ? "Native SAR analysis" : "Native optical analysis", role: "Primary observed-image evidence", model: role === "sar" ? "Relative intensity and texture candidates" : "Visible-spectrum, edge and texture candidates", status: "used", disclosure: "Candidate support; not confirmed semantic classes" });
    }
  }
  if (result.sar_water_analysis || result.sar_scene_analysis) byId.set("native-sar", { id: "native-sar", label: "Native SAR analysis", role: "Primary observed-image evidence", model: result.sar_water_analysis?.method ?? result.sar_scene_analysis?.method ?? "Native SAR evidence", status: "used" });
  if (result.sar_translated_optical_analysis) {
    const translated = result.sar_translated_optical_analysis;
    const product = authoritativeTranslatedEvidence(result);
    const status: SourceStatus = translated.generation_state === "FAILED" ? "failed" : translated.generation_state === "NOT_ELIGIBLE" || translated.generation_state === "NOT_REQUESTED" ? "skipped" : product ? "used" : "available";
    byId.set("translated-sar", { id: "translated-sar", label: `${translated.model ?? "SAR translation"}-generated representation`, role: "Secondary interpretive evidence", model: translated.model ?? "Model unavailable", status, disclosure: product ? "Not observed optical imagery" : "No generated evidence product published" });
    if (translated.semantic_comparison_state === "FAILED") byId.set("translated-semantics", { id: "translated-semantics", label: "Translation semantic comparison", role: "Optional downstream interpretation", model: translated.model ?? "Translation model", status: "failed", disclosure: "No failed semantic claim was used" });
  }
  if (result.grounding_result) byId.set("grounding", { id: "grounding", label: "Grounding DINO", role: "Model-produced localization", model: result.grounding_result.model.base_architecture, status: "used" });
  const engine = result.change_engine ?? changeResult?.change_engine;
  if (engine?.fallback_used) byId.set("change-fallback", { id: "change-fallback", label: "Deterministic Change Analyzer", role: "Automatic fallback evidence", model: specialistLabel(engine.primary_tool), status: "fallback", disclosure: engine.fallback_reason ? titleCase(engine.fallback_reason) : undefined });

  const sources = Array.from(byId.values());
  if (!sources.length) return null;
  return <section className="result-section" aria-labelledby={`sources-${result.request_id}`}>
    <div className="result-section-heading"><div><p className="result-section-number">SOURCES</p><h2 id={`sources-${result.request_id}`}>Evidence Sources</h2></div><p>Only sources published by the completed workflow are listed.</p></div>
    <div className="sq-source-grid">{sources.map(source => {
      const Icon = source.status === "failed" ? CircleAlert : source.status === "fallback" ? RotateCcw : CheckCircle2;
      return <article key={source.id} className={cn("sq-source-card", `is-${source.status}`)}>
        <span><Icon size={16}/></span>
        <div><h3>{source.label}</h3><p>{source.role}</p><small>{source.model}{source.disclosure ? ` · ${source.disclosure}` : ""}</small></div>
        <em>{source.status}</em>
      </article>;
    })}</div>
    <p className="sq-source-disclosure"><Database size={13}/>Evidence participation is derived from the response execution trace and published result fields.</p>
  </section>;
}

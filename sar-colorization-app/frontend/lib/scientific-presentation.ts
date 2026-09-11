import type { AgentResponse, ChangeAnalysisResponse, ChangeEngine, TTPResult } from "@/types/agent";

export const titleCase = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());

export function changeEngineName(engine?: ChangeEngine | null, learned?: TTPResult | null): "ChangerEx" | "TTP" | "Learned change detector" {
  const identity = `${engine?.mode ?? ""} ${engine?.primary_tool ?? ""} ${engine?.fallback_reason ?? ""} ${learned?.model ?? ""} ${learned?.architecture ?? ""}`.toLowerCase();
  if (identity.includes("ttp")) return "TTP";
  if (identity.includes("changer")) return "ChangerEx";
  return engine ? "ChangerEx" : "Learned change detector";
}

export function specialistLabel(tool: string): string {
  const value = tool.toLowerCase();
  if (value.includes("changerex")) return "ChangerEx Change Detector";
  if (value.includes("ttp")) return "TTP Change Detector";
  if (value.includes("deterministic") && value.includes("change")) return "Deterministic Change Analyzer";
  if (value.includes("ground")) return "Grounding Specialist";
  if (value.includes("rsvqa") || value.includes("vqa")) return "Remote Sensing VQA Specialist";
  if (value.includes("vision_encoder") || value === "sve") return "SatQuery Vision Encoder";
  if (value.includes("caption")) return "Remote Sensing Captioner";
  return titleCase(tool);
}

const infrastructureTool = (tool: string) => /upload|ingest|valid|metadata|router|classif|report|response|cache|preview/i.test(tool);

export function presentedSpecialists(result: AgentResponse, change?: ChangeAnalysisResponse) {
  const engine = result.change_engine ?? change?.change_engine;
  const learned = result.ttp_result ?? change?.ttp_result;
  if (engine) {
    if (engine.fallback_used || engine.primary_tool === "deterministic_change_analyzer") {
      return {
        primary: "Deterministic Change Analyzer",
        supporting: [] as string[],
        fallbackReason: `${changeEngineName(engine, learned)} unavailable${engine.fallback_reason ? ` · ${titleCase(engine.fallback_reason)}` : ""}`,
      };
    }
    const primary = `${changeEngineName(engine, learned)} Change Detector`;
    const supporting = engine.supporting_tool ? [specialistLabel(engine.supporting_tool)] : [];
    return { primary, supporting, fallbackReason: null };
  }
  const tools = result.execution.selected_tools.filter(tool => !infrastructureTool(tool));
  const primaryTool = tools.find(tool => !/deterministic/i.test(tool)) ?? tools[0];
  const primary = primaryTool ? specialistLabel(primaryTool) : "No specialist selected";
  return { primary, supporting: tools.filter(tool => tool !== primaryTool).map(specialistLabel), fallbackReason: null };
}

export function workflowLabel(inputMode: string): string {
  if (inputMode === "single") return "Single Image";
  if (inputMode === "cross_modal") return "Optical + SAR";
  if (inputMode === "bi_temporal") return "Bi-temporal";
  return titleCase(inputMode);
}

export function learnedEvidenceLabel(engine?: ChangeEngine | null, learned?: TTPResult | null) {
  return `${changeEngineName(engine, learned)} learned mask`;
}

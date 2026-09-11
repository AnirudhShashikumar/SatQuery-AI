import type { AgentResponse, DemoManifest, DemoWorkflow } from "@/types/agent";

export type SingleImagePresentationAction = "captioning" | "vqa";

export const singleImageDefaultQueries: Record<SingleImagePresentationAction, string> = {
  captioning: "Describe the land cover and major objects visible in this image.",
  vqa: "What is the dominant land-cover type?",
};

export const singleImageActionDetails = {
  captioning: {
    label: "Describe Scene",
    method: "Learned remote-sensing-adapted vision-language model",
    toolId: "rs_captioner",
  },
  vqa: {
    label: "Ask a Question",
    method: "Deterministic evidence-grounded question answering",
    toolId: "rs_vqa",
  },
} as const satisfies Record<SingleImagePresentationAction, { label: string; method: string; toolId: string }>;

export type EvidencePreview = { label: string; path: string };

export function approvedSingleImageWorkflow(manifest: DemoManifest, workflowId: string): DemoWorkflow {
  if (!manifest.enabled) throw new Error("Local demo mode is disabled. Start the backend with SATQUERY_DEMO_MODE=true to use the approved presentation sample.");
  const workflow = manifest.workflows.find(item => item.id === workflowId);
  if (!workflow || workflow.input_mode !== "single") throw new Error("The approved single-image demo sample is unavailable from the local manifest.");
  if (!workflow.files.some(file => file.role === "primary")) throw new Error("The approved single-image workflow does not provide a primary image.");
  return workflow;
}

export function vqaEvidencePreviews(response: AgentResponse): EvidencePreview[] {
  const previews = response.vqa_details?.single_image_evidence?.previews;
  if (!previews) return [];
  const candidates: Array<{ label: string; path: string | null }> = [
    { label: "Water support", path: previews.water_support },
    { label: "Vegetation support", path: previews.vegetation_support },
    { label: "Structural support", path: previews.built_up_support },
    { label: "Agriculture support", path: previews.agriculture_support },
    { label: "Combined evidence overlay", path: previews.combined_overlay },
  ];
  return candidates.filter((item): item is EvidencePreview => Boolean(item.path));
}

export function responseFailureMessage(response: AgentResponse, action: SingleImagePresentationAction): string | null {
  if (response.status === "success" || response.status === "partial") return null;
  const backendMessage = response.answer || response.warnings[0] || response.confidence.reason;
  if (action === "captioning" && /checkpoint|captioner unavailable|not cached|unavailable/i.test(backendMessage)) {
    return "Captioning is unavailable because the local checkpoint is not cached or could not be loaded. Switch to Ask a Question to continue with the deterministic offline VQA workflow.";
  }
  if (action === "vqa") return backendMessage || "The controlled VQA workflow could not produce a supported answer. Review the question and backend warnings.";
  return backendMessage || "The selected local specialist did not produce a result.";
}

export function formatPresentationLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

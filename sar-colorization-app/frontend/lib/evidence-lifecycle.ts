import type { AgentResponse } from "@/types/agent";

export function authoritativeTranslatedEvidence(result: AgentResponse) {
  const translated = result.sar_translated_optical_analysis;
  if (!translated || translated.generation_state !== "SUCCEEDED") return undefined;
  return translated.evidence_products?.find(product =>
    product.status === "SUCCEEDED"
    && product.evidence_type === "generated_optical_like"
    && product.generator === translated.model
    && product.evidence_run_id === result.request_id
    && product.source_observation_id === result.primary_image_metadata?.file_id
    && product.source_modality?.startsWith("sar")
    && Boolean(product.reference)
  );
}

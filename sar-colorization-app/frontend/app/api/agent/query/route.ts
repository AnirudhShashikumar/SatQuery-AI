import { NextResponse } from "next/server";
import {
  buildSpaceCall,
  parseJsonOutput,
  rewriteArtifactReferences,
  SpaceContractError,
} from "@/lib/hf-space-contract";
import { artifactProxyUrl, callSpace, proxyErrorBody, SpaceProxyError } from "@/lib/server/hf-space";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

type ResultEnvelope = {
  ok: boolean;
  status_code: number;
  response?: unknown;
  error?: unknown;
};

function envelope(value: unknown): ResultEnvelope {
  if (!value || typeof value !== "object") {
    throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space returned an invalid result envelope.");
  }
  const candidate = value as Partial<ResultEnvelope>;
  if (typeof candidate.ok !== "boolean" || typeof candidate.status_code !== "number") {
    throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space returned an invalid result envelope.");
  }
  return candidate as ResultEnvelope;
}

export async function POST(request: Request) {
  try {
    if (!request.headers.get("content-type")?.toLowerCase().startsWith("multipart/form-data")) {
      throw new SpaceContractError("Expected multipart/form-data.", 415);
    }
    const call = buildSpaceCall(await request.formData());
    const outputs = await callSpace(call.endpoint, call.payload);
    if (outputs.length < 3) {
      throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space response is incomplete.");
    }

    const result = envelope(parseJsonOutput(outputs[0]));
    if (!result.ok) {
      const status = Number.isInteger(result.status_code) && result.status_code >= 400 && result.status_code <= 599
        ? result.status_code
        : 502;
      return NextResponse.json({ detail: result.error ?? { code: "INFERENCE_FAILED", message: "Inference failed." } }, { status });
    }
    if (result.response === undefined) {
      throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space response payload is missing.");
    }

    const response = rewriteArtifactReferences(
      result.response,
      parseJsonOutput(outputs[1]),
      outputs[2],
      artifactProxyUrl,
    );
    return NextResponse.json(response, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof SpaceContractError) {
      return NextResponse.json({ detail: { code: "INVALID_REQUEST", message: error.message } }, { status: error.status });
    }
    const failure = error instanceof SpaceProxyError
      ? error
      : new SpaceProxyError(502, "SPACE_REQUEST_FAILED", "The Hugging Face Space request failed.");
    return NextResponse.json(proxyErrorBody(failure), { status: failure.status });
  }
}

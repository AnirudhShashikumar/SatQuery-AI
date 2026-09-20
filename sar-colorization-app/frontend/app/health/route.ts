import { NextResponse } from "next/server";
import { parseJsonOutput } from "@/lib/hf-space-contract";
import { callSpace, proxyErrorBody, SpaceProxyError } from "@/lib/server/hf-space";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET() {
  try {
    const outputs = await callSpace("/health", {}, "health");
    const health = parseJsonOutput(outputs[0]);
    if (!health || typeof health !== "object") {
      throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space returned invalid health data.");
    }
    const payload = health as { status?: unknown; backend?: unknown };
    if (!payload.backend || typeof payload.backend !== "object") {
      return NextResponse.json(
        { detail: { code: "SPACE_INITIALIZING", message: "The ZeroGPU Space is initializing." } },
        { status: 503, headers: { "Cache-Control": "no-store" } },
      );
    }
    return NextResponse.json(payload.backend, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const failure = error instanceof SpaceProxyError
      ? error
      : new SpaceProxyError(502, "SPACE_HEALTH_FAILED", "The Space health check failed.");
    return NextResponse.json(proxyErrorBody(failure), { status: failure.status });
  }
}

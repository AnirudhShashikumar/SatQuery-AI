import { NextResponse } from "next/server";
import {
  proxyErrorBody,
  SpaceProxyError,
  spaceAuthorizationHeaders,
  verifiedArtifactUrl,
} from "@/lib/server/hf-space";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(request: Request) {
  try {
    const requestUrl = new URL(request.url);
    const remoteUrl = verifiedArtifactUrl(
      requestUrl.searchParams.get("source"),
      requestUrl.searchParams.get("signature"),
    );
    const upstream = await fetch(remoteUrl, {
      cache: "no-store",
      headers: spaceAuthorizationHeaders(),
      redirect: "error",
      signal: AbortSignal.timeout(45_000),
    });
    if (!upstream.ok || !upstream.body) {
      const status = upstream.status === 404 || upstream.status === 410 ? 404 : 502;
      throw new SpaceProxyError(status, "ARTIFACT_UNAVAILABLE", "The evidence artifact is unavailable.");
    }

    const headers = new Headers({
      "Cache-Control": "private, no-store, max-age=0",
      "Content-Type": upstream.headers.get("content-type") ?? "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
    });
    const length = upstream.headers.get("content-length");
    if (length) headers.set("Content-Length", length);
    return new NextResponse(upstream.body, { status: 200, headers });
  } catch (error) {
    const failure = error instanceof SpaceProxyError
      ? error
      : new SpaceProxyError(502, "ARTIFACT_REQUEST_FAILED", "The evidence artifact request failed.");
    return NextResponse.json(proxyErrorBody(failure), { status: failure.status });
  }
}

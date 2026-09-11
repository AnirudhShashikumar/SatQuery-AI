import { NextResponse } from "next/server";

const buildInfo = Object.freeze({
  app: "SatQuery AI",
  build_id: process.env.NEXT_PUBLIC_SATQUERY_BUILD_ID ?? "development",
  git_sha: process.env.NEXT_PUBLIC_SATQUERY_GIT_SHA ?? "unknown",
  api_url: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010",
});

export const dynamic = "force-static";
export const revalidate = false;

export function GET() {
  return NextResponse.json(buildInfo, {
    headers: {
      "Cache-Control": "no-store, max-age=0",
    },
  });
}

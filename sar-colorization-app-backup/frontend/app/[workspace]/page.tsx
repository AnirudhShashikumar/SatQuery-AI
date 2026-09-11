import type { Metadata } from "next";
import { Workspace } from "@/components/workspace";

const routeTitles: Record<string, string> = {
  assistant: "Assistant",
  pix2pix: "Pix2Pix",
  structure: "SARFusionFormer",
  comparison: "Model Comparison",
  architecture: "Architecture",
  benchmark: "Benchmarks",
  reports: "Reports",
  settings: "AI Providers",
  api: "Local API",
  logs: "Execution Logs",
  about: "About",
};

export async function generateMetadata({ params }: { params: Promise<{ workspace: string }> }): Promise<Metadata> {
  const { workspace } = await params;
  const title = routeTitles[workspace];
  return title ? { title } : { title: { absolute: "SatQuery AI" } };
}

export default function WorkspacePage() { return <Workspace />; }

"use client";
import { Suspense, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Activity, Braces, CheckCircle2, Clock3, MessageSquarePlus } from "lucide-react";
import { ProductShell } from "@/components/product-shell";
import { Pix2PixWorkspace } from "@/components/pix2pix-workspace";
import { StructureWorkspace } from "@/components/structure-workspace";
import { ComparisonWorkspace } from "@/components/comparison-workspace";
import { ProviderSettings } from "@/components/provider-settings";
import { ArchitectureView } from "@/components/architecture-view";
import { BenchmarkView } from "@/components/benchmark-view";
import { ReportsCenter } from "@/components/reports-center";
import { AssistantWorkspace } from "@/components/assistant-workspace";
import { PresentationReturnBanner } from "@/components/presentation-page-handoff";

const applications = ["Flood Monitoring", "Disaster Response", "Agriculture", "Urban Mapping", "Military Intelligence", "Forest Monitoring", "Climate Research", "Infrastructure Planning", "Coastal Monitoring"];

function ResearchView({ page }: { page: string }) {
 if (page === "architecture") return <ArchitectureView/>;
 if (page === "benchmark") return <BenchmarkView/>;
 if (page === "reports") return <ReportsCenter/>;
 if (page === "api" || page === "logs") return <DeveloperView page={page}/>;
 return <section><p className="eyebrow">Earth intelligence workspace</p><h1 className="mt-3 text-5xl font-semibold">SatQuery AI</h1><p className="muted mt-5 max-w-3xl">Ask Earth Anything.</p><h2 className="mt-10 text-xl font-semibold">Applications</h2><div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{applications.map(x => <div key={x} className="panel p-5 transition hover:-translate-y-1"><h3 className="font-medium">{x}</h3><p className="mt-2 text-sm text-zinc-500">Multimodal remote sensing intelligence for mission-critical analysis.</p></div>)}</div></section>;
}

function DeveloperView({ page }: { page: "api" | "logs" }) {
 const [latestRequest, setLatestRequest] = useState("");
 useEffect(() => setLatestRequest(window.sessionStorage.getItem("satquery-latest-request-id") ?? ""), []);
 if (page === "api") return <section className="max-w-6xl"><p className="eyebrow">Developer workspace</p><h1 className="mt-3 text-5xl font-semibold tracking-tight">Local API</h1><p className="muted mt-4 max-w-3xl">The same evidence-first workflows behind the Assistant are available through the local FastAPI service.</p><div className="mt-8 grid gap-4 md:grid-cols-2">{[["POST", "/api/analysis/image", "Run routed single, cross-modal, grounding, and temporal image analysis."], ["POST", "/api/agent/inspect", "Inspect image representation and modality before specialist execution."], ["GET", "/health", "Read local runtime and model availability."], ["GET", "/api/agent/demo-manifest", "List approved local demonstration workflows."]].map(([method, endpoint, copy]) => <article key={endpoint} className="panel p-5"><div className="flex items-center justify-between gap-3"><span className="rounded-md border border-sky-300/20 bg-sky-300/[.06] px-2 py-1 font-mono text-xs text-sky-200">{method}</span><Braces size={17} className="text-zinc-600"/></div><h2 className="mt-4 break-all font-mono text-sm text-zinc-100">{endpoint}</h2><p className="mt-2 text-sm leading-6 text-zinc-500">{copy}</p></article>)}</div></section>;
 return <section className="max-w-6xl"><p className="eyebrow">Developer workspace</p><h1 className="mt-3 text-5xl font-semibold tracking-tight">Execution logs</h1><p className="muted mt-4 max-w-3xl">Inspect the latest request context without exposing imagery or model internals outside the local runtime.</p>{latestRequest ? <article className="panel mt-8 p-5"><div className="flex flex-wrap items-center justify-between gap-3"><span className="flex items-center gap-2 text-sm font-semibold"><Activity size={16} className="text-sky-300"/>Latest Assistant request</span><span className="flex items-center gap-1.5 text-xs text-emerald-300"><CheckCircle2 size={14}/>Completed</span></div><dl className="mt-5 grid gap-3 sm:grid-cols-2"><div className="glass rounded-xl p-4"><dt className="text-xs text-zinc-500">Request ID</dt><dd className="mt-2 break-all font-mono text-xs text-zinc-200">{latestRequest}</dd></div><div className="glass rounded-xl p-4"><dt className="flex items-center gap-1.5 text-xs text-zinc-500"><Clock3 size={13}/>Persistence</dt><dd className="mt-2 text-sm text-zinc-200">Current browser session</dd></div></dl></article> : <div className="mt-8 grid justify-items-center rounded-2xl border border-dashed border-white/[.09] px-6 py-16 text-center"><span className="grid h-14 w-14 place-items-center rounded-2xl border border-sky-300/15 bg-sky-300/[.04] text-sky-300"><Activity size={22}/></span><h2 className="mt-5 text-lg font-semibold">No request in this session</h2><p className="mt-2 max-w-md text-sm leading-6 text-zinc-500">Run an Assistant analysis to populate the latest local execution record.</p><Link href="/assistant" className="mt-5 inline-flex items-center gap-2 rounded-xl border border-sky-300/20 bg-sky-300/[.05] px-4 py-2.5 text-sm text-sky-100"><MessageSquarePlus size={15}/>Start a conversation</Link></div>}</section>;
}
export function Workspace() { const params = useParams<{ workspace: string }>(); const content = params.workspace === "assistant" ? <Suspense fallback={<div className="panel h-64 animate-pulse" aria-label="Loading Assistant"/>}><AssistantWorkspace/></Suspense> : params.workspace === "structure" ? <StructureWorkspace/> : params.workspace === "comparison" ? <ComparisonWorkspace/> : params.workspace === "pix2pix" ? <Pix2PixWorkspace/> : params.workspace === "settings" ? <ProviderSettings/> : <ResearchView page={params.workspace}/>; return <ProductShell><PresentationReturnBanner route={`/${params.workspace}`}/>{content}</ProductShell>; }

"use client";

import { AlertTriangle, CheckCircle2, Database, FileArchive, FileCheck2, LoaderCircle, RefreshCw, ShieldCheck, Wrench } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { PresentationOpenPageButton } from "@/components/presentation-page-handoff";
import { analyticsLabel } from "@/lib/analytics";
import { getAgentAnalytics, getAgentCompliance, getAgentTools } from "@/services/api";
import type { AnalyticsResponse, ComplianceResponse, ToolDefinition } from "@/types/agent";

type ComplianceData = { compliance: ComplianceResponse; tools: ToolDefinition[]; analytics: AnalyticsResponse };
const statusTone = (status: string) => status === "available" || status === "optional_available" ? "border-emerald-300/25 text-emerald-200" : "border-amber-300/25 text-amber-100";

export function PresentationComplianceScene({ onContinue }: { onContinue: () => void }) {
  const [data, setData] = useState<ComplianceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const mountedRef = useRef(true);
  const sequenceRef = useRef(0);
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; sequenceRef.current += 1; }; }, []);
  const load = useCallback(async () => {
    const sequence = ++sequenceRef.current; setLoading(true); setError("");
    try {
      const [compliance, tools, analytics] = await Promise.all([getAgentCompliance(), getAgentTools(), getAgentAnalytics()]);
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      setData({ compliance, tools, analytics });
    } catch (caught) {
      if (mountedRef.current && sequence === sequenceRef.current) setError(caught instanceof Error ? caught.message : "Compliance data is unavailable.");
    } finally { if (mountedRef.current && sequence === sequenceRef.current) setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const requirement = (name: string) => data?.compliance.requirements.find(item => item.requirement.toLowerCase().includes(name));
  const grounding = requirement("grounding");
  const evidence = requirement("evidence");
  const confidence = requirement("confidence");
  const trace = requirement("execution trace");
  const reports = requirement("mission report");
  const adaptation = requirement("remote-sensing-adapted");
  const formats = data ? Array.from(new Set(data.analytics.capabilities.flatMap(item => item.evidence.match(/GeoTIFF|TIFF|PNG|JPEG/gi) ?? []).map(value => value.toUpperCase().replace("GEOTIFF", "GeoTIFF")))) : [];
  const reportFormats = reports ? ["PDF", "JSON", "CSV", "ZIP"].filter(format => reports.implementation.includes(format)) : [];
  const workflows = data ? Array.from(new Set(data.tools.filter(tool => tool.status === "available").flatMap(tool => tool.supported_tasks).filter(task => task !== "unsupported"))).map(analyticsLabel) : [];
  const modalities = data ? Array.from(new Set(data.tools.filter(tool => tool.status === "available").flatMap(tool => tool.supported_modalities))).map(analyticsLabel) : [];

  return <section className="presentation-compliance-scene mt-4 min-h-0 flex-1" aria-label="SIH Compliance presentation summary">
    <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-[9px] font-bold uppercase tracking-[.18em] text-sky-300">Backend-authoritative readiness</p><div className="flex flex-wrap gap-2"><button type="button" onClick={() => void load()} disabled={loading} className="presentation-summary-secondary"><RefreshCw size={12} className={loading ? "animate-spin" : ""}/>Refresh</button><PresentationOpenPageButton route="/assistant/compliance" stageIndex={13}>Open Full Compliance</PresentationOpenPageButton><button type="button" onClick={onContinue} className="presentation-summary-secondary">Continue</button></div></div>
    {loading && !data && <ComplianceState icon={LoaderCircle} title="Loading SIH readiness" message="Reading the authoritative compliance matrix and public registry." loading/>}
    {error && !data && <ComplianceState icon={AlertTriangle} title="Compliance unavailable" message={`${error} No readiness values are substituted.`}/>} 
    {data && <div className="presentation-compliance-layout mt-3 grid min-h-0 gap-3 lg:grid-cols-[minmax(240px,.7fr)_minmax(0,1.3fr)]">
      <article className="presentation-summary-panel min-h-0 rounded-2xl border p-3"><div className="flex items-center justify-between gap-2"><div><p className="text-[8px] font-bold uppercase tracking-[.16em] text-zinc-500">SIH compliance badge</p><h2 className="mt-1 text-sm font-semibold">Mandatory readiness</h2></div><span className="grid h-10 w-10 place-items-center rounded-xl border border-emerald-300/25 bg-emerald-300/[.07] text-emerald-200"><ShieldCheck size={20}/></span></div><div className="mt-3 rounded-2xl border border-emerald-300/20 bg-emerald-300/[.05] p-3 text-center"><p className="text-3xl font-semibold text-emerald-200">{data.compliance.mandatory_satisfied}/{data.compliance.mandatory_total}</p><p className="mt-1 text-[9px] uppercase tracking-[.14em] text-emerald-100/70">Mandatory</p></div><dl className="mt-2 grid grid-cols-2 gap-1.5"><ComplianceFact icon={Wrench} label="Registry" value={`${data.tools.filter(tool => tool.status === "available").length}/${data.tools.length} available`}/><ComplianceFact icon={CheckCircle2} label="Grounding optional" value={grounding ? analyticsLabel(grounding.status) : "Not declared"}/><ComplianceFact icon={Database} label="Modalities" value={modalities.join(" · ") || "Not declared"}/><ComplianceFact icon={FileArchive} label="Formats" value={formats.join(" · ") || "Not declared"}/></dl><p className="mt-2 text-[8px] leading-3 text-zinc-500">Generated {new Date(data.compliance.generated_at).toLocaleString()} · {data.compliance.project.replace(/GeoVision/gi, "SatQuery AI")}</p></article>
      <article className="presentation-summary-panel flex min-h-0 flex-col rounded-2xl border p-3"><div><p className="text-[8px] font-bold uppercase tracking-[.16em] text-zinc-500">Supported and auditable scope</p><h2 className="mt-1 text-sm font-semibold">Evidence-backed requirement summary</h2></div><div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4">{[
        ["Downloadable reports", reports, reportFormats.join(" · ") || reports?.implementation],
        ["Evidence", evidence, evidence?.implementation],
        ["Confidence", confidence, confidence?.implementation],
        ["Execution trace", trace, trace?.implementation],
        ["RS adaptation", adaptation, adaptation?.implementation],
      ].map(([label, item, value]) => { const row = item as ReturnType<typeof requirement>; return <div key={label as string} className="min-w-0 rounded-xl border border-white/[.06] bg-white/[.02] p-2"><p className="text-[7px] uppercase tracking-wide text-zinc-600">{label as string}</p><span className={`mt-1 inline-flex rounded-full border px-1.5 py-0.5 text-[7px] ${statusTone(row?.status ?? "unavailable")}`}>{row ? analyticsLabel(row.status) : "Not declared"}</span><p className="mt-1 line-clamp-2 text-[8px] leading-3 text-zinc-400" title={String(value ?? "Not declared")}>{String(value ?? "Not declared")}</p></div>; })}</div><div className="mt-2 grid min-h-0 flex-1 gap-2 sm:grid-cols-2"><div className="min-h-0 rounded-xl border border-white/[.06] p-2"><p className="text-[8px] uppercase tracking-wide text-zinc-500">Supported workflows</p><div className="mt-1.5 flex max-h-24 flex-wrap gap-1 overflow-auto">{workflows.map(value => <span key={value} className="rounded-lg bg-sky-400/[.07] px-2 py-1 text-[8px] text-sky-200">{value}</span>)}</div></div><div className="min-h-0 rounded-xl border border-white/[.06] p-2"><p className="text-[8px] uppercase tracking-wide text-zinc-500">Declared limitations remain visible</p><ul className="mt-1.5 max-h-24 space-y-1 overflow-auto text-[8px] leading-3 text-zinc-500">{data.compliance.requirements.filter(item => item.limitation).slice(0, 6).map(item => <li key={item.requirement}>• {item.requirement}: {item.limitation}</li>)}</ul></div></div></article>
    </div>}
  </section>;
}

function ComplianceFact({ icon: Icon, label, value }: { icon: typeof FileCheck2; label: string; value: string }) { return <div className="min-w-0 rounded-xl border border-white/[.05] bg-white/[.02] p-2"><dt className="flex items-center gap-1 text-[7px] uppercase tracking-wide text-zinc-600"><Icon size={9}/>{label}</dt><dd className="mt-1 line-clamp-2 text-[8px] font-medium leading-3 text-zinc-300" title={value}>{value}</dd></div>; }
function ComplianceState({ icon: Icon, title, message, loading = false }: { icon: typeof ShieldCheck; title: string; message: string; loading?: boolean }) { return <div className="presentation-summary-panel mt-3 grid min-h-56 place-items-center rounded-2xl border p-5 text-center"><div><Icon className={`mx-auto ${loading ? "animate-spin text-sky-300" : "text-rose-200"}`} size={20}/><h2 className="mt-2 text-sm font-semibold">{title}</h2><p className="mt-1 text-[10px] text-zinc-500">{message}</p></div></div>; }

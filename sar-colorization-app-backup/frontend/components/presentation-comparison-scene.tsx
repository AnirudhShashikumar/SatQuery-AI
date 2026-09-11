"use client";

import { AlertTriangle, CheckCircle2, Clock3, GitCompareArrows, LoaderCircle, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { comparabilityLabel, presentationComparisonError, storePresentationComparisonHandoff } from "@/lib/presentation-comparison";
import { presentationRuntimeState, storePresentationHandoff } from "@/lib/presentation-handoff";
import { comparisonTaskLabel, factualDifferences, pairAssessment } from "@/lib/mission-comparison";
import { assessComparison, getComparisonItem, getComparisonItems } from "@/services/api";
import type { ComparisonAssessment, ComparisonItem, ComparisonItemSummary } from "@/types/agent";

type LoadState = "loading" | "empty" | "ready" | "error";
type LoadFailure = { phase: "items" | "assessment"; cause: unknown };

const nice = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
const timestamp = (value: string) => new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

export function PresentationComparisonScene({ onContinue }: { onContinue: () => void }) {
  const router = useRouter();
  const [state, setState] = useState<LoadState>("loading");
  const [summaries, setSummaries] = useState<ComparisonItemSummary[]>([]);
  const [items, setItems] = useState<ComparisonItem[]>([]);
  const [assessment, setAssessment] = useState<ComparisonAssessment | null>(null);
  const [error, setError] = useState("");
  const mountedRef = useRef(true);
  const sequenceRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; sequenceRef.current += 1; };
  }, []);

  const load = useCallback(async () => {
    const sequence = ++sequenceRef.current;
    setState("loading"); setError(""); setSummaries([]); setItems([]); setAssessment(null);
    try {
      const history = [...await getComparisonItems()].sort((left, right) => +new Date(right.created_at) - +new Date(left.created_at));
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      if (history.length < 2) { setSummaries(history); setState("empty"); return; }
      const pair = history.slice(0, 2);
      const ids = pair.map(item => item.request_id);
      const [details, authoritativeAssessment] = await Promise.all([
        Promise.all(ids.map(id => getComparisonItem(id))).catch(cause => Promise.reject({ phase: "items", cause } satisfies LoadFailure)),
        assessComparison(ids).catch(cause => Promise.reject({ phase: "assessment", cause } satisfies LoadFailure)),
      ]);
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      setSummaries(pair); setItems(details); setAssessment(authoritativeAssessment); setState("ready");
    } catch (caught) {
      if (!mountedRef.current || sequence !== sequenceRef.current) return;
      const failure = caught as Partial<LoadFailure>;
      setError(presentationComparisonError(failure.cause ?? caught, failure.phase === "assessment"));
      setState("error");
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const relation = assessment && summaries.length === 2
    ? pairAssessment(assessment.assessments, summaries[0].request_id, summaries[1].request_id)
    : undefined;
  const facts = useMemo(() => assessment ? factualDifferences(items, assessment.assessments) : [], [assessment, items]);

  const openWorkspace = () => {
    if (summaries.length < 2) return;
    storePresentationComparisonHandoff(window.sessionStorage, summaries.map(item => item.request_id));
    storePresentationHandoff(window.sessionStorage, { route: "/assistant/compare", stageIndex: 10, ...presentationRuntimeState(document) });
    router.push("/assistant/compare");
  };

  return <section className="presentation-comparison-scene mt-4 min-h-0 flex-1" aria-label="Mission Comparison presentation summary">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div><p className="text-[9px] font-bold uppercase tracking-[.18em] text-sky-300">Authoritative Stored Results</p><p className="mt-1 text-[10px] text-zinc-500">{state === "ready" ? `${summaries.length} recent results assessed` : state === "empty" ? `${summaries.length} recent result${summaries.length === 1 ? "" : "s"}` : state === "error" ? "Comparison unavailable" : "Loading recent results"}</p></div>
      <div className="flex gap-2"><button type="button" onClick={() => void load()} disabled={state === "loading"} className="presentation-comparison-secondary"><RefreshCw size={12} className={state === "loading" ? "animate-spin" : ""}/>Refresh</button><button type="button" onClick={onContinue} className="presentation-comparison-secondary">Continue</button></div>
    </div>

    {state === "loading" && <div className="presentation-comparison-empty mt-3 grid min-h-56 place-items-center rounded-2xl border text-xs text-zinc-400"><span className="flex items-center gap-2"><LoaderCircle className="animate-spin" size={16}/>Loading authoritative comparison data…</span></div>}
    {state === "empty" && <div className="presentation-comparison-empty mt-3 grid min-h-56 place-items-center rounded-2xl border p-6 text-center"><div><GitCompareArrows className="mx-auto text-zinc-600" size={25}/><h2 className="mt-3 text-base font-semibold">Two stored results are required</h2><p className="mt-2 max-w-md text-[11px] leading-5 text-zinc-500">Fewer than two authoritative results are available. Complete another workflow, then refresh this stage. No analysis is started here.</p></div></div>}
    {state === "error" && <div className="presentation-comparison-empty mt-3 rounded-2xl border border-rose-300/20 p-5" role="alert"><div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 shrink-0 text-rose-200" size={17}/><div><h2 className="text-sm font-semibold text-rose-100">Mission comparison unavailable</h2><p className="mt-1 text-[10px] leading-4 text-rose-100/80">{error}</p><button type="button" onClick={() => void load()} className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-rose-300/20 px-2.5 py-1.5 text-[10px] text-rose-100"><RefreshCw size={11}/>Retry</button></div></div></div>}

    {state === "ready" && relation && <div className="presentation-comparison-layout mt-3 grid min-h-0 gap-3">
      <div className="presentation-comparison-cards grid grid-cols-2 gap-3">
        {summaries.map((item, index) => <ResultSummary key={item.request_id} item={item} index={index}/>) }
      </div>

      <article className="presentation-comparison-assessment min-h-0 rounded-2xl border p-3">
        <div className="flex flex-wrap items-start justify-between gap-2"><div className="min-w-0"><p className="text-[8px] font-bold uppercase tracking-[.16em] text-zinc-500">Authoritative comparability assessment</p><div className="mt-1.5 flex flex-wrap items-center gap-2"><span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[9px] font-semibold ${relation.level === "direct" ? "border-emerald-300/25 text-emerald-200" : relation.level === "partial" ? "border-amber-300/25 text-amber-100" : "border-rose-300/25 text-rose-200"}`}>{relation.level === "direct" ? <CheckCircle2 size={11}/> : <AlertTriangle size={11}/>} {comparabilityLabel(relation.level)}</span><span className="text-[9px] text-zinc-500">Shared input identity: <strong className="text-zinc-300">{relation.shared_inputs ? "Yes" : "No"}</strong></span><span className="text-[9px] text-zinc-500">Shared task family: <strong className="text-zinc-300">{relation.shared_task_family ? "Yes" : "No"}</strong></span></div></div><button type="button" onClick={openWorkspace} className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-xl bg-sky-300 px-3 text-[10px] font-semibold text-zinc-950"><GitCompareArrows size={12}/>Open Full Comparison</button></div>
        <p className="mt-2 text-[10px] leading-4 text-zinc-400">{relation.reason}</p>
        {!relation.shared_inputs && <p className="mt-1 text-[9px] text-amber-100">No common input identity was found; measurements remain workflow-specific.</p>}
        <p className="mt-1 text-[9px] font-medium text-zinc-400">Factual differences only. No universal winner is inferred.</p>
        <div className="presentation-comparison-facts mt-2 overflow-auto rounded-xl border border-white/[.06]">
          <table className="w-full min-w-[560px] text-left text-[9px]"><thead><tr className="bg-white/[.025]"><th className="px-2.5 py-1.5 text-zinc-500">Fact</th>{items.map(item => <th key={item.request_id} className="max-w-48 truncate px-2.5 py-1.5 text-zinc-300">{item.display_name}</th>)}</tr></thead><tbody>{facts.map(fact => <tr key={fact.label} className="border-t border-white/[.05]"><th className="px-2.5 py-1.5 font-medium text-zinc-400">{fact.label}{fact.caution && <span className="block text-[8px] font-normal text-amber-100">{fact.caution}</span>}</th>{items.map(item => <td key={item.request_id} className="px-2.5 py-1.5 text-zinc-300">{fact.values[item.request_id]}</td>)}</tr>)}</tbody></table>
        </div>
      </article>
    </div>}
  </section>;
}

function ResultSummary({ item, index }: { item: ComparisonItemSummary; index: number }) {
  return <article className="presentation-comparison-card min-w-0 rounded-2xl border p-3">
    <div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="text-[8px] font-bold uppercase tracking-[.15em] text-sky-300">Result {index + 1}</p><h2 className="mt-1 truncate text-sm font-semibold" title={item.display_name}>{comparisonTaskLabel(item.task) ?? nice(item.task)}</h2></div><span className={`rounded-full border px-2 py-1 text-[8px] ${item.status === "success" ? "border-emerald-300/25 text-emerald-200" : item.status === "failed" ? "border-rose-300/25 text-rose-200" : "border-amber-300/25 text-amber-100"}`}>{nice(item.status)}</span></div>
    <p className="mt-1 truncate font-mono text-[8px] text-zinc-600" title={item.request_id}>{item.request_id}</p>
    <dl className="mt-2 grid grid-cols-3 gap-1.5 text-[9px]">
      <Fact label="Input" value={nice(item.input_mode)}/><Fact label="Timestamp" value={timestamp(item.created_at)}/><Fact label="Runtime" value={item.execution_duration_ms == null ? "Unavailable" : `${item.execution_duration_ms} ms`} icon/><Fact label="Evidence" value={`${item.evidence_product_count} product${item.evidence_product_count === 1 ? "" : "s"}`}/><Fact label="Cache" value={item.cached ? "Cached" : "Fresh"}/><Fact label="Report" value={item.report_available ? "Available" : "Not generated"}/>
    </dl>
  </article>;
}

function Fact({ label, value, icon = false }: { label: string; value: string; icon?: boolean }) {
  return <div className="min-w-0 rounded-lg bg-white/[.025] p-1.5"><dt className="text-[7px] uppercase tracking-wide text-zinc-600">{label}</dt><dd className="mt-0.5 truncate text-[8px] font-medium text-zinc-300" title={value}>{icon && <Clock3 className="mr-1 inline" size={8}/>} {value}</dd></div>;
}

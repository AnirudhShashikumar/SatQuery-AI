import { AlertTriangle, ArrowRight, BookOpenCheck, Database, FlaskConical, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { benchmarkData } from "@/lib/benchmarks";
import { BenchmarkComparison } from "@/components/benchmarks/benchmark-comparison";
import { BenchmarkExportPanel } from "@/components/benchmarks/benchmark-export-panel";
import { BenchmarkOverview } from "@/components/benchmarks/benchmark-overview";
import { BenchmarkStatusBadge } from "@/components/benchmarks/benchmark-status-badge";
import { SpecialistBenchmarkCard } from "@/components/benchmarks/specialist-benchmark-card";

export function BenchmarkView() {
  const { suite, preferred, records, historyBySpecialist, warnings, demos } = benchmarkData;
  return <section className="benchmark-page"><header className="benchmark-header"><div><p className="eyebrow">Evaluation intelligence</p><h1>Benchmarks</h1><p>Task-appropriate results with explicit datasets, splits, checkpoint fingerprints, hardware context, and limitations.</p></div><span><ShieldCheck size={15}/>Evidence-backed · suite {suite.suite_version}</span></header>
    {warnings.length > 0 && <div className="benchmark-loader-warning" role="alert"><AlertTriangle size={17}/><div><strong>Some benchmark records were rejected</strong>{warnings.map(warning => <p key={warning}>{warning}</p>)}</div></div>}
    <BenchmarkOverview suite={suite} preferred={preferred}/>
    <section className="benchmark-principles" aria-label="Benchmark interpretation rules"><article><Database size={17}/><strong>Versioned data</strong><p>The page reads validated repository JSON and never runs evaluation.</p></article><article><BookOpenCheck size={17}/><strong>Protocol visible</strong><p>Dataset, split, sample count, timestamp, and provenance stay attached.</p></article><article><FlaskConical size={17}/><strong>Smoke stays smoke</strong><p>Operational evidence is never presented as benchmark accuracy.</p></article></section>
    <section className="benchmark-specialists" aria-labelledby="specialist-results-title"><div className="benchmark-section-heading"><p className="eyebrow">Specialist model cards</p><h2 id="specialist-results-title">Measured results and honest gaps</h2><p>Expand each card for protocol, checkpoint provenance, source artifacts, and scientific limitations.</p></div>{preferred.map(record => <SpecialistBenchmarkCard key={record.benchmark_id} record={record} history={historyBySpecialist[record.specialist_id] ?? []}/>)}</section>
    <BenchmarkComparison records={records} performanceRecords={preferred.filter(record => record.performance.mean_latency_ms !== null || record.performance.peak_memory_mb !== null)}/>
    <section className="benchmark-history" aria-labelledby="benchmark-history-title"><header><div><p className="eyebrow">Historical runs</p><h2 id="benchmark-history-title">Preserved context, not blended scores</h2></div><span>{records.length} normalized records</span></header><div>{records.map(record => <article key={record.benchmark_id}><div><strong>{record.display_name}</strong><small>{record.benchmark_id}</small></div><BenchmarkStatusBadge status={record.evaluation.status}/><span>{record.evaluation.dataset ?? "No dataset"} · {record.evaluation.split ?? "No split"}</span><span>n={record.evaluation.sample_count?.toLocaleString() ?? "—"}</span></article>)}</div></section>
    <BenchmarkExportPanel suite={suite} records={preferred}/>
    <section className="benchmark-gallery-callout"><div><p className="eyebrow">Curated demonstrations</p><h2>Review real assets through real workflows</h2><p>{demos.cases.length} attributed cases are available. Curated expectations and benchmark annotations are visibly distinguished.</p></div><Link href="/demos">Open Demo Gallery <ArrowRight size={15}/></Link></section>
    <section className="benchmark-limitations" aria-labelledby="benchmark-gaps-title"><h2 id="benchmark-gaps-title">Known benchmark gaps</h2><ul>{suite.known_gaps.map(gap => <li key={gap}>{gap}</li>)}</ul><p>No combined SatQuery “accuracy” is calculated because task metrics are not commensurate.</p></section>
  </section>;
}

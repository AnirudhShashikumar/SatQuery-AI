import { CalendarDays, CheckCircle2, Clock3, Database, ShieldCheck } from "lucide-react";
import { formatTimestamp, type BenchmarkSuite, type BenchmarkResult } from "@/lib/benchmarks";

export function BenchmarkOverview({ suite, preferred }: { suite: BenchmarkSuite; preferred: BenchmarkResult[] }) {
  const validationOnly = suite.coverage.verified_validation + suite.coverage.partial_validation;
  const timestamps = preferred.map(record => record.evaluation.timestamp).filter((value): value is string => Boolean(value)).sort();
  const latest = timestamps.at(-1) ?? null;
  const facts = [
    { label: "Verified test", value: suite.coverage.verified_test, note: "Official prescribed test split", icon: ShieldCheck },
    { label: "Validation only", value: validationOnly, note: "Known validation splits", icon: CheckCircle2 },
    { label: "Operational only", value: suite.coverage.operational_only, note: "Runtime or parity evidence", icon: Clock3 },
    { label: "Unavailable", value: suite.coverage.unavailable, note: "No valid quality benchmark", icon: Database },
  ];
  return <section className="benchmark-suite-overview" aria-labelledby="benchmark-overview-title"><div className="benchmark-overview-copy"><p className="eyebrow">Research readiness</p><h2 id="benchmark-overview-title">Evidence coverage, without a synthetic system score</h2><p>Each specialist keeps its own task-appropriate protocol. Test, validation, smoke, and operational results remain visibly separate.</p><div className="benchmark-suite-meta"><span><CalendarDays size={14}/>Last measured {formatTimestamp(latest)}</span><span>Suite v{suite.suite_version}</span><span>{preferred.length} specialist records</span></div></div><div className="benchmark-coverage-grid">{facts.map(({ label, value, note, icon: Icon }) => <article key={label}><Icon size={17}/><strong>{value}</strong><span>{label}</span><small>{note}</small></article>)}</div></section>;
}

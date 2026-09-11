import { CircleSlash2 } from "lucide-react";

export function MissingBenchmarkState({ note }: { note?: string }) {
  return <div className="benchmark-missing" role="note"><CircleSlash2 size={20}/><div><strong>Benchmark unavailable</strong><p>Operational integration may be complete, but no verified quality benchmark has been imported. Smoke tests are not accuracy evaluations.</p>{note && <small>{note}</small>}</div></div>;
}

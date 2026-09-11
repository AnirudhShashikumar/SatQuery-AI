export type MetricCardProps = {
  title?: string;
  label?: string;
  value: string | number;
  unit?: string;
  description?: string;
  detail?: string;
  trend?: "higher" | "lower" | "neutral";
  source?: "measured" | "derived" | "qualitative" | "demo";
};

const sourceLabel = { measured: "Measured", derived: "Derived", qualitative: "Qualitative", demo: "Demo" };

export function MetricCard({ title, label, value, unit, description, detail, trend = "neutral", source }: MetricCardProps) {
  const heading = title ?? label ?? "Metric";
  const helper = description ?? detail;
  const trendText = trend === "higher" ? "Higher is better" : trend === "lower" ? "Lower is better" : helper;
  return <div className="glass rounded-2xl p-4"><div className="flex items-start justify-between gap-3"><p className="text-xs font-medium uppercase tracking-wider text-zinc-500">{heading}</p>{source && <span className="rounded-full border border-white/[.08] bg-white/[.04] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-zinc-400">{sourceLabel[source]}</span>}</div><p className="mt-2 flex items-baseline gap-1.5 tabular-nums"><span className="text-xl font-semibold tracking-tight">{value}</span>{unit && <span className="text-sm font-medium text-zinc-400">{unit}</span>}</p>{trendText && <p className="mt-1 text-xs text-zinc-500">{trendText}</p>}</div>;
}

import type { AgentHealth } from "@/types/agent";

export type PresentedHealthStatus = "ready" | "loading" | "unavailable" | "failed" | "disabled" | "warning";

export function presentedHealthStatus(status: string | undefined, enabled: boolean | undefined): PresentedHealthStatus {
  if (enabled === false || status?.toLowerCase() === "disabled") return "disabled";
  const value = (status ?? "unavailable").toLowerCase();
  if (["ready", "success", "available", "loaded", "unloaded", "lazy", "not_loaded"].includes(value)) return "ready";
  if (["loading", "initializing"].includes(value)) return "loading";
  if (["failed", "error", "model_error", "checksum_failure"].includes(value)) return "failed";
  if (["degraded", "warning"].includes(value)) return "warning";
  return "unavailable";
}

export function isServiceHealthEntry(name: string) {
  return /service|translation|ttp/i.test(name);
}

export function summarizeAgentHealth(health?: AgentHealth) {
  const entries = Object.entries(health?.specialists ?? {});
  const summarize = (subset: typeof entries) => ({
    total: subset.length,
    ready: subset.filter(([, item]) => presentedHealthStatus(item.status, item.enabled) === "ready").length,
  });
  const specialists = entries.filter(([name]) => !isServiceHealthEntry(name));
  const services = entries.filter(([name]) => isServiceHealthEntry(name));
  const warningCount = entries.filter(([, item]) => presentedHealthStatus(item.status, item.enabled) === "warning" || Boolean(item.warnings?.length)).length;
  const unavailableCount = entries.filter(([, item]) => ["unavailable", "failed", "disabled"].includes(presentedHealthStatus(item.status, item.enabled))).length;
  const failedCount = entries.filter(([, item]) => presentedHealthStatus(item.status, item.enabled) === "failed").length;
  return { entries, specialists: summarize(specialists), services: summarize(services), unavailableCount, warningCount, failedCount };
}

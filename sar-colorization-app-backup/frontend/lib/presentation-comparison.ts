import { ApiRequestError } from "@/services/api";
import type { ComparabilityLevel } from "@/types/agent";
import { presentationStorageKeys } from "@/lib/presentation";

export const presentationComparisonStorageKeys = {
  returnPending: presentationStorageKeys.comparisonReturn,
  requestIds: presentationStorageKeys.comparisonRequestIds,
} as const;

export const missionComparisonStageIndex = 10;

export function storePresentationComparisonHandoff(storage: Storage, requestIds: string[]): void {
  storage.setItem(presentationStorageKeys.step, String(missionComparisonStageIndex));
  storage.setItem(presentationComparisonStorageKeys.returnPending, "true");
  storage.setItem(presentationComparisonStorageKeys.requestIds, JSON.stringify(requestIds.slice(0, 2)));
}

export function hasPresentationComparisonReturn(storage: Storage): boolean {
  return storage.getItem(presentationComparisonStorageKeys.returnPending) === "true";
}

export function readPresentationComparisonRequestIds(storage: Storage): string[] {
  try {
    const value = JSON.parse(storage.getItem(presentationComparisonStorageKeys.requestIds) ?? "[]");
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim())).slice(0, 2) : [];
  } catch {
    return [];
  }
}

export function clearPresentationComparisonHandoff(storage: Storage): void {
  storage.removeItem(presentationComparisonStorageKeys.returnPending);
  storage.removeItem(presentationComparisonStorageKeys.requestIds);
}

export function comparabilityLabel(level: ComparabilityLevel): string {
  if (level === "not_direct") return "Not directly comparable";
  return level === "direct" ? "Direct" : "Partial";
}

export function presentationComparisonError(error: unknown, assessment = false): string {
  if (error instanceof ApiRequestError && (error.status === 404 || error.status === 410 || /EXPIRED|NOT_FOUND/.test(error.code))) {
    return "One of the authoritative results has expired or is no longer available. Refresh to select the latest stored pair.";
  }
  const message = error instanceof Error ? error.message : "Mission comparison is unavailable.";
  if (/fetch|network|backend|connect/i.test(message)) {
    return "The local SatQuery backend is offline. Restart it on port 8010, then refresh this stage.";
  }
  if (assessment) return "The authoritative comparability assessment could not be completed. Refresh without rerunning either workflow.";
  return "The comparison API is unavailable. The stored results were not changed or rerun.";
}

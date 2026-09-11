export const analyticsLabel = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());

export function formatAnalyticsDuration(milliseconds: number | null) {
  if (milliseconds === null) return "Not measured";
  if (milliseconds < 1000) return `${milliseconds.toLocaleString()} ms`;
  return `${(milliseconds / 1000).toFixed(milliseconds < 10_000 ? 2 : 1)} s`;
}

export function formatUptime(seconds: number) {
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
}

export function runtimeBarPercent(value: number | null, maximum: number) {
  if (value === null || maximum <= 0) return 0;
  return Math.max(3, Math.min(100, (value / maximum) * 100));
}

export function isPositiveStatus(status: string) {
  return ["available", "active", "ready", "registered", "success", "partial", "healthy", "model provenance"].includes(status.toLowerCase());
}

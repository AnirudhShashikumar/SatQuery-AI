import type { InputMode } from "@/types/agent";

export const RECENT_CONVERSATIONS_KEY = "satquery-recent-conversations";

export type RecentConversation = {
  id: string;
  prompt: string;
  mode?: InputMode;
  createdAt: string;
};

export function readRecentConversations(storage: Pick<Storage, "getItem">): RecentConversation[] {
  try {
    const value = JSON.parse(storage.getItem(RECENT_CONVERSATIONS_KEY) ?? "[]");
    if (!Array.isArray(value)) return [];
    return value.filter((item): item is RecentConversation => Boolean(item && typeof item.id === "string" && typeof item.prompt === "string" && typeof item.createdAt === "string")).slice(0, 5);
  } catch {
    return [];
  }
}

export function rememberConversation(storage: Pick<Storage, "getItem" | "setItem">, prompt: string, mode?: InputMode) {
  const clean = prompt.trim();
  if (!clean) return;
  const existing = readRecentConversations(storage).filter(item => item.prompt.toLowerCase() !== clean.toLowerCase());
  const item: RecentConversation = { id: `${Date.now()}-${clean.slice(0, 24)}`, prompt: clean, mode, createdAt: new Date().toISOString() };
  storage.setItem(RECENT_CONVERSATIONS_KEY, JSON.stringify([item, ...existing].slice(0, 5)));
}

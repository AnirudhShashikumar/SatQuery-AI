import { describe, expect, it } from "vitest";
import { readRecentConversations, RECENT_CONVERSATIONS_KEY, rememberConversation } from "./recent-conversations";

function memoryStorage(seed: Record<string, string> = {}) {
  const data = new Map(Object.entries(seed));
  return { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, value) };
}

describe("recent conversations", () => {
  it("stores newest unique questions first and caps history", () => {
    const storage = memoryStorage();
    for (let index = 0; index < 7; index += 1) rememberConversation(storage, `Question ${index}`, "single");
    rememberConversation(storage, "Question 6", "cross_modal");
    const recent = readRecentConversations(storage);
    expect(recent).toHaveLength(5);
    expect(recent[0]).toMatchObject({ prompt: "Question 6", mode: "cross_modal" });
    expect(new Set(recent.map(item => item.prompt)).size).toBe(5);
  });

  it("safely ignores malformed persisted data", () => {
    expect(readRecentConversations(memoryStorage({ [RECENT_CONVERSATIONS_KEY]: "not-json" }))).toEqual([]);
  });
});

import { presentationStorageKeys } from "@/lib/presentation";

export type PresentationHandoff = {
  route: string;
  stageIndex: number;
  notesVisible: boolean;
  theme: "dark" | "light" | "system";
  fullscreen: boolean;
};

export function storePresentationHandoff(
  storage: Storage,
  input: { route: string; stageIndex: number; theme?: string; fullscreen?: boolean },
): void {
  const notesVisible = storage.getItem(presentationStorageKeys.notes) === "true";
  const theme = input.theme === "light" || input.theme === "dark" ? input.theme : "system";
  storage.setItem(presentationStorageKeys.step, String(input.stageIndex));
  storage.setItem(presentationStorageKeys.notes, String(notesVisible));
  storage.setItem(presentationStorageKeys.handoffRoute, input.route);
  storage.setItem(presentationStorageKeys.handoffTheme, theme);
  storage.setItem(presentationStorageKeys.handoffFullscreen, String(Boolean(input.fullscreen)));
}

export function readPresentationHandoff(storage: Storage): PresentationHandoff | null {
  const route = storage.getItem(presentationStorageKeys.handoffRoute);
  if (!route?.startsWith("/") || route.startsWith("//")) return null;
  const stageIndex = Number.parseInt(storage.getItem(presentationStorageKeys.step) ?? "", 10);
  const savedTheme = storage.getItem(presentationStorageKeys.handoffTheme);
  return {
    route,
    stageIndex: Number.isFinite(stageIndex) ? stageIndex : 0,
    notesVisible: storage.getItem(presentationStorageKeys.notes) === "true",
    theme: savedTheme === "light" || savedTheme === "dark" ? savedTheme : "system",
    fullscreen: storage.getItem(presentationStorageKeys.handoffFullscreen) === "true",
  };
}

export function hasPresentationHandoff(storage: Storage, route: string): boolean {
  return readPresentationHandoff(storage)?.route === route;
}

export function clearPresentationHandoff(storage: Storage): void {
  storage.removeItem(presentationStorageKeys.handoffRoute);
  storage.removeItem(presentationStorageKeys.handoffTheme);
  storage.removeItem(presentationStorageKeys.handoffFullscreen);
  storage.removeItem(presentationStorageKeys.comparisonReturn);
  storage.removeItem(presentationStorageKeys.comparisonRequestIds);
}

export function presentationRuntimeState(documentValue: Document): { theme: "dark" | "light"; fullscreen: boolean } {
  return {
    theme: documentValue.documentElement.classList.contains("light") ? "light" : "dark",
    fullscreen: Boolean(documentValue.fullscreenElement),
  };
}

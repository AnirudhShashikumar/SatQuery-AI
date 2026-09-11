import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));
export const dataUrl = (value?: string | null) => value ? (value.startsWith("blob:") || value.startsWith("data:") || value.startsWith("http") ? value : `data:image/png;base64,${value}`) : undefined;
export const formatMetric = (value?: number | null, digits = 4) => value == null ? "—" : value.toFixed(digits);

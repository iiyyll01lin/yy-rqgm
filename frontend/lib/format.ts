/** Small formatting + styling helpers shared across the wizard UI. */

import type { TierClass, VramBreakdown } from "./types";

/** Join truthy class name fragments (tiny clsx replacement). */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function fmtGB(n: number, dp = 1): string {
  return `${n.toFixed(dp)} GB`;
}

/** Compact tokens/s formatting, e.g. 1,325 tok/s. */
export function fmtTps(n: number): string {
  return `${Math.round(n).toLocaleString("en-US")} tok/s`;
}

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function fmtUsd(n?: number): string {
  return n == null ? "—" : `$${n.toLocaleString("en-US")}`;
}

/** Fixed colors for each VRAM segment in the stacked bar (hex for gradients). */
export const SEGMENT_COLORS: Record<keyof VramBreakdown, string> = {
  weights: "#ed1c24", // AMD red
  kv_cache: "#f59e0b", // amber
  activations: "#22d3ee", // cyan
  overhead: "#64748b", // slate
};

export interface TierAccent {
  /** solid text color class */
  text: string;
  /** subtle background tint class */
  bg: string;
  /** border/ring class */
  border: string;
  /** hex used for inline bars / dots */
  hex: string;
  /** short badge label */
  label: string;
}

/** Per-family accent palette, ordered as an upgrade ladder. */
export const TIER_ACCENT: Record<TierClass, TierAccent> = {
  "Ryzen AI": {
    text: "text-emerald-300",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    hex: "#34d399",
    label: "Edge / NPU",
  },
  Radeon: {
    text: "text-red-300",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    hex: "#f87171",
    label: "桌面 dGPU",
  },
  "Radeon PRO": {
    text: "text-sky-300",
    bg: "bg-sky-500/10",
    border: "border-sky-500/30",
    hex: "#38bdf8",
    label: "工作站",
  },
  Instinct: {
    text: "text-amber-300",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    hex: "#fbbf24",
    label: "資料中心",
  },
};

/** Instinct results are simulated in the demo box. */
export function isSimulatedClass(cls: TierClass): boolean {
  return cls === "Instinct";
}

/** Human labels for the supported precisions. */
export const DTYPE_LABEL: Record<string, string> = {
  int4: "INT4 (最省顯存)",
  fp8: "FP8 (平衡)",
  fp16: "FP16 (最高精度)",
  bf16: "BF16",
};

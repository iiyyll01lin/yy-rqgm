/** Shared control option sets + defaults for the wizard forms. */

/** Sequence-length steps (used by log-style sliders + selects). */
export const SEQ_OPTIONS = [
  512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
];

export function seqToIndex(seq: number): number {
  let best = 0;
  let bestDiff = Infinity;
  SEQ_OPTIONS.forEach((v, i) => {
    const d = Math.abs(v - seq);
    if (d < bestDiff) {
      bestDiff = d;
      best = i;
    }
  });
  return best;
}

export function indexToSeq(i: number): number {
  return SEQ_OPTIONS[Math.max(0, Math.min(SEQ_OPTIONS.length - 1, i))];
}

export const WORKLOAD_TYPES: { value: string; label: string }[] = [
  { value: "realtime", label: "即時 / 低延遲 (Real-time)" },
  { value: "interactive", label: "互動式 (Interactive)" },
  { value: "batch", label: "批次處理 (Batch)" },
  { value: "agentic", label: "多步驟 Agent (Agentic / 搜尋)" },
];

export const POPULATION_MAX = 128;

/** Prefilled anchor scenario so the demo is compelling from first load. */
export const DEFAULT_DOMAIN = "智慧製造 / 工廠品質檢測 (Smart Manufacturing QC)";
export const DEFAULT_DESCRIPTION =
  "產線需要一個能即時判讀相機影像、標記表面瑕疵並給出可稽核判定理由的 AI Agent；資料須留在廠內，且要能在現有硬體上先驗證，再評估升級到更高吞吐的方案。Visual defect detection / surface scratch inspection on the production line (camera vision QC).";

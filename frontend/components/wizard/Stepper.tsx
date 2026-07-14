"use client";

import { Check } from "lucide-react";
import { cn } from "@/lib/format";
import { useWizard, type StepId } from "./WizardContext";

const STEPS: { id: StepId; title: string; en: string }[] = [
  { id: 1, title: "領域定義", en: "Domain" },
  { id: 2, title: "限制診斷", en: "Constraint Diagnostic" },
  { id: 3, title: "硬體模擬實驗室", en: "Simulation Lab" },
  { id: 4, title: "匯出提案", en: "Export" },
];

export function Stepper() {
  const { step, maxStep, goTo } = useWizard();

  return (
    <nav aria-label="精靈步驟" className="w-full">
      <ol className="flex items-center">
        {STEPS.map((s, i) => {
          const done = s.id < step;
          const current = s.id === step;
          const unlocked = s.id <= maxStep;
          return (
            <li key={s.id} className="flex flex-1 items-center last:flex-none">
              <button
                type="button"
                disabled={!unlocked}
                onClick={() => unlocked && goTo(s.id)}
                className={cn(
                  "group flex items-center gap-3 rounded-xl px-1 py-1 text-left transition-opacity",
                  unlocked ? "cursor-pointer" : "cursor-not-allowed opacity-50",
                )}
              >
                <span
                  className={cn(
                    "flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-sm font-semibold tabular-nums transition-colors",
                    current &&
                      "border-amd bg-amd text-white shadow-[0_0_18px_-3px_rgba(237,28,36,0.8)]",
                    done && "border-emerald-500/50 bg-emerald-500/15 text-emerald-300",
                    !current && !done && "border-line bg-surface-2 text-muted",
                  )}
                >
                  {done ? <Check className="h-4 w-4" /> : s.id}
                </span>
                <span className="hidden min-w-0 sm:block">
                  <span
                    className={cn(
                      "block truncate text-sm font-semibold",
                      current ? "text-white" : "text-zinc-300",
                    )}
                  >
                    {s.title}
                  </span>
                  <span className="block truncate text-[11px] uppercase tracking-wide text-muted">
                    {s.en}
                  </span>
                </span>
              </button>
              {i < STEPS.length - 1 ? (
                <span
                  className={cn(
                    "mx-2 h-px flex-1 sm:mx-4",
                    s.id < step ? "bg-emerald-500/40" : "bg-line",
                  )}
                />
              ) : null}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

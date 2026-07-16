"use client";

/** Shared primitives + formatting for the epoch-admin (RQGM evolution) surface. */

import * as React from "react";
import { Check, Minus, X } from "lucide-react";
import { cn } from "@/lib/format";

/* --------------------------------------------------------------------- */
/* Formatting                                                             */
/* --------------------------------------------------------------------- */

export function pct(n: number, dp = 1): string {
  return `${(n * 100).toFixed(dp)}%`;
}

/** Signed fixed-precision number, e.g. "+0.3000" / "-0.0250". */
export function signed(n: number, dp = 4): string {
  return `${n >= 0 ? "+" : ""}${n.toFixed(dp)}`;
}

export function fixed(n: number | null | undefined, dp = 4): string {
  return n == null ? "—" : n.toFixed(dp);
}

/** Human label for a frontier objective key. */
export function objectiveLabel(key: string): string {
  if (key === "parsimony") return "精簡 Parsimony";
  if (key === "adversarial") return "對抗 Adversarial";
  if (key.startsWith("sep::")) return `分離 · ${key.slice(5)}`;
  return key;
}

/* --------------------------------------------------------------------- */
/* Check row (P1 / P2 / boolean predicates)                               */
/* --------------------------------------------------------------------- */

export function CheckRow({
  ok,
  label,
  value,
  neutral,
}: {
  ok: boolean;
  label: React.ReactNode;
  value?: React.ReactNode;
  neutral?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-line bg-surface-2/50 px-3 py-2">
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={cn(
            "flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
            neutral
              ? "bg-zinc-600/30 text-zinc-400"
              : ok
                ? "bg-emerald-500/20 text-emerald-300"
                : "bg-red-500/20 text-red-300",
          )}
        >
          {neutral ? (
            <Minus className="h-3 w-3" />
          ) : ok ? (
            <Check className="h-3 w-3" />
          ) : (
            <X className="h-3 w-3" />
          )}
        </span>
        <span className="truncate text-sm text-zinc-200">{label}</span>
      </div>
      {value != null ? (
        <span className="shrink-0 font-mono text-xs tabular-nums text-zinc-400">
          {value}
        </span>
      ) : null}
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* Champion vs challenger separation comparison                           */
/* --------------------------------------------------------------------- */

function CompareRow({
  label,
  value,
  color,
  max,
}: {
  label: string;
  value: number;
  color: string;
  max: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs text-muted">{label}</span>
      <div className="relative h-4 flex-1 overflow-hidden rounded-md bg-surface-2">
        <div
          className="h-full rounded-md transition-[width] duration-500"
          style={{
            width: `${Math.max(3, (Math.abs(value) / max) * 100)}%`,
            backgroundColor: color,
          }}
        />
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums text-zinc-200">
        {value.toFixed(4)}
      </span>
    </div>
  );
}

export function SeparationCompare({
  champion,
  challenger,
  delta,
}: {
  champion: number;
  challenger: number;
  delta: number;
}) {
  const max = Math.max(Math.abs(champion), Math.abs(challenger), 0.001);
  const better = delta > 0;

  return (
    <div className="space-y-2">
      <CompareRow label="現任 Champion" value={champion} color="#64748b" max={max} />
      <CompareRow
        label="挑戰者 Challenger"
        value={challenger}
        color={better ? "#34d399" : "#f87171"}
        max={max}
      />
      <div className="flex items-center justify-end gap-1.5 pt-0.5 text-xs">
        <span className="text-muted">Δ separation</span>
        <span
          className={cn(
            "rounded px-1.5 py-0.5 font-mono font-semibold tabular-nums",
            better
              ? "bg-emerald-500/15 text-emerald-300"
              : "bg-red-500/15 text-red-300",
          )}
        >
          {signed(delta)}
        </span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* RQGM tolerance ladder (before -> after)                                */
/* --------------------------------------------------------------------- */

export function ToleranceLadder({
  before,
  after,
}: {
  before: number[];
  after: number[];
}) {
  // `before` is the full active schedule; tightening drops levels, so the
  // universe of levels to render is `before` (∪ after for safety).
  const universe = Array.from(new Set([...before, ...after])).sort(
    (a, b) => a - b,
  );
  const afterSet = new Set(after);

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {universe.map((t) => {
        const active = afterSet.has(t);
        return (
          <span
            key={t}
            className={cn(
              "rounded-md border px-1.5 py-0.5 font-mono text-[11px] tabular-nums",
              active
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300 line-through",
            )}
            title={active ? "active tolerance level" : "dropped (tightened)"}
          >
            {t}
          </span>
        );
      })}
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* Small key/value line                                                   */
/* --------------------------------------------------------------------- */

export function KeyVal({
  k,
  v,
  mono,
}: {
  k: React.ReactNode;
  v: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-xs text-muted">{k}</span>
      <span
        className={cn(
          "text-right text-sm text-zinc-200",
          mono && "font-mono tabular-nums",
        )}
      >
        {v}
      </span>
    </div>
  );
}

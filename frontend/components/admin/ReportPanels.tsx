"use client";

/** Report-driven panels for GET /api/admin/report (RQGM transparency). */

import * as React from "react";
import {
  Brain,
  Database,
  Layers,
  Ruler,
  Scale,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/format";
import { Badge, Card } from "@/components/ui";
import type {
  DataSplits,
  JudgeAgreement,
  MemoryStats,
  SplitSeparation,
} from "@/lib/types";
import { pct } from "./shared";

function PanelHeader({
  icon,
  title,
  desc,
  right,
}: {
  icon: React.ReactNode;
  title: React.ReactNode;
  desc?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 text-amd">{icon}</span>
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {desc ? <p className="mt-0.5 text-xs text-muted">{desc}</p> : null}
        </div>
      </div>
      {right}
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* Data splits (train / val / test isolation)                            */
/* --------------------------------------------------------------------- */

const SPLIT_META: Record<
  keyof DataSplits,
  { label: string; role: string }
> = {
  train: { label: "Train", role: "GEPA 進化只讀" },
  val: { label: "Val", role: "Code gate 判定" },
  test: { label: "Test", role: "僅供報告" },
};

export function DataSplitsCard({ splits }: { splits: DataSplits }) {
  const order: (keyof DataSplits)[] = ["train", "val", "test"];
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Layers className="h-4 w-4" />}
        title="錨點資料切分 Data splits"
        desc="嚴格隔離：進化只看 train、gate 只看 val、test 只供誠實報告。"
      />
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {order.map((key) => {
          const s = splits[key];
          const total = s.total || 1;
          return (
            <div
              key={key}
              className="rounded-xl border border-line bg-surface-2/50 p-3.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-white">
                  {SPLIT_META[key].label}
                </span>
                <span className="text-2xl font-semibold tabular-nums text-white">
                  {s.total}
                </span>
              </div>
              <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full bg-amber-400/80"
                  style={{ width: `${(s.weak / total) * 100}%` }}
                  title={`weak: ${s.weak}`}
                />
                <div
                  className="h-full bg-emerald-400/80"
                  style={{ width: `${(s.strong / total) * 100}%` }}
                  title={`strong: ${s.strong}`}
                />
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] tabular-nums">
                <span className="text-amber-300">weak {s.weak}</span>
                <span className="text-emerald-300">strong {s.strong}</span>
              </div>
              <div className="mt-2 text-[11px] text-muted">
                {SPLIT_META[key].role}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Separation (val / test)                                               */
/* --------------------------------------------------------------------- */

function SepBlock({ label, s }: { label: string; s: SplitSeparation }) {
  const max = Math.max(s.mean_weak_deficit, s.mean_strong_deficit, 0.001);
  return (
    <div className="rounded-xl border border-line bg-surface-2/50 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-zinc-200">{label}</span>
        <span className="text-xs text-muted">n={s.n}</span>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-3xl font-semibold tabular-nums text-white">
          {s.separation.toFixed(3)}
        </span>
        <span className="text-xs text-muted">separation</span>
      </div>
      <div className="mt-3 space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-12 shrink-0 text-[11px] text-amber-300">weak</span>
          <div className="relative h-3 flex-1 overflow-hidden rounded bg-surface-2">
            <div
              className="h-full rounded bg-amber-400/80"
              style={{ width: `${(s.mean_weak_deficit / max) * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-zinc-300">
            {s.mean_weak_deficit.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-12 shrink-0 text-[11px] text-emerald-300">
            strong
          </span>
          <div className="relative h-3 flex-1 overflow-hidden rounded bg-surface-2">
            <div
              className="h-full rounded bg-emerald-400/80"
              style={{ width: `${(s.mean_strong_deficit / max) * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-zinc-300">
            {s.mean_strong_deficit.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

export function SeparationCard({
  val,
  test,
}: {
  val: SplitSeparation;
  test: SplitSeparation;
}) {
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Ruler className="h-4 w-4" />}
        title="弱/強分離度 Separation"
        desc="separation = mean deficit(weak) − mean deficit(strong)；越高代表 rubric 越能區分好壞架構。"
      />
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <SepBlock label="Val (held-out)" s={val} />
        <SepBlock label="Test (held-out)" s={test} />
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Judge / human agreement                                               */
/* --------------------------------------------------------------------- */

function AgreementBlock({ label, a }: { label: string; a: JudgeAgreement }) {
  const kappaTone =
    a.cohen_kappa >= 0.8 ? "green" : a.cohen_kappa >= 0.6 ? "amber" : "red";
  const accTone =
    a.accuracy >= 0.9 ? "green" : a.accuracy >= 0.75 ? "amber" : "red";
  return (
    <div className="rounded-xl border border-line bg-surface-2/50 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-zinc-200">{label}</span>
        <span className="text-xs text-muted">
          τ={a.tau} · {a.correct}/{a.n}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Badge tone={accTone}>accuracy {pct(a.accuracy)}</Badge>
        <Badge tone={kappaTone}>κ {a.cohen_kappa.toFixed(2)}</Badge>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {a.per_anchor.map((p) => {
          const correct = p.predicted === p.label;
          return (
            <span
              key={p.id}
              title={`${p.id}\ndeficit=${p.deficit} · 判定=${p.predicted} · 真值=${p.label}`}
              className={cn(
                "h-3.5 w-3.5 rounded-sm",
                correct ? "bg-emerald-500/70" : "bg-red-500/80 ring-1 ring-red-300",
              )}
            />
          );
        })}
      </div>
      <div className="mt-2 text-[11px] text-muted">
        每格 = 一個錨點；紅色代表判定與人類標註不一致。
      </div>
    </div>
  );
}

export function JudgeAgreementCard({
  val,
  test,
}: {
  val: JudgeAgreement;
  test: JudgeAgreement;
}) {
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Scale className="h-4 w-4" />}
        title="評審一致率 Judge agreement"
        desc="錨點指標改為『與人類標註一致率』(accuracy / Cohen's κ)，而非原始分離度。"
      />
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <AgreementBlock label="Val" a={val} />
        <AgreementBlock label="Test" a={test} />
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Memory                                                                 */
/* --------------------------------------------------------------------- */

export function MemoryCard({ memory }: { memory: MemoryStats }) {
  if (memory.error) {
    return (
      <Card className="p-5">
        <PanelHeader
          icon={<Brain className="h-4 w-4" />}
          title="記憶 Memory"
        />
        <p className="mt-3 text-sm text-red-300">{memory.error}</p>
      </Card>
    );
  }
  const cells = [
    { label: "總計 Total", value: memory.total, icon: <Database className="h-3.5 w-3.5" />, tone: "text-white" },
    { label: "heuristic_failure", value: memory.heuristic_failure, icon: null, tone: "text-amber-300" },
    { label: "physics_truth", value: memory.physics_truth, icon: <ShieldCheck className="h-3.5 w-3.5" />, tone: "text-emerald-300" },
  ];
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Brain className="h-4 w-4" />}
        title="記憶 Memory"
        desc="soft-delete + reconfirm；physics_truth 永久保留。"
        right={<Badge tone="neutral">mode · {memory.mode}</Badge>}
      />
      <div className="mt-4 grid grid-cols-3 gap-3">
        {cells.map((c) => (
          <div
            key={c.label}
            className="rounded-xl border border-line bg-surface-2/50 px-3 py-3 text-center"
          >
            <div className={cn("text-2xl font-semibold tabular-nums", c.tone)}>
              {c.value}
            </div>
            <div className="mt-1 flex items-center justify-center gap-1 text-[11px] text-muted">
              {c.icon}
              <span className="truncate">{c.label}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

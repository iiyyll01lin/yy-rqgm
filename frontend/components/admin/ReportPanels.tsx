"use client";

/** Report-driven panels for GET /api/admin/report (RQGM transparency). */

import * as React from "react";
import {
  Brain,
  Bug,
  Database,
  Fingerprint,
  Gauge,
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
  OverAcceptance,
  OverOptimization,
  Provenance,
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
/* Data splits (train / dev / val / test isolation)                      */
/* --------------------------------------------------------------------- */

const SPLIT_META: Record<
  keyof DataSplits,
  { label: string; role: string }
> = {
  train: { label: "Train", role: "GEPA 變異只讀" },
  dev: { label: "Dev", role: "Frontier 選擇" },
  val: { label: "Val", role: "Code gate 判定" },
  test: { label: "Test", role: "僅供報告" },
};

export function DataSplitsCard({ splits }: { splits: DataSplits }) {
  const order: (keyof DataSplits)[] = ["train", "dev", "val", "test"];
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Layers className="h-4 w-4" />}
        title="錨點資料切分 Data splits"
        desc="四路嚴格隔離：train 驅動變異、dev 選 frontier、val 只給 gate、test 只供報告（dev 選擇 → 消除 winner's curse）。"
      />
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {order.map((key) => {
          const s = splits[key];
          if (!s) return null;
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
/* Over-optimization (proxy(val) − gold(test) separation gap)            */
/* --------------------------------------------------------------------- */

export function OverOptimizationCard({ data }: { data: OverOptimization }) {
  const gap = data.separation_gap;
  // A large positive gap = champion looks much sharper on the split closer to
  // the optimization loop (val) than on the untouched gold split (test).
  const tone = gap <= 0.05 ? "green" : gap <= 0.15 ? "amber" : "red";
  const toneText =
    tone === "green"
      ? "text-emerald-300"
      : tone === "amber"
        ? "text-amber-300"
        : "text-red-300";
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Gauge className="h-4 w-4" />}
        title="過度最佳化監測 Over-optimization"
        desc="proxy(val) − gold(test) 分離度落差；落差越大 = champion 在靠近進化迴圈的 val 上比未動過的 gold test 明顯更利。"
        right={
          <Badge tone={tone}>
            gap {gap >= 0 ? "+" : ""}
            {gap.toFixed(4)}
          </Badge>
        }
      />
      <div className="mt-4 grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-3 text-center">
          <div className="text-2xl font-semibold tabular-nums text-white">
            {data.proxy_val_separation.toFixed(3)}
          </div>
          <div className="mt-1 text-[11px] text-muted">proxy · val</div>
        </div>
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-3 text-center">
          <div className="text-2xl font-semibold tabular-nums text-white">
            {data.gold_test_separation.toFixed(3)}
          </div>
          <div className="mt-1 text-[11px] text-muted">gold · test</div>
        </div>
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-3 text-center">
          <div className={cn("text-2xl font-semibold tabular-nums", toneText)}>
            {gap >= 0 ? "+" : ""}
            {gap.toFixed(3)}
          </div>
          <div className="mt-1 text-[11px] text-muted">gap</div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        gap ≤ 0 代表 gold test 分離度不輸 proxy val（健康）；大的正 gap 才是 val 被過度最佳化的警訊。
      </p>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Over-acceptance (gamed samples wrongly scored "strong")               */
/* --------------------------------------------------------------------- */

export function OverAcceptanceCard({ data }: { data: OverAcceptance }) {
  const rate = data.over_acceptance_rate;
  const tone = rate <= 0.001 ? "green" : rate <= 0.2 ? "amber" : "red";
  const rateText =
    tone === "green"
      ? "text-emerald-300"
      : tone === "amber"
        ? "text-amber-300"
        : "text-red-300";
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Bug className="h-4 w-4" />}
        title="過度接受監測 Over-acceptance"
        desc="被 rubric 誤判為 strong (deficit < τ) 的 gamed/weak 對抗樣本比例（含 out-of-catalog 偽裝）；高 = 判官被偽裝的 poison pill 騙過。"
        right={<Badge tone={tone}>{pct(rate)}</Badge>}
      />
      <div className="mt-4 flex items-baseline gap-2">
        <span className={cn("text-3xl font-semibold tabular-nums", rateText)}>
          {pct(rate)}
        </span>
        <span className="text-[11px] text-muted">
          {data.accepted_as_strong}/{data.n} 樣本被誤放行 · τ={data.tau}
        </span>
      </div>
      <div className="mt-3 space-y-1.5">
        {data.per_sample.map((s) => (
          <div
            key={s.id}
            className={cn(
              "flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5",
              s.accepted_as_strong
                ? "border-red-500/30 bg-red-500/[0.05]"
                : "border-line bg-surface-2/50",
            )}
          >
            <div className="flex min-w-0 items-center gap-1.5">
              <span
                className="truncate font-mono text-[11px] text-zinc-300"
                title={`${s.id} · targets: ${(Array.isArray(s.targets) ? s.targets : [s.targets]).join(", ")}`}
              >
                {s.id}
              </span>
              {s.out_of_catalog ? (
                <Badge tone="neutral">OOC</Badge>
              ) : null}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <span className="font-mono text-[11px] tabular-nums text-muted">
                deficit {s.deficit.toFixed(3)}
              </span>
              {s.accepted_as_strong ? (
                <Badge tone="red">誤放行</Badge>
              ) : (
                <Badge tone="green">擋下</Badge>
              )}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2.5 text-[11px] leading-relaxed text-muted">
        <span className="font-mono">OOC</span> = out-of-catalog（新型/未見過的 gaming），用來衡量泛化而非 id 記憶。
      </p>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Provenance (judge model id + git SHA + backend + using_mock)          */
/* --------------------------------------------------------------------- */

export function ProvenanceCard({ provenance }: { provenance: Provenance }) {
  const shortSha = provenance.git_sha ? provenance.git_sha.slice(0, 10) : "—";
  return (
    <Card className="p-5">
      <PanelHeader
        icon={<Fingerprint className="h-4 w-4" />}
        title="來源與可重現 Provenance"
        desc="這份報告由哪個 judge 模型 + 哪個原始碼版本產生（可稽核 / 可重現）。"
        right={
          <Badge tone={provenance.using_mock ? "amber" : "green"}>
            {provenance.using_mock ? "Mock (offline)" : "Live model"}
          </Badge>
        }
      />
      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-2.5">
          <div className="text-[11px] text-muted">judge model</div>
          <div className="mt-0.5 truncate font-mono text-sm text-zinc-200" title={provenance.judge_model}>
            {provenance.judge_model}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-2.5">
          <div className="text-[11px] text-muted">rqgm backend</div>
          <div className="mt-0.5 font-mono text-sm text-zinc-200">
            {provenance.rqgm_backend}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-2.5">
          <div className="text-[11px] text-muted">git sha</div>
          <div className="mt-0.5 font-mono text-sm text-zinc-200" title={provenance.git_sha ?? undefined}>
            {shortSha}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-surface-2/50 px-3 py-2.5">
          <div className="text-[11px] text-muted">using_mock</div>
          <div className="mt-0.5 font-mono text-sm text-zinc-200">
            {provenance.using_mock == null ? "—" : String(provenance.using_mock)}
          </div>
        </div>
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

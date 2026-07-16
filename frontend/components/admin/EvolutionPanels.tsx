"use client";

/** Panels driven by /epoch/propose + /epoch/approve: the two-stage gate, the
 *  Pareto frontier, and champion-vs-challenger exploitation (hack ratio). */

import * as React from "react";
import {
  ArrowDown,
  Crosshair,
  GitBranch,
  Lock,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Swords,
  Trophy,
} from "lucide-react";
import { cn } from "@/lib/format";
import { Badge, Callout, Card } from "@/components/ui";
import type {
  EpochApproveResponse,
  ExploitationReport,
  Frontier,
} from "@/lib/types";
import {
  CheckRow,
  KeyVal,
  SeparationCompare,
  ToleranceLadder,
  fixed,
  objectiveLabel,
  signed,
} from "./shared";

function StageTitle({
  n,
  title,
  right,
}: {
  n: string;
  title: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-full border border-line bg-surface-2 text-xs font-bold text-zinc-300">
          {n}
        </span>
        <span className="text-sm font-semibold text-white">{title}</span>
      </div>
      {right}
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* Two-stage gate                                                         */
/* --------------------------------------------------------------------- */

export function GatePanel({ result }: { result: EpochApproveResponse }) {
  const { gate, hitl, applied } = result;

  const outcome = applied
    ? { tone: "green" as const, label: "已晉升 Promoted", icon: <Trophy className="h-4 w-4" /> }
    : !gate.passed
      ? { tone: "red" as const, label: "被 Code Gate 擋下 Rejected", icon: <ShieldAlert className="h-4 w-4" /> }
      : { tone: "amber" as const, label: "HITL 否決 Vetoed", icon: <Lock className="h-4 w-4" /> };

  return (
    <Card className="animate-fade-in-up p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <span className="mt-0.5 text-amd">
            <ShieldCheck className="h-4 w-4" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-white">
              兩段式晉升門檻 Two-stage gate
            </h3>
            <p className="mt-0.5 text-xs text-muted">
              Code gate 先行（held-out val 統計把關），HITL 只是通過後的否決安全鎖。
            </p>
          </div>
        </div>
        <Badge tone={outcome.tone}>
          {outcome.icon}
          {outcome.label}
        </Badge>
      </div>

      <div className="mt-4 space-y-3">
        {/* Stage 1 — code gate */}
        <div
          className={cn(
            "rounded-xl border p-4",
            gate.passed
              ? "border-emerald-500/25 bg-emerald-500/[0.04]"
              : "border-red-500/25 bg-red-500/[0.04]",
          )}
        >
          <StageTitle
            n="1"
            title="Code gate（統計把關）"
            right={
              <Badge tone={gate.passed ? "green" : "red"}>
                {gate.passed ? "PASSED" : "FAILED"}
              </Badge>
            }
          />
          <div className="mt-3">
            <SeparationCompare
              champion={gate.champion_separation}
              challenger={gate.challenger_separation}
              delta={gate.separation_delta}
            />
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <CheckRow
              ok={gate.p1_non_inferior}
              label="P1 非劣性 (平手偏袒現任)"
              value={`Δ ${signed(gate.separation_delta)}`}
            />
            <CheckRow
              ok={gate.p2_passed}
              label="P2 bootstrap CI 下界 > 0"
              value={`${signed(gate.bootstrap_lower_bound)} @ α=${gate.bootstrap_alpha}`}
            />
          </div>
          <p className="mt-2.5 font-mono text-[11px] leading-relaxed text-muted">
            {gate.reason}
          </p>
        </div>

        <div className="flex justify-center">
          <ArrowDown className="h-4 w-4 text-muted" />
        </div>

        {/* Stage 2 — HITL */}
        <div
          className={cn(
            "rounded-xl border p-4",
            hitl.consulted
              ? "border-line bg-surface-2/40"
              : "border-zinc-700/40 bg-surface-2/20 opacity-90",
          )}
        >
          <StageTitle
            n="2"
            title="HITL 安全鎖（僅能否決）"
            right={
              !hitl.consulted ? (
                <Badge tone="neutral">未諮詢 Not consulted</Badge>
              ) : hitl.vetoed ? (
                <Badge tone="amber">否決 Vetoed</Badge>
              ) : (
                <Badge tone="green">加簽 Approved</Badge>
              )
            }
          />
          {!hitl.consulted ? (
            <p className="mt-2.5 text-xs leading-relaxed text-red-200/90">
              Code gate 未通過，因此 <strong>完全沒有諮詢人類</strong>——
              人類永遠無法覆寫失敗的 code gate。這正是重點：先用 code 統計把關，HITL 只是安全鎖。
            </p>
          ) : (
            <p className="mt-2.5 text-xs leading-relaxed text-muted">
              Code gate 已通過，才交由人類加簽。人類可以
              <strong className="text-zinc-200"> 否決 </strong>
              一個通過的挑戰者，但不能救回被 gate 擋下的挑戰者。
            </p>
          )}
        </div>
      </div>

      <div className="mt-4">
        <Callout
          tone={applied ? "success" : gate.passed ? "warning" : "danger"}
          title={
            applied
              ? `已晉升至 epoch ${result.epoch_id} · champion = ${result.champion_version}`
              : !gate.passed
                ? "現任 champion 保留（code gate 擋下）"
                : "現任 champion 保留（HITL 否決）"
          }
        >
          <span className="text-xs leading-relaxed">{result.reason}</span>
          {applied ? (
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge tone="neutral">soft-deleted {result.erased_memories}</Badge>
              <Badge tone="neutral">reconfirmed {result.reconfirmed_memories}</Badge>
              <Badge tone="green">physics_truth 永久保留</Badge>
            </div>
          ) : null}
        </Callout>
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Pareto frontier                                                        */
/* --------------------------------------------------------------------- */

function ObjectiveChip({ k, v }: { k: string; v: number }) {
  const tone =
    k === "parsimony"
      ? "text-slate-300"
      : v > 0
        ? "text-emerald-300"
        : v < 0
          ? "text-red-300"
          : "text-zinc-400";
  return (
    <div className="flex items-center justify-between gap-2 rounded-md border border-line bg-surface-2/50 px-2 py-1">
      <span className="truncate text-[11px] text-muted" title={k}>
        {objectiveLabel(k)}
      </span>
      <span className={cn("font-mono text-[11px] tabular-nums", tone)}>
        {v.toFixed(2)}
      </span>
    </div>
  );
}

export function FrontierPanel({
  frontier,
  title = "Pareto frontier（population 搜尋）",
  desc = "top-K 非 dominated 的挑戰者 rubric；最佳者（BBε 下界最高）送 code gate。",
}: {
  frontier: Frontier;
  title?: string;
  desc?: string;
}) {
  if (!frontier || !frontier.members?.length) {
    return (
      <Card className="p-5">
        <div className="flex items-center gap-2.5">
          <GitBranch className="h-4 w-4 text-amd" />
          <h3 className="text-sm font-semibold text-white">{title}</h3>
        </div>
        <p className="mt-3 text-sm text-muted">尚無 frontier 資料（先提出挑戰者）。</p>
      </Card>
    );
  }

  const sorted = [...frontier.members].sort((a, b) => b.bbe - a.bbe);

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <GitBranch className="mt-0.5 h-4 w-4 text-amd" />
          <div>
            <h3 className="text-sm font-semibold text-white">{title}</h3>
            <p className="mt-0.5 text-xs text-muted">{desc}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="amd">size {frontier.size}</Badge>
          <Badge tone="neutral">top-K {frontier.top_k}</Badge>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {frontier.objectives.map((o) => (
          <span
            key={o}
            className="rounded-full border border-line bg-surface-2/60 px-2 py-0.5 text-[11px] text-zinc-300"
          >
            {objectiveLabel(o)}
          </span>
        ))}
      </div>

      <div className="mt-4 space-y-3">
        {sorted.map((m) => {
          const isBest = m.version === frontier.best_version;
          return (
            <div
              key={m.version}
              className={cn(
                "rounded-xl border p-3.5",
                isBest
                  ? "border-emerald-500/40 bg-emerald-500/[0.05]"
                  : "border-line bg-surface-2/40",
              )}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-zinc-200">
                    {m.version}
                  </span>
                  {isBest ? (
                    <Badge tone="green">
                      <Trophy className="h-3 w-3" />
                      BEST → gate
                    </Badge>
                  ) : null}
                </div>
                <span className="font-mono text-xs tabular-nums text-zinc-300">
                  BBε {signed(m.bbe)}
                </span>
              </div>

              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-muted">新增準則:</span>
                {m.added_criteria.length ? (
                  m.added_criteria.map((c) => (
                    <span
                      key={c}
                      className="rounded bg-amd/15 px-1.5 py-0.5 font-mono text-[10px] text-red-200"
                    >
                      {c}
                    </span>
                  ))
                ) : (
                  <span className="text-[11px] text-muted">（無 · 現任）</span>
                )}
                {m.parent_version ? (
                  <span className="ml-1 text-[11px] text-muted">
                    ← {m.parent_version}
                  </span>
                ) : null}
              </div>

              <div className="mt-2.5 grid gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
                {Object.entries(m.objectives)
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([k, v]) => (
                    <ObjectiveChip key={k} k={k} v={v} />
                  ))}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Exploitation (hack ratio)                                             */
/* --------------------------------------------------------------------- */

export function ExploitationCell({
  report,
  title,
}: {
  report: ExploitationReport;
  title: React.ReactNode;
}) {
  const detected = report.exploitation_detected;
  return (
    <div
      className={cn(
        "rounded-xl border p-4",
        detected
          ? "border-red-500/30 bg-red-500/[0.05]"
          : "border-emerald-500/25 bg-emerald-500/[0.04]",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-zinc-200">{title}</span>
        <Badge tone={detected ? "red" : "green"}>
          {detected ? (
            <>
              <Swords className="h-3 w-3" />
              EXPLOITATION
            </>
          ) : (
            <>
              <ShieldCheck className="h-3 w-3" />
              clean
            </>
          )}
        </Badge>
      </div>

      <div className="mt-2 flex items-baseline gap-2">
        <span
          className={cn(
            "text-3xl font-semibold tabular-nums",
            detected ? "text-red-300" : "text-emerald-300",
          )}
        >
          {fixed(report.mean_hack_ratio, 3)}
        </span>
        <span className="text-[11px] text-muted">
          hack ratio (strict/loose) · n={report.n_samples}
        </span>
      </div>

      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[11px] text-muted">容忍度 tolerance（before → after）</span>
          <Badge tone={report.tightened ? "amber" : "neutral"}>
            strictness · {report.strictness_level}
          </Badge>
        </div>
        <ToleranceLadder
          before={report.tolerances_before}
          after={report.tolerances_after}
        />
      </div>

      <div className="mt-3 space-y-0.5">
        <KeyVal
          k="tightened"
          v={report.tightened ? "是 收緊一級" : "否"}
        />
        <KeyVal
          k="adversarial injection"
          v={report.trigger_adversarial_injection ? "觸發" : "未觸發"}
        />
        <KeyVal k="reason" v={<span className="font-mono text-[11px]">{report.reason}</span>} />
      </div>
    </div>
  );
}

export function ExploitationPanel({
  champion,
  challenger,
}: {
  champion: ExploitationReport;
  challenger: ExploitationReport;
}) {
  return (
    <Card className="animate-fade-in-up p-5">
      <div className="flex items-start gap-2.5">
        <Crosshair className="mt-0.5 h-4 w-4 text-amd" />
        <div>
          <h3 className="text-sm font-semibold text-white">
            對抗 / 剝削偵測 Exploitation（hack ratio）
          </h3>
          <p className="mt-0.5 text-xs text-muted">
            hack ratio 過低 = rubric 有 poison-pill 盲點（loose 放行 strict 會擋的）；
            偵測到就自動收緊 tolerance。backend: {champion.backend}
          </p>
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <ExploitationCell report={champion} title="現任 Champion" />
        <ExploitationCell report={challenger} title="挑戰者 Challenger" />
      </div>
    </Card>
  );
}

/** Single-report exploitation card (used for the report's current-champion hack ratio). */
export function ExploitationSummaryCard({
  report,
}: {
  report: ExploitationReport;
}) {
  return (
    <Card className="p-5">
      <div className="flex items-start gap-2.5">
        <Sparkles className="mt-0.5 h-4 w-4 text-amd" />
        <div>
          <h3 className="text-sm font-semibold text-white">
            現任 rubric 剝削偵測 Hack ratio
          </h3>
          <p className="mt-0.5 text-xs text-muted">
            對當前 champion 的 held-out 評估（不落盤，僅顯示若偵測到會如何收緊）。
          </p>
        </div>
      </div>
      <div className="mt-4">
        <ExploitationCell report={report} title="現任 Champion" />
      </div>
    </Card>
  );
}

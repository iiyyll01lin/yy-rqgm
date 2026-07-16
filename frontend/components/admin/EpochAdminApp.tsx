"use client";

import * as React from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Ban,
  Check,
  Cpu,
  FlaskConical,
  GitBranch,
  RefreshCw,
} from "lucide-react";
import {
  approveEpoch,
  getHealth,
  getReport,
  proposeEpoch,
} from "@/lib/api";
import { cn } from "@/lib/format";
import type {
  EpochApproveResponse,
  EpochProposeResponse,
  EpochReport,
  EvaluatorHealth,
  HealthResponse,
} from "@/lib/types";
import { Badge, Button, Callout, Card, Spinner, Stat } from "@/components/ui";
import { SourceBanner } from "@/components/SourceBanner";
import {
  DataSplitsCard,
  JudgeAgreementCard,
  MemoryCard,
  OverAcceptanceCard,
  OverOptimizationCard,
  ProvenanceCard,
  SeparationCard,
} from "./ReportPanels";
import {
  ExploitationPanel,
  ExploitationSummaryCard,
  FrontierPanel,
  GatePanel,
} from "./EvolutionPanels";
import { KeyVal, SeparationCompare, pct, signed } from "./shared";

/* --------------------------------------------------------------------- */
/* Header                                                                 */
/* --------------------------------------------------------------------- */

function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-amd to-amd-bright shadow-[0_6px_18px_-6px_rgba(237,28,36,0.8)]">
            <Cpu className="h-5 w-5 text-white" />
          </div>
          <div className="leading-tight">
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-tight text-white">
                AgentForge
              </span>
              <span className="rounded-full border border-line px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted">
                Epoch Admin
              </span>
            </div>
            <div className="hidden text-[11px] text-muted sm:block">
              RQGM 進化控制台 · code-gate 先行、HITL 為安全鎖
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <SourceBanner />
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface-2 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            回精靈
          </Link>
        </div>
      </div>
    </header>
  );
}

/* --------------------------------------------------------------------- */
/* Evaluator summary strip (consumes GET /health `evaluator`)             */
/* --------------------------------------------------------------------- */

function evaluatorFrom(
  health: HealthResponse | null,
  report: EpochReport | null,
): EvaluatorHealth | null {
  if (health?.evaluator && !("error" in health.evaluator)) {
    return health.evaluator;
  }
  if (report) {
    return {
      rqgm_backend: report.rqgm_backend,
      val_separation: report.separation.val.separation,
      test_separation: report.separation.test.separation,
      proxy_gold_separation_gap: report.over_optimization.separation_gap,
      over_acceptance_rate: report.over_acceptance.over_acceptance_rate,
      hack_ratio: report.hack_ratio.mean_hack_ratio,
      exploitation_detected: report.hack_ratio.exploitation_detected,
      tolerance_levels: report.hack_ratio.tolerances_after.length,
      val_judge_accuracy: report.judge_agreement.val.accuracy,
      val_judge_kappa: report.judge_agreement.val.cohen_kappa,
    };
  }
  return null;
}

function SummaryStrip({
  report,
  health,
}: {
  report: EpochReport;
  health: HealthResponse | null;
}) {
  const ev = evaluatorFrom(health, report);
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="amd">epoch {report.epoch_id}</Badge>
          <Badge tone="neutral">
            champion · <span className="font-mono">{report.champion_version}</span>
          </Badge>
          <Badge tone="sky">RQGM · {report.rqgm_backend}</Badge>
          <Badge tone={ev?.exploitation_detected ? "red" : "green"}>
            {ev?.exploitation_detected ? "exploitation detected" : "no exploitation"}
          </Badge>
        </div>
      </div>
      {ev ? (
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Val separation" value={ev.val_separation.toFixed(3)} />
          <Stat label="Test separation" value={ev.test_separation.toFixed(3)} />
          <Stat
            label="Proxy−Gold gap"
            value={`${ev.proxy_gold_separation_gap >= 0 ? "+" : ""}${ev.proxy_gold_separation_gap.toFixed(3)}`}
            accent={ev.proxy_gold_separation_gap > 0.15 ? "#f87171" : undefined}
          />
          <Stat
            label="Over-acceptance"
            value={pct(ev.over_acceptance_rate)}
            accent={ev.over_acceptance_rate > 0.2 ? "#f87171" : undefined}
          />
          <Stat
            label="Hack ratio"
            value={ev.hack_ratio == null ? "—" : ev.hack_ratio.toFixed(3)}
            accent={ev.exploitation_detected ? "#f87171" : "#34d399"}
          />
          <Stat label="Judge accuracy" value={pct(ev.val_judge_accuracy)} />
          <Stat label="Cohen's κ" value={ev.val_judge_kappa.toFixed(2)} />
          <Stat label="Tolerance levels" value={ev.tolerance_levels} />
        </div>
      ) : null}
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Diff view (rubric_diff)                                                */
/* --------------------------------------------------------------------- */

function DiffView({ diff }: { diff: string }) {
  return (
    <pre className="max-h-56 overflow-auto rounded-xl border border-line bg-surface-2/50 p-3 text-[11px] leading-relaxed">
      {diff.split("\n").map((line, i) => {
        const tone = line.startsWith("+")
          ? "text-emerald-300"
          : line.startsWith("-")
            ? "text-red-300"
            : line.startsWith("@@")
              ? "text-sky-300"
              : "text-muted";
        return (
          <div key={i} className={cn("font-mono", tone)}>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}

/* --------------------------------------------------------------------- */
/* Proposal (pending challenger) card                                     */
/* --------------------------------------------------------------------- */

function ProposalCard({
  proposal,
  approving,
  onApprove,
  onVeto,
}: {
  proposal: EpochProposeResponse;
  approving: "approve" | "veto" | null;
  onApprove: () => void;
  onVeto: () => void;
}) {
  const m = proposal.metrics;
  return (
    <Card className="animate-fade-in-up p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <FlaskConical className="mt-0.5 h-4 w-4 text-amd" />
          <div>
            <h3 className="text-sm font-semibold text-white">
              待審挑戰者 Pending challenger
            </h3>
            <p className="mt-0.5 font-mono text-xs text-zinc-300">
              {proposal.challenger_id}
            </p>
          </div>
        </div>
        <Badge tone="neutral">split · {m.split}</Badge>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <SeparationCompare
            champion={m.champion_separation}
            challenger={m.challenger_separation}
            delta={m.separation_delta}
          />
          <div className="mt-3 rounded-xl border border-line bg-surface-2/40 px-3 py-2">
            <KeyVal
              k="BBε lower bound"
              v={m.bbe_lower_bound == null ? "—" : signed(m.bbe_lower_bound)}
              mono
            />
            <KeyVal k="frontier size" v={m.frontier_size ?? "—"} mono />
            <KeyVal
              k="added criteria"
              v={
                m.added_criteria?.length ? m.added_criteria.join(", ") : "（無）"
              }
            />
          </div>
        </div>
        <div>
          <div className="mb-1.5 text-[11px] text-muted">rubric diff</div>
          <DiffView diff={proposal.rubric_diff} />
        </div>
      </div>

      <Callout tone="info" title="送交兩段式門檻">
        <span className="text-xs leading-relaxed">
          點下方任一鍵送出你的 HITL 決定：系統會
          <strong> 先跑 code gate </strong>
          （held-out val 統計）；只有通過 gate 才會採納你的加簽，任何情況下人類都無法覆寫失敗的 gate。
        </span>
      </Callout>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button onClick={onApprove} disabled={approving != null}>
          {approving === "approve" ? (
            <>
              <Spinner /> 送出中…
            </>
          ) : (
            <>
              <Check className="h-4 w-4" /> 加簽核准 Approve
            </>
          )}
        </Button>
        <Button variant="danger" onClick={onVeto} disabled={approving != null}>
          {approving === "veto" ? (
            <>
              <Spinner /> 送出中…
            </>
          ) : (
            <>
              <Ban className="h-4 w-4" /> 否決 Veto
            </>
          )}
        </Button>
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------- */
/* Main app                                                               */
/* --------------------------------------------------------------------- */

export function EpochAdminApp() {
  const [report, setReport] = React.useState<EpochReport | null>(null);
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  const [proposal, setProposal] = React.useState<EpochProposeResponse | null>(
    null,
  );
  const [approveResult, setApproveResult] =
    React.useState<EpochApproveResponse | null>(null);
  const [proposing, setProposing] = React.useState(false);
  const [approving, setApproving] = React.useState<"approve" | "veto" | null>(
    null,
  );
  const [actionError, setActionError] = React.useState<string | null>(null);

  const refresh = React.useCallback(async () => {
    const [rep, hp] = await Promise.all([getReport(), getHealth()]);
    setReport(rep);
    setHealth(hp);
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [rep, hp] = await Promise.all([getReport(), getHealth()]);
        if (cancelled) return;
        setReport(rep);
        setHealth(hp);
      } catch (err) {
        if (cancelled) return;
        setLoadError(
          err instanceof Error ? err.message : "無法載入 RQGM 報告",
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onPropose() {
    setProposing(true);
    setActionError(null);
    setApproveResult(null);
    try {
      const res = await proposeEpoch();
      setProposal(res);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "提出挑戰者失敗，請重試。",
      );
    } finally {
      setProposing(false);
    }
  }

  async function onDecision(approve: boolean) {
    setApproving(approve ? "approve" : "veto");
    setActionError(null);
    try {
      const res = await approveEpoch(approve);
      setApproveResult(res);
      if (res.applied) {
        setProposal(null);
        await refresh();
      }
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "送出決定失敗，請重試。",
      );
    } finally {
      setApproving(null);
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <div className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-24 text-muted">
            <Spinner /> 載入 RQGM 進化報告…
          </div>
        ) : loadError || !report ? (
          <Callout tone="danger" title="載入失敗">
            {loadError ?? "沒有可用的報告資料。"}
          </Callout>
        ) : (
          <div className="space-y-6">
            <SummaryStrip report={report} health={health} />

            {/* --- Evolution loop --- */}
            <section className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-widest text-amd">
                    Evolution loop
                  </div>
                  <h2 className="mt-0.5 text-lg font-semibold text-white">
                    進化迴圈：propose → code gate → HITL
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    onClick={() => refresh()}
                    aria-label="重新整理"
                  >
                    <RefreshCw className="h-4 w-4" /> 重新整理
                  </Button>
                  <Button onClick={onPropose} disabled={proposing}>
                    {proposing ? (
                      <>
                        <Spinner /> 搜尋 frontier…
                      </>
                    ) : (
                      <>
                        <GitBranch className="h-4 w-4" /> 提出挑戰者
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {actionError ? (
                <Callout tone="danger">{actionError}</Callout>
              ) : null}

              {!proposal && !approveResult ? (
                <Callout tone="neutral" icon={<GitBranch className="h-4 w-4" />}>
                  點「提出挑戰者」以 GEPA Pareto frontier population 搜尋出最佳挑戰者，
                  再交由 code gate + HITL 兩段式審核。
                </Callout>
              ) : null}

              {proposal ? (
                <ProposalCard
                  proposal={proposal}
                  approving={approving}
                  onApprove={() => onDecision(true)}
                  onVeto={() => onDecision(false)}
                />
              ) : null}

              {proposal ? <FrontierPanel frontier={proposal.frontier} /> : null}

              {approveResult ? <GatePanel result={approveResult} /> : null}
              {approveResult ? (
                <ExploitationPanel
                  champion={approveResult.champion_exploitation}
                  challenger={approveResult.challenger_exploitation}
                />
              ) : null}
            </section>

            {/* --- Transparency report --- */}
            <section className="space-y-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-widest text-amd">
                  Transparency report
                </div>
                <h2 className="mt-0.5 text-lg font-semibold text-white">
                  透明報告：GET /api/admin/report
                </h2>
              </div>

              <DataSplitsCard splits={report.data_splits} />
              <div className="grid gap-4 lg:grid-cols-2">
                <SeparationCard
                  val={report.separation.val}
                  test={report.separation.test}
                />
                <OverOptimizationCard data={report.over_optimization} />
              </div>
              <OverAcceptanceCard data={report.over_acceptance} />
              <div className="grid gap-4 lg:grid-cols-2">
                <JudgeAgreementCard
                  val={report.judge_agreement.val}
                  test={report.judge_agreement.test}
                />
                <ExploitationSummaryCard report={report.hack_ratio} />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <MemoryCard memory={report.memory} />
                {report.provenance ? (
                  <ProvenanceCard provenance={report.provenance} />
                ) : null}
              </div>
              {!proposal ? (
                <FrontierPanel
                  frontier={report.frontier}
                  title="最近的 Pareto frontier（已持久化）"
                  desc="上一輪 propose 存到 data/frontier 的 frontier 摘要；提出新挑戰者會即時更新。"
                />
              ) : null}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import {
  ArrowLeft,
  ArrowRight,
  CircleAlert,
  CircleCheck,
  Cpu,
  Gauge,
  Stethoscope,
  TriangleAlert,
} from "lucide-react";
import { DTYPES } from "@/lib/types";
import { SEQ_OPTIONS } from "@/lib/options";
import { TIER_ACCENT, cn, fmtGB, fmtTps } from "@/lib/format";
import {
  Badge,
  Button,
  Callout,
  Card,
  Field,
  SectionTitle,
  Segmented,
  Select,
  Spinner,
  Stat,
  TextInput,
} from "@/components/ui";
import { VramBar, VramLegend } from "@/components/viz";
import { sortTiers, useWizard } from "./WizardContext";

export function Step2Diagnostic() {
  const {
    tiers,
    models,
    hwMode,
    hwTierId,
    customMem,
    customBw,
    reqModelId,
    reqSeqLen,
    reqConcurrency,
    reqDtype,
    diagnoseResult,
    diagnoseLoading,
    update,
    runDiagnose,
    goTo,
    back,
    unlock,
  } = useWizard();

  const orderedTiers = sortTiers(tiers);
  const tier = tiers.find((t) => t.id === hwTierId);
  const hwMem = hwMode === "tier" ? tier?.memory_gb ?? 0 : customMem;
  const hwBw = hwMode === "tier" ? tier?.bandwidth_tbs ?? 0 : customBw;

  async function diagnose() {
    await runDiagnose({
      current_hardware:
        hwMode === "tier"
          ? { tier_id: hwTierId }
          : { custom: { memory_gb: customMem, bandwidth_tbs: customBw } },
      requirements: {
        model_id: reqModelId,
        seq_len: reqSeqLen,
        concurrency: reqConcurrency,
        dtype: reqDtype,
      },
    });
  }

  function continueToLab() {
    update({
      simModelId: reqModelId,
      simSeqLen: reqSeqLen,
      simDtype: reqDtype,
      simPopulation: Math.max(reqConcurrency, 8),
    });
    unlock(3);
    goTo(3);
  }

  return (
    <div className="space-y-6">
      <Card className="animate-fade-in-up p-6">
        <SectionTitle
          eyebrow="Step 2 · Constraint Diagnostic"
          title="診斷你目前硬體能跑什麼"
          desc="這一步由 deterministic 的「靜態硬體閘門」把關：純物理的 VRAM / 頻寬計算，永不進化，是可信度的地基。"
        />

        {/* Current hardware */}
        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
              <Cpu className="h-4 w-4 text-amd" /> 目前硬體
            </h3>
            <Segmented
              size="sm"
              value={hwMode}
              onChange={(v) => update({ hwMode: v })}
              options={[
                { value: "tier", label: "AMD 硬體層級" },
                { value: "custom", label: "自訂規格" },
              ]}
            />
          </div>

          {hwMode === "tier" ? (
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
              {orderedTiers.map((t) => {
                const accent = TIER_ACCENT[t.class];
                const selected = hwTierId === t.id;
                const simulated = t.class === "Instinct";
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => update({ hwTierId: t.id })}
                    className={cn(
                      "rounded-xl border p-3 text-left transition-all",
                      selected
                        ? "border-amd bg-amd/10"
                        : "border-line bg-surface-2/50 hover:border-zinc-600",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-semibold text-white">
                        {t.name}
                      </span>
                      {simulated ? (
                        <Badge tone="amber" className="shrink-0">
                          SIM
                        </Badge>
                      ) : null}
                    </div>
                    <div className="mt-1 flex items-center gap-1.5">
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium",
                          accent.bg,
                          accent.text,
                        )}
                      >
                        {t.class}
                      </span>
                      <span className="text-xs tabular-nums text-muted">
                        {t.memory_gb} GB · {t.bandwidth_tbs} TB/s
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="記憶體 / VRAM (GB)" htmlFor="cmem">
                <TextInput
                  id="cmem"
                  type="number"
                  min={1}
                  value={customMem}
                  onChange={(e) =>
                    update({ customMem: Number(e.target.value) || 0 })
                  }
                />
              </Field>
              <Field label="記憶體頻寬 (TB/s)" htmlFor="cbw">
                <TextInput
                  id="cbw"
                  type="number"
                  min={0.01}
                  step={0.01}
                  value={customBw}
                  onChange={(e) =>
                    update({ customBw: Number(e.target.value) || 0 })
                  }
                />
              </Field>
            </div>
          )}
        </div>

        {/* Requirements */}
        <div className="mt-6">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-zinc-200">
            <Gauge className="h-4 w-4 text-amd" /> 工作負載需求
          </h3>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="模型 (Model)" htmlFor="model">
              <Select
                id="model"
                value={reqModelId}
                onChange={(e) => update({ reqModelId: e.target.value })}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · {m.params_b}B
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="脈絡長度 (Sequence Length)" htmlFor="seq">
              <Select
                id="seq"
                value={reqSeqLen}
                onChange={(e) => update({ reqSeqLen: Number(e.target.value) })}
              >
                {SEQ_OPTIONS.map((s) => (
                  <option key={s} value={s}>
                    {s.toLocaleString("en-US")} tokens
                  </option>
                ))}
              </Select>
            </Field>
            <Field
              label="並發數 (Concurrency)"
              hint={`${reqConcurrency} 路`}
              htmlFor="conc"
            >
              <Select
                id="conc"
                value={reqConcurrency}
                onChange={(e) =>
                  update({ reqConcurrency: Number(e.target.value) })
                }
              >
                {[1, 2, 4, 8, 16, 32, 64].map((c) => (
                  <option key={c} value={c}>
                    {c} 路並發
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="精度 (Dtype)">
              <Segmented
                value={reqDtype}
                onChange={(v) => update({ reqDtype: v })}
                options={DTYPES.map((d) => ({
                  value: d,
                  label: d.toUpperCase(),
                }))}
              />
            </Field>
          </div>
        </div>

        <div className="mt-6">
          <Button onClick={diagnose} disabled={diagnoseLoading}>
            {diagnoseLoading ? (
              <>
                <Spinner /> 計算中…
              </>
            ) : (
              <>
                <Stethoscope className="h-4 w-4" /> 執行可行性診斷
              </>
            )}
          </Button>
        </div>
      </Card>

      {diagnoseResult ? (
        <Card className="animate-fade-in-up p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SectionTitle
              eyebrow="Gatekeeper Verdict"
              title="可行性判定"
            />
            {diagnoseResult.feasible ? (
              <Badge tone="green" className="text-sm">
                <CircleCheck className="h-4 w-4" /> 可行 · 硬體足以承載
              </Badge>
            ) : (
              <Badge tone="red" className="text-sm">
                <CircleAlert className="h-4 w-4" /> 超出硬體邊界
              </Badge>
            )}
          </div>

          <div className="mt-5 grid gap-6 lg:grid-cols-5">
            <div className="lg:col-span-3">
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className="font-medium text-zinc-200">VRAM 分解</span>
                <span className="text-xs text-muted">
                  目前硬體：{fmtGB(hwMem)} @ {hwBw} TB/s
                </span>
              </div>
              <VramBar
                breakdown={diagnoseResult.report.vram_breakdown}
                total={diagnoseResult.report.vram_total_gb}
                capacityGb={hwMem}
                height={30}
              />
              <div className="mt-4">
                <VramLegend
                  breakdown={diagnoseResult.report.vram_breakdown}
                  total={diagnoseResult.report.vram_total_gb}
                  columns={4}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 lg:col-span-2">
              <Stat
                label="VRAM 總需求"
                value={fmtGB(diagnoseResult.report.vram_total_gb)}
              />
              <Stat
                label="餘裕 Headroom"
                value={fmtGB(diagnoseResult.report.headroom_gb)}
                accent={
                  diagnoseResult.report.headroom_gb < 0
                    ? "#f87171"
                    : diagnoseResult.report.headroom_gb < 2
                      ? "#fbbf24"
                      : "#34d399"
                }
              />
              <Stat
                label="解碼吞吐 (估)"
                value={fmtTps(diagnoseResult.report.tokens_per_s_est)}
                sub="memory-bandwidth 上界"
              />
              <Stat
                label="精度 / 並發"
                value={`${reqDtype.toUpperCase()}`}
                sub={`${reqConcurrency} 路 · ${reqSeqLen.toLocaleString("en-US")} tok`}
              />
            </div>
          </div>

          {diagnoseResult.gaps.length > 0 ? (
            <div className="mt-6 space-y-3">
              <h3 className="text-sm font-semibold text-zinc-200">
                能力缺口（以你的領域語言說明）
              </h3>
              {diagnoseResult.gaps.map((gap, i) => (
                <Callout
                  key={i}
                  tone={diagnoseResult.feasible ? "warning" : "danger"}
                  icon={<TriangleAlert className="h-4 w-4" />}
                  title={
                    <span className="flex flex-wrap items-center gap-2">
                      {gap.constraint}
                      <span className="text-xs font-normal opacity-80">
                        需要 {gap.needed} · 目前 {gap.have}
                      </span>
                    </span>
                  }
                >
                  {gap.explanation_domain}
                </Callout>
              ))}
            </div>
          ) : (
            <div className="mt-6">
              <Callout tone="success" icon={<CircleCheck className="h-4 w-4" />}>
                目前硬體可穩定承載此工作負載。前往模擬實驗室，看看升級 AMD 層級還能解鎖多少能力。
              </Callout>
            </div>
          )}

          <div className="mt-6 flex items-center justify-between gap-4">
            <Button variant="ghost" onClick={back}>
              <ArrowLeft className="h-4 w-4" /> 上一步
            </Button>
            <Button onClick={continueToLab}>
              下一步：硬體模擬實驗室 <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      ) : (
        <div className="flex justify-start">
          <Button variant="ghost" onClick={back}>
            <ArrowLeft className="h-4 w-4" /> 上一步
          </Button>
        </div>
      )}
    </div>
  );
}

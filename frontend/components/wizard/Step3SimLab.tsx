"use client";

import * as React from "react";
import {
  ArrowLeft,
  ArrowRight,
  Boxes,
  CircleAlert,
  CircleCheck,
  Layers,
  Sparkles,
  Zap,
} from "lucide-react";
import { DTYPES } from "@/lib/types";
import {
  POPULATION_MAX,
  SEQ_OPTIONS,
  indexToSeq,
  seqToIndex,
} from "@/lib/options";
import { TIER_ACCENT, cn, fmtInt, fmtTps } from "@/lib/format";
import {
  Badge,
  Button,
  Callout,
  Card,
  Field,
  SectionTitle,
  Segmented,
  Select,
  Range,
  Spinner,
} from "@/components/ui";
import { MetricBars, VramBar, VramLegend } from "@/components/viz";
import { sortTiers, useWizard } from "./WizardContext";

export function Step3SimLab() {
  const {
    sessionId,
    tiers,
    models,
    simModelId,
    simSeqLen,
    simPopulation,
    simDtype,
    simPrefixRatio,
    simTierIds,
    simResult,
    simLoading,
    update,
    runSimulate,
    goTo,
    back,
    unlock,
  } = useWizard();

  const timer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Live, debounced simulation whenever a control changes.
  React.useEffect(() => {
    if (!sessionId || simTierIds.length === 0) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      runSimulate({
        model_id: simModelId,
        seq_len: simSeqLen,
        population: simPopulation,
        dtype: simDtype,
        prefix_ratio: simPrefixRatio,
        tier_ids: simTierIds,
      });
    }, 220);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [
    sessionId,
    simModelId,
    simSeqLen,
    simPopulation,
    simDtype,
    simPrefixRatio,
    simTierIds,
    runSimulate,
  ]);

  const byTier = new Map((simResult?.per_tier ?? []).map((r) => [r.tier_id, r]));
  const selectedTiers = sortTiers(
    tiers.filter((t) => simTierIds.includes(t.id)),
  );
  const kvSavings = simResult?.per_tier[0]?.kv_savings_from_prefix_pct ?? 0;

  const popItems = selectedTiers
    .map((t) => {
      const r = byTier.get(t.id);
      if (!r) return null;
      return {
        id: t.id,
        label: t.name,
        value: r.max_population,
        display: fmtInt(r.max_population),
        color: TIER_ACCENT[t.class].hex,
        highlight: t.id === "mi300x",
        simulated: t.class === "Instinct",
      };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);

  const tpsItems = selectedTiers
    .map((t) => {
      const r = byTier.get(t.id);
      if (!r) return null;
      return {
        id: t.id,
        label: t.name,
        value: r.tokens_per_s_est,
        display: fmtTps(r.tokens_per_s_est),
        color: TIER_ACCENT[t.class].hex,
        highlight: t.id === "mi300x",
        simulated: t.class === "Instinct",
      };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);

  // Headline advantage: MI300X max population vs a consumer baseline.
  const mi = byTier.get("mi300x");
  const base = byTier.get("rx_7900_xtx");
  const advantage =
    mi && base && base.max_population > 0
      ? mi.max_population / base.max_population
      : null;

  function toggleTier(id: string) {
    const next = simTierIds.includes(id)
      ? simTierIds.filter((t) => t !== id)
      : [...simTierIds, id];
    update({ simTierIds: next });
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Controls */}
        <Card className="animate-fade-in-up p-5 lg:col-span-4">
          <SectionTitle
            eyebrow="Step 3 · Simulation Lab"
            title="硬體模擬實驗室"
          />
          <p className="mt-2 text-sm leading-relaxed text-muted">
            調整下列參數，即時觀察 VRAM 組成與各 AMD 層級能解鎖的能力。物理計算是硬邊界。
          </p>

          <div className="mt-5 space-y-5">
            <Field label="模型 (Model)" htmlFor="sim-model">
              <Select
                id="sim-model"
                value={simModelId}
                onChange={(e) => update({ simModelId: e.target.value })}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · {m.params_b}B
                  </option>
                ))}
              </Select>
            </Field>

            <Field
              label="脈絡長度 (Sequence Length)"
              hint={`${simSeqLen.toLocaleString("en-US")} tokens`}
            >
              <Range
                aria-label="sequence length"
                min={0}
                max={SEQ_OPTIONS.length - 1}
                value={seqToIndex(simSeqLen)}
                onChange={(i) => update({ simSeqLen: indexToSeq(i) })}
              />
            </Field>

            <Field
              label="Population / 批次 (Batch)"
              hint={`${simPopulation} 條並行序列`}
            >
              <Range
                aria-label="population"
                min={1}
                max={POPULATION_MAX}
                value={simPopulation}
                onChange={(v) => update({ simPopulation: v })}
              />
            </Field>

            <Field label="精度 (Dtype)">
              <Segmented
                className="w-full"
                value={simDtype}
                onChange={(v) => update({ simDtype: v })}
                options={DTYPES.map((d) => ({
                  value: d,
                  label: d.toUpperCase(),
                }))}
              />
            </Field>

            <Field
              label="Prefix 共享比例 (Prefix Caching)"
              hint={`${Math.round(simPrefixRatio * 100)}% 共享`}
            >
              <Range
                aria-label="prefix ratio"
                min={0}
                max={0.9}
                step={0.05}
                value={simPrefixRatio}
                onChange={(v) => update({ simPrefixRatio: v })}
              />
            </Field>

            {simPopulation > 1 && simPrefixRatio > 0 ? (
              <Callout tone="info" icon={<Sparkles className="h-4 w-4" />}>
                Prefix caching 在此設定下省下約{" "}
                <strong>{kvSavings}%</strong> 的 KV 快取——這正是大 population
                搜尋 (如 MCTS) 能在 AMD 大顯存上規模化的關鍵。
              </Callout>
            ) : null}

            <div>
              <div className="mb-2 text-sm font-medium text-zinc-200">
                比較的層級
              </div>
              <div className="grid grid-cols-1 gap-1.5">
                {sortTiers(tiers).map((t) => {
                  const on = simTierIds.includes(t.id);
                  const accent = TIER_ACCENT[t.class];
                  return (
                    <label
                      key={t.id}
                      className={cn(
                        "flex cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-2 text-sm transition-colors",
                        on
                          ? "border-line bg-surface-2"
                          : "border-transparent opacity-60 hover:opacity-100",
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() => toggleTier(t.id)}
                        className="h-4 w-4 accent-[var(--color-amd)]"
                      />
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: accent.hex }}
                      />
                      <span className="truncate text-zinc-200">{t.name}</span>
                      <span className="ml-auto shrink-0 text-xs tabular-nums text-muted">
                        {t.memory_gb}GB
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-6 lg:col-span-8">
          {advantage && advantage > 1.2 ? (
            <Card className="animate-fade-in-up border-amber-500/30 bg-gradient-to-br from-amber-500/10 to-transparent p-5">
              <div className="flex items-start gap-3">
                <div className="rounded-xl bg-amber-500/15 p-2.5">
                  <Zap className="h-5 w-5 text-amber-300" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-white">
                      Instinct MI300X 優勢：192GB HBM3 / 5.3TB/s
                    </h3>
                    <Badge tone="amber">SIMULATED</Badge>
                  </div>
                  <p className="mt-1 text-sm leading-relaxed text-amber-100/90">
                    在目前設定下，MI300X 可承載約{" "}
                    <strong className="tabular-nums">
                      {fmtInt(mi!.max_population)}
                    </strong>{" "}
                    條並行序列，約為 Radeon RX 7900 XTX 的{" "}
                    <strong className="tabular-nums">
                      {advantage.toFixed(1)}×
                    </strong>
                    ——巨大顯存與頻寬讓大規模 population 搜尋與長期負結果記憶成為可能。
                  </p>
                </div>
              </div>
            </Card>
          ) : null}

          <Card className="animate-fade-in-up p-5">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                <Boxes className="h-4 w-4 text-amd" /> 各層級能解鎖的最大 population
              </h3>
              {simLoading ? (
                <span className="flex items-center gap-1.5 text-xs text-muted">
                  <Spinner className="h-3.5 w-3.5" /> 計算中
                </span>
              ) : null}
            </div>
            <div className="mt-4">
              {popItems.length ? (
                <MetricBars items={popItems} />
              ) : (
                <p className="text-sm text-muted">請至少選擇一個層級比較。</p>
              )}
            </div>

            <div className="mt-6 flex items-center gap-2 text-sm font-semibold text-zinc-200">
              <Zap className="h-4 w-4 text-amd" /> 解碼吞吐估計 (tokens/s)
            </div>
            <div className="mt-4">
              {tpsItems.length ? <MetricBars items={tpsItems} /> : null}
            </div>
          </Card>

          <Card className="animate-fade-in-up p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                <Layers className="h-4 w-4 text-amd" /> 各層級 VRAM 佔用 vs 容量
              </h3>
              <VramLegend
                breakdown={
                  simResult?.per_tier[0]?.vram_breakdown ?? {
                    weights: 0,
                    kv_cache: 0,
                    activations: 0,
                    overhead: 0,
                  }
                }
                total={simResult?.per_tier[0]?.vram_total_gb ?? 0}
                columns={4}
              />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {selectedTiers.map((t) => {
                const r = byTier.get(t.id);
                if (!r) return null;
                const accent = TIER_ACCENT[t.class];
                const simulated = t.class === "Instinct";
                return (
                  <div
                    key={t.id}
                    className={cn(
                      "rounded-xl border p-3.5",
                      t.id === "mi300x"
                        ? "border-amber-500/40 bg-amber-500/5"
                        : "border-line bg-surface-2/50",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: accent.hex }}
                        />
                        <span className="text-sm font-semibold text-white">
                          {t.name}
                        </span>
                        {simulated ? <Badge tone="amber">SIM</Badge> : null}
                      </div>
                      {r.feasible ? (
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400">
                          <CircleCheck className="h-3.5 w-3.5" /> 可行
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-red-400">
                          <CircleAlert className="h-3.5 w-3.5" /> 超載
                        </span>
                      )}
                    </div>
                    <div className="mt-3">
                      <VramBar
                        breakdown={r.vram_breakdown}
                        total={r.vram_total_gb}
                        capacityGb={t.memory_gb}
                        height={22}
                      />
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <div className="text-muted">max population</div>
                        <div className="font-semibold tabular-nums text-white">
                          {fmtInt(r.max_population)}
                        </div>
                      </div>
                      <div>
                        <div className="text-muted">tokens/s (估)</div>
                        <div className="font-semibold tabular-nums text-white">
                          {fmtTps(r.tokens_per_s_est)}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="mt-4 text-xs leading-relaxed text-muted">
              <Badge tone="amber" className="mr-1">
                SIMULATED
              </Badge>
              Instinct (MI300X / MI325X) 為資料中心硬體，此展示機僅具備 Ryzen AI +
              Radeon；Instinct 結果由第一性原理物理公式模擬，非實機量測。
            </p>
          </Card>

          <div className="flex items-center justify-between gap-4">
            <Button variant="ghost" onClick={back}>
              <ArrowLeft className="h-4 w-4" /> 上一步
            </Button>
            <Button
              onClick={() => {
                // Recommend the cheapest tier that is feasible at this population.
                const feasible = selectedTiers.find(
                  (t) => byTier.get(t.id)?.feasible,
                );
                update({
                  exportTierId: feasible?.id ?? "mi300x",
                });
                unlock(4);
                goTo(4);
              }}
            >
              下一步：匯出提案 <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { ArrowRight, Check, Layers, Sparkles, Wand2 } from "lucide-react";
import { WORKLOAD_TYPES } from "@/lib/options";
import { cn } from "@/lib/format";
import {
  Badge,
  Button,
  Callout,
  Card,
  Field,
  SectionTitle,
  Select,
  Spinner,
  TextArea,
  TextInput,
} from "@/components/ui";
import { useWizard } from "./WizardContext";

export function Step1Domain() {
  const {
    domain,
    description,
    workloadType,
    domainResult,
    selectedTemplateId,
    domainLoading,
    update,
    submitDomain,
    goTo,
    unlock,
  } = useWizard();

  const canAnalyze = domain.trim().length > 1 && description.trim().length > 3;

  async function analyze() {
    if (!canAnalyze) return;
    await submitDomain({
      domain: domain.trim(),
      description: description.trim(),
      workload_type: workloadType || undefined,
    });
  }

  return (
    <div className="space-y-6">
      <Card className="animate-fade-in-up p-6">
        <SectionTitle
          eyebrow="Step 1 · Domain"
          title="定義你的領域需求"
          desc="用你自己的產業語言描述需求即可。系統會把「教育使用者」的負擔接手過來——先理解你的場景，再對應到可執行的 Agent 工作流程範本。"
        />

        <div className="mt-6 grid gap-5">
          <Field label="領域 / 產業 (Domain)" htmlFor="domain">
            <TextInput
              id="domain"
              value={domain}
              onChange={(e) => update({ domain: e.target.value })}
              placeholder="例如：智慧製造 / 工廠品質檢測、金融風控、醫療影像…"
            />
          </Field>

          <Field
            label="需求描述 (Description)"
            hint="越具體，匹配越準"
            htmlFor="desc"
          >
            <TextArea
              id="desc"
              rows={4}
              value={description}
              onChange={(e) => update({ description: e.target.value })}
              placeholder="描述你想讓 AI Agent 做什麼、資料型態、延遲/隱私要求…"
            />
          </Field>

          <Field label="工作負載型態 (Workload Type)" htmlFor="workload">
            <Select
              id="workload"
              value={workloadType}
              onChange={(e) => update({ workloadType: e.target.value })}
            >
              {WORKLOAD_TYPES.map((w) => (
                <option key={w.value} value={w.value}>
                  {w.label}
                </option>
              ))}
            </Select>
          </Field>

          <div>
            <Button onClick={analyze} disabled={!canAnalyze || domainLoading}>
              {domainLoading ? (
                <>
                  <Spinner /> 分析中…
                </>
              ) : (
                <>
                  <Wand2 className="h-4 w-4" /> 分析領域並匹配工作流程
                </>
              )}
            </Button>
          </div>
        </div>
      </Card>

      {domainResult ? (
        <Card className="animate-fade-in-up p-6">
          <SectionTitle
            eyebrow="Matched Workflows"
            title="匹配到的工作流程範本"
            desc="選擇最貼近你需求的範本；此範本會決定後續的評估重點與匯出的部署骨架。"
            right={
              <Badge tone="amd">
                <Layers className="h-3.5 w-3.5" />
                {domainResult.matched_templates.length} 個候選
              </Badge>
            }
          />

          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {domainResult.matched_templates.map((tpl) => {
              const selected = selectedTemplateId === tpl.id;
              const recommended =
                tpl.id === domainResult.recommended_template_id;
              return (
                <button
                  key={tpl.id}
                  type="button"
                  onClick={() => update({ selectedTemplateId: tpl.id })}
                  className={cn(
                    "group relative rounded-xl border p-4 text-left transition-all",
                    selected
                      ? "border-amd bg-amd/10 shadow-[0_0_0_1px_rgba(237,28,36,0.5)]"
                      : "border-line bg-surface-2/50 hover:border-zinc-600",
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="font-semibold text-white">{tpl.name}</h3>
                    <span
                      className={cn(
                        "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border transition-colors",
                        selected
                          ? "border-amd bg-amd text-white"
                          : "border-zinc-600 text-transparent",
                      )}
                    >
                      <Check className="h-3.5 w-3.5" />
                    </span>
                  </div>
                  {recommended ? (
                    <div className="mt-1">
                      <Badge tone="green">
                        <Sparkles className="h-3 w-3" /> 推薦
                      </Badge>
                    </div>
                  ) : null}
                  <p className="mt-2 text-sm leading-relaxed text-muted">
                    {tpl.description}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {tpl.needs.map((need) => (
                      <span
                        key={need}
                        className="rounded-md border border-line bg-surface px-2 py-0.5 text-[11px] text-zinc-400"
                      >
                        {need}
                      </span>
                    ))}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-6 flex items-center justify-between gap-4">
            <Callout tone="info" icon={<Sparkles className="h-4 w-4" />}>
              硬體物理是<strong>不可進化的硬邊界</strong>；領域適配則交由會進化的評估器判斷。
            </Callout>
            <Button
              className="shrink-0"
              disabled={!selectedTemplateId}
              onClick={() => {
                unlock(2);
                goTo(2);
              }}
            >
              下一步：限制診斷 <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      ) : null}
    </div>
  );
}

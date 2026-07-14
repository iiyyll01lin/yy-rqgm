"use client";

import * as React from "react";
import {
  ArrowLeft,
  Check,
  Copy,
  Download,
  FileText,
  MessageSquareHeart,
  Package,
  Rocket,
} from "lucide-react";
import { TIER_ACCENT } from "@/lib/format";
import {
  Button,
  Callout,
  Card,
  Field,
  SectionTitle,
  Select,
  Spinner,
} from "@/components/ui";
import { Markdown } from "@/components/Markdown";
import { CodeFileViewer } from "@/components/CodeFileViewer";
import { FeedbackWidget } from "@/components/FeedbackWidget";
import { sortTiers, useWizard } from "./WizardContext";

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function Step4Export() {
  const {
    tiers,
    models,
    domainResult,
    selectedTemplateId,
    simModelId,
    simSeqLen,
    simPopulation,
    simDtype,
    simPrefixRatio,
    exportTierId,
    exportResult,
    exportLoading,
    exportError,
    update,
    runExport,
    back,
  } = useWizard();

  const [copied, setCopied] = React.useState(false);

  const orderedTiers = sortTiers(tiers);
  const templateOptions = domainResult?.matched_templates ?? [];
  const tier = tiers.find((t) => t.id === exportTierId);
  const model = models.find((m) => m.id === simModelId);

  async function generate() {
    if (!selectedTemplateId) return;
    // Forward the simulated sizing so the exported TCO/deploy files reflect the
    // exact scenario the user explored in the lab (not the backend defaults).
    await runExport({
      target_tier_id: exportTierId,
      model_id: simModelId,
      template_id: selectedTemplateId,
      seq_len: simSeqLen,
      concurrency: simPopulation,
      dtype: simDtype,
      prefix_ratio: simPrefixRatio,
    });
  }

  async function copyMarkdown() {
    if (!exportResult) return;
    try {
      await navigator.clipboard.writeText(exportResult.tco_markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="space-y-6">
      <Card className="animate-fade-in-up p-6">
        <SectionTitle
          eyebrow="Step 4 · Export"
          title="匯出 AMD 提案與可部署模板"
          desc="產生給決策者看的 TCO / ROI 提案，以及可直接在 AMD ROCm 上啟動的部署模板 (docker-compose + LangGraph)。"
        />

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <Field label="目標採購層級 (Target Tier)" htmlFor="ex-tier">
            <Select
              id="ex-tier"
              value={exportTierId}
              onChange={(e) => update({ exportTierId: e.target.value })}
            >
              {orderedTiers.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} · {t.memory_gb}GB
                  {t.class === "Instinct" ? " (SIM)" : ""}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="模型 (Model)" htmlFor="ex-model">
            <Select
              id="ex-model"
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
          <Field label="工作流程範本 (Template)" htmlFor="ex-tpl">
            <Select
              id="ex-tpl"
              value={selectedTemplateId ?? ""}
              onChange={(e) => update({ selectedTemplateId: e.target.value })}
            >
              {templateOptions.length ? (
                templateOptions.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))
              ) : (
                <option value={selectedTemplateId ?? ""}>
                  {selectedTemplateId ?? "—"}
                </option>
              )}
            </Select>
          </Field>
        </div>

        {tier ? (
          <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted">
            <span>目標配置：</span>
            <span
              className="rounded px-1.5 py-0.5 font-medium"
              style={{
                backgroundColor: `${TIER_ACCENT[tier.class].hex}22`,
                color: TIER_ACCENT[tier.class].hex,
              }}
            >
              {tier.name}
            </span>
            <span className="tabular-nums">
              {tier.memory_gb}GB @ {tier.bandwidth_tbs}TB/s
            </span>
            {model ? (
              <span>
                · {model.name} ({model.params_b}B)
              </span>
            ) : null}
          </div>
        ) : null}

        <div className="mt-5">
          <Button onClick={generate} disabled={exportLoading || !selectedTemplateId}>
            {exportLoading ? (
              <>
                <Spinner /> 產生中…
              </>
            ) : (
              <>
                <Rocket className="h-4 w-4" /> 生成提案與部署模板
              </>
            )}
          </Button>
          {exportError ? (
            <p className="mt-2 text-sm text-red-400">{exportError}</p>
          ) : null}
        </div>
      </Card>

      {exportResult ? (
        <>
          <Card className="animate-fade-in-up p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="flex items-center gap-2 text-base font-semibold text-white">
                <FileText className="h-5 w-5 text-amd" />
                AMD TCO &amp; 採購 ROI 提案
              </h3>
              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={copyMarkdown}>
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 text-emerald-400" /> 已複製
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4" /> 複製 Markdown
                    </>
                  )}
                </Button>
                <Button
                  variant="secondary"
                  onClick={() =>
                    downloadText(
                      "AMD_TCO_Proposal.md",
                      exportResult.tco_markdown,
                    )
                  }
                >
                  <Download className="h-4 w-4" /> 下載 .md
                </Button>
              </div>
            </div>
            <div className="mt-4 max-h-[520px] overflow-auto rounded-xl border border-line bg-surface-2/40 p-5">
              <Markdown>{exportResult.tco_markdown}</Markdown>
            </div>
          </Card>

          <Card className="animate-fade-in-up p-6">
            <h3 className="mb-4 flex items-center gap-2 text-base font-semibold text-white">
              <Package className="h-5 w-5 text-amd" />
              可部署模板 (Deployment Template)
            </h3>
            <CodeFileViewer files={exportResult.deploy_files} />
          </Card>

          <Card className="animate-fade-in-up p-6">
            <h3 className="mb-1 flex items-center gap-2 text-base font-semibold text-white">
              <MessageSquareHeart className="h-5 w-5 text-amd" />
              回饋 (教系統變聰明)
            </h3>
            <div className="mt-3">
              <FeedbackWidget />
            </div>
          </Card>
        </>
      ) : (
        <Callout tone="neutral" icon={<Package className="h-4 w-4" />}>
          尚未生成。選好目標層級後點擊上方按鈕，即可產生可交付的提案與部署檔案。
        </Callout>
      )}

      <div className="flex items-center justify-start">
        <Button variant="ghost" onClick={back}>
          <ArrowLeft className="h-4 w-4" /> 上一步
        </Button>
      </div>
    </div>
  );
}

"use client";

import Link from "next/link";
import { Cpu, GitBranch, ShieldCheck, Sparkles } from "lucide-react";
import { SourceBanner } from "@/components/SourceBanner";
import { Spinner } from "@/components/ui";
import { Stepper } from "./Stepper";
import { Step1Domain } from "./Step1Domain";
import { Step2Diagnostic } from "./Step2Diagnostic";
import { Step3SimLab } from "./Step3SimLab";
import { Step4Export } from "./Step4Export";
import { WizardProvider, useWizard } from "./WizardContext";

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
              <span className="hidden rounded-full border border-line px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted sm:inline">
                AMD Ecosystem
              </span>
            </div>
            <div className="hidden text-[11px] text-muted sm:block">
              自我進化的 AI Agent 選型與教育平台
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <SourceBanner />
          <Link
            href="/admin"
            className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface-2 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600"
            title="RQGM 進化控制台：code-gate、frontier、對抗樣本"
          >
            <GitBranch className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">進化控制台</span>
          </Link>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="mt-auto border-t border-line">
      <div className="mx-auto max-w-6xl px-4 py-6 text-xs leading-relaxed text-muted sm:px-6">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            靜態硬體閘門：deterministic 物理，永不進化
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-amd" />
            評估器：fuzzy 領域判斷，靠回饋自我進化
          </span>
        </div>
        <p className="mt-3 text-muted/80">
          Instinct (MI300X / MI325X) 結果為 SIMULATED；展示機僅具 Ryzen AI +
          Radeon。VRAM / 吞吐為第一性原理估算，實測以現場為準。
        </p>
      </div>
    </footer>
  );
}

function StepRouter() {
  const { step, catalogLoading, models } = useWizard();

  if (step > 1 && catalogLoading && models.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-24 text-muted">
        <Spinner /> 初始化硬體與模型資料…
      </div>
    );
  }

  switch (step) {
    case 1:
      return <Step1Domain />;
    case 2:
      return <Step2Diagnostic />;
    case 3:
      return <Step3SimLab />;
    case 4:
      return <Step4Export />;
    default:
      return null;
  }
}

function WizardShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
        <div className="mb-8 rounded-2xl border border-line bg-surface/60 px-4 py-4 sm:px-6">
          <Stepper />
        </div>
        <main>
          <StepRouter />
        </main>
      </div>
      <Footer />
    </div>
  );
}

export function WizardApp() {
  return (
    <WizardProvider>
      <WizardShell />
    </WizardProvider>
  );
}

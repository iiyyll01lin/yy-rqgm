"use client";

import * as React from "react";
import {
  createSession,
  getModels,
  getTiers,
  postDiagnose,
  postDomain,
  postExport,
  postSimulate,
} from "@/lib/api";
import {
  DEFAULT_DESCRIPTION,
  DEFAULT_DOMAIN,
} from "@/lib/options";
import type {
  DiagnoseRequest,
  DiagnoseResponse,
  DomainRequest,
  DomainResponse,
  ExportRequest,
  ExportResponse,
  ModelSpec,
  SimulateRequest,
  SimulateResponse,
  Tier,
} from "@/lib/types";

export type StepId = 1 | 2 | 3 | 4;

export interface WizardState {
  step: StepId;
  maxStep: StepId;
  sessionId: string | null;

  tiers: Tier[];
  models: ModelSpec[];
  catalogLoading: boolean;
  catalogError: string | null;

  // Step 1 — Domain
  domain: string;
  description: string;
  workloadType: string;
  domainResult: DomainResponse | null;
  selectedTemplateId: string | null;
  domainLoading: boolean;

  // Step 2 — Diagnostic
  hwMode: "tier" | "custom";
  hwTierId: string;
  customMem: number;
  customBw: number;
  reqModelId: string;
  reqSeqLen: number;
  reqConcurrency: number;
  reqDtype: string;
  diagnoseResult: DiagnoseResponse | null;
  diagnoseLoading: boolean;

  // Step 3 — Simulation Lab
  simModelId: string;
  simSeqLen: number;
  simPopulation: number;
  simDtype: string;
  simPrefixRatio: number;
  simTierIds: string[];
  simResult: SimulateResponse | null;
  simLoading: boolean;

  // Step 4 — Export
  exportTierId: string;
  exportResult: ExportResponse | null;
  exportLoading: boolean;
  exportError: string | null;
}

const INITIAL: WizardState = {
  step: 1,
  maxStep: 1,
  sessionId: null,

  tiers: [],
  models: [],
  catalogLoading: true,
  catalogError: null,

  domain: DEFAULT_DOMAIN,
  description: DEFAULT_DESCRIPTION,
  workloadType: "realtime",
  domainResult: null,
  selectedTemplateId: null,
  domainLoading: false,

  hwMode: "tier",
  hwTierId: "rx_7900_xtx",
  customMem: 24,
  customBw: 0.96,
  reqModelId: "llama-3.1-8b",
  reqSeqLen: 8192,
  reqConcurrency: 8,
  reqDtype: "fp16",
  diagnoseResult: null,
  diagnoseLoading: false,

  simModelId: "llama-3.1-8b",
  simSeqLen: 8192,
  simPopulation: 16,
  simDtype: "fp16",
  simPrefixRatio: 0.5,
  simTierIds: [],
  simResult: null,
  simLoading: false,

  exportTierId: "mi300x",
  exportResult: null,
  exportLoading: false,
  exportError: null,
};

interface WizardActions {
  update: (patch: Partial<WizardState>) => void;
  goTo: (step: StepId) => void;
  back: () => void;
  unlock: (step: StepId) => void;
  submitDomain: (body: DomainRequest) => Promise<DomainResponse | null>;
  runDiagnose: (body: DiagnoseRequest) => Promise<DiagnoseResponse | null>;
  runSimulate: (body: SimulateRequest) => Promise<SimulateResponse | null>;
  runExport: (body: ExportRequest) => Promise<ExportResponse | null>;
}

type WizardContextValue = WizardState & WizardActions;

const WizardContext = React.createContext<WizardContextValue | null>(null);

export function useWizard(): WizardContextValue {
  const ctx = React.useContext(WizardContext);
  if (!ctx) throw new Error("useWizard must be used within WizardProvider");
  return ctx;
}

export function WizardProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<WizardState>(INITIAL);

  const update = React.useCallback((patch: Partial<WizardState>) => {
    setState((s) => ({ ...s, ...patch }));
  }, []);

  const goTo = React.useCallback((step: StepId) => {
    setState((s) => ({
      ...s,
      step,
      maxStep: (Math.max(s.maxStep, step) as StepId),
    }));
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, []);

  const back = React.useCallback(() => {
    setState((s) => ({ ...s, step: (Math.max(1, s.step - 1) as StepId) }));
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, []);

  const unlock = React.useCallback((step: StepId) => {
    setState((s) => ({ ...s, maxStep: (Math.max(s.maxStep, step) as StepId) }));
  }, []);

  // Load catalogs + open a session once on mount.
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [sess, tiersRes, modelsRes] = await Promise.all([
          createSession(),
          getTiers(),
          getModels(),
        ]);
        if (cancelled) return;
        setState((s) => ({
          ...s,
          sessionId: sess.session_id,
          tiers: tiersRes.tiers,
          models: modelsRes.models,
          simTierIds: tiersRes.tiers.map((t) => t.id),
          catalogLoading: false,
          catalogError: null,
        }));
      } catch (err) {
        if (cancelled) return;
        setState((s) => ({
          ...s,
          catalogLoading: false,
          catalogError:
            err instanceof Error ? err.message : "無法載入硬體與模型資料",
        }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const submitDomain = React.useCallback(
    async (body: DomainRequest) => {
      if (!state.sessionId) return null;
      update({ domainLoading: true });
      try {
        const res = await postDomain(state.sessionId, body);
        setState((s) => ({
          ...s,
          domainResult: res,
          selectedTemplateId:
            s.selectedTemplateId ?? res.recommended_template_id,
          domainLoading: false,
        }));
        return res;
      } catch {
        update({ domainLoading: false });
        return null;
      }
    },
    [state.sessionId, update],
  );

  const runDiagnose = React.useCallback(
    async (body: DiagnoseRequest) => {
      if (!state.sessionId) return null;
      update({ diagnoseLoading: true });
      try {
        const res = await postDiagnose(state.sessionId, body);
        update({ diagnoseResult: res, diagnoseLoading: false });
        return res;
      } catch {
        update({ diagnoseLoading: false });
        return null;
      }
    },
    [state.sessionId, update],
  );

  const runSimulate = React.useCallback(
    async (body: SimulateRequest) => {
      if (!state.sessionId) return null;
      update({ simLoading: true });
      try {
        const res = await postSimulate(state.sessionId, body);
        update({ simResult: res, simLoading: false });
        return res;
      } catch {
        update({ simLoading: false });
        return null;
      }
    },
    [state.sessionId, update],
  );

  const runExport = React.useCallback(
    async (body: ExportRequest) => {
      if (!state.sessionId) return null;
      update({ exportLoading: true, exportError: null });
      try {
        const res = await postExport(state.sessionId, body);
        update({ exportResult: res, exportLoading: false });
        return res;
      } catch (err) {
        update({
          exportLoading: false,
          exportError:
            err instanceof Error ? err.message : "匯出產生失敗，請重試。",
        });
        return null;
      }
    },
    [state.sessionId, update],
  );

  const value: WizardContextValue = {
    ...state,
    update,
    goTo,
    back,
    unlock,
    submitDomain,
    runDiagnose,
    runSimulate,
    runExport,
  };

  return (
    <WizardContext.Provider value={value}>{children}</WizardContext.Provider>
  );
}

/* --------------------------------------------------------------------- */
/* Selectors / helpers                                                    */
/* --------------------------------------------------------------------- */

export function tierOrder(cls: Tier["class"]): number {
  return (
    { "Ryzen AI": 0, Radeon: 1, "Radeon PRO": 2, Instinct: 3 }[cls] ?? 99
  );
}

/** Sort tiers along the upgrade ladder (edge -> data center). */
export function sortTiers(tiers: Tier[]): Tier[] {
  return [...tiers].sort((a, b) => {
    const d = tierOrder(a.class) - tierOrder(b.class);
    return d !== 0 ? d : a.memory_gb - b.memory_gb;
  });
}

/**
 * Centralized AgentForge API client.
 *
 * Talks to the backend REST API at NEXT_PUBLIC_API_BASE (default
 * http://localhost:8000). Because the backend may not be running while the UI
 * is developed or demoed, this client has a graceful MOCK fallback:
 *
 *   NEXT_PUBLIC_USE_MOCK = "true"   -> always use local mock data
 *   NEXT_PUBLIC_USE_MOCK = "false"  -> always hit the live API (no fallback)
 *   (unset)                         -> "auto": try live, fall back to mock on
 *                                       any network/HTTP error
 *
 * The active data source is observable so the UI can surface a "demo data"
 * banner (see subscribeSource / getSource).
 */

import {
  mockApproveEpoch,
  mockCreateSession,
  mockDiagnose,
  mockDomain,
  mockExport,
  mockFeedback,
  mockGetModels,
  mockGetTiers,
  mockHealth,
  mockProposeEpoch,
  mockReport,
  mockSimulate,
} from "./mock";
import type {
  DiagnoseRequest,
  DiagnoseResponse,
  DomainRequest,
  DomainResponse,
  EpochApproveResponse,
  EpochProposeResponse,
  EpochReport,
  ExportRequest,
  ExportResponse,
  FeedbackRequest,
  FeedbackResponse,
  Gap,
  HealthResponse,
  ModelSpec,
  SessionResponse,
  SimulateRequest,
  SimulateResponse,
  Tier,
  TierClass,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

type MockMode = "always" | "never" | "auto";

function resolveMockMode(): MockMode {
  const v = (process.env.NEXT_PUBLIC_USE_MOCK ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(v)) return "always";
  if (["0", "false", "no", "off"].includes(v)) return "never";
  return "auto";
}

export const MOCK_MODE: MockMode = resolveMockMode();

/* --------------------------------------------------------------------- */
/* Observable data source (for the demo-mode banner)                      */
/* --------------------------------------------------------------------- */

export type DataSource = "unknown" | "live" | "mock";

let currentSource: DataSource =
  MOCK_MODE === "always" ? "mock" : "unknown";
const listeners = new Set<(s: DataSource) => void>();

function setSource(next: DataSource) {
  if (next === currentSource) return;
  currentSource = next;
  listeners.forEach((cb) => cb(next));
}

export function getSource(): DataSource {
  return currentSource;
}

export function subscribeSource(cb: (s: DataSource) => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/* --------------------------------------------------------------------- */
/* Core request helper                                                    */
/* --------------------------------------------------------------------- */

const MOCK_LATENCY_MS = 320;

function delay<T>(value: T, ms = MOCK_LATENCY_MS): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

/**
 * Circuit breaker for "auto" mode: once the live backend is found unreachable,
 * stop attempting (and timing out on) it so the standalone demo stays snappy.
 */
let autoFellBack = false;

interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  timeoutMs?: number;
}

/**
 * Perform a live request, transparently falling back to `mock` when allowed.
 * `mock` is a lazy factory so mock work is only done when actually needed.
 */
async function request<T>(
  path: string,
  options: RequestOptions,
  mock: () => T,
): Promise<T> {
  if (MOCK_MODE === "always" || (MOCK_MODE === "auto" && autoFellBack)) {
    setSource("mock");
    return delay(mock());
  }

  const { method = "GET", body, timeoutMs = 2500 } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${path}`);
    }
    const data = (await res.json()) as T;
    setSource("live");
    return data;
  } catch (err) {
    if (MOCK_MODE === "never") {
      throw err;
    }
    // "auto": degrade gracefully to mock data so the UI stays demoable, and
    // trip the circuit breaker so later calls skip the live attempt.
    autoFellBack = true;
    if (typeof console !== "undefined") {
      console.warn(
        `[AgentForge] Live API unavailable for ${path}; serving mock data.`,
        err,
      );
    }
    setSource("mock");
    return delay(mock());
  } finally {
    clearTimeout(timer);
  }
}

/* --------------------------------------------------------------------- */
/* Contract normalization (reconcile live backend shapes <-> UI shapes)   */
/* --------------------------------------------------------------------- */

/**
 * The live backend returns machine-value tier classes (e.g. "radeon"), while
 * the UI keys palettes/labels/sim badges off display values (e.g. "Radeon").
 * Map machine -> display; pass display values through unchanged (idempotent,
 * so the mock — which already uses display values — is unaffected).
 */
const TIER_CLASS_NORMALIZE: Record<string, TierClass> = {
  ryzen_ai: "Ryzen AI",
  radeon: "Radeon",
  radeon_pro: "Radeon PRO",
  instinct: "Instinct",
  "Ryzen AI": "Ryzen AI",
  Radeon: "Radeon",
  "Radeon PRO": "Radeon PRO",
  Instinct: "Instinct",
};

function normalizeTierClass(cls: string): TierClass {
  return TIER_CLASS_NORMALIZE[cls] ?? (cls as TierClass);
}

function normalizeTiers(res: { tiers: Tier[] }): { tiers: Tier[] } {
  return {
    tiers: (res.tiers ?? []).map((t) => ({
      ...t,
      class: normalizeTierClass(t.class as unknown as string),
    })),
  };
}

/** Friendly, bilingual labels for the backend's machine-value gap constraints. */
const GAP_CONSTRAINT_LABEL: Record<string, string> = {
  vram_gb: "VRAM 記憶體容量 (VRAM)",
  throughput_tokens_per_s: "解碼吞吐 / 延遲 (Throughput)",
};

const GAP_CONSTRAINT_UNIT: Record<string, string> = {
  vram_gb: "GB",
  throughput_tokens_per_s: "tokens/s",
};

/**
 * The live backend returns gap `needed`/`have` as bare numbers and `constraint`
 * as a machine id; the mock returns pre-formatted strings + friendly labels.
 * Normalize the live shape into the UI's string shape (idempotent for the mock).
 */
function normalizeGap(raw: Gap): Gap {
  const unit = GAP_CONSTRAINT_UNIT[raw.constraint] ?? "";
  const fmt = (v: unknown): string => {
    if (typeof v === "number") {
      const n = Number.isInteger(v) ? v.toString() : v.toFixed(1);
      return unit ? `${n} ${unit}` : n;
    }
    return String(v);
  };
  return {
    constraint: GAP_CONSTRAINT_LABEL[raw.constraint] ?? raw.constraint,
    needed: fmt(raw.needed),
    have: fmt(raw.have),
    explanation_domain: raw.explanation_domain,
  };
}

function normalizeDiagnose(res: DiagnoseResponse): DiagnoseResponse {
  return { ...res, gaps: (res.gaps ?? []).map(normalizeGap) };
}

/* --------------------------------------------------------------------- */
/* Typed endpoint methods (mirror the backend contract)                   */
/* --------------------------------------------------------------------- */

export function createSession(): Promise<SessionResponse> {
  return request("/api/session", { method: "POST" }, mockCreateSession);
}

export function getTiers(): Promise<{ tiers: Tier[] }> {
  return request("/api/tiers", { method: "GET" }, mockGetTiers).then(
    normalizeTiers,
  );
}

export function getModels(): Promise<{ models: ModelSpec[] }> {
  return request("/api/models", { method: "GET" }, mockGetModels);
}

export function postDomain(
  sessionId: string,
  body: DomainRequest,
): Promise<DomainResponse> {
  return request(
    `/api/session/${sessionId}/domain`,
    { method: "POST", body },
    () => mockDomain(sessionId, body),
  );
}

export function postDiagnose(
  sessionId: string,
  body: DiagnoseRequest,
): Promise<DiagnoseResponse> {
  return request(
    `/api/session/${sessionId}/diagnose`,
    { method: "POST", body },
    () => mockDiagnose(sessionId, body),
  ).then(normalizeDiagnose);
}

export function postSimulate(
  sessionId: string,
  body: SimulateRequest,
): Promise<SimulateResponse> {
  return request(
    `/api/session/${sessionId}/simulate`,
    { method: "POST", body },
    () => mockSimulate(sessionId, body),
  );
}

export function postExport(
  sessionId: string,
  body: ExportRequest,
): Promise<ExportResponse> {
  return request(
    `/api/session/${sessionId}/export`,
    { method: "POST", body },
    () => mockExport(sessionId, body),
  );
}

export function postFeedback(
  sessionId: string,
  body: FeedbackRequest,
): Promise<FeedbackResponse> {
  return request(
    `/api/session/${sessionId}/feedback`,
    { method: "POST", body },
    () => mockFeedback(sessionId, body),
  );
}

/* --------------------------------------------------------------------- */
/* Admin / RQGM epoch-evolution endpoints                                 */
/* --------------------------------------------------------------------- */

/** GET /api/admin/report — RQGM transparency report for the current champion. */
export function getReport(): Promise<EpochReport> {
  return request("/api/admin/report", { method: "GET" }, mockReport);
}

/** POST /api/admin/epoch/propose — GEPA Pareto-frontier population search. */
export function proposeEpoch(): Promise<EpochProposeResponse> {
  return request("/api/admin/epoch/propose", { method: "POST" }, mockProposeEpoch);
}

/**
 * POST /api/admin/epoch/approve — two-stage gate. The code gate runs first and
 * can block regardless of `approve`; the HITL boolean is a veto-only safety lock
 * that is consulted ONLY after the code gate passes.
 */
export function approveEpoch(approve: boolean): Promise<EpochApproveResponse> {
  return request(
    "/api/admin/epoch/approve",
    { method: "POST", body: { approve } },
    () => mockApproveEpoch(approve),
  );
}

/** GET /health — includes the compact `evaluator` summary (val/test sep, κ, …). */
export function getHealth(): Promise<HealthResponse> {
  return request("/health", { method: "GET" }, mockHealth);
}

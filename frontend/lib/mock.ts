/**
 * Mock / fallback data layer.
 *
 * Mirrors the backend REST contract so the wizard is fully demoable with no
 * server running. VRAM/throughput numbers are computed from the same
 * first-principles physics as the (future) deterministic gatekeeper, so every
 * control in the Simulation Lab produces believable, responsive results.
 */

import {
  computeVram,
  maxPopulation,
  prefixSavingsPct,
  tokensPerSecond,
  type VramInput,
} from "./vram";
import type {
  DiagnoseRequest,
  DiagnoseResponse,
  DomainRequest,
  DomainResponse,
  EpochApproveResponse,
  EpochProposeResponse,
  EpochReport,
  ExploitationReport,
  ExportRequest,
  ExportResponse,
  FeedbackRequest,
  FeedbackResponse,
  Frontier,
  GateResult,
  HealthResponse,
  ModelSpec,
  Gap,
  OverAcceptance,
  OverAcceptanceSample,
  Provenance,
  SessionResponse,
  SimulateRequest,
  SimulateResponse,
  Tier,
  TierSimResult,
  WorkflowTemplate,
} from "./types";

/* --------------------------------------------------------------------- */
/* Static hardware + model catalogs                                       */
/* --------------------------------------------------------------------- */

/**
 * AMD tier ladder from edge NPU up to data-center Instinct. Instinct entries
 * are SIMULATED (the demo box only ships Ryzen AI + Radeon silicon).
 */
export const MOCK_TIERS: Tier[] = [
  {
    id: "ryzen_ai_max_395",
    name: "Ryzen AI Max+ 395",
    class: "Ryzen AI",
    memory_gb: 128,
    bandwidth_tbs: 0.256,
    form_factor: "APU · 迷你工作站/筆電 (XDNA2 NPU + RDNA3.5 iGPU)",
    has_npu: true,
    tops_npu: 50,
    price_usd_est: 1999,
    notes: "統一記憶體架構，最高 128GB 可作 VRAM；NPU 適合低功耗常駐推論與邊緣部署。",
  },
  {
    id: "rx_7900_xtx",
    name: "Radeon RX 7900 XTX",
    class: "Radeon",
    memory_gb: 24,
    bandwidth_tbs: 0.96,
    form_factor: "桌上型 dGPU (RDNA3, GDDR6)",
    has_npu: false,
    price_usd_est: 999,
    notes: "消費級 dGPU，ROCm 支援；高頻寬適合單機低延遲原型驗證 (PoC)。",
  },
  {
    id: "w7900",
    name: "Radeon PRO W7900",
    class: "Radeon PRO",
    memory_gb: 48,
    bandwidth_tbs: 0.864,
    form_factor: "工作站 dGPU (RDNA3, 48GB GDDR6)",
    has_npu: false,
    price_usd_est: 3999,
    notes: "工作站級 48GB 大顯存，單卡可跑中型模型或較大並發，適合部門級部署。",
  },
  {
    id: "mi300x",
    name: "Instinct MI300X",
    class: "Instinct",
    memory_gb: 192,
    bandwidth_tbs: 5.3,
    form_factor: "資料中心 OAM (CDNA3, 192GB HBM3)",
    has_npu: false,
    price_usd_est: 15000,
    notes:
      "SIMULATED — 資料中心級 192GB HBM3 / 5.3TB/s；巨大顯存與頻寬支撐大規模 population 搜尋與長期負結果記憶。",
  },
  {
    id: "mi325x",
    name: "Instinct MI325X",
    class: "Instinct",
    memory_gb: 256,
    bandwidth_tbs: 6.0,
    form_factor: "資料中心 OAM (CDNA3, 256GB HBM3E)",
    has_npu: false,
    price_usd_est: 20000,
    notes:
      "SIMULATED — 256GB HBM3E / 6.0TB/s；為最嚴苛的高並發與超長脈絡工作負載預留成長空間。",
  },
];

/** Representative open-weight models spanning edge to data-center scale. */
export const MOCK_MODELS: ModelSpec[] = [
  {
    id: "llama-3.2-3b",
    name: "Llama 3.2 3B Instruct",
    params_b: 3,
    n_layers: 28,
    n_kv_heads: 8,
    head_dim: 128,
    hidden: 3072,
    context_len: 131072,
    dtype_default: "fp16",
  },
  {
    id: "qwen2.5-7b",
    name: "Qwen2.5 7B Instruct",
    params_b: 7,
    n_layers: 28,
    n_kv_heads: 4,
    head_dim: 128,
    hidden: 3584,
    context_len: 131072,
    dtype_default: "fp16",
  },
  {
    id: "llama-3.1-8b",
    name: "Llama 3.1 8B Instruct",
    params_b: 8,
    n_layers: 32,
    n_kv_heads: 8,
    head_dim: 128,
    hidden: 4096,
    context_len: 131072,
    dtype_default: "fp16",
  },
  {
    id: "deepseek-r1-distill-14b",
    name: "DeepSeek-R1-Distill-Qwen 14B",
    params_b: 14,
    n_layers: 48,
    n_kv_heads: 8,
    head_dim: 128,
    hidden: 5120,
    context_len: 131072,
    dtype_default: "fp16",
  },
  {
    id: "qwen2.5-32b",
    name: "Qwen2.5 32B Instruct",
    params_b: 32,
    n_layers: 64,
    n_kv_heads: 8,
    head_dim: 128,
    hidden: 5120,
    context_len: 131072,
    dtype_default: "fp16",
  },
  {
    id: "llama-3.1-70b",
    name: "Llama 3.1 70B Instruct",
    params_b: 70,
    n_layers: 80,
    n_kv_heads: 8,
    head_dim: 128,
    hidden: 8192,
    context_len: 131072,
    dtype_default: "fp16",
  },
];

/* --------------------------------------------------------------------- */
/* Domain workflow templates (drop-in "domain pack" flavour)              */
/* --------------------------------------------------------------------- */

interface TemplateSeed extends WorkflowTemplate {
  keywords: string[];
}

const TEMPLATE_SEEDS: TemplateSeed[] = [
  {
    id: "smart-manufacturing-qc",
    name: "工廠即時瑕疵檢測 Agent (Visual QC)",
    description:
      "產線影像即時判讀與缺陷分類，將品檢知識轉為可稽核的自動化決策，支援邊緣低延遲部署。",
    needs: ["低延遲影像判讀", "產線邊緣部署", "可稽核決策軌跡"],
    keywords: [
      "製造", "工廠", "瑕疵", "品檢", "品質", "qc", "defect", "產線",
      "manufacturing", "inspection", "檢測", "良率", "缺陷", "vision", "影像",
    ],
  },
  {
    id: "predictive-maintenance",
    name: "設備預測性維護 Agent (Predictive Maintenance)",
    description:
      "融合多路時序感測資料，偵測異常並推論根因，主動生成維修工單，降低非計畫停機。",
    needs: ["時序感測資料融合", "異常根因分析", "維修工單生成"],
    keywords: [
      "維護", "保養", "maintenance", "設備", "感測", "sensor", "異常",
      "anomaly", "預測", "故障", "停機", "振動", "溫度", "predictive",
    ],
  },
  {
    id: "doc-intelligence",
    name: "技術文件智能問答 Agent (Document Intelligence)",
    description:
      "對規格書、SOP 與知識庫做長文脈檢索問答，對齊多語專業術語，縮短現場查詢時間。",
    needs: ["長文脈 RAG", "規格書擷取", "多語術語對齊"],
    keywords: [
      "文件", "手冊", "規格", "document", "sop", "知識庫", "rag", "問答",
      "檢索", "manual", "說明書", "客服知識", "retrieval",
    ],
  },
  {
    id: "process-optimization",
    name: "製程參數最佳化 Agent (Process Optimization)",
    description:
      "以大規模平行搜尋 (population) 探索製程參數空間，透過代理模型模擬並保留負結果長期記憶。",
    needs: ["大規模平行搜尋 (population)", "代理模型模擬", "負結果長期記憶"],
    keywords: [
      "最佳化", "optimization", "參數", "製程", "recipe", "模擬",
      "simulation", "mcts", "搜尋", "配方", "調校", "population", "實驗",
    ],
  },
  {
    id: "support-triage",
    name: "客服/工單分流 Agent (Support Triage)",
    description:
      "對客戶請求做意圖分類與知識庫檢索，支援多輪對話並自動分流至正確處理流程。",
    needs: ["意圖分類", "知識庫檢索", "多輪對話"],
    keywords: [
      "客服", "support", "工單", "ticket", "客戶", "對話", "chatbot",
      "分流", "triage", "intent", "服務台", "helpdesk",
    ],
  },
];

/** Public template catalog (without internal keyword metadata). */
export const MOCK_TEMPLATES: WorkflowTemplate[] = TEMPLATE_SEEDS.map((t) => ({
  id: t.id,
  name: t.name,
  description: t.description,
  needs: t.needs,
}));

/* --------------------------------------------------------------------- */
/* Lightweight in-memory session context (enriches domain-language copy)  */
/* --------------------------------------------------------------------- */

interface SessionContext {
  domain: string;
  description: string;
  workload_type?: string;
  template_id: string;
}

const sessionStore = new Map<string, SessionContext>();

export function getModelById(id: string): ModelSpec | undefined {
  return MOCK_MODELS.find((m) => m.id === id);
}

export function getTierById(id: string): Tier | undefined {
  return MOCK_TIERS.find((t) => t.id === id);
}

function vramInputFor(
  model: ModelSpec,
  seqLen: number,
  population: number,
  dtype: string,
  prefixRatio = 0,
): VramInput {
  return {
    paramsB: model.params_b,
    nLayers: model.n_layers,
    nKvHeads: model.n_kv_heads,
    headDim: model.head_dim,
    hidden: model.hidden,
    seqLen,
    population,
    dtype,
    prefixRatio,
  };
}

/* --------------------------------------------------------------------- */
/* Endpoint mocks                                                         */
/* --------------------------------------------------------------------- */

export function mockCreateSession(): SessionResponse {
  const session_id = `sess_${Math.random().toString(36).slice(2, 10)}`;
  return { session_id };
}

export function mockGetTiers(): { tiers: Tier[] } {
  return { tiers: MOCK_TIERS };
}

export function mockGetModels(): { models: ModelSpec[] } {
  return { models: MOCK_MODELS };
}

export function mockDomain(
  sessionId: string,
  req: DomainRequest,
): DomainResponse {
  const haystack =
    `${req.domain} ${req.description} ${req.workload_type ?? ""}`.toLowerCase();

  const scored = TEMPLATE_SEEDS.map((seed) => {
    const score = seed.keywords.reduce(
      (acc, kw) => (haystack.includes(kw.toLowerCase()) ? acc + 1 : acc),
      0,
    );
    return { seed, score };
  }).sort((a, b) => b.score - a.score);

  const matched = scored.filter((s) => s.score > 0);
  // Always surface at least the top three candidates for a useful demo.
  const chosen = (matched.length >= 2 ? matched : scored).slice(0, 4);
  const recommended_template_id = chosen[0]?.seed.id ?? "smart-manufacturing-qc";

  sessionStore.set(sessionId, {
    domain: req.domain,
    description: req.description,
    workload_type: req.workload_type,
    template_id: recommended_template_id,
  });

  return {
    matched_templates: chosen.map(({ seed }) => ({
      id: seed.id,
      name: seed.name,
      description: seed.description,
      needs: seed.needs,
    })),
    recommended_template_id,
  };
}

export function mockDiagnose(
  sessionId: string,
  req: DiagnoseRequest,
): DiagnoseResponse {
  const model = getModelById(req.requirements.model_id) ?? MOCK_MODELS[2];
  const { concurrency, seq_len, dtype } = req.requirements;

  const hw = req.current_hardware;
  const tier = hw.tier_id ? getTierById(hw.tier_id) : undefined;
  const memory_gb = tier?.memory_gb ?? hw.custom?.memory_gb ?? 24;
  const bandwidth_tbs = tier?.bandwidth_tbs ?? hw.custom?.bandwidth_tbs ?? 0.96;

  const input = vramInputFor(model, seq_len, concurrency, dtype);
  const { breakdown, total } = computeVram(input);
  const tps = tokensPerSecond(bandwidth_tbs, {
    paramsB: model.params_b,
    nLayers: model.n_layers,
    nKvHeads: model.n_kv_heads,
    headDim: model.head_dim,
    seqLen: seq_len,
    dtype,
  });
  const headroom = Math.round((memory_gb - total) * 10) / 10;
  const feasible = total <= memory_gb;

  const ctx = sessionStore.get(sessionId);
  const domain = ctx?.domain?.trim() || "你的應用場景";

  const gaps: Gap[] = [];
  if (!feasible) {
    gaps.push({
      constraint: "VRAM 記憶體容量",
      needed: `${total} GB`,
      have: `${memory_gb} GB`,
      explanation_domain: `以「${domain}」的部署需求，模型權重加上 ${concurrency} 路並發、${seq_len.toLocaleString("en-US")} tokens 脈絡的 KV 快取共需約 ${total} GB，但目前硬體僅有 ${memory_gb} GB。這代表在產線尖峰或多工單同時湧入時會發生記憶體不足 (OOM)，導致 Agent 服務中斷。`,
    });
  } else if (headroom < 2) {
    gaps.push({
      constraint: "記憶體餘裕 (Headroom)",
      needed: "≥ 2 GB 緩衝",
      have: `${headroom} GB`,
      explanation_domain: `目前配置雖可勉強執行，但僅剩 ${headroom} GB 餘裕。一旦「${domain}」出現突發流量或脈絡變長，系統會非常接近 OOM 邊界，穩定性風險高。`,
    });
  }
  if (tps < 20) {
    gaps.push({
      constraint: "解碼吞吐 / 延遲",
      needed: "互動式體驗 ≥ 20 tokens/s",
      have: `${tps} tokens/s`,
      explanation_domain: `此硬體的記憶體頻寬為 ${bandwidth_tbs} TB/s，估計解碼速度約 ${tps} tokens/s。對「${domain}」要求的即時回應而言偏慢，現場人員會感到明顯等待；升級至更高頻寬的 AMD 層級可顯著縮短延遲。`,
    });
  }

  return {
    feasible,
    report: {
      vram_total_gb: total,
      vram_breakdown: breakdown,
      tokens_per_s_est: tps,
      headroom_gb: headroom,
    },
    gaps,
  };
}

export function mockSimulate(
  _sessionId: string,
  req: SimulateRequest,
): SimulateResponse {
  const model = getModelById(req.model_id) ?? MOCK_MODELS[2];
  const tierIds =
    req.tier_ids && req.tier_ids.length > 0
      ? req.tier_ids
      : MOCK_TIERS.map((t) => t.id);
  const prefixRatio = req.prefix_ratio ?? 0;

  const per_tier: TierSimResult[] = tierIds
    .map((tierId): TierSimResult | null => {
      const tier = getTierById(tierId);
      if (!tier) return null;
      const input = vramInputFor(
        model,
        req.seq_len,
        req.population,
        req.dtype,
        prefixRatio,
      );
      const { breakdown, total } = computeVram(input);
      const tps = tokensPerSecond(tier.bandwidth_tbs, {
        paramsB: model.params_b,
        nLayers: model.n_layers,
        nKvHeads: model.n_kv_heads,
        headDim: model.head_dim,
        seqLen: req.seq_len,
        dtype: req.dtype,
      });
      return {
        tier_id: tier.id,
        feasible: total <= tier.memory_gb,
        vram_total_gb: total,
        vram_breakdown: breakdown,
        tokens_per_s_est: tps,
        max_population: maxPopulation(tier.memory_gb, input),
        kv_savings_from_prefix_pct: prefixSavingsPct(
          req.population,
          prefixRatio,
        ),
      };
    })
    .filter((r): r is TierSimResult => r !== null);

  return { per_tier };
}

export function mockExport(
  sessionId: string,
  req: ExportRequest,
): ExportResponse {
  const tier = getTierById(req.target_tier_id) ?? MOCK_TIERS[3];
  const model = getModelById(req.model_id) ?? MOCK_MODELS[2];
  const template =
    MOCK_TEMPLATES.find((t) => t.id === req.template_id) ?? MOCK_TEMPLATES[0];
  const ctx = sessionStore.get(sessionId);
  const domain = ctx?.domain?.trim() || "智慧製造";

  return {
    tco_markdown: buildTcoMarkdown(domain, tier, model, template),
    deploy_files: buildDeployFiles(tier, model, template),
  };
}

export function mockFeedback(
  _sessionId: string,
  req: FeedbackRequest,
): FeedbackResponse {
  return {
    ok: true,
    stored_as: `anchor_feedback_${Date.now().toString(36)}_r${req.rating}`,
  };
}

/* --------------------------------------------------------------------- */
/* Export artifact generators                                            */
/* --------------------------------------------------------------------- */

const usd = (n?: number) =>
  n == null ? "—" : `$${n.toLocaleString("en-US")}`;

function buildTcoMarkdown(
  domain: string,
  tier: Tier,
  model: ModelSpec,
  template: WorkflowTemplate,
): string {
  const simulated = tier.class === "Instinct";
  const capex = tier.price_usd_est ?? 0;
  const power3yr = Math.round(((tier.class === "Instinct" ? 0.75 : 0.35) * 24 * 365 * 3 * 0.12));
  const maint3yr = Math.round(capex * 0.15);
  const tco3yr = capex + power3yr + maint3yr;

  return `# AMD TCO & 採購 ROI 提案

## 執行摘要 (Executive Summary)

針對 **${domain}** 場景導入 **${template.name}**，本提案建議採用 **AMD ${tier.name}** 作為部署平台。
此配置以開源 ROCm 技術棧執行 **${model.name}**，在資料主權、單位算力成本與長期擴充性上提供最佳的整體擁有成本 (TCO)。

> ${simulated ? "⚠️ 本層級 (Instinct) 之數據為 **模擬 (SIMULATED)**，供資料中心規模規劃參考。" : "本層級硬體可於現場實機驗證 (real hardware)。"}

## 1. 業務需求對應

| 業務需求 | 技術能力 |
| --- | --- |
${template.needs.map((n) => `| ${n} | 由 ${model.name} + AMD ${tier.class} 平台承載 |`).join("\n")}

## 2. 建議硬體配置

| 項目 | 規格 |
| --- | --- |
| 平台 | AMD ${tier.name} (${tier.class}) |
| 記憶體 | **${tier.memory_gb} GB** ${tier.class === "Instinct" ? "HBM3" : tier.has_npu ? "統一記憶體" : "GDDR6"} |
| 記憶體頻寬 | **${tier.bandwidth_tbs} TB/s** |
| 型態 | ${tier.form_factor} |
| NPU | ${tier.has_npu ? `是 (${tier.tops_npu} TOPS, XDNA2)` : "否"} |
| 建議模型 | ${model.name} (${model.params_b}B) |

## 3. 三年 TCO 估算 (每節點, 示意)

| 成本項目 | 金額 (USD) |
| --- | --- |
| 硬體資本支出 (CapEx) | ${usd(capex)} |
| 電力 (3 年, 估) | ${usd(power3yr)} |
| 維運 (3 年, 估) | ${usd(maint3yr)} |
| **三年總持有成本** | **${usd(tco3yr)}** |

## 4. ROI 論述

- **資料主權**：全程 on-prem / 私有雲，敏感的 ${domain} 資料不外流第三方 API。
- **邊際成本趨零**：一次性採購後，推論邊際成本遠低於按 token 計費的雲端 API。
- **開源棧無授權綁定**：ROCm + vLLM + LangGraph，避免專有平台的授權與退出成本。
- **升級路徑清晰**：沿 Ryzen AI → Radeon → Radeon PRO → Instinct 逐級擴充，投資可分階段。

## 5. 部署技術棧 (開源)

- **推論引擎**：${tier.has_npu ? "Lemonade (Ryzen AI NPU) / " : ""}vLLM on ROCm (\`rocm/vllm-dev\`)
- **編排**：LangGraph (Task Agent → Static Gatekeeper → Evaluator → HITL)
- **記憶體最佳化**：Prefix Caching (\`--enable-prefix-caching\`) 大幅降低共享 prefix 的 KV 成本
- **量化**：AMD Quark (int4 / fp8)

## 6. 風險與後續步驟

1. 以隨附的 \`docker-compose.yml\` 於目標硬體啟動 PoC，實測吞吐與延遲。
2. 收集 ${domain} 領域專家回饋，作為 Evaluator 自我進化的 ground-truth anchor。
3. 依實測結果決定量產層級與節點數。

---
*本提案由 AgentForge 自動生成；數值為第一性原理估算，實測數據以現場為準。*
`;
}

function buildDeployFiles(
  tier: Tier,
  model: ModelSpec,
  template: WorkflowTemplate,
): Record<string, string> {
  const modelRepo =
    {
      "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
      "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
      "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
      "deepseek-r1-distill-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
      "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
      "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    }[model.id] ?? "Qwen/Qwen2.5-7B-Instruct";

  const dockerCompose = `# AgentForge deploy template
# Target: AMD ${tier.name} (${tier.class})${
    tier.class === "Instinct" ? "  # SIMULATED tier" : ""
  }
# Workflow: ${template.name}
services:
  vllm:
    image: rocm/vllm-dev:main
    container_name: agentforge-vllm
    devices:
      - "/dev/kfd"
      - "/dev/dri"
    group_add:
      - "video"
      - "render"
    ipc: host
    security_opt:
      - "seccomp=unconfined"
    environment:
      - VLLM_ROCM_USE_AITER=1
      - HIP_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
    command: >
      vllm serve ${modelRepo}
      --dtype ${model.dtype_default}
      --max-model-len ${Math.min(model.context_len, 32768)}
      --enable-prefix-caching
      --gpu-memory-utilization 0.90

  qdrant:
    image: qdrant/qdrant:latest
    container_name: agentforge-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
`;

  const appPy = `"""AgentForge — LangGraph orchestration skeleton.

Flow: Task Agent -> Static Hardware Gatekeeper -> RQGM Evaluator -> HITL.
Target tier: ${tier.name} (${tier.class})
Model: ${model.name}
Workflow: ${template.name}
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI

# vLLM exposes an OpenAI-compatible endpoint on ROCm.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "${modelRepo}"


class AgentState(TypedDict, total=False):
    task: str
    hardware_ok: bool
    draft: str
    verdict: str


def gatekeeper(state: AgentState) -> AgentState:
    """Deterministic physics gate (never evolves). Replace with vram.py call."""
    # TODO: call backend/gatekeeper/vram.py to validate VRAM/bandwidth budget.
    state["hardware_ok"] = True
    return state


def task_agent(state: AgentState) -> AgentState:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a ${template.name} agent."},
            {"role": "user", "content": state["task"]},
        ],
    )
    state["draft"] = resp.choices[0].message.content or ""
    return state


def evaluator(state: AgentState) -> AgentState:
    """Fuzzy, evolving RQGM evaluator (deficit-scoring rubric)."""
    # TODO: wire backend/evaluator/judge.py (XML rubric + forced CoT).
    state["verdict"] = "accepted"
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("gatekeeper", gatekeeper)
    g.add_node("task_agent", task_agent)
    g.add_node("evaluator", evaluator)
    g.add_edge(START, "gatekeeper")
    g.add_edge("gatekeeper", "task_agent")
    g.add_edge("task_agent", "evaluator")
    g.add_edge("evaluator", END)
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        return g.compile(checkpointer=saver)


if __name__ == "__main__":
    app = build_graph()
    out = app.invoke(
        {"task": "Inspect the attached part for surface defects."},
        config={"configurable": {"thread_id": "demo"}},
    )
    print(out)
`;

  const readme = `# AgentForge Deployment — ${template.name}

Generated deployment template targeting **AMD ${tier.name}** (${tier.class}).
${tier.class === "Instinct" ? "\n> ⚠️ This tier is **SIMULATED** in the demo box. Sizing is derived from first-principles physics.\n" : ""}
## Prerequisites

- AMD ROCm-capable host (\`/dev/kfd\`, \`/dev/dri\` accessible)
- Docker + Docker Compose
- HuggingFace access to \`${modelRepo}\` (set \`HF_TOKEN\` if gated)

## 1. Start the inference + memory stack

\`\`\`bash
docker compose up -d
\`\`\`

vLLM will serve \`${modelRepo}\` on an OpenAI-compatible API at \`http://localhost:8000/v1\`.
Prefix caching is enabled (\`--enable-prefix-caching\`) to minimize KV cost on shared prompts.

## 2. Run the agent graph

\`\`\`bash
pip install langgraph langgraph-checkpoint-sqlite openai
python app.py
\`\`\`

## Sizing summary

| Field | Value |
| --- | --- |
| Model | ${model.name} (${model.params_b}B, ${model.dtype_default}) |
| Memory | ${tier.memory_gb} GB @ ${tier.bandwidth_tbs} TB/s |
| Context | up to ${Math.min(model.context_len, 32768).toLocaleString("en-US")} tokens |

Physics is a hard boundary; domain fit is judged by the evolving evaluator.
`;

  return {
    "docker-compose.yml": dockerCompose,
    "app.py": appPy,
    "README.md": readme,
  };
}

/* --------------------------------------------------------------------- */
/* RQGM epoch-evolution mocks (admin surface)                            */
/*                                                                        */
/* A small, DETERMINISTIC state machine so the epoch-admin UI renders     */
/* end-to-end with no backend. Narrative: champion-0 has a poison-pill    */
/* blind spot (exploitation detected). The first proposed challenger adds  */
/* the missing criteria and PASSES the code gate — approving it promotes   */
/* the epoch; vetoing freezes it. After promotion the champion is strong,  */
/* so the next challenger FAILS the code gate (and HITL is never even       */
/* consulted — a human cannot override a failed gate).                     */
/* --------------------------------------------------------------------- */

/** RQGM tolerance schedule (loosest → tightest); dropping a level = stricter. */
const RQGM_TOLERANCES = [0.0, 0.001, 0.01, 0.025, 0.05, 0.1];

const VAL_WEAK_IDS = [
  "anchor_val_thermal_ignored",
  "anchor_val_noise_gamed",
  "anchor_val_no_fallback",
  "anchor_val_overfit_prompt",
];
const VAL_STRONG_IDS = [
  "anchor_val_robust_qc",
  "anchor_val_layered_safety",
  "anchor_val_calibrated",
];
/** Planted flaw family per weak val anchor (for the P2 per-flaw breakdown). */
const VAL_WEAK_FLAWS = [
  "thermal_runaway",
  "sensor_noise",
  "no_fallback",
  "prompt_overfit",
];

interface MockEpochState {
  epoch: number;
  championVersion: string;
  // Separation on each held-out split: dev = frontier SELECTION, val = code
  // GATE (independent re-test), test = reporting-only GOLD. Kept distinct so the
  // mock honestly shows dev/val/test isolation (a dev win can still fail the val
  // gate — no winner's curse).
  championDevSep: number;
  championValSep: number;
  championTestSep: number;
}

const mockEpoch: MockEpochState = {
  epoch: 0,
  championVersion: "champion-0",
  championDevSep: 0.47,
  championValSep: 0.505,
  championTestSep: 0.6,
};

interface PendingChallenger {
  version: string;
  parentVersion: string;
  devSep: number; // selection split (propose metrics)
  valSep: number; // gate split (gate result)
  bbe: number; // dev-split BBε lower bound (frontier ranking)
  addedCriteria: string[];
  objectives: Record<string, number>;
  rubricDiff: string;
  frontier: Frontier;
  passes: boolean;
}

let mockPending: PendingChallenger | null = null;

function round(n: number, dp = 4): number {
  const p = 10 ** dp;
  return Math.round(n * p) / p;
}

function shortId(): string {
  return Math.random().toString(36).slice(2, 8);
}

function deficitMap(weak: number[], strong: number[]): Record<string, number> {
  const m: Record<string, number> = {};
  VAL_WEAK_IDS.forEach((id, i) => (m[id] = round(weak[i] ?? weak[weak.length - 1])));
  VAL_STRONG_IDS.forEach(
    (id, i) => (m[id] = round(strong[i] ?? strong[strong.length - 1])),
  );
  return m;
}

/** Champion-0 has a poison-pill blind spot (loose passes what strict fails). */
function championExploitation(epoch: number): ExploitationReport {
  if (epoch === 0) {
    return {
      backend: "rqgm",
      mean_hack_ratio: 0.3118,
      exploitation_detected: true,
      reason: "EXPLOITATION_DETECTED",
      tolerances_before: RQGM_TOLERANCES,
      tolerances_after: RQGM_TOLERANCES.slice(0, 5),
      tightened: true,
      trigger_adversarial_injection: false,
      strictness_level: 1,
      n_samples: 4,
    };
  }
  // Promoted champion has closed the blind spot: no further tightening.
  return {
    backend: "rqgm",
    mean_hack_ratio: 0.8021,
    exploitation_detected: false,
    reason: "NO_TRANSITION",
    tolerances_before: RQGM_TOLERANCES.slice(0, 5),
    tolerances_after: RQGM_TOLERANCES.slice(0, 5),
    tightened: false,
    trigger_adversarial_injection: false,
    strictness_level: 1,
    n_samples: 4,
  };
}

function challengerExploitation(): ExploitationReport {
  return {
    backend: "rqgm",
    mean_hack_ratio: 0.8021,
    exploitation_detected: false,
    reason: "NO_TRANSITION",
    tolerances_before: RQGM_TOLERANCES.slice(0, 5),
    tolerances_after: RQGM_TOLERANCES.slice(0, 5),
    tightened: false,
    trigger_adversarial_injection: false,
    strictness_level: 1,
    n_samples: 4,
  };
}

function buildChallenger(): PendingChallenger {
  const epoch = mockEpoch.epoch;
  const passes = epoch === 0; // first challenger fixes the blind spot; later ones plateau
  const version = `challenger-e${epoch}-${shortId()}`;
  const parentVersion = mockEpoch.championVersion;

  // The plateau challenger is a genuine (frontier-selected) dev win, but a TINY
  // one: it clears P1 (Δsep>0) yet the Bayesian posterior on the weak val
  // anchors stays below threshold — so the code gate rejects it on P2. This is
  // exactly the winner's-curse guard: a dev-selected improvement is not
  // rubber-stamped by the independent val gate.
  const devSep = passes
    ? round(mockEpoch.championDevSep + 0.28)
    : round(mockEpoch.championDevSep + 0.06);
  const valSep = passes
    ? round(mockEpoch.championValSep + 0.3)
    : round(mockEpoch.championValSep + 0.05);
  const bbe = passes ? 0.275 : 0.09;
  const addedCriteria = passes
    ? ["thermal_budget", "noise_resilience"]
    : ["verbosity_penalty"];
  const objectives = passes
    ? {
        "sep::thermal_budget": 0.46,
        "sep::noise_resilience": 0.38,
        adversarial: 0.62,
        parsimony: -2,
      }
    : {
        "sep::thermal_budget": 0.44,
        "sep::noise_resilience": 0.31,
        adversarial: 0.5,
        parsimony: -1,
      };

  const runnerUpVersion = `challenger-e${epoch}-${shortId()}`;
  const frontier: Frontier = {
    size: 3,
    top_k: 8,
    objectives: [
      "adversarial",
      "parsimony",
      "sep::noise_resilience",
      "sep::thermal_budget",
    ],
    best_version: version,
    members: [
      {
        version,
        objectives,
        bbe,
        added_criteria: addedCriteria,
        parent_version: parentVersion,
        // Thompson-sampling bandit state (child successes / trials as a parent).
        child_successes: 2,
        child_trials: 3,
      },
      {
        version: runnerUpVersion,
        objectives: {
          "sep::thermal_budget": 0.51,
          "sep::noise_resilience": 0.09,
          adversarial: 0.55,
          parsimony: -1,
        },
        bbe: round(bbe - 0.08),
        added_criteria: ["thermal_budget"],
        parent_version: parentVersion,
        child_successes: 1,
        child_trials: 2,
      },
      {
        version: parentVersion,
        objectives: {
          "sep::thermal_budget": 0.12,
          "sep::noise_resilience": 0.08,
          adversarial: 0.33,
          parsimony: 0,
        },
        bbe: 0.02,
        added_criteria: [],
        parent_version: "",
        child_successes: 1,
        child_trials: 4,
      },
    ],
  };

  const rubricDiff = passes
    ? `--- ${parentVersion}.xml\n+++ ${version}.xml\n@@ <criteria> @@\n+  <criterion id="thermal_budget" weight="0.15">\n+    Penalise architectures that ignore the thermal envelope of the target AMD tier.\n+  </criterion>\n+  <criterion id="noise_resilience" weight="0.12">\n+    Require an explicit fallback path when sensor SNR degrades below spec.\n+  </criterion>`
    : `--- ${parentVersion}.xml\n+++ ${version}.xml\n@@ <criteria> @@\n+  <criterion id="verbosity_penalty" weight="0.05">\n+    Penalise padded, low-signal justifications.\n+  </criterion>`;

  return {
    version,
    parentVersion,
    devSep,
    valSep,
    bbe,
    addedCriteria,
    objectives,
    rubricDiff,
    frontier,
    passes,
  };
}

/** P2 posterior threshold + minimum detectable effect (mirrors GateConfig). */
const POSTERIOR_THRESHOLD = 0.95;
const MIN_DETECTABLE_EFFECT = 0.1;

/** Per-flaw {win,loss,tie} breakdown from a per-weak-anchor outcome list. */
function perFlawWins(
  outcomes: ("win" | "loss" | "tie")[],
): Record<string, { win: number; loss: number; tie: number }> {
  const out: Record<string, { win: number; loss: number; tie: number }> = {};
  VAL_WEAK_FLAWS.forEach((flaw, i) => {
    const slot = { win: 0, loss: 0, tie: 0 };
    slot[outcomes[i] ?? "tie"] += 1;
    out[flaw] = slot;
  });
  return out;
}

function buildGate(pending: PendingChallenger): GateResult {
  const championSep = mockEpoch.championValSep;
  const challengerSep = pending.valSep;
  const delta = round(challengerSep - championSep);

  // The promoted champion has already closed the blind spot, so its weak-anchor
  // deficits are high (mean = championValSep + strong mean). Strong stays clean.
  const championSharp = mockEpoch.epoch > 0;
  const championWeak = championSharp
    ? [0.85, 0.82, 0.8, 0.83]
    : [0.55, 0.52, 0.5, 0.53];
  const strong = [0.03, 0.02, 0.01];

  // Per-weak-anchor paired win/loss/tie outcomes (challenger deficit vs champion
  // on each held-out gamed architecture) — the raw evidence the Beta-Binomial
  // posterior tests the sign of.
  const challengerWeak = pending.passes
    ? [0.85, 0.82, 0.8, 0.83] // all penalise the gamed anchors MORE → 4 wins
    : [0.96, 0.92, 0.79, 0.83]; // 2 win / 1 loss / 1 tie: a real but tiny, inconsistent gain
  const outcomes: ("win" | "loss" | "tie")[] = challengerWeak.map((d, i) => {
    const diff = d - championWeak[i];
    return diff > 1e-6 ? "win" : diff < -1e-6 ? "loss" : "tie";
  });
  const nWins = outcomes.filter((o) => o === "win").length;
  const nLosses = outcomes.filter((o) => o === "loss").length;
  const nTies = outcomes.filter((o) => o === "tie").length;

  // P(Δsep>0) = P(θ>0.5) under a Beta(1+wins, 1+losses) posterior (Bayes-Laplace
  // prior). Precomputed for the two mock scenarios: Beta(5,1)→0.9688, Beta(3,2)→0.6875.
  const posterior = pending.passes ? 0.9688 : 0.6875;
  const posteriorOk = posterior >= POSTERIOR_THRESHOLD;
  const effectOk = delta >= MIN_DETECTABLE_EFFECT;
  const p1 = delta > 0; // tie favours incumbent
  const p2 = posteriorOk && effectOk;
  const passed = p1 && p2;

  const championDeficits = deficitMap(championWeak, strong);
  const challengerDeficits = deficitMap(challengerWeak, strong);

  const sign = delta >= 0 ? "+" : "";
  const reason = passed
    ? `code gate PASSED: challenger separation ${challengerSep.toFixed(4)} > champion ${championSep.toFixed(4)} (delta ${sign}${delta.toFixed(4)}); posterior P(Δsep>0)=${posterior.toFixed(4)} >= ${POSTERIOR_THRESHOLD} on ${nWins}W/${nLosses}L/${nTies}T weak anchors and effect ${delta.toFixed(4)} >= MDE ${MIN_DETECTABLE_EFFECT}.`
    : !p1
      ? `code gate FAILED (P1 non-inferiority): challenger separation ${challengerSep.toFixed(4)} does not exceed champion ${championSep.toFixed(4)} (delta ${sign}${delta.toFixed(4)}); tie favours incumbent.`
      : !posteriorOk
        ? `code gate FAILED (P2 posterior): P(Δsep>0)=${posterior.toFixed(4)} < ${POSTERIOR_THRESHOLD} (only ${nWins}W/${nLosses}L/${nTies}T weak anchors moved; underpowered / inconsistent gain).`
        : `code gate FAILED (P2 effect size): posterior P(Δsep>0)=${posterior.toFixed(4)} is sufficient but the effect ${delta.toFixed(4)} is below the minimum practical size (MDE ${MIN_DETECTABLE_EFFECT}).`;

  return {
    passed,
    reason,
    champion_separation: round(championSep),
    challenger_separation: round(challengerSep),
    separation_delta: delta,
    p1_non_inferior: p1,
    p2_passed: p2,
    posterior_prob_improvement: posterior,
    posterior_threshold: POSTERIOR_THRESHOLD,
    effect_size: delta,
    min_detectable_effect: MIN_DETECTABLE_EFFECT,
    n_wins: nWins,
    n_losses: nLosses,
    n_ties: nTies,
    n_val: 7,
    n_weak: VAL_WEAK_IDS.length,
    per_flaw_wins: perFlawWins(outcomes),
    champion_deficits: championDeficits,
    challenger_deficits: challengerDeficits,
  };
}

/**
 * Reproducibility provenance: which judge model + source revision produced the
 * numbers. Offline this is the configured default model with `using_mock` true.
 */
function provenanceBlock(): Provenance {
  return {
    judge_model: "AgentForge-Local",
    using_mock: true,
    rqgm_backend: "rqgm",
    git_sha: "f36e83967eeb510f4d03a675572a589cb6401461",
  };
}

/**
 * Over-acceptance monitor: fraction of gamed/weak adversarial samples (INCLUDING
 * out-of-catalog gaming) the current rubric wrongly scores "strong" (deficit <
 * tau). champion-0 waves two disguised poison pills through; the promoted
 * champion closes the in-catalog blind spots but a novel out-of-catalog gaming
 * sample still slips (honest: generalization is imperfect).
 */
function overAcceptanceBlock(exploited: boolean): OverAcceptance {
  const tau = 0.3;
  const perSample: OverAcceptanceSample[] = exploited
    ? [
        { id: "adv_thermal_masked", targets: ["thermal_runaway"], deficit: 0.12, accepted_as_strong: true, out_of_catalog: false },
        { id: "adv_noise_gamed", targets: ["sensor_noise"], deficit: 0.22, accepted_as_strong: true, out_of_catalog: false },
        { id: "adv_sensor_disable", targets: ["kpi_gaming"], deficit: 0.55, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_fallback_missing", targets: ["no_fallback"], deficit: 0.61, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_ooc_disguised_a", targets: ["novel_gaming"], deficit: 0.48, accepted_as_strong: false, out_of_catalog: true },
        { id: "adv_ooc_disguised_b", targets: ["novel_gaming"], deficit: 0.51, accepted_as_strong: false, out_of_catalog: true },
      ]
    : [
        { id: "adv_thermal_masked", targets: ["thermal_runaway"], deficit: 0.72, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_noise_gamed", targets: ["sensor_noise"], deficit: 0.68, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_sensor_disable", targets: ["kpi_gaming"], deficit: 0.63, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_fallback_missing", targets: ["no_fallback"], deficit: 0.66, accepted_as_strong: false, out_of_catalog: false },
        { id: "adv_ooc_disguised_a", targets: ["novel_gaming"], deficit: 0.24, accepted_as_strong: true, out_of_catalog: true },
        { id: "adv_ooc_disguised_b", targets: ["novel_gaming"], deficit: 0.55, accepted_as_strong: false, out_of_catalog: true },
      ];
  const accepted = perSample.filter((s) => s.accepted_as_strong).length;
  return {
    over_acceptance_rate: round(accepted / perSample.length),
    accepted_as_strong: accepted,
    n: perSample.length,
    tau,
    per_sample: perSample,
  };
}

export function mockReport(): EpochReport {
  const epoch = mockEpoch.epoch;
  const valSep = mockEpoch.championValSep;
  const testSep = mockEpoch.championTestSep;
  const exploited = epoch === 0;

  const perAnchor = (
    ids: string[],
    label: "weak" | "strong",
    deficits: number[],
    predictedOverride?: ("weak" | "strong")[],
  ) =>
    ids.map((id, i) => ({
      id,
      deficit: round(deficits[i] ?? deficits[deficits.length - 1]),
      predicted: predictedOverride?.[i] ?? (deficits[i] >= 0.3 ? "weak" : "strong"),
      label,
    }));

  // On champion-0 one weak anchor is mis-judged as strong (the blind spot).
  const valWeakPred: ("weak" | "strong")[] = exploited
    ? ["weak", "weak", "strong", "weak"]
    : ["weak", "weak", "weak", "weak"];

  return {
    epoch_id: epoch,
    champion_version: mockEpoch.championVersion,
    rqgm_backend: "rqgm",
    provenance: provenanceBlock(),
    data_splits: {
      train: { weak: 5, strong: 5, total: 10 },
      dev: { weak: 3, strong: 2, total: 5 },
      val: { weak: 4, strong: 3, total: 7 },
      test: { weak: 3, strong: 2, total: 5 },
    },
    separation: {
      val: {
        separation: round(valSep),
        mean_weak_deficit: round(exploited ? 0.525 : 0.845),
        mean_strong_deficit: 0.02,
        n: 7,
      },
      test: {
        separation: round(testSep),
        mean_weak_deficit: round(exploited ? 0.62 : 0.86),
        mean_strong_deficit: 0.02,
        n: 5,
      },
    },
    // Over-optimization monitor: proxy(val) − gold(test) separation gap.
    over_optimization: {
      proxy_val_separation: round(valSep),
      gold_test_separation: round(testSep),
      separation_gap: round(valSep - testSep),
    },
    // Over-acceptance monitor: gamed samples wrongly scored "strong".
    over_acceptance: overAcceptanceBlock(exploited),
    hack_ratio: championExploitation(epoch),
    judge_agreement: {
      val: {
        n: 7,
        tau: 0.3,
        accuracy: round(exploited ? 0.857 : 1.0),
        cohen_kappa: round(exploited ? 0.72 : 1.0),
        correct: exploited ? 6 : 7,
        per_anchor: [
          ...perAnchor(
            VAL_WEAK_IDS,
            "weak",
            exploited ? [0.55, 0.52, 0.18, 0.53] : [0.85, 0.82, 0.8, 0.83],
            valWeakPred,
          ),
          ...perAnchor(VAL_STRONG_IDS, "strong", [0.03, 0.02, 0.01]),
        ],
      },
      test: {
        n: 5,
        tau: 0.3,
        accuracy: 1.0,
        cohen_kappa: 1.0,
        correct: 5,
        per_anchor: [
          ...perAnchor(
            ["anchor_test_gamed_a", "anchor_test_gamed_b", "anchor_test_gamed_c"],
            "weak",
            [0.64, 0.61, 0.6],
          ),
          ...perAnchor(
            ["anchor_test_robust_a", "anchor_test_robust_b"],
            "strong",
            [0.03, 0.01],
          ),
        ],
      },
    },
    frontier: mockPending
      ? mockPending.frontier
      : {
          size: 3,
          top_k: 8,
          objectives: [
            "adversarial",
            "parsimony",
            "sep::noise_resilience",
            "sep::thermal_budget",
          ],
          best_version: `challenger-e${Math.max(0, epoch - 1)}-seed01`,
          members: [
            {
              version: `challenger-e${Math.max(0, epoch - 1)}-seed01`,
              objectives: {
                "sep::thermal_budget": 0.46,
                "sep::noise_resilience": 0.38,
                adversarial: 0.62,
                parsimony: -2,
              },
              bbe: 0.275,
              added_criteria: ["thermal_budget", "noise_resilience"],
              parent_version: "champion-0",
              child_successes: 2,
              child_trials: 3,
            },
            {
              version: `challenger-e${Math.max(0, epoch - 1)}-seed02`,
              objectives: {
                "sep::thermal_budget": 0.51,
                "sep::noise_resilience": 0.09,
                adversarial: 0.55,
                parsimony: -1,
              },
              bbe: 0.19,
              added_criteria: ["thermal_budget"],
              parent_version: "champion-0",
              child_successes: 1,
              child_trials: 2,
            },
            {
              version: "champion-0",
              objectives: {
                "sep::thermal_budget": 0.12,
                "sep::noise_resilience": 0.08,
                adversarial: 0.33,
                parsimony: 0,
              },
              bbe: 0.02,
              added_criteria: [],
              parent_version: "",
              child_successes: 1,
              child_trials: 4,
            },
          ],
        },
    memory: {
      mode: "local",
      collection: "agentforge_memory",
      total: exploited ? 12 : 11,
      heuristic_failure: exploited ? 7 : 6,
      physics_truth: 5,
    },
  };
}

export function mockProposeEpoch(): EpochProposeResponse {
  const pending = buildChallenger();
  mockPending = pending;
  return {
    challenger_id: pending.version,
    rubric_diff: pending.rubricDiff,
    metrics: {
      // Selection is on the `dev` split (val is reserved for the gate).
      split: "dev",
      frontier_size: pending.frontier.size,
      added_criteria: pending.addedCriteria,
      champion_separation: round(mockEpoch.championDevSep),
      challenger_separation: round(pending.devSep),
      separation_delta: round(pending.devSep - mockEpoch.championDevSep),
      bbe_lower_bound: round(pending.bbe),
      objectives: pending.objectives,
      note: "best of the Pareto frontier on the dev selection split by BBε lower bound; the code gate re-checks on the held-out val split.",
    },
    frontier: pending.frontier,
  };
}

export function mockApproveEpoch(approve: boolean): EpochApproveResponse {
  const pending = mockPending ?? buildChallenger();
  mockPending = pending;
  const gate = buildGate(pending);
  const championExploit = championExploitation(mockEpoch.epoch);
  const challengerExploit = challengerExploitation();

  // Stage 1: the code gate can block regardless of the human decision.
  if (!gate.passed) {
    return {
      epoch_id: mockEpoch.epoch,
      applied: false,
      champion_version: mockEpoch.championVersion,
      gate,
      hitl: { consulted: false, approved: null, vetoed: false },
      champion_exploitation: championExploit,
      challenger_exploitation: challengerExploit,
      erased_memories: 0,
      reconfirmed_memories: 0,
      reason:
        "CODE GATE FAILED; HITL not consulted (a human cannot override a failed gate).",
    };
  }

  // Stage 2: HITL veto-only safety lock.
  if (!approve) {
    return {
      epoch_id: mockEpoch.epoch,
      applied: false,
      champion_version: mockEpoch.championVersion,
      gate,
      hitl: { consulted: true, approved: false, vetoed: true },
      champion_exploitation: championExploit,
      challenger_exploitation: challengerExploit,
      erased_memories: 0,
      reconfirmed_memories: 0,
      reason: "challenger PASSED the code gate but was vetoed by HITL; champion frozen.",
    };
  }

  // Approved: promote the challenger and advance the epoch.
  const priorEpoch = mockEpoch.epoch;
  mockEpoch.epoch = priorEpoch + 1;
  mockEpoch.championVersion = pending.version;
  mockEpoch.championDevSep = pending.devSep;
  mockEpoch.championValSep = pending.valSep;
  mockEpoch.championTestSep = round(mockEpoch.championTestSep + 0.23);
  mockPending = null;

  return {
    epoch_id: mockEpoch.epoch,
    applied: true,
    champion_version: mockEpoch.championVersion,
    gate,
    hitl: { consulted: true, approved: true, vetoed: false },
    champion_exploitation: championExploit,
    challenger_exploitation: challengerExploit,
    erased_memories: 1,
    reconfirmed_memories: 1,
    reason: `promoted challenger to epoch ${mockEpoch.epoch}; physics_truth memories preserved, obsolete heuristic_failure soft-deleted.`,
  };
}

export function mockHealth(): HealthResponse {
  const epoch = mockEpoch.epoch;
  const exploited = epoch === 0;
  return {
    status: "ok",
    epoch_id: epoch,
    champion_version: mockEpoch.championVersion,
    inference: { using_mock: true, base_url: "mock://deterministic" },
    memory: {
      mode: "local",
      collection: "agentforge_memory",
      total: exploited ? 12 : 11,
      heuristic_failure: exploited ? 7 : 6,
      physics_truth: 5,
    },
    evaluator: {
      rqgm_backend: "rqgm",
      val_separation: round(mockEpoch.championValSep),
      test_separation: round(mockEpoch.championTestSep),
      proxy_gold_separation_gap: round(
        mockEpoch.championValSep - mockEpoch.championTestSep,
      ),
      over_acceptance_rate: exploited ? 0.3333 : 0.1667,
      hack_ratio: exploited ? 0.3118 : 0.8021,
      exploitation_detected: exploited,
      tolerance_levels: 5,
      val_judge_accuracy: round(exploited ? 0.857 : 1.0),
      val_judge_kappa: round(exploited ? 0.72 : 1.0),
    },
  };
}

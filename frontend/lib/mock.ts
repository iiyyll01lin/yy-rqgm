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
  ExportRequest,
  ExportResponse,
  FeedbackRequest,
  FeedbackResponse,
  Gap,
  ModelSpec,
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
      explanation_domain: `以「${domain}」的部署需求，模型權重加上 ${concurrency} 路並發、${seq_len.toLocaleString()} tokens 脈絡的 KV 快取共需約 ${total} GB，但目前硬體僅有 ${memory_gb} GB。這代表在產線尖峰或多工單同時湧入時會發生記憶體不足 (OOM)，導致 Agent 服務中斷。`,
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
| Context | up to ${Math.min(model.context_len, 32768).toLocaleString()} tokens |

Physics is a hard boundary; domain fit is judged by the evolving evaluator.
`;

  return {
    "docker-compose.yml": dockerCompose,
    "app.py": appPy,
    "README.md": readme,
  };
}

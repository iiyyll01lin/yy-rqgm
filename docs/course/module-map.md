<!--
衍生自 canonical 範本（zkp-final/docs/module-map.md · Layer 2）。
單一表格：每列一個模組，欄位 = 模組 | 主題 | 對應藍圖/Part | 軌 | 先修 | 教材(doc/原始碼) | Hands-on | 驗證指令(test/endpoint)。
每個路徑（docs/*、backend/*、tests/*、scripts/*）都必須真的存在於 repo。
Diátaxis 類型：reference。
-->

# Module Map · 模組地圖

> **6 層課程標準的 L2。** 一張表把 AgentForge 的四個架構模組（A1–A4）＋端到端走查（DEMO）攤成
> **M0–M9 的教學序列**，讓教授/審稿人一眼看到*哪個模組、對應哪份藍圖、哪一軌、先修什麼、教材（doc + 原始碼）在哪、
> 動手做什麼、用哪條指令驗證*。它**彙整** [`../blueprint.md`](../blueprint.md) §4 的模組索引與
> [`README.md`](../../README.md) 的 REST API 契約，不重寫它們。
>
> **This is the module map**: the four architecture modules + the end-to-end demo, flattened into an
> M0–M9 teaching sequence with a hands-on task and a real validation command per module.

---

## The map · 地圖

軌 · Track：**core** = 產品主線（hot path + 物理 + 棧 + 匯出 + capstone）· **adv** = cold-path 進化。
`先修` 用 [`course-contract.md`](course-contract.md) §2 的 DAG。驗證指令為**真實**的 `pytest` / REST endpoint / 腳本。

| 模組 | 主題 · Theme | 藍圖 / Part | 軌 | 先修 | 教材 · Material | Hands-on 任務 | 驗證 · Validate |
|---|---|---|---|---|---|---|---|
| **M0** | 雙閘門與誠實邊界（live/mock/simulated） | blueprint · I | core | — | [`../blueprint.md`](../blueprint.md) §0–3 + [`README.md`](../../README.md) | 讀懂 `gatekeeper/` vs `evaluator/` 的目錄級分離 | `uv run pytest -q`（56 passed，離線） |
| **M1** | 預備知識與環境 | blueprint · I | core | M0 | [`README.md`](../../README.md) Quickstart | 起服務、開 UI、看 Live/Mock 徽章 | `./scripts/dev.sh` → 開 `http://localhost:3000` |
| **M2** | Deterministic Gatekeeper 物理數學 | A2 · II | core | M1 | [`../02-sizing-math.md`](../02-sizing-math.md) + [`backend/gatekeeper/vram.py`](../../backend/gatekeeper/vram.py) · [`bandwidth.py`](../../backend/gatekeeper/bandwidth.py) | 手算 `VRAM=W+KV+Act+Overhead`、`tokens/s=BW/bytes`，對照 backend | `uv run pytest tests/test_vram.py tests/test_bandwidth.py tests/test_feasibility.py` |
| **M3** | LangGraph 編排與 HITL | A1 · III | core | M1 | [`../01-orchestration.md`](../01-orchestration.md) + [`backend/graph/orchestrator.py`](../../backend/graph/orchestrator.py) · [`state.py`](../../backend/graph/state.py) | 跑 `orchestrate` → HITL interrupt → `resume{approved:true}` | `POST /api/session/{id}/orchestrate` + `/orchestrate/resume`；`uv run pytest tests/test_domains_router.py` |
| **M4** | RQGM Evaluator：deficit scoring + poison pill | A3 · III | core | M3 | [`../03-evaluator.md`](../03-evaluator.md) + [`backend/evaluator/rubric.xml`](../../backend/evaluator/rubric.xml) · [`judge.py`](../../backend/evaluator/judge.py) | 對弱設計跑 `evaluate`，讀 `deficit_score` + `red_flags` | `POST /api/session/{id}/evaluate`；`uv run pytest tests/test_inference_mock.py` |
| **M5** | Prefix caching / KV explosion / MI300X 記憶體優勢 | A2 · II | core | M2 | [`../02-sizing-math.md`](../02-sizing-math.md) §4–5 | 掃跨 tier `simulate`，看 `max_population` 與 prefix 節省（Instinct 標 SIM） | `POST /api/session/{id}/simulate`（population, prefix_ratio） |
| **M6** | Epoch freeze + GEPA + selective erasure | A1 §6 / A3 §5–8 · IV | adv | M4, M5 | [`../03-evaluator.md`](../03-evaluator.md) §5–8 + [`backend/evaluator/evolve.py`](../../backend/evaluator/evolve.py) · [`versioning.py`](../../backend/evaluator/versioning.py) | `epoch/propose`（challenger separation）→ `epoch/approve`（HITL 閘）→ selective erasure | `POST /api/admin/epoch/propose` + `/approve`；`uv run pytest tests/test_evolution.py` |
| **M7** | ROCm 開源棧與部署 | A4 §1,3 · V | core | M2 | [`../04-stack-export.md`](../04-stack-export.md) §1,3 + [`infra/docker-compose.rocm.yml`](../../infra/docker-compose.rocm.yml) | 讀 ROCm 旗標（`/dev/kfd`、`VLLM_ROCM_USE_AITER=1`、`--enable-prefix-caching`）；量化骨架 | `docker compose -f infra/docker-compose.rocm.yml up`；`python scripts/quantize_quark.py`（骨架） |
| **M8** | Export：可跑 PoC + C-Level TCO/ROI | A4 §2,4 · V | core | M2, M7 | [`../04-stack-export.md`](../04-stack-export.md) §4 + [`backend/export/tco.py`](../../backend/export/tco.py) | 跑 `export`，讀 TCO markdown（每個數字回溯 Gatekeeper 數學）+ 6 個可跑檔 | `POST /api/session/{id}/export` |
| **M9** | Capstone：智慧製造 anchor 端到端 | DEMO · VI | core | M2,M3,M4,M8 | [`../DEMO.md`](../DEMO.md) + [`data/anchor/anchor_architectures.json`](../../data/anchor/anchor_architectures.json) | 4-step wizard 全跑 + HITL + epoch 進化（Instinct 標 SIM） | `./scripts/demo.sh`；`uv run pytest tests/test_api_smoke.py` |

> **驗證說明 · Validation note.** 每個模組的**筆電安全**總入口是 deterministic 測試套件與 mock 降級：
>
> ```bash
> uv run pytest              # backend 56 passed（deterministic，離線，無需 GPU）
> ./scripts/demo.sh          # 端到端契約走查（推論無 server 時自動 MOCK）
> ```
>
> 表中「驗證」欄是該模組的**真實**指令（`pytest` 檔、REST endpoint、或腳本）。Gatekeeper 物理數字在任何機器上
> 都是 **live**（純算術，由 `tests/` 鎖住）；模型輸出在無本地推論 server 時為 **mock**；**Tier 4 Instinct 一律
> `SIMULATED`**。三分法逐項見 L5 [`validation-ledger.md`](validation-ledger.md)。誠實邊界見
> [`course-contract.md`](course-contract.md) §6。

---

## Hands-on 對應原始碼 · Where the code lives

| 主題 | 目錄 / 檔 | 對應模組 |
|---|---|---|
| deterministic 物理（trust anchor） | [`backend/gatekeeper/`](../../backend/gatekeeper/vram.py)（`vram.py`, `bandwidth.py`, `feasibility.py`, `tiers.json`） | M2, M5 |
| LangGraph 編排 + HITL | [`backend/graph/`](../../backend/graph/orchestrator.py)（`orchestrator.py`, `router.py`, `state.py`, `nodes/`） | M3 |
| RQGM evaluator + 進化 | [`backend/evaluator/`](../../backend/evaluator/judge.py)（`rubric.xml`, `judge.py`, `evolve.py`, `versioning.py`） | M4, M6 |
| 本地推論（mock fallback） | [`backend/inference/lemonade_client.py`](../../backend/inference/lemonade_client.py) | M4, M7 |
| 演化記憶（in-memory fallback） | [`backend/memory/qdrant_store.py`](../../backend/memory/qdrant_store.py) | M6 |
| TCO 匯出 + 可跑模板 | [`backend/export/`](../../backend/export/tco.py)（`tco.py`, `deploy_template/`） | M8 |
| domain pack（anchor 情境） | [`backend/domains/smart_manufacturing/`](../../backend/domains/smart_manufacturing/domain.yaml)（`poison_pills.yaml`, `rubric_fragment.xml`, `workflow_templates/`） | M4, M9 |

---

## 結構備註 · Structure notes

- **藍圖 ↔ 模組對應**。A1(orchestration)→M3/M6、A2(sizing-math)→M2/M5、A3(evaluator)→M4/M6、
  A4(stack-export)→M7/M8、DEMO→M9；導論（M0–M1）是藍圖之前的地基。權威定義在
  [`../blueprint.md`](../blueprint.md) §4。
- **每個模組頁的骨架一致**（學習目標 → 概念 → hands-on → live/mock/simulated 說明 → 練習 → 誠實邊界），
  這正是 L4 [`grading-rubric.md`](grading-rubric.md) 評分的對象。
- **產品主線 vs 進階**。M0–M5, M7–M9 是 hot path + 物理 + 棧 + 匯出 + capstone（P0 MVP 即可交付價值）；
  M6 是 cold-path 進化（藍圖路線圖 P1→P2，[`../blueprint.md`](../blueprint.md) §7）。
- **REST 契約**。逐 endpoint 見 [`README.md`](../../README.md) 的「REST API 契約」節；`orchestrate`/`epoch` 為
  驅動 LangGraph 與進化的補充端點。

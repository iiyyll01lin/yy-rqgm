<!--
衍生自 canonical 範本（zkp-final/docs/validation-ledger.md · Layer 5）。
本 repo 的關鍵誠實需求：把每個數字分成 live（真跑 / deterministic 物理）· mock（本地推論降級）·
simulated（Tier 4 Instinct）。複製了：出處標籤 legend、證據表、三分法專節、待補區、交叉連結。
永遠不在沒有 backend 計算 / committed 測試 / 明確 mock|SIMULATED 標籤時新增數字。
Diátaxis 類型：reference。
-->

# Validation Ledger · 驗證帳本

> **6 層課程標準的 L5。** 一份索引：*每個宣稱的數字來自哪裡、在什麼條件下成立*，讓課程每個數字都能回溯。
> 本頁**不重跑**任何量測，只索引 backend 的 deterministic 計算、`tests/`、與端到端走查
> [`../DEMO.md`](../DEMO.md)。**核心紀律：把每個數字分成 live / mock / simulated 三類。**
>
> **This ledger separates every figure into live (real hardware / deterministic physics), mock
> (local-inference fallback), and SIMULATED (Tier 4 Instinct).** It never re-measures.

---

## 出處標籤 · Provenance labels

| 標籤 · Label | 意義 · Meaning |
|---|---|
| `deterministic` | 純算術的 Gatekeeper 輸出（VRAM / 頻寬 / tokens-s）；**live 與 mock 恆等**，由 `tests/` 鎖住。 |
| `live-hw` | 在真 AMD 硬體（**T1–T3**：Ryzen AI / Radeon / Radeon PRO）上由本地推論（Lemonade / vLLM ROCm）服務。 |
| `mock` | 無本地推論 server 時的 **deterministic MOCK 降級**（judge 輸出可重現）；即「local-inference fallback」。 |
| `SIMULATED` | **Tier 4 Instinct（MI300X / MI325X）** 由公式模擬，**無實機**；UI 與匯出一律掛 `SIM` 徽章。 |
| `anchor` | 來自 [`data/anchor/`](../../data/anchor/anchor_architectures.json) 的 ground-truth（HITL 種子），供 epoch gating。 |

---

## 驗證怎麼跑 · How validation runs

- **deterministic 閘。** `uv run pytest` = **74 passed**（離線、無需 GPU）。物理數學測試
  [`tests/test_vram.py`](../../tests/test_vram.py)、[`tests/test_bandwidth.py`](../../tests/test_bandwidth.py)、
  [`tests/test_feasibility.py`](../../tests/test_feasibility.py) 把 Gatekeeper 公式鎖死 —— 這是 `deterministic` 數字的權威。
- **端到端契約。** `./scripts/demo.sh` curl 驅動整條契約（domain→diagnose→simulate→export + orchestrate/epoch）；
  推論無 server 時自動 `mock`，Qdrant 缺席時 in-memory。
- **前後端一致。** 前端 MOCK [`frontend/lib/vram.ts`](../../frontend/lib/vram.ts) 與 backend
  [`backend/gatekeeper/vram.py`](../../backend/gatekeeper/vram.py) 採**相同公式**，故 live / mock 物理數字恆等。

---

## 證據帳本 · Evidence ledger

每列都可回溯到 backend 計算 / committed 測試 / DEMO 走查。數字後面的標籤即其**出處三分類**。

| 宣稱 · Claim | 層 · Layer | 值 · Value | 證據 · Artefact | 出處 · Provenance | 權威 |
|---|---|---|---|---|---|
| **Diagnose**（RX 7900 XTX 24 GB, Llama 3.1 8B @8k ×8） | Gatekeeper | `VRAM = 16.1+8.6+2.5+1.0 = 28.1 GB` → headroom **−4.1 GB**（infeasible）；~**39 tok/s** | [`backend/gatekeeper/vram.py`](../../backend/gatekeeper/vram.py) + [`tests/test_vram.py`](../../tests/test_vram.py) + [`../DEMO.md`](../DEMO.md) Step 2 | `deterministic`（live=mock） | DEMO |
| **Simulate T1–T3**（pop=16, prefix=0.5） | Gatekeeper | Ryzen AI 128 GB `max_pop 184 @10`；RX 7900 XTX `8 @39`；W7900 48 GB `48 @35` | [`backend/gatekeeper/feasibility.py`](../../backend/gatekeeper/feasibility.py) + [`tests/test_feasibility.py`](../../tests/test_feasibility.py) | `deterministic` + `live-hw`（T1–T3 真硬體級別） | DEMO Step 3 |
| **Simulate T4 Instinct** ⚠️ | Gatekeeper | MI300X 192 GB `max_pop 292 @217`；MI325X 256 GB `400 @245`；≈ **36.5×** 於 RX 7900 XTX（容量比） | [`backend/gatekeeper/tiers.json`](../../backend/gatekeeper/tiers.json) + [`../DEMO.md`](../DEMO.md) Step 3 | **`SIMULATED`**（公式，無實機） | DEMO Step 3 |
| **Prefix caching 節省** | sizing 數學 | `savings = φ(P−1)/P`；demo @(pop 16, r=0.5) ≈ **46.9%**；φ=0.8 → 78.8% @P=64 | [`../02-sizing-math.md`](../02-sizing-math.md) §4 | `deterministic`（恆等式） | 02-sizing-math |
| **tokens/s 上界**（Llama-3-70B int4 35 GB, batch=1） | 頻寬數學 | H100 ~67 · H200 ~96 · MI300X ~106 · MI325X ~120（×0.7 MBU） | [`../02-sizing-math.md`](../02-sizing-math.md) §3.3 | `deterministic`（解析上界；T4 規格值） | 02-sizing-math |
| **RQGM evaluate**（弱設計） | Evaluator | `deficit_score ≈ 0.6`；red_flags = `physics_common_sense`（duct-tape）+ `diagnostic_resilience`（相關噪聲） | [`backend/evaluator/judge.py`](../../backend/evaluator/judge.py) + [`rubric.xml`](../../backend/evaluator/rubric.xml) | `mock`（deterministic judge）/ `live-hw`（有 Lemonade 時） | DEMO |
| **Orchestrate + HITL** | LangGraph | `router→task_agent→gatekeeper→(feasible)→evaluator→hitl`；infeasible **HARD REJECT**（省 GPU）；`resume{approved:true}` 完成 | [`backend/graph/orchestrator.py`](../../backend/graph/orchestrator.py) + [`tests/test_domains_router.py`](../../tests/test_domains_router.py) | `deterministic`（路由/閘）+ `mock`（LLM node） | DEMO |
| **Epoch 進化** | Evaluator | `epoch/propose` 跑 GEPA Pareto frontier（Thompson 父代）+ 紅隊 → frontier best（`dev` BBε）；`epoch/approve` **先過 code gate**（val P1 非劣 + P2 Bayesian 後驗/MDE）再 HITL 加簽（只能否決）→ **epoch 0→1** + selective erasure（soft-delete + reconfirm；`physics_truth` 永久保留）；`report` 另含 over-acceptance / over-optimization gap / provenance | [`backend/evaluator/evolve.py`](../../backend/evaluator/evolve.py) + [`gate.py`](../../backend/evaluator/gate.py) + [`frontier.py`](../../backend/evaluator/frontier.py) + [`rqgm_adapter.py`](../../backend/evaluator/rqgm_adapter.py) + [`tests/test_rqgm_phases.py`](../../tests/test_rqgm_phases.py) | `deterministic` / `mock`（可重現） | DEMO + 03-evaluator |
| **Export**（目標 MI300X） | Export | **FEASIBLE**；TCO/ROI markdown + 6 個可跑檔（compose/Dockerfile/app.py/README/requirements/.env） | [`backend/export/tco.py`](../../backend/export/tco.py) + [`deploy_template/renderers.py`](../../backend/export/deploy_template/renderers.py) | `SIMULATED`（目標 T4）+ `deterministic`（sizing） | DEMO Step 4 |
| **後端測試套件** | 全平台 | **74 passed**（deterministic，離線，無 GPU） | [`tests/`](../../tests/test_api_smoke.py) | `deterministic` | README |

---

## Live / Mock / Simulated 三分法 · The three-way provenance split

這張表是本 repo 誠實邊界的核心：每個子系統的數字屬於哪一類、缺硬體時如何降級。

| 子系統 · Subsystem | 有硬體 / server 時 | 缺硬體 / server 時（fallback） | 標籤 |
|---|---|---|---|
| **Gatekeeper 物理**（VRAM / 頻寬 / tokens-s） | 純算術，任何機器一致 | 同左（不依賴硬體） | `deterministic`（live=mock） |
| **Tier 1–3**（Ryzen AI / Radeon / Radeon PRO） | 真跑（Lemonade / vLLM ROCm 服務模型） | —（規格 + deterministic 數學） | `live-hw` |
| **本地推論**（LLM 輸出） | `live-hw`（Lemonade server 在場） | **deterministic MOCK**（judge 可重現） | `live-hw` → `mock` |
| **演化記憶**（Qdrant） | 真 Qdrant server | in-memory 降級 | `live-hw` → `mock` |
| **Tier 4 Instinct**（MI300X / MI325X） | **無實機** | 公式模擬（縮小 population 驗真） | **`SIMULATED`** |
| **LLM judge**（evaluator deficit score） | `live-hw`（Lemonade 在場） | deterministic MOCK（可重現） | `live-hw` → `mock` |

> 讀法：**物理永遠 live**（deterministic，live=mock 恆等）；**模型輸出**在無本地推論 server 時降級為
> `mock`（這就是「local-inference fallback」，見 [`backend/inference/lemonade_client.py`](../../backend/inference/lemonade_client.py)）；
> **Tier 4 Instinct 永遠 `SIMULATED`**（無實機，UI/匯出掛 `SIM` 徽章）。demo 中「MI300X ≈ 36.5×」是一個
> **SIMULATED 的容量比**，不是真硬體實測 speedup（見 [`grading-rubric.md`](grading-rubric.md) M5 評分示例）。

---

## 待補 · Open items

以下為誠實聲明的邊界，非錯誤：

| 項目 · Item | 現況 | 待補 |
|---|---|---|
| **Tier 4 Instinct 實機** | 一切 `SIMULATED`（公式 + 縮小 population） | 取得真 MI300X/MI325X 後，把 `SIMULATED` 列升級為 `live-hw`（數字可能收斂，方向不變） |
| **Live 本地推論** | 預設 `mock`（筆電） | 在真 AMD box + 模型權重上開 Lemonade / vLLM ROCm，judge 走 `live-hw` |
| **production 記憶 / checkpointer** | Qdrant in-memory + `SqliteSaver` | 換 production Qdrant + `PostgresSaver`（[`../01-orchestration.md`](../01-orchestration.md) §4） |
| **live provenance 欄位** | `build_report` 產生 `provenance`（judge model + git sha + using_mock），但 `EpochReportResponse` 未宣告該欄位，Live 模式會被 pydantic 過濾（Mock 模式可見） | backend 於 `EpochReportResponse` 補上 `provenance` 欄位即可在 Live 端曝露（本輪 docs+frontend-only，未動 backend） |

---

## 交叉連結 · Cross-references

- 誠實邊界與三分法 → [`course-contract.md`](course-contract.md) §5–6
- sizing 公式與 prefix caching 恆等式 → [`../02-sizing-math.md`](../02-sizing-math.md)
- evaluator 設計 / 防 reward-hacking → [`../03-evaluator.md`](../03-evaluator.md)
- 端到端走查（本帳本數字的來源） → [`../DEMO.md`](../DEMO.md)
- 模組 → hands-on → 驗證指令 → [`module-map.md`](module-map.md)

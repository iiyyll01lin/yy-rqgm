<!--
本檔衍生自 6 層課程標準的 canonical 範本（zkp-final/docs/course-contract.md · Layer 1）。
它把 AgentForge 的架構藍圖（docs/blueprint.md、01-04）與端到端走查（docs/DEMO.md）彙整成
「學生/教授面向」的一頁課程契約，不重寫藍圖，只交叉連結。
誠實紀律：Tier 1–3 為真跑；本地推論缺席時降級為 deterministic MOCK；Tier 4 Instinct 一律 SIMULATED。
Diátaxis 類型：reference + explanation。
-->

# Course Contract · 課程契約

> **6 層課程標準的 L1。** 本頁是課程與學習者之間的一頁「合約」：講清楚適用對象、先修、學習成果、
> 硬體假設、**live / mock / simulated 政策**，以及本平台拒絕跨越的誠實邊界。它**彙整**（不取代）
> [`../blueprint.md`](../blueprint.md) 的主索引、[`../01-orchestration.md`](../01-orchestration.md)–[`../04-stack-export.md`](../04-stack-export.md)
> 四個架構模組、與 [`../DEMO.md`](../DEMO.md) 的端到端走查，以及根目錄 [`README.md`](../../README.md)。
>
> **This is the course contract**: one page on audience, prerequisites, outcomes, hardware
> assumptions, the live/mock/simulated policy, and the honesty boundary — consolidating (not
> rewriting) the AgentForge architecture blueprint it cross-links.

---

## 1. Audience · 適用對象

| 讀者 · Reader | 為什麼讀 · Why | 從哪開始 · Start at |
|---|---|---|
| **大學教授（排課）** | 依課型（選型/物理數學、Agent 編排、可信度 seminar、產品實作）挑一條現成路線。 | [`teaching-paths.md`](teaching-paths.md) |
| **自學者 / Agent 工程師** | 從 deterministic 物理閘門一路走到受控進化的 evaluator 與可跑 PoC 匯出。 | [M0](module-map.md) |
| **平台 / SRE / 決策者** | 看懂「AMD 大記憶體 → 更少節點 → 更低 TCO」的可稽核推導，並區分真跑/模擬。 | [M2](module-map.md) → [M8](module-map.md) |
| **審稿人 / reviewer** | 在**任何筆電**上端到端跑（無需 GPU；推論自動降級 MOCK、Qdrant 降級 in-memory）。 | [`../DEMO.md`](../DEMO.md) |

---

## 2. Prerequisites · 先修

**帶進來的知識 · Knowledge you bring**

- 熟悉 Linux shell；能讀 Python（FastAPI / LangGraph）與 TypeScript（Next.js 前端為選讀）。
- 基本的 LLM 推論直覺（weights / KV cache / decode）；**不需要**先懂 ROCm —— 硬體數學由 Gatekeeper 的角度從頭引入。
- AMD GPU / ROCm 經驗 **選配**：整個平台在無 GPU 的筆電上也能端到端跑（推論降級 deterministic MOCK）。
- 進化理論（RQGM / GEPA / DGM）**不需先修**；名詞表見 [`../blueprint.md`](../blueprint.md) §6。

**各模組先修（DAG）· Per-module prerequisites**

箭頭 `Mx → My` 讀作「**Mx 需先修 My**」：

```text
M1 → M0
M2 → M1
M3 → M1
M4 → M3
M5 → M2
M6 → M4, M5
M7 → M2
M8 → M2, M7
M9 → M2, M3, M4, M8
```

> 產品主線 = **M0–M5, M7–M9**（hot path + 物理 + 棧 + 匯出 + capstone）；進階軌 = **M6**（cold-path 進化）。
> 逐模組的 hands-on / 對應原始碼 / 驗證指令見 L2 [`module-map.md`](module-map.md)。

---

## 3. Learning outcomes · 學習成果

走完產品主線（M0–M5, M7–M9）你能：

- 一句話說出本平台的**信任模型**：把**物理**（deterministic、永不進化的 `Static Hardware Gatekeeper`）
  與**品味**（fuzzy、受控進化的 `RQGM Evaluator`）實體分離（`backend/gatekeeper/` vs `backend/evaluator/`）。
- 手算 deterministic sizing：`VRAM = Weights + KV + Activations + Overhead`、
  `tokens/s ≈ bandwidth / bytes_per_token`，並解釋為何 decode 是 memory-bound（見 [`../02-sizing-math.md`](../02-sizing-math.md)）。
- 推導 prefix-caching 節省 `savings = φ·(P−1)/P` 並說明 MI300X 192 GB 如何把「大 population × 長熱記憶 × 長 context」
  的零和取捨消掉。
- 讀懂 evaluator 的 **deficit-scoring** XML rubric 與三大 failure mode（physics common sense、diagnostic resilience、
  modularity drift）＋ poison pill，並說明它為何能抗諂媚（見 [`../03-evaluator.md`](../03-evaluator.md)）。
- 對每個數字正確標註 **live / mock / SIMULATED**：Tier 1–3 真跑、推論缺席時 deterministic MOCK、
  Tier 4 Instinct 一律 SIMULATED（見 §5、§6）。

進階軌（M6）再加上：epoch freeze、GEPA reflective mutation、held-out anchor gating、selective erasure，
以及「為何 evaluator 必須在進化 harness 之外」的防 reward-hacking 論證。

---

## 4. Hardware assumptions · 硬體假設

平台把 AMD 產品線抽象成四個 tier（規格為 single source of truth，見 [`../blueprint.md`](../blueprint.md) §5）；
**真跑**與**模擬**的分界就是誠實邊界的一部分：

| Tier | 代表硬體 | 記憶體 / 頻寬 | 本平台狀態 |
|---|---|---|---|
| **T1 — Ryzen AI** | Ryzen AI Max+ 395（Strix Halo, XDNA2 NPU） | ≤128 GB unified / 0.256 TB/s | ✅ **真跑**（Lemonade `flm`/`rocm`） |
| **T2 — Radeon** | RX 7900 XTX | 24 GB / 0.96 TB/s | ✅ **真跑**（Lemonade / llama.cpp ROCm） |
| **T3 — Radeon PRO** | W7900 | 48 GB / 0.86 TB/s | ✅ **真跑**（vLLM / llama.cpp ROCm） |
| **T4 — Instinct** | MI300X / MI325X（CDNA3, HBM3） | 192 / 256 GB / 5.3–6.0 TB/s | ⚠️ **SIMULATED**（Gatekeeper 數學 + 縮小 population） |

> 開發機為 **Ryzen AI + Radeon**；T4 Instinct **無實機**，一切數字由 deterministic 物理公式模擬，
> UI 與匯出皆標 `SIMULATED` / `SIM`（見 [`README.md`](../../README.md) 的 SIMULATED disclaimer）。
> **你不需要任何 GPU 就能上整門課** —— 見下方政策。

---

## 5. Live / Mock / Simulated 政策 · Live / Mock / Simulated Policy

本平台的每個數字都落在三種來源之一，這條三分法是課程的核心誠實紀律：

- **Live（真跑）** —— 在真 AMD 硬體（T1–T3）上由本地推論（Lemonade / vLLM ROCm）服務模型；
  且 **deterministic Gatekeeper 數學**（VRAM / 頻寬 / tokens-s）在**任何機器**上都是 live，因為它是純算術、由 `tests/` 鎖住。
- **Mock（deterministic 降級）** —— 沒有本地推論 server 時，`backend/inference/lemonade_client.py` 自動降級為
  **deterministic MOCK**（judge 輸出可重現）；`Qdrant` 缺席時降級為 in-memory。前端以 `NEXT_PUBLIC_USE_MOCK` 控制，
  header 右上角有 **Live / Mock 徽章**。這讓整個平台在一般筆電上端到端可跑。
- **Simulated（模擬）** —— **Tier 4 Instinct（MI300X / MI325X）的所有數字**（max_population、tokens/s、TCO）
  由 deterministic 公式**模擬**，因為沒有實機。UI 與匯出文件一律標 `SIMULATED`。

一鍵在筆電上端到端跑（無 GPU）：

```bash
./scripts/dev.sh     # 同時起 backend(:8000) + frontend(:3000)
./scripts/demo.sh    # 另一終端 curl 驅動整條契約（含 HITL 與 epoch 進化）
```

> 關鍵不變式：前端 MOCK 的 [`frontend/lib/vram.ts`](../../frontend/lib/vram.ts) 與 backend
> [`backend/gatekeeper/vram.py`](../../backend/gatekeeper/vram.py) 採用**相同公式**，所以 Live / Mock 數字**一致可信**。
> 差別只在「模型輸出」是否為真本地推論；**物理數字兩邊恆等**。

---

## 6. Honesty boundary · 誠實邊界

這是本平台可信度的地基，也是沒有任何一頁可以跨越的線：

> **物理是可信度的地基，永不進化；領域適配交由會進化的評估器判斷，但升級須先過 held-out anchor 上的 code 統計 gate，人類只能否決、不能覆寫。
> Instinct（MI300X / MI325X）的數字一律標 SIMULATED。**
>
> **Physics is the trust foundation and never evolves; domain fit is judged by an evolving
> evaluator that only upgrades after passing a statistical code gate on held-out anchors — humans can veto but never override. Instinct figures are SIMULATED.**

具體規則，每個模組都被檢查：

- **物理不可被 LLM「講贏」**：feasible/infeasible、VRAM、tokens/s 由 deterministic Gatekeeper 決定，
  擋在任何 LLM 判斷之前（[`../01-orchestration.md`](../01-orchestration.md) §3）。
- **來源標清楚**：每個數字標 `live` / `mock` / `SIMULATED`；**絕不**把模擬的 Instinct 數字當成真硬體實測。
- **進化受控，不是「自動變聰明」**：evaluator 在 epoch 內凍結、升級須先過 held-out anchor 上的 **code 統計 gate（P1/P2）**、**HITL 只能否決**，
  且恆在進化 harness 之外（防 reward hacking，見 [`../03-evaluator.md`](../03-evaluator.md) §7）。

課程反覆示範的具體誠實點（來自 [`../DEMO.md`](../DEMO.md)）：MI300X 在 demo 中可承載約 **292** 條並行序列
≈ **36.5×** 於 RX 7900 XTX —— 這是一個 **SIMULATED 的容量比**（由容量公式推導），**不是**真硬體上量到的吞吐 speedup；
UI 與匯出都掛 `SIM` 徽章。

> **每一頁的數字規則**：引用 backend 的 deterministic 計算 / 一個 committed 測試，或標
> `mock` / `SIMULATED`。永遠不要把前端估算或模擬值當成真硬體量測貼出來。

---

## 7. Time budget · 時間預算

自定進度的規劃值（權威的逐路線節奏見 L3 [`teaching-paths.md`](teaching-paths.md)）：

| 範圍 · Scope | 模組 · Modules | 建議時間 · Suggested |
|---|---|---|
| 單堂 demo / 客座（筆電可跑） | M0 + M9（跑 `demo.sh`） | ~1 小時（路線 A） |
| 選型 / 物理數學 | M0, M2, M5, M8 | ~2–3 堂 |
| Agent 編排 / LangGraph 系統 | M0, M3, M4, M6 | ~3–4 週 |
| 可信度 / anti-reward-hacking seminar | M0, M4, M6 | ~2–3 堂 |
| 完整一學期產品實作 | M0–M9 | ~12–14 週 |

> 這些是**規劃估計**，不是量測到的時長；逐模組目標與練習見 L2/L4。

---

## 8. The 6-layer standard · 六層標準（本課程的自描述）

本契約是可重複標準的第 1 層。各層在本 repo 的落點：

| 層 · Layer | 檔案 · Artifact |
|---|---|
| **L1** Course contract | **本檔** · [`docs/course/course-contract.md`](course-contract.md) |
| **L2** Module map | [`docs/course/module-map.md`](module-map.md)（模組 + hands-on + 驗證指令） |
| **L3** Teaching routes | [`docs/course/teaching-paths.md`](teaching-paths.md) |
| **L4** Assessment kit | [`docs/course/grading-rubric.md`](grading-rubric.md) + [`docs/course/submission-checklist.md`](submission-checklist.md) |
| **L5** Validation ledger | [`docs/course/validation-ledger.md`](validation-ledger.md)（live / mock / simulated 證據索引 + `tests/`） |
| **L6** Platform metadata | [`course.yaml`](../../course.yaml) |

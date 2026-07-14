# 智慧製造 Anchor Demo — 端到端走查

> 一個具體、可重現的走查：工廠 QC 視覺瑕疵檢測 Agent，從「現有硬體跑不動」到「升級路徑 + 可交付 PoC」。
> 全程用 **deterministic 物理**把關可信度，數字皆來自 **backend**（非前端 client-side 估算）。

## TL;DR — 一鍵重現

```bash
# 終端 1：起 backend + frontend
./scripts/dev.sh

# 終端 2：curl 驅動整條契約（含 HITL 與 epoch 進化）
./scripts/demo.sh
#   BASE=http://localhost:8000 ./scripts/demo.sh
#   ./scripts/demo.sh --no-evolve      # 不動 epoch 持久狀態
```

或直接開 UI：<http://localhost:3000>（右上角資料來源徽章應顯示 **Live API**）。

> 無需 GPU：推論在無 Lemonade server 時自動降級為 deterministic MOCK；judge 輸出可重現。
> Instinct (MI300X / MI325X) 數字為 **SIMULATED**，UI 與匯出皆標註。

---

## 情境設定 (Scenario)

| 項目 | 值 |
| --- | --- |
| 領域 (Domain) | 智慧製造 / 工廠品質檢測（freeform，UI 用產業語言輸入即可） |
| 需求 (Need) | 產線相機影像即時判讀、標記表面瑕疵、給可稽核判定；資料留廠內 |
| 現有硬體 | **Radeon RX 7900 XTX — 24 GB**（強力消費級 dGPU） |
| 想跑的模型 | **Llama 3.1 8B Instruct**，8,192-token context |
| 產線並發 | **8 條**（尖峰同時湧入） |

---

## Step 1 — Domain（需求 → workflow 路由）

用產業語言描述需求，`POST /api/session/{id}/domain` 以 keyword + embedding 混合路由到最合適的
workflow template（不需額外 LLM 呼叫）。

**結果**：推薦 **Visual Quality Control (defect detection)**，另列 Closed-loop Process Optimization、
Predictive Maintenance 共 3 個候選。

> 契約細節：即使 `domain` 傳入的是 freeform 產業標籤（非已註冊的 pack id），後端也會跨所有 domain 搜尋，
> 確保一定路由得到範本（整合修復點之一）。

## Step 2 — Diagnose（現有硬體可行性；deterministic gatekeeper）

`POST /diagnose`：對 RX 7900 XTX 24 GB 跑純物理計算。

```
VRAM = Weights 16.1 + KV_cache 8.6 + Activations 2.5 + Overhead 1.0 = 28.1 GB
容量 24.0 GB  →  headroom −4.1 GB  →  ❌ 超出硬體邊界
解碼吞吐 ≈ 39 tokens/s（memory-bandwidth 上界）
```

**缺口（以領域語言說明）**：

> To run Llama 3.1 8B Instruct for 智慧製造 / 工廠品質檢測 at 8,192-token context (×8), you need ~28 GB
> of VRAM, but Radeon RX 7900 XTX has only 24 GB. Move up to a higher-VRAM AMD tier (e.g. **W7900 48 GB**
> or **MI300X 192 GB**), quantize to int4, or reduce context.

UI 會畫出堆疊 VRAM bar、容量標線與**溢出 (OOM) 斜線區**，一眼看懂「差多少」。

## Step 3 — Simulate（升級路徑；跨 tier 模擬）

`POST /simulate`（population=16, prefix_ratio=0.5）掃整條 AMD 階梯：

| Tier | Feasible | max_population | tokens/s |
| --- | --- | ---: | ---: |
| Ryzen AI Max+ 395 (128 GB) | ✅ | 184 | 10 |
| Radeon RX 7800 XT (16 GB) | ❌ | 0 | 25 |
| **Radeon RX 7900 XTX (24 GB)** | ❌ | 8 | 39 |
| Radeon PRO W7800 (32 GB) | ✅ | 21 | 24 |
| Radeon PRO W7900 (48 GB) | ✅ | 48 | 35 |
| **Instinct MI300X (192 GB)** ⚠️ SIM | ✅ | **292** | **217** |
| Instinct MI325X (256 GB) ⚠️ SIM | ✅ | 400 | 245 |

**Headline**：MI300X 可承載約 **292** 條並行序列 ≈ **36.5×** 於 RX 7900 XTX——巨大顯存 + 頻寬讓
大規模 population 搜尋（如 MCTS）與長期 negative-result 記憶成為可能。Prefix caching @ (pop 16, r=0.5)
省下約 **46.9%** 的 KV。

## Step 4 — Export（可交付 PoC + AMD TCO/ROI）

`POST /export`（目標 = MI300X，帶入 Step 3 的 sizing）：**FEASIBLE**，產出

- **TCO / ROI 提案**（markdown）：sizing 由 deterministic gatekeeper 背書；含開源棧論述、
  三年 TCO vs「不受控 open-ended 進化」雲端成本（Darwin Gödel Machine 資料點 ~$22k/run）對照。
- **6 個可跑檔案**：`docker-compose.yml`（`rocm/vllm-dev` + `--enable-prefix-caching` + Qdrant）、
  `Dockerfile`、`app.py`（LangGraph 骨架）、`README.md`、`requirements.txt`、`.env.example`。

---

## 進階：HITL 編排 + RQGM 進化（`evaluate` / `orchestrate` / `epoch`）

### RQGM Evaluator（`POST /evaluate`）
對一個「單一靜態閾值就自動停機、無 state schema、無 HITL」的弱設計評分：
`deficit_score ≈ 0.6`，紅旗包含 `physics_common_sense`（數值 duct-tape、缺物理根因）與
`diagnostic_resilience`（未處理相關性感測噪聲）。

### LangGraph HITL（`POST /orchestrate` → `/orchestrate/resume`）
`router → task_agent → gatekeeper →（feasible）→ rqgm_evaluator → hitl`。
gatekeeper 若判 infeasible 直接 **HARD REJECT**（省下 GPU 成本，不進 evaluator）；
feasible 則在 `hitl` 節點 **interrupt** 等待人類核准，`resume{approved:true}` 後完成。

### GEPA epoch 進化（`POST /epoch/propose` → `/epoch/approve`）
`propose` 讀取累積回饋 + evaluator traces 當「文字梯度」，reflectively mutate champion rubric 成
challenger，並在 held-out anchor set 上比 *separation*（`deficit(weak) − deficit(strong)`，越高越能分辨好壞）。
`approve{approve:true}` 是 **HITL 閘門**：核准才 `epoch 0 → 1`、晉升 champion，並對舊 epoch 的
`heuristic_failure` 記憶做 selective erasure（`physics_truth` 永久保留）。

> `epoch/approve` 會寫入 `data/epoch_state.json`。回到 epoch-0 seed champion：
> `rm -f data/epoch_state.json data/rubric_history/challenger-*.xml`。

---

## 這個 demo 證明了什麼

1. **可信度地基**：所有 VRAM / tokens/s / max-population 都來自 deterministic 物理，可稽核、Live/Mock 一致。
2. **教育性 upsell**：不是「買最貴的」，而是用**你的領域語言**說明「差哪塊、升級解鎖什麼」。
3. **AMD 全開源路徑**：從 Ryzen AI / Radeon 真跑，到 Instinct 模擬，一條 ROCm 棧走到底，零供應商鎖定。
4. **受控進化**：evaluator 會進化，但被 epoch 凍結 + HITL 核准雙重約束，結構性地防 reward-hacking。

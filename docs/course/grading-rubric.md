<!--
衍生自 canonical 範本（zkp-final/docs/grading-rubric.md · Layer 4a）。
criteria × levels 結構，最高權重、不可協商的是「誠實紀律」（標 live/mock/simulated、絕不把模擬當實測）。
Diátaxis 類型：reference。
-->

# Grading Rubric · 評分量表

> **6 層課程標準的 L4（評量）。** 用來評每個模組頁末「練習題」的作答（M0–M9）。評的是**推理**，
> 不是字句：能否用 deterministic 物理歸因、能否誠實標 live/mock/simulated、能否分清「受控進化」與「自動變聰明」。
>
> **This rubric grades the reasoning** behind each module's exercises — physics-first attribution,
> honest live/mock/simulated labeling, and the controlled-evolution distinction.

---

## 評分原則 · Grading philosophy

1. **誠實優先。** 本平台的整個論點是「物理永不進化、Instinct 一律 SIMULATED」。一個答案若把**模擬的** Instinct
   數字當成真硬體實測、編造 benchmark、或把前端估算當 backend 計算，該項無論多漂亮都不能高於 *Developing*。
2. **物理先於品味。** 獎勵先用 deterministic Gatekeeper 數學（`VRAM=W+KV+Act+Overhead`、`tokens/s=BW/bytes`）
   判可行性，再談 LLM 的 fuzzy 判斷；反過來（讓 LLM 判物理）扣分。
3. **受控進化不是自動變聰明。** 最重要的觀念是：evaluator 會進化，但被 **epoch freeze + held-out anchor 上的 code 統計 gate + HITL 否決**
   三重約束，且恆在進化 harness 之外（見 [M6](module-map.md)）。

---

## 評分標準與權重 · Criteria and weights

各標準用下面 4 級量表；建議權重（每份作業可微調，總和 = 100）：

| # | 標準 · Criterion | 權重 |
|---|---|---|
| C1 | **物理歸因** —— 先用 deterministic Gatekeeper 判 feasible/infeasible + VRAM 分解 + tokens/s，不讓 LLM「講贏」物理；分清 deterministic（gatekeeper）與 fuzzy（evaluator）。 | 30 |
| C2 | **誠實來源標註** —— 每個數字標 `live` / `mock` / `SIMULATED`；引用 backend 計算或 committed 測試，不把模擬 Instinct 當實測、不把前端估算當 backend。 | 30 |
| C3 | **first-principles 推理** —— decode 為 memory-bound、`savings=φ(P-1)/P`、KV explosion、容量→吞吐正相關→更少節點→更低 TCO、deficit scoring 抗諂媚。 | 20 |
| C4 | **可重現** —— 指名正確的 `pytest` 檔 / REST endpoint / 腳本；主張在筆電（mock 降級）上可重跑。 | 10 |
| C5 | **表達** —— 區分「受控進化」與「自動變聰明」；用 domain 語言做 upsell（缺哪塊、升級解鎖什麼）；不 vendor 話術。 | 10 |

### 等級 · Level scale

| 等級 · Level | 意義 · Meaning |
|---|---|
| **4 優 · Exemplary** | 正確、first-principles、誠實邊界範圍正確；每個數字都掛對 live/mock/SIM 標籤。 |
| **3 良 · Proficient** | 結論正確，但來源標註或物理歸因有小缺口。 |
| **2 待加強 · Developing** | 方向對但有實質錯誤（讓 LLM 判物理、漏標 SIM、混用前端/後端數字）。 |
| **1 / 0 缺 · Missing** | 缺答，或違反誠實 rule（把 SIMULATED 當實測、編數字、宣稱 evaluator 不可能被 hack 卻不引用三道防線）。 |

> **誠實 auto-cap.** 任何把**模擬的** Instinct（MI300X/MI325X）數字當成真硬體實測、編造 benchmark、
> 或宣稱「evaluator 不會被 reward-hack」卻不引用三道結構性防線，都把 **C2**（若它驅動結論則連 **C1**）
> 直接壓到 level 1 —— 對應本 repo 在 [`course-contract.md`](course-contract.md) §6 的不可協商規則。

---

## 評分示例 · Worked example（M5）

> 練習 M5.2 要學生解讀 demo 中「MI300X max_population = 292 ≈ 36.5× 於 RX 7900 XTX」。

- **Level 4：** 「292 來自 deterministic 容量公式 `P_max = (capacity − weights − overhead − shared_prefix_KV) / branch_KV`
  （與 `tests/` 同一套數學），而 MI300X **無實機** → 這是 **SIMULATED**；36.5× 是**容量比**（能同時開多少 branch），
  **不是**真硬體上量到的吞吐 speedup。價值是『大 population + 長熱記憶不再零和』，UI 與匯出都掛 `SIM` 徽章。」
  —— 引用 backend 公式、標 SIMULATED、講清楚「容量比 ≠ 實測 speedup」。
- **Level 2：** 「MI300X 可以開 292 條，比較多」—— 方向對，但沒標 SIMULATED、沒說是容量比。
- **Level 1：** 「MI300X 實測快 36.5×」—— 把模擬的容量比當成真硬體實測 speedup。Auto-capped。

---

## 使用方式 · Applying the rubric

- 每題按 C1–C5 打分、加權、加總成 0–100 的模組分數。
- 整體成績可依你的 syllabus 對 **產品主線（M0–M5, M7–M9）** 與 **進階軌（M6）** 分別加權；
  每題背後的模組/教材/hands-on 見 L2 [`module-map.md`](module-map.md)。
- 回傳成績前，拿學生引用的數字對照 L5 [`validation-ledger.md`](validation-ledger.md)：確認它標對了 live/mock/SIMULATED，
  且引用的是 backend 計算 / committed 測試，而不是模擬值或前端估算。

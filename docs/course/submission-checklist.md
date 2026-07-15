<!--
衍生自 canonical 範本（zkp-final/docs/submission-checklist.md · Layer 4b）。
兩種讀者：交練習題的學生、改課程/平台內容的貢獻者。誠實（標 live/mock/simulated）與「引用 backend 計算/committed 測試」
是不可協商的閘。Diátaxis 類型：how-to。
-->

# Submission Checklist · 繳交檢查表

> **6 層課程標準的 L4（評量）。** 交件前的一道逐項打勾閘，兩種讀者：(A) 交模組練習題的**學生**、
> (B) 改課程/平台內容的**貢獻者**。對應 [`grading-rubric.md`](grading-rubric.md)；證據索引在
> [`validation-ledger.md`](validation-ledger.md)。

---

## A. 學生 —— 練習題 · Students

- [ ] 我先用 **deterministic Gatekeeper** 判可行性（`VRAM=W+KV+Act+Overhead`、`tokens/s=BW/bytes`），
  沒有讓 LLM「講贏」物理；我分清 deterministic（gatekeeper）與 fuzzy（evaluator）。
- [ ] 我引用的每個數字都**標了來源**：`live`（真跑 / deterministic 物理）· `mock`（無本地推論 server 的降級）·
  `SIMULATED`（Tier 4 Instinct）。沒有把模擬值當實測。
- [ ] 我引用的是 **backend 計算 / committed 測試**（如 `tests/test_vram.py`），不是前端估算或憑空的數字。
- [ ] 我解釋了**推理**（decode 為 memory-bound、`savings=φ(P-1)/P`、容量→更少節點→更低 TCO），不只給結論。
- [ ] 我沒有 over-claim：把「**受控進化**」（epoch freeze + held-out anchor + HITL）與「自動變聰明」分清楚；
  談 evaluator 不被 hack 時我引用了三道結構性防線。
- [ ] Instinct（MI300X/MI325X）相關的一律標 `SIMULATED`，並說明它是**容量/公式模擬**而非真硬體實測。

---

## B. 貢獻者 —— 修改課程/平台內容 · Contributors

### B1. 正確性與誠實 · Correctness & honesty

- [ ] 變更遵守 [`course-contract.md`](course-contract.md) §6 的誠實邊界（物理永不進化、Instinct 一律 SIMULATED）。
- [ ] 任何新增/修改的數字都引用 backend 的 deterministic 計算或 committed 測試，且與
  [`../02-sizing-math.md`](../02-sizing-math.md) 的公式一致；新證據已加進 L5 [`validation-ledger.md`](validation-ledger.md)。
- [ ] 沒有把模擬的 Instinct 數字當實測；沒有把前端 MOCK（`frontend/lib/vram.ts`）當 backend 權威。
- [ ] 前後端公式維持一致（`frontend/lib/vram.ts` ↔ [`backend/gatekeeper/vram.py`](../../backend/gatekeeper/vram.py)），
  所以 Live / Mock 數字仍恆等。

### B2. 可重現 · Reproducibility

- [ ] `uv run pytest` **GREEN（56 passed）** —— deterministic、離線、無需 GPU；物理數學測試（`test_vram` / `test_bandwidth` /
  `test_feasibility`）必過。
- [ ] `./scripts/demo.sh` 在筆電上端到端跑通（推論無 server 時自動 MOCK，Qdrant 缺席時 in-memory）。
- [ ] 我引用的每個 REST endpoint / `pytest` 檔 / 腳本都真的存在且行為如描述。

### B3. 文件衛生 · Docs hygiene

- [ ] 所有內部連結可解析（相對路徑；GitHub 與本機皆可）。`docs/course/` 內連到藍圖用 `../`，連到 repo 根用 `../../`。
- [ ] Markdown 語法正確；mermaid 圖可 render（label 一律加雙引號以容納中文與括號/冒號，見
  [`../01-orchestration.md`](../01-orchestration.md) §3 註）。
- [ ] 我沒有修改 `docs/` 的架構藍圖（`blueprint.md`、`01-04`、`DEMO.md` 為 final，勿改）；課程層只加在 `docs/course/`。
- [ ] 我沒有手改生成物（`frontend/.next/**`、`data/epoch_state.json`、`data/rubric_history/challenger-*.xml` 等 runtime）。

### B4. 範圍與提交紀律 · Scope & commit discipline

- [ ] 我**只**動我任務擁有的檔案（此課程層 = `docs/course/` + `course.yaml`），用明確 pathspec。
- [ ] 若有專門的 commit 階段擁有 git 操作，我**沒有**自己跑任何 `git` 變更（add/commit/stash/reset/push）；
  唯讀 `git status` 可以。
- [ ] 真的要 commit 時，用 typed **Conventional Commit**，且 body 以 `Signed-off-by:` trailer 結尾（repo 慣例）。

---

## 一鍵筆電閘 · The one-command gate

```bash
uv run pytest          # backend 56 passed（deterministic，離線，無需 GPU）
./scripts/demo.sh      # 端到端契約走查（推論無 server 時自動 MOCK；Instinct 標 SIMULATED）
```

> 若 `pytest` 綠燈、`demo.sh` 端到端跑通、且你引用的每個數字都標對 live/mock/SIMULATED 並回溯到 backend 計算 /
> committed 測試，這份繳交就通過機械閘；接著由 [`grading-rubric.md`](grading-rubric.md) 評推理。

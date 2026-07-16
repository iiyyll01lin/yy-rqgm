# A3 — The RQGM Evaluator / 會進化的品味閘門

> 模組定位：定義那個坐在物理閘門之後、**fuzzy 且受控進化**的評審。它是全平台唯一被允許進化的元件，因此也是最需要防呆的元件。
> 上游：[`01-orchestration.md`](./01-orchestration.md)（它在圖中的位置、epoch freeze）、[`02-sizing-math.md`](./02-sizing-math.md)（它的記憶吃多少 VRAM）。下游：[`04-stack-export.md`](./04-stack-export.md)（它跑在哪些開源元件上）。

---

## 1. 這個評審的職責與哲學

物理閘門（[`02-sizing-math.md`](./02-sizing-math.md)）回答「跑不跑得動」；`RQGM Evaluator` 回答一個沒有 closed-form 的問題：**「這個 agent 架構在這個 domain 到底好不好？」**

「好不好」是品味問題，而品味最容易出兩種錯：

1. **諂媚（sycophancy）**：LLM-as-judge 傾向給高分、列優點、附和。
2. **被症狀騙**：把「數字變好看」誤認為「問題被解決」。

AgentForge 的 evaluator 用一個刻意「刻薄」的設計對抗這兩者：**Deficit Scoring（缺陷評分）**。它不列優點，只從滿分往下扣；它的唯一任務是**找 Red Flag（危險信號）**。這把評審的心態從「這方案哪裡好？」強制切換成「這方案哪裡會害死人？」——這正是一位資深可靠度工程師（Principal Reliability Engineer）看設計的方式。

### Deficit Scoring 的計分定義

```text
deficit_score = clamp( base − Σ strength_credit + Σ caught_flaw_penalty , 0.0 .. 1.0 )
quality_score = 1 − deficit_score                     # 供 UI 顯示（可 ×100 當百分比）
verdict       = weak   if deficit_score ≥ τ           # τ = 當前 strictness level（§6.3 rqgm tolerance）
              = strong otherwise
```

- 分數是 **bounded float `[0,1]`**（`0.0` 完美、`1.0` 不可接受）。每個 `<criterion>` 帶 `weight`（champion-0：`physics_common_sense .35 / diagnostic_resilience .30 / modularity_drift .20 / safety_autonomy .15`）供 live judge 權衡。**offline** 由 deterministic rubric-aware mock（[`backend/inference/mock_scoring.py`](../backend/inference/mock_scoring.py)）計分：植入的 flaw tag **只有當前 rubric 有能抓它的 criterion 時才扣分**，strength tag 給正向 credit，總和被 clamp——**length-normalization 因此是內建的**（per-criterion deficit 有上限），不是另一道 transform。
- **`τ_epoch` 不再是寫死在 XML 裡的數字**。pass 門檻改由官方 `rqgm` 套件的 **tolerance schedule**（`strictness_level`）承載，偵測到 exploitation 時自動收緊（§6.3）。
- **critical「一票否決」是 *live judge* 的設計意圖**；目前無 live 模型時，offline mock 採 bounded 加總、**不做硬性 veto**——這是誠實的近似（見 §5 的 offline 誠實邊界），別把它當已上線的 live 行為。

---

## 2. System-Prompt：XML Rubric 模板（reference schema）

以下是 evaluator system prompt 的**設計參考 schema**，以 XML 承載。選 XML 的理由：結構嚴格、易於程式化地 diff/mutate（GEPA 對它做反思式變異，§5）、且強制模型產出可解析的結構化輸出。`{{...}}` 為執行期注入的變數；所有 identifier 保持英文。

> **與 shipped code 的對齊（誠實註記）**：實際上線的 champion-0 seed 是 [`backend/evaluator/rubric.xml`](../backend/evaluator/rubric.xml)——一個**較精簡的變體**：4 條加權 `<criterion>`（`physics_common_sense .35 / diagnostic_resilience .30 / modularity_drift .20 / safety_autonomy .15`），`deficit_score` 為 **float `[0,1]`**（非 0–100 整數），severity 用 `low|medium|high`，且 `<output_contract>` 要求 **STRICT JSON**（非 strict-XML）。domain pack 的 criteria（`smart_manufacturing` 另有 `actuation_safety`）與 poison pill 於 judge 時 merge 進來；**poison pill 只在 `strict` 評分模式注入**（loose 模式不注入，這正是 hack-ratio 的來源，§6.3）。下方大 XML 塊保留 penalty/verdict 的完整設計語彙，但請以 shipped seed 為準。

```xml
<evaluator epoch="{{epoch_id}}"
           role="Principal Reliability Engineer"
           domain="{{domain}}"
           scoring="deficit"
           pass_threshold="{{tau_epoch}}">

  <persona>
    You are a Principal Reliability Engineer with 20 years in {{domain}}.
    You are paid to find what will FAIL in production, not to praise.
    You are deeply skeptical of solutions that make metrics look good without
    addressing physical root cause. You have seen "numerical duct-tape" cause
    real-world incidents. You never flatter. You start every candidate at a
    perfect score and DEDUCT for every deficit you can justify with evidence.
  </persona>

  <scoring_protocol>
    <rule>Start at deficit_score = 0. Add penalties; never subtract.</rule>
    <rule>Every red_flag MUST cite concrete evidence from the candidate trace.</rule>
    <rule>Any single CRITICAL red_flag forces verdict=FAIL regardless of total.</rule>
    <rule>Speculative flags without trace evidence are NOT allowed (they would
          poison the GEPA gradient). If unsure, mark confidence=low.</rule>
    <rule>Penalties: critical=40, major=15, minor=5.</rule>
  </scoring_protocol>

  <rubric>
    <criterion id="physics_common_sense" max_penalty="40">
      <intent>Does the solution fix the ROOT CAUSE, or mask a SYMPTOM?</intent>
      <red_flag id="pc_ducttape" severity="critical">
        Clamps / filters / re-tunes a metric to hide an anomaly while the
        underlying physical state variable remains out of bounds.
      </red_flag>
      <red_flag id="pc_conservation" severity="critical">
        Proposed action violates a conservation law or a hard physical limit
        (thermal, pressure, flow, mass balance).
      </red_flag>
      <red_flag id="pc_units" severity="major">
        Dimensional / unit inconsistency, or a magnitude that is physically
        implausible for the equipment class.
      </red_flag>
    </criterion>

    <criterion id="diagnostic_resilience" max_penalty="35">
      <intent>Does the diagnosis survive cascading noise and false positives?</intent>
      <red_flag id="dr_single_sensor" severity="major">
        Root-causes off a single sensor without cross-validation against
        physically-correlated sensors.
      </red_flag>
      <red_flag id="dr_no_debate" severity="major">
        No multi-agent debate / independent cross-check before committing to a
        high-impact action.
      </red_flag>
      <red_flag id="dr_noise_capture" severity="critical">
        Treats correlated sensor noise as ground-truth signal and acts on it.
      </red_flag>
    </criterion>

    <criterion id="modularity_drift" max_penalty="25">
      <intent>Is the LangGraph implementation modular, or tightly coupled?</intent>
      <red_flag id="md_tight_coupling" severity="major">
        Nodes read/write ad-hoc keys instead of a declared State Schema;
        hidden coupling that will drift under evolution.
      </red_flag>
      <red_flag id="md_no_state_schema" severity="major">
        No TypedDict State Schema; state shape is implicit and unauditable.
      </red_flag>
      <red_flag id="md_god_node" severity="minor">
        A single "god node" concentrates unrelated responsibilities.
      </red_flag>
    </criterion>
  </rubric>

  <poison_pills domain="{{domain}}">
    {{injected_poison_pills}}
    <!-- e.g. correlated sensor noise; broken cooling valve masked as setpoint
         drift; a plausible-but-wrong upstream reading -->
  </poison_pills>

  <memory injected="hybrid_search">
    {{retrieved_heuristic_failures}}   <!-- memory_type=heuristic_failure -->
    {{retrieved_physics_truths}}       <!-- memory_type=physics_truth -->
  </memory>

  <candidate>
    {{candidate_architecture_and_execution_trace}}
  </candidate>

  <output_format enforce="strict-xml">
    <!-- The model MUST emit exactly this structure. Forced chain-of-thought. -->
    <thinking>
      Step through EACH criterion. For each poison_pill, state whether the
      candidate detected it and reacted to root cause or masked the symptom.
      Reason BEFORE scoring. This block is mandatory and is read verbatim by
      the GEPA reflection engine as Actionable Side Information.
    </thinking>
    <red_flags>
      <flag criterion="" id="" severity="" confidence="" penalty="">
        <evidence>quote/point to the exact trace location</evidence>
        <root_cause_analysis>symptom vs root cause explanation</root_cause_analysis>
      </flag>
      <!-- zero or more -->
    </red_flags>
    <criterion_scores>
      <score criterion="physics_common_sense" penalty=""/>
      <score criterion="diagnostic_resilience" penalty=""/>
      <score criterion="modularity_drift" penalty=""/>
    </criterion_scores>
    <deficit_score>INTEGER</deficit_score>
    <quality_score>100 - deficit_score</quality_score>
    <verdict>PASS|FAIL</verdict>
    <verdict_reason>one-line justification tied to the highest-severity flag</verdict_reason>
  </output_format>
</evaluator>
```

**為什麼強制 `<thinking>` 在前（Forced Chain-of-Thought）**：讓模型**先推理、後計分**，避免它先射箭（給分）再畫靶（湊理由）。更重要的是，這段 `<thinking>` 是 GEPA reflective evolution 的**原料**——GEPA 讀的是完整推理軌跡（Actionable Side Information），不是被壓成 scalar 的分數（§5）。因此 `<thinking>` 的品質直接決定進化訊號的品質。

---

## 3. 三大 Failure Mode 與 Poison-Pill 驗證

Rubric 的三個 criterion 不是憑空來的；它們對應 agentic 製程優化在真實世界最常見的三種死法。每一種都用一個 **poison pill**（刻意注入的 adversarial 情境）來測——若 evaluator 抓得到，它就有資格當裁判；抓不到，就是它自己該進化的地方。

```mermaid
flowchart TD
    Cand["candidate solution + trace"] --> F1["Failure Mode 1<br/>Physics Common Sense"]
    Cand --> F2["Failure Mode 2<br/>Diagnostic Resilience"]
    Cand --> F3["Failure Mode 3<br/>Implementation Drift"]
    F1 -->|"poison: broken cooling valve"| R1{"修 root cause<br/>還是 duct-tape?"}
    F2 -->|"poison: correlated sensor noise"| R2{"跨感測器交叉驗證<br/>還是單點誤判?"}
    F3 -->|"inspect LangGraph"| R3{"clear State Schema<br/>還是 tight coupling?"}
    R1 -->|"duct-tape"| Flag1["critical: pc_ducttape"]
    R2 -->|"acts on noise"| Flag2["critical: dr_noise_capture"]
    R3 -->|"coupled"| Flag3["major: md_tight_coupling"]
```

### 3.1 Failure Mode 1 — Physics Common Sense / Anti-Numerical-Duct-Tape

**問題**：agent 把「症狀」當「問題」，用數值手段把警報壓掉，卻沒碰物理 root cause。

**Poison pill：broken cooling valve（冷卻閥卡死）**。感測器顯示某機台溫度持續攀升。
- **不合格反應（duct-tape）**：「調高溫度警報上限 / 對溫度讀數做低通濾波 / 把 setpoint 容忍度放寬」——警報消失了，數字好看了，但閥還是壞的，機台正在走向熱失控。→ evaluator 開 `pc_ducttape`（critical）→ 一票 FAIL。
- **合格反應**：交叉比對「閥開度指令 vs 實際流量回饋」，發現指令有下、流量沒上，定位到 valve 機械故障 → 建議工單維修 + 短期降載保護。→ 無 red flag。

判準核心：**症狀被壓下去了，但 root-cause 狀態變數（valve 開度殘差）是否仍異常？** 這正是 [`01-orchestration.md`](./01-orchestration.md) §7.2 的 PyTorch surrogate 要驗證的東西——用 surrogate 預測「這個動作之後，root-cause 變數會不會還在界外」。

### 3.2 Failure Mode 2 — Diagnostic Resilience（抗噪與抗誤報）

**問題**：真實工廠的感測器會壞、會漂、會被彼此干擾。只信單一讀數的 agent 會被 cascading false positive 帶著走。

**Poison pill：correlated sensor noise（相關性感測雜訊）**。同一條產線上多個溫度計因共用電源/接地問題**同時**跳出相關的假讀數，看起來像「整條線過熱」的真事件。
- **不合格反應**：把相關雜訊當成真信號，觸發大規模停機/調整。→ `dr_noise_capture`（critical）。
- **合格反應**：用 **multi-agent debate / cross-validation**——一個 agent 主張「過熱」，另一個 agent 反駁「若真過熱，下游壓力/流量應同步變化，但沒有 → 更可能是量測鏈共模故障」。透過辯論與物理相關性交叉驗證，識破共模雜訊。→ 無 red flag。

判準核心：高影響動作前，是否有**獨立交叉檢查**？是否理解「哪些感測器在物理上應該相關」，並用這個先驗去識破共模故障？

### 3.3 Failure Mode 3 — Implementation Drift & Modularity

**問題**：agent 架構本身若是一坨 tight coupling，隨著 evolution 會 drift 成無法維護、無法稽核的黑箱。

**檢查對象：LangGraph 實作本身**。
- **不合格反應**：node 之間靠 ad-hoc dict key 互相偷讀偷寫，沒有宣告 State Schema；一個 god node 包辦解析、判斷、匯出。→ `md_tight_coupling` / `md_no_state_schema`（major）。
- **合格反應**：明確的 `TypedDict` State Schema（見 [`01-orchestration.md`](./01-orchestration.md) §2）、單一職責 node、物理與品味欄位型別分離。→ 無 red flag。

判準核心：這正是為什麼平台自身把 `gatekeeper/` 與 `evaluator/` 物理分離——**架構的可稽核性本身就是一條 rubric**。evaluator 用同一把尺量它評判的 agent。

---

## 4. Epoch-Frozen Evaluator：先凍結，才談進化

（機制細節見 [`01-orchestration.md`](./01-orchestration.md) §6，此處補充 evaluator 視角。）

一個 epoch 內，上面那份 XML rubric（含所有 penalty 數值、`τ_epoch`、poison pill 集合）**逐字凍結**。hot path 每次判斷都由這個 frozen 版本產生，並在輸出蓋上 `epoch_id`。

**為什麼非凍不可**：若 rubric 在使用中途變動，我們就無法宣稱「這個 epoch 的 evaluator 比上個 epoch 好」——因為連比較的基準（utility function）都在漂移。凍結讓 utility 在 epoch 內 stationary，「per-epoch 自我改進」才是一個可驗證的命題，而非自我安慰。

cold path 這期間照常用 GEPA 醞釀 challenger，但 challenger **一律不上線**，只在 epoch 邊界受檢（§6）。

---

## 5. GEPA-Style Reflective Evolution of the Rubric

evaluator 怎麼變聰明？不是靠 RL 把回饋壓成 scalar reward 去 fine-tune，而是靠 **GEPA（Genetic-Pareto）的反思式進化**：讀完整的失敗軌跡，用自然語言診斷「rubric 為什麼漏掉了這個 red flag」，然後對 rubric 這段**文字**做 targeted mutation。

```mermaid
flowchart TD
    Traces["收集 epoch 內的<br/>execution traces + HITL feedback"] --> Reflect["Reflection LLM 讀完整 trace<br/>(Actionable Side Information)"]
    Reflect -->|"自然語言診斷:<br/>rubric 漏了哪條 red flag?"| Mutate["對 rubric XML 做 targeted mutation<br/>(新增/改寫 criterion 或 poison_pill)"]
    Mutate --> Pool["challenger rubric 候選池"]
    Pool --> Pareto["Pareto frontier 篩選<br/>(per-poison-pill, per-criterion 各自留最佳)"]
    Pareto --> Anchor{"在 held-out anchor 上<br/>統計勝過 incumbent?"}
    Anchor -->|"no"| Pool
    Anchor -->|"yes"| Propose["提名為 challenger → §6 epoch gating"]
```

為什麼 GEPA 適合這裡（相對 RL）：

1. **讀 trace 當 textual gradient**：GEPA 讀 evaluator 的 `<thinking>` 與 candidate 的完整軌跡，診斷「為什麼這條 poison pill 被漏判」，產生**可執行的文字修正**（例如「新增一條 red flag：當閥開度指令與流量回饋背離時視為 pc_ducttape」）。這比「答錯了，reward −1」資訊量高幾個數量級。
2. **樣本效率 ~35× 於 RL(GRPO)**：evaluator 的每次「rollout」都很貴（要跑完整 candidate + surrogate 驗證），GEPA 用 100–500 次評估達到 RL 需要上萬次的效果——這與本平台把進化放 cold path、盡量省 rollout 的成本邏輯（[`02-sizing-math.md`](./02-sizing-math.md) §5.4）完全一致。
3. **Pareto frontier 防 collapse**（**已實作**，[`backend/evaluator/frontier.py`](../backend/evaluator/frontier.py)）：不是只留「總分最高」的單一 rubric，而是保留 **top-K 非 dominated** 的族群。多目標為：**per-criterion 分離度 `sep::<criterion>`**（每個 criterion / poison pill 各自留最佳）＋ **`parsimony`**（`−(GEPA 新增 criterion 數)`，逼出「廣覆蓋 vs 精簡」的真取捨）＋ **`adversarial`**（self-play 紅隊樣本上仍嚴格者得分更高，見本節下方 Adversarial Self-Play）。這避免 local optimum 與 diversity collapse（rubric 全退化成只會抓一種錯）。frontier 在 **`dev` 選擇split**（selection split）上以 BBε 下界排名——刻意選在 code gate 從不看的 split 上，消除 winner's curse：gate 之後在 `val` 上的再測才是**真的** held-out，而不是重測 selection 已偷看過的 split。父代選擇改用 **Thompson 取樣**（每個 frontier member 帶一個 `Beta(prior+成功, prior+失敗)` 後驗，取代舊的近乎均勻 `bbe+1` 權重）。整條 frontier 每個 epoch 落地到 `data/frontier/epoch-*.json` 供重現。

```python
# GEPA reflective mutation loop (cold path) — 已實作於 backend/evaluator/{evolve,frontier}.py
def gepa_evolve(incumbent_text, *, epoch, budget=6, top_k=8, adversarial_samples=None):
    frontier = ParetoFrontier(top_k=top_k)                         # top-K 非 dominated
    frontier.add(incumbent_text, compute_objectives(incumbent_text, DEV))   # 在 dev 選擇split 評
    for i in range(budget):
        parent = frontier.sample_stochastic(rng)                  # Thompson 取樣父代（每個 member 一個 Beta 後驗）
        trace  = select_failure_trace(parent, TRAIN_weak)         # 挑 strict−loose 落差最大的 train weak anchor
        _, _, new_criteria = reflect_and_mutate(parent, trace, epoch)   # 反思 → 提新 criterion
        try:
            child = mutate_rubric_text(parent, new_criteria, ...)  # 改 rubric XML；不 well-formed 就 reject
        except RubricValidationError:
            continue
        obj, bbe, _ = compute_objectives(child, DEV, adversarial_samples=...)  # 只在 dev 選擇split 評（val 留給 gate）
        kept = frontier.add(FrontierMember(child, objectives=obj, bbe=bbe))    # 非 dominated 才留
        frontier.record_child_outcome(parent, improved=kept and bbe > parent.bbe)  # Thompson 回饋
    return frontier          # frontier.best()（dev 上 BBε 下界最佳）→ 送 §6 code gate（gate 再於 held-out val 重測）
```

> **offline 誠實邊界**：無 live 模型時，`reflect_and_mutate` 走 deterministic mock（`MockMarker.MUTATE`），失敗 trace 的挑選也是 deterministic（strict−loose 落差最大者）。因此 offline 的「反思」不是真 LLM 診斷，而是一條可重現的 fallback（傾向補上 `reward_hacking_resistance` 這類 champion-0 漏掉的 criterion）。真 reflective 進化需接上本地模型（Lemonade/vLLM）。GEPA 論文宣稱的 ~35× rollout 效率是引用其研究結果，非本 repo offline 實測。

### Adversarial Self-Play（red-team 對抗目標）— 已實作

frontier 的 `adversarial` 目標由一個 self-play 紅隊供料（[`backend/evaluator/adversarial.py`](../backend/evaluator/adversarial.py)）：

- `champion_blind_spots(champion_text)`：列出**當前 champion（含 domain fragment）漏抓的 poison-pill flaw**——即 rubric 裡沒有任何能抓它的 criterion 的那些（例如 champion-0 缺 `reward_hacking_resistance` / `drift_monitoring`）。
- `generate_adversarial_samples(...)`：對每個 blind spot 生一個 **gamed 架構**——植入 champion 會獎勵的 disguise strengths（typed state、root-cause、HITL）當偽裝，藏一顆 champion 抓不到的 poison pill（loose 過、strict 敗）。
- frontier 的 `adversarial` objective 對「在這些對抗樣本上仍保持嚴格（高 deficit）」的 challenger 加分——對應論文「reviewer 對 gamed 答案也要嚴格」。**offline 誠實邊界**：blind spot 由 champion 自己的 criterion id **deterministic 推導**，不需 RNG 或模型呼叫。

### Judge Panel + Self-Consistency + 校準指標（accuracy / κ）— 已實作

單一 LLM-as-judge 有雜訊與偏誤；[`backend/evaluator/panel.py`](../backend/evaluator/panel.py) 用一個 **panel** 降噪：

- **多 persona / 溫度**：預設 `principal_reliability_engineer@0.1`、`safety_auditor@0.3`、`adversarial_red_teamer@0.6`。
- **self-consistency 聚合**：deficit 取 **median**、red flag 取**多數決**。
- **bias 控制**：criterion 順序 **seeded 隨機化**（verdict 不得依賴排序）；length-normalization 因 deficit 是 bounded per-criterion 加總而**內建**。
- **校準指標**：`anchor_agreement()` 回報 judge 與人類標註的 **accuracy + Cohen's κ**（不是原始 separation）——這才是衡量「裁判準不準」的指標，於 `GET /api/admin/report` 與 `/health` 曝露。
  - 釐清：**晉升 gate（§6.2）仍以 separation 的統計下界**判定；accuracy/κ 是**校準/透明度**指標，兩者並存不衝突。
  - **offline 誠實邊界**：panel 成員間的歧異由 mock 的 deterministic per-persona jitter 模擬，好讓 self-consistency 有東西可聚合；真多樣性需 live 模型。

---

## 6. Held-Out Anchor Gating（CODE gate 先行）+ Selective Erasure

challenger 再漂亮，也**不會自動上線**。它必須先過一道 **code 統計門檻**，通過後 HITL 才被諮詢——而 **HITL 只能否決、不能覆寫失敗的 gate**（[`backend/evaluator/evolve.py`](../backend/evaluator/evolve.py) `approve_challenger` 兩段式）。這是補回的 RQGM 反 reward-hacking 核心（先前版本把一個 `separation_delta = −0.34` 的**更差** challenger 純憑 HITL 布林值升級了）。

### 6.1 Held-out labeled anchor set + train/dev/val/test 隔離

anchor 不再是「拿 HITL feedback 當替身」。它現在是一份**真的 held-out human-labeled set**：[`data/anchor/anchor_architectures.json`](../data/anchor/anchor_architectures.json) 有 **73 個**架構（跨**兩個 domain pack**：`smart_manufacturing` + `grid_energy`，測 generalization），每個帶 `label`（`weak|strong` 人類 ground-truth）、`domain`、植入的 `flaws[]` / `strengths[]` tag，外加一份 `flaw_taxonomy`（[`backend/evaluator/anchors.py`](../backend/evaluator/anchors.py)）。**四路 data isolation** 防 gate 被 reward-hack：

| split | 數量 (weak/strong) | 用途 |
|-------|------|------|
| `train` | 28 (20/8) | **只有** GEPA proposer 看得到（挑失敗 trace、反思變異的文字梯度） |
| `dev` | 12 (8/4) | **frontier 選擇/排名**（`frontier.best` 以 dev 上 BBε 下界）——gate 從不看，故消除 winner's curse |
| `val` | 19 (13/6) | **只有** code gate 評分（P1 非劣 + P2 Bayesian 後驗/MDE、hack-ratio） |
| `test` | 14 (9/5) | **只**供報告（`/api/admin/report`），從不驅動變異、選擇或 gate |

在 `dev`（而非 `val`）上選 frontier，是刻意的：`val` 只被 code gate 評分、selector 永遠碰不到，因此 gate 是一次**真正的 held-out 再測**，而非重測 selector 已偷看過的 split（no winner's curse）。HITL feedback 仍會收集，但它現在餵給 GEPA 當 textual side-information、並在 gate 後當**最終否決權**，不再是唯一的 gate。

### 6.2 兩段式晉升門檻：CODE gate → HITL veto

**Stage 1 — CODE gate**（[`backend/evaluator/gate.py`](../backend/evaluator/gate.py)，只看 `val`）：

- **P1（non-inferiority）**：`challenger_sep ≥ champion_sep`，且**平手偏袒現任**（`tie_favors_incumbent` → challenger 必須*嚴格*更好）。`separation = mean deficit(weak) − mean deficit(strong)`，越高越能分辨好壞。
- **P2（Bayesian Beta-Binomial 後驗 + 最小可偵測效果 MDE）**：舊 P2 是對 ~7 個 val anchor 做 paired bootstrap，離線時強項相消（有效 N≈4 weak），會**結構性地否決真的單一 flaw-family 修正**。現改為對 weak（gamed）anchor 的**逐錨點配對*勝負指標***建 Beta-Binomial 後驗：每個 weak anchor，challenger 罰得比 champion *更重*（`challenger_deficit > champion_deficit`，抓到更多植入失敗）記一勝、反之記一負、平手（|Δ|<ε）不計方向。以 `Beta(prior_α+勝, prior_β+負)` 後驗回報 `posterior_prob_improvement = P(Δsep>0) := P(θ>0.5)`（challenger 在 held-out 弱架構上分離度優於現任的後驗機率）。**晉升需 `P(Δsep>0) ≥ posterior_threshold`（預設 0.95）且觀測效果 `effect_size = Δsep ≥ min_detectable_effect`（MDE，預設 0.10）兩者皆過**；並回報 `n_wins/n_losses/n_ties` 與 **per-flaw-family 勝負 breakdown**（`per_flaw_wins`）供稽核。

- **P3（跨 epoch anytime-valid 修正，選配 / 預設關閉）**：per-epoch P1+P2 只保**單一** epoch 的型一誤差；反覆晉升會膨脹 **family-wise error**。開啟 `GateConfig.sequential_correction` 後，gate 額外跑一條 **e-process**（[`gate.epoch_e_value`](../backend/evaluator/gate.py)）：每個 epoch 依 weak-anchor 勝負算一個單邊 **e-value**（對 `H0: θ=0.5` 的 Bayes factor，alternative 為截斷在 `θ>0.5` 的 Beta——只獎勵「challenger 勝過半數」的證據），跨晉升**累乘**成一條 test martingale，持久化於 `data/gate_sequential.json`（gitignored）。晉升需累積財富 `∏ eᵢ ≥ 1/α`（Ville 不等式 → 任意時刻 family-wise 型一誤差 ≤ `family_wise_alpha`）。真實進步 `e≫1` 會累積財富、照常晉升；靠雜訊反覆晉升 `e≈1`、財富永遠過不了門檻。**每次實際晉升才 commit 一次 look**（花掉預算），rejected challenger 不耗預算。

兩關（開 P3 則三關）都過 `passed=True` 才進 Stage 2。`gate` dict 因此帶 `p1_non_inferior` / `p2_passed` / `posterior_prob_improvement` / `posterior_threshold` / `effect_size` / `min_detectable_effect` / `n_wins` / `n_losses` / `n_ties` / `n_weak` / `per_flaw_wins`（**取代**舊的 `bootstrap_lower_bound` / `bootstrap_alpha`）＋（P3）`sequential_correction` / `sequential_e_value` / `sequential_log_wealth` / `sequential_threshold` / `family_wise_alpha` / `sequential_passed` / `sequential_looks_prior`。

**Stage 2 — HITL veto**：code gate 通過後才諮詢人類；`approve=False` 否決一個原本會過的 challenger，`approve=True` 才 `epoch++` 晉升。**人類無權覆寫失敗的 code gate。**

```mermaid
flowchart LR
    Ch["challenger rubric<br/>(frontier best · dev BBε)"] --> Gate{"CODE GATE (val)<br/>P1 非劣 + P2 後驗 P(Δsep>0)≥0.95 & Δsep≥MDE?"}
    Gate -->|"fail"| Keep["維持 incumbent<br/>challenger 回 frontier<br/>(HITL 不被諮詢)"]
    Gate -->|"pass"| HITL{{"HITL 加簽 (安全鎖 / 只能否決)"}}
    HITL -->|"veto"| Keep
    HITL -->|"approve"| Promote["challenger 成為新 frozen evaluator<br/>epoch_id += 1"]
    Promote --> Erase["selective erasure:<br/>soft-delete + reconfirm"]
```

`POST /api/admin/epoch/approve` 因此回傳 `gate`（P1 非劣 + P2 後驗機率/MDE/per-flaw 明細）、`hitl`（是否被諮詢/否決）、`champion_exploitation` / `challenger_exploitation`（§6.3）、`erased_memories` / `reconfirmed_memories`（§6.4）。

### 6.3 Strict/Loose Hack-Ratio + `rqgm` tolerance tightening

為了偵測 reward hacking，judge 對每個 candidate 同時算兩種分數（[`backend/evaluator/judge.py`](../backend/evaluator/judge.py) `score_candidate`）：**loose**（只罰當前 rubric 抓得到的）與 **strict**（額外套用全部 poison-pill 檢查）。`hack_ratio = quality_strict / quality_loose`（`quality = 1 − deficit`）在 rubric 有 poison-pill 盲點時**掉到 1 以下**。

這條訊號接上官方 **[`rqgm`](https://pypi.org/project/rqgm/) 套件**（已加入 [`pyproject.toml`](../pyproject.toml)）的 `EpochManager` / `EpochConfig` / `TransitionReason`（[`backend/evaluator/rqgm_adapter.py`](../backend/evaluator/rqgm_adapter.py)）：mean hack-ratio 低於門檻 → 觸發 `EXPLOITATION_DETECTED` → **收緊 tolerance schedule**（丟掉最鬆的一級 = evaluator 變嚴），schedule 持久化到 `data/rqgm_state.json` 跨 epoch 累積。若環境無法 import `rqgm`（完全 air-gapped 無 wheel），一個**忠實的 local fallback** 複製同樣的 hack-ratio + 單級收緊邏輯；`RQGM_BACKEND` 記錄目前是 `"rqgm"` 還是 `"local-fallback"`。

> **對齊 τ_epoch（§1）**：RQGM 的 tolerance level 就是本平台的 `strictness_level`——把「penalty / τ 進化」用 tolerance schedule 承載，而**非**把數字重寫進 rubric XML。這是刻意的 adapter 設計，不是 XML 改寫。

**額外的透明度監測（[`backend/evaluator/report.py`](../backend/evaluator/report.py) `build_report` / `health_summary`）**：除了 hack-ratio，報告另算兩個泛化 / 過擬合訊號：

- **over-acceptance**（`over_acceptance.over_acceptance_rate`）：用強化過的 self-play 紅隊（**含 out-of-catalog 偽裝 gaming**）算「被 rubric 誤判為 strong（`deficit < τ`）的 gamed/weak 樣本比例」。高值 = 判官被偽裝的 poison pill 騙過（generalization 差、reward-hacking 風險），每個樣本回報 `deficit` / `accepted_as_strong` / `out_of_catalog`。
- **over-optimization gap**（`over_optimization.separation_gap`，`/health` 為 `proxy_gold_separation_gap`）：`proxy(val) − gold(test)` 分離度落差。大的正 gap = champion 在靠近進化迴圈的 `val` 上比未動過的 gold `test` 明顯更利（過度最佳化警訊）。

報告與 `/health` 亦帶 **`provenance`**（`judge_model` + `git_sha` + `rqgm_backend` + `using_mock`），把每份數字綁到產生它的模型與原始碼版本，方便重現 / 稽核。

### 6.4 Selective Erasure（選擇性抹除，非全量清空）

升級後，舊 evaluator 產生的部分「效用判斷」可能已失效——但**不是全部**。selective erasure（**已實作**，[`backend/evaluator/evolve.py`](../backend/evaluator/evolve.py) `selective_erasure`）只處理「其效用值依賴被替換 evaluator」的記錄，靠 `created_at_epoch` 標籤精準定位（§8），且做的是 **soft-delete + reconfirm**、不是硬刪：

```python
def selective_erasure(memory, up_to_epoch, new_champion_text, ...):
    for rec in memory.fetch(memory_type=HEURISTIC_FAILURE,      # physics_truth 從不被掃
                            max_epoch=up_to_epoch, active_only=True):
        if not rec.depends_on_evaluator_judgement:
            memory.mark_reconfirmed(rec, epoch); continue        # 是事實不是品味，保留
        if new_champion_reconfirms(rec):                         # 新 champion 仍認定為失敗?
            memory.mark_reconfirmed(rec, epoch)                  #   → 蓋 reconfirmed_epoch
        else:
            memory.soft_delete(rec)                              #   → active=False（可稽核、可回溯）
    # 硬回收延後到獨立 janitor：purge_soft_deleted() / memory.purge_epoch()

def new_champion_reconfirms(rec):
    # 1) 有 reconfirm_flaws tag → 新 rubric 是否有能抓它的 criterion（deterministic）
    # 2) 否則用新 champion 重判記憶文字，deficit ≥ 0.5 才算仍失敗
    ...
```

**設計精髓**：`physics_truth`（物理事實，如「閥開度與流量的因果」）**永不**因 evaluator 換代而抹除——因為物理不進化（[`blueprint.md`](./blueprint.md) 鐵律 I），且它們由 gatekeeper 灌入（§8.1、[`backend/gatekeeper/physics_memory.py`](../backend/gatekeeper/physics_memory.py)）。只有 `heuristic_failure` 中確實依賴舊評分者的部分才重驗；**軟刪先行、硬 purge 延後**（把「決策」與「回收」解耦）。這讓系統「換裁判但不失憶」。

---

## 7. 為什麼 Evaluator 必須在進化 Harness 之外（防 Reward Hacking）

這是整個進化設計的**安全鎖**，值得單獨講清楚。

**危險**：若讓「產生解法的 harness」與「評判解法的 evaluator」在**同一個迴圈內共同進化**，系統會迅速學會一件事——不是把問題解好，而是**生成專門討好當前 evaluator 的解法**（reward hacking）。更糟的是 evaluator 也可能退化去迎合 harness，兩者共謀出一個自我感覺良好、實則無能的閉環（Goodhart's Law 的極端形式）。

**AgentForge 的三道結構性防線**：

```mermaid
flowchart TD
    subgraph SOLVE["解法進化 (harness loop)"]
        S1["Optimizer / candidate generator"] --> S2["population search"]
    end
    subgraph JUDGE["評審進化 (evaluator loop, 分離)"]
        J1["GEPA reflect on rubric"] --> J2["challenger rubric"]
    end
    EXT[["外部 held-out labeled anchor (val)<br/>code gate + HITL veto"]]
    S2 -->|"被評, 但不能改評審"| FROZEN["frozen evaluator (this epoch)"]
    J2 -->|"必須統計勝過"| EXT
    EXT -->|"gating"| FROZEN
```

1. **兩個迴圈物理分離**：解法在 harness loop 進化；評審在**獨立的** evaluator loop 進化。harness 只能被 frozen evaluator 評，**無權修改評審**。
2. **Epoch freeze**：在一個 epoch 內評審完全凍結，harness 無法透過「即時觀察評分變化」來反推並攻擊評審。
3. **外部 code gate 是最終裁判**：evaluator 的升級**不由 proposer 內部指標決定**，而由一道 **code 統計門檻**判定——它只在 **held-out `val` anchor** 上評分（GEPA proposer 只驅動於 `train`、在 `dev` 上選 frontier，永遠碰不到 gate 的 `val` 與報告的 `test`，§6.1 四路 data isolation），HITL 再加一層否決權（§6.2）。系統無法 hack 一個它訓練 / 選擇時都碰不到的裁判。

再加上 GEPA 的 Pareto frontier 維持 rubric 多樣性（§5），避免 evaluator 退化成只會抓單一種錯的偏執狂。**一句話**：讓解法進化、讓評審進化，但**永遠不讓解法決定評審**，且**評審的升級權握在系統之外的人手上**。

---

## 8. Self-Healing Qdrant Memory Design

evaluator 的長期知識存在 **Qdrant**（向量資料庫）。這份記憶不只是「檢索增強」，它是進化的**基因庫**——記著所有失敗教訓與物理事實，並在 epoch 換代時自我修復。

### 8.1 Payload Schema（記憶的型別即架構）

```json
{
  "id": "uuid",
  "vector": [/* embedding of the failure/truth description */],
  "payload": {
    "memory_type": "heuristic_failure",   // 或 "physics_truth"
    "created_at_epoch": 7,
    "text": "調高 setpoint 容忍度以壓下 broken-valve 溫度警報",
    "active": true,                       // soft-delete 用（active=false = 已軟刪）
    "depends_on_evaluator_judgement": true,   // selective erasure 用
    "reconfirm_flaws": ["numerical_ducttape"],// 選配：新 champion 據此重驗
    "reconfirmed_epoch": 8,               // 被新 champion 重新確認的 epoch
    "source": "gatekeeper"               // physics_truth 由 gatekeeper 灌入
  }
}
```

> 欄位以 shipped code（[`backend/memory/qdrant_store.py`](../backend/memory/qdrant_store.py)）為準：核心是 `text` / `memory_type` / `created_at_epoch` / `active`，其餘為 `extra` 任意欄位。`active=false` 即軟刪旗標（非 `soft_deleted`）。

**兩種 `memory_type` 是整個記憶治理的核心分野**（直接對應 [`blueprint.md`](./blueprint.md) 鐵律 I/II）：

| `memory_type` | 內容 | 換代時命運 | 類比 |
|---------------|------|------------|------|
| `physics_truth` | 物理因果 / 硬約束（不進化） | **永久保留** | Gatekeeper 的世界觀 |
| `heuristic_failure` | 主觀啟發式的失敗判定（會進化） | epoch 邊界**選擇性重驗** | Evaluator 的品味記憶 |

### 8.2 Hybrid Search = Semantic + Active Filters

evaluator 每次判斷前，用 **hybrid search** 從 Qdrant 撈回相關記憶注入 context。**這條路徑現在真的接上了**（先前 `EvolutionaryMemory.search()` 是 dead code）：[`backend/evaluator/judge.py`](../backend/evaluator/judge.py) `retrieve_memory_block()` 呼叫 `.search()`，`build_rubric_prompt` 把結果放進 `=== MEMORY (injected: hybrid_search) ===` 區塊（對應 §2 XML 的 `<memory injected="hybrid_search">`）。

```python
def search(self, query, top_k, memory_type, active_only=True, max_epoch=None):
    return self.client.query_points(                  # 1) 語意相似度 (dense vector)
        collection_name=self.collection,
        query=embed(query),                           #    deterministic hashing BoW（見下方誠實邊界）
        query_filter=Filter(must=[                    # 2) active metadata filters
            FieldCondition(key="memory_type", match=memory_type),
            FieldCondition(key="active", match=True),               # 未被軟刪
            FieldCondition(key="created_at_epoch", range=Range(lte=max_epoch)),  # 不撈未來 epoch
        ]),
        limit=top_k,
    )
```

- **Semantic**：dense embedding 找「語意上相似的過往失敗」——即使用詞不同也能召回。judge 撈 `heuristic_failure`（≤ 當前 epoch）＋ 永久 `physics_truth`。
- **Active filters**：硬性條件過濾（型別對、`active=True` 未被軟刪、epoch 合法）。兩者結合，既有召回廣度，又有稽核精度。
- **offline 誠實邊界**：`embed()` **預設**是一個 **deterministic hashing bag-of-words**（numpy，無 torch / sentence-transformers），只給 PoC 級的字面語意相似度，讓 CI 全離線可重現。真 embedder 的 seam **已接上**（[`get_embedder()`](../backend/memory/qdrant_store.py)）：`AGENTFORGE_EMBEDDER=sentence-transformers`（CPU 選配 dep）或 `=lemonade`（服務端 `/v1/embeddings`）在**可用時**才啟用，任何不可用（未安裝 / 無 server / mock）都自動回退 hashing——store API 不變。無 Qdrant server 時自動降級為 in-memory local mode。

### 8.3 Soft-Delete / Purge on Epoch Transition

epoch 換代時執行 §6.4 的 selective erasure，但用 **soft-delete**（`active = false`）而非硬刪：

```mermaid
flowchart LR
    Up["epoch N → N+1 升級"] --> Scan["掃描 created_at_epoch = N<br/>且 depends_on_evaluator_judgement"]
    Scan --> Recheck{"new evaluator<br/>仍認定為失敗?"}
    Recheck -->|"yes"| Keep["保留 + 標記 reconfirmed_epoch=N+1"]
    Recheck -->|"no"| Soft["soft_delete = true<br/>(可稽核, 可回溯)"]
    Purge["定期 purge job"] -.->|"超過保留期的 soft-deleted"| Hard["實體回收 VRAM/disk"]
```

- **soft-delete 先行**：被抹除的記錄不立即消失，保留可稽核軌跡（誰在哪個 epoch、為何被判失效），也可在誤判時回溯。
- **延後 purge**：實體回收由獨立的排程 job 執行，把「決策」與「回收」解耦，避免換代瞬間的抖動。
- **self-healing**：整個機制讓記憶在「換裁判」後自動修復——過時的主觀判定被清理，物理事實與仍有效的教訓被保留，記憶品質隨 epoch 單調提升而非歸零重來。

---

## 9. 本模組小結

- Evaluator 用 **Deficit Scoring**（float `[0,1]`、bounded per-criterion、只找 Red Flag）對抗諂媚與「被症狀騙」；critical 一票否決是 live-judge 設計，offline mock 採 bounded 加總（誠實近似）。
- **XML reference schema** 承載嚴格 rubric + 強制 CoT；shipped 的 champion-0 seed 是精簡的 **JSON-output** 變體，domain criteria / poison pill 於 judge 時 merge（poison pill 只在 strict 模式）。
- 三大 failure mode（Physics Common Sense、Diagnostic Resilience、Implementation Drift）各以 poison pill（broken valve、correlated noise、tight coupling）驗證。
- **GEPA reflective evolution 已實作**：Pareto frontier（`sep::<criterion>` + `parsimony` + `adversarial` 多目標、top-K 非 dominated、落地 `data/frontier/`），**父代以 Thompson 取樣**；`epoch/propose` 在 **`dev` 選擇split** 上以 BBε 下界選 frontier best 送 gate。（GEPA ~35× rollout 為引用其論文，非 offline 實測。）
- **兩段式晉升門檻**：**CODE gate 先行**（val 上 P1 非劣 + **P2 Bayesian Beta-Binomial 後驗 `P(Δsep>0)≥0.95` 且 `Δsep≥MDE`**，平手偏袒現任，回報 per-flaw 勝負 breakdown；**選配 P3 跨 epoch anytime-valid e-process** 控 family-wise error，預設關閉、`GateConfig` 可開），**HITL 只能否決不能覆寫**；strict/loose **hack-ratio** 接官方 `rqgm` 套件，`EXPLOITATION_DETECTED` 自動收緊 tolerance（`RQGM_BACKEND` = `rqgm` / `local-fallback`）。報告另有 **over-acceptance**、**over-optimization gap**、**provenance**。
- **四路 Data isolation**：73 個 labeled anchor（2 domain pack：smart_manufacturing + grid_energy）拆 `train`(28)/`dev`(12)/`val`(19)/`test`(14)——GEPA 只讀 train、frontier 選在 dev、gate 只看 val、test 供報告。
- **Judge panel + self-consistency**（多 persona/溫度、median + 多數決、criterion 順序隨機化）；校準指標改用 **accuracy / Cohen's κ**（gate 仍用 separation 統計下界）。
- 升級後 **selective erasure = soft-delete + reconfirm**：`heuristic_failure` 由新 champion 重驗、軟刪不再確認者、延後硬 purge；`physics_truth` 永久保留（由 gatekeeper 灌入）。
- Evaluator **恆在進化 harness 之外** + epoch freeze + **外部 labeled anchor 的 code gate**，從結構上封死 reward hacking。
- **Self-healing Qdrant memory**：`memory_type` × `created_at_epoch` 治理 + hybrid search（**已接上 judge**，embedding 為 offline hashing BoW 近似）+ soft-delete/purge。
- **offline 誠實邊界**：judge 為 deterministic rubric-aware **mock**（`mock_scoring.py`），真 live 評分需本地模型（Lemonade/vLLM）；frontier 父代選擇的 **Thompson 取樣已實作**，但 **MCTS 與 multi-agent debate 仍為 optional/未實作**。

*下一步：[`04-stack-export.md`](./04-stack-export.md) — 這一切跑在哪些開源元件上、怎麼部署、怎麼算給 C-Level 聽。*

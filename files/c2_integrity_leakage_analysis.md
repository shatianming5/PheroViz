# C2 proposer⊥judge：泄漏边界与可接受方案

## 结论先行

1. **C2 headline 应只用程序化 `data_fidelity` / `series_cohesion`。** 在 sealed
   manifest 固定后，这两个分数由代码和 source-data/expectation 计算，评分时不调用
   Claude 或 Gemini。因此不存在“同一 VLM 既给 C2 分数又定义 C2 分数”的**评分调用
   泄漏**。
2. 这不等于 curation 标签完全独立：`chart_family`、`x/y` 和
   `evaluation_expectation` 最初由 `claude-sonnet-4.6` +
   `gemini-3.5-flash` 的双模型审核确认。它是一个必须披露的**标签/选样依赖**，但对
   所有比较方法固定且不会在评分时按方法重新生成，故不是 differential scoring
   leakage。
3. 若把 C5 的 Claude/Gemini visual-form rejudge 当作 C2 结论或排名证据，则存在
   潜在的 judge-reuse 风险。即使 curation 是“是否有效”、C5 是“视觉质量”，也不能以
   “任务不同”证明模型独立。现有 C5 的两个模型只能作支持性、非 headline 证据。
4. 最稳妥的发布策略是：**C2 headline = 程序化指标；C5 不进入 C2 的 Holm family。**
   若必须报告 C2 visual-form，先冻结一个未参与 curation 的第三方独立 VLM（或盲态
   人类评审）及其 prompt、served identity、版本和 hash，再以它作为独立 rejudge。

## 已核对的信任链

`build-cases --evidence` 才会把 evidence 中的 verification 重新绑定到 candidate：
`nature_download/corpus/cases.py` 将其写为 `curation_status="verified"`、
`eligible_for_experiment=true` 和 `verification_evidence`。没有 evidence 时，候选明确为
`curation-not-verified`。`assemble-benchmark` 随后重新验证 candidate、proposal、
review、evidence、summary、source-data hash，并只从全部已审核 source singles 重建
canonical multi-panel case（`nature_download/corpus/benchmark.py`）。

若 multi 来自 `derive-multi-proposals`，该 derived batch（含其 source singles）必须再次
`review-proposals`；用**新的** evidence 重跑 `build-cases`。不能同时传入旧 single review
bundle 和含相同 candidate IDs 的 derived bundle：assembler 会把跨 bundle 重复 ID 拒绝，
这正是防止旧证据被偷渡到新 multi 的保护。

运行端也会重复检查 source binding：`agent/experiments/manifest.py` 验证 evidence、
reviews、proposed、candidate summaries、数据 hash、`code_dirty=false` 与每个 case 的
verified binding。`run_c2_v3.py --dataset-manifest` 只有通过这条路径才可 materialize
matrix。故不能把 proposed JSONL 改字串、也不能把 unverified candidate 的
`curation_status` 手改成 verified；两者均不是 sealed benchmark。

## 分情形判断

| 情形 | 是否为评分泄漏 | 可作 headline 吗 | 原因 / 约束 |
|---|---:|---:|---|
| (a) C2 程序化 `data_fidelity`、`series_cohesion` | 否（条件于已冻结 expectation） | 可以 | scorer 不请求 Claude/Gemini；输入为已 sealed 的 render、source data 和 expectation。必须在执行前冻结 manifest，不能根据方法输出修订 expectation。 |
| Curation 的双模型确认 chart family/x/y | 不是 scorer reuse；是标签依赖 | 可以，但须披露 | 可能影响哪些题进入集合或 target 怎样定义；它不会因被比较方法不同而改变。应报告 review models、review hash、接受/拒绝数和 selection rule。 |
| (b) C5 用同一 Claude/Gemini 给 visual-form 分 | 潜在泄漏 / shared-model bias | 不可作为独立 C2 headline | 同一模型族既参与 curation 又对 render 排名。任务和输入不同可降低直接污染，但不能证明 judge independence。 |
| C5 第三方、未参与 curation 的冻结 VLM | 可避免同一 judge reuse | 仅在预注册后可作支持性 endpoint | 仍应与程序化 correctness 分开报告；visual quality 不可替代 data correctness。 |

## 条件独立性的精确定义

记 `E` 为在执行前由 review evidence 固定并哈希绑定的 expectation，`R_m,d` 为方法 `m`
在 DOI `d` 的 render，`S_m,d` 为程序化分数。方案 A 的评分路径是：

```text
curation judges ──> E (sealed) ──> deterministic scorer(E, source_data, R_m,d) = S_m,d
method m ────────────────────────────────> R_m,d
```

在固定 `E`、source data 和 render 后，`S_m,d` 是无模型 API 调用的确定性函数；curation
judge 的 request/served identity、文字 rationale 和 score 都不是该函数的输入。因此，
**条件于 sealed benchmark 的 scorer independence 成立**。与此同时，`E` 的来源仍包含
curation judge，故不能把它写成“ground truth 与所有模型完全独立”。正确措辞是：
“双模型 curation 定义并冻结任务语义；方法间差异由不调用该 curation judge 的程序化
scorer 计算。”

## Reclassified wide-melt 的附加完整性门

当前 `simple-2d-v4` reclassifier 的**标签构造**仅经过
`read_candidate_table → analyze_table → propose_single_candidate`：源文件先以 candidate 中的
SHA-256 校验，再由表头、dtype、非空值、基数、单调性及重复值结构决定
`chart_family`、`x/y`、wide-melt source columns 和 expectation。历史 `reviews.jsonl` 不在这条
调用链中。对两个 batch 的对抗性测试将输入 proposal 中所有旧的 `experiment_case`、`x/y`、
chart family、expectation、proposal analysis 和 curation fields 替换为伪造 judge 决定后，
200 和 116 个重建 records 均逐字节相同；详见
`files/c2_reclassify_integrity/reclassifier_leakage_audit.json`。

但这不允许把所有 reclassify 产物都当作无条件可封存：

* `reproposed_strict_rejects*.jsonl` 的成员由历史 reject status 选择，故它只能用于诊断性
  triage，**绝不能**定义最终 C2 universe；
* 当前审计到的 source 已声明 `simple-2d-v4`，但已发布 report/artifacts 仍声明
  `simple-2d-v3`，且 A2 输入 hash 不匹配当前 priority artifact。必须 freeze V4、重写
  per-batch **full** review input、重新 review/evidence/build-cases；
* 两个 strict batch 有重叠 candidate IDs，不能作为两个 bundle 一起封存。最终 pool 必须选一个
  non-overlapping source universe，或在 review 前做确定性去重并重新绑定 evidence。

Wide melt 不改变评分的 ground truth，而是明确 ground-truth representation：raw hash-bound
宽表在内存中按 sealed `source_value_columns` 变换为
`__wide_group__` / `__wide_value__`；raw 文件不被修改。运行端在 profile、render 和
`evaluate_figure` 前使用同一 fail-closed transform，并在 artifacts 中记录 `data_binding`。
因此 expectation 对虚拟列的 programmatic fidelity/series-cohesion 是对这个固定变换后的
source data 打分，而不是对 judge 文本或 prior label 打分。虚拟 series GID 现在也被 manifest
extractor 保留，避免 multi-panel palette cohesion 因 `__` 前缀被错误丢弃。

## “validation ≠ scoring”论证的准确边界

它有事实基础但不能被夸大：

* curation review 的输入是 proposal、source data、figure/caption，并回答 valid、
  chart family、x/y；
* C5 的固定 rubric 只看一个 sealed image，明令不推断 data fidelity/provenance；
  `agent/experiments/rejudge.py` 的 input manifest 也声明 request fields 仅为
  `image + fixed_visual_form_prompt`，排除 `source_data`、`method`、prior scores、
  generation/editing feedback；
* 因而没有把 reviewer 的文字输出直接拼入 visual rejudge prompt，也没有把 method
  名交给 VLM。

这说明两次调用的**任务、输入和 rubric 分离**，可作为“没有直接 prompt 泄漏”的证据；
但它不能推出“同一模型没有共享偏好或校准偏差”。因此不能把这一点包装成
proposer⊥judge 的充分证明。

## C2 发布时的角色隔离规则

| 角色 | 当前/建议身份 | 可见信息 | 不可见信息 |
|---|---|---|---|
| curation verifier | Claude 4.6 + Gemini 3.5 Flash | source assets、proposal、固定 review rubric | C2 run output、方法胜负、后验统计 |
| C2 proposer / generator | `gpt-5.6-sol`（当前 harness 默认） | sealed task、source table、固定 budget | review 原文、accept/reject rationale、其他方法的 render/score |
| C2 headline scorer | deterministic programmatic evaluator | render、source data、sealed expectation | model API、reviewer identity/output、方法名称 |
| optional visual rejudge | **第三独立模型**或盲态人类 | 仅 render + 预注册 visual rubric | source data、method 名、prior score、curation text |

若未来将 Claude 或 Gemini 本身作为 C2 generator/backbone，也应把该方法视为与 curation
judge 不独立：要么不以该比较作 confirmatory claim，要么使用未参与 curation 的
independent evaluator，并把这一例外预注册。

## 具体 integrity-safe 执行方案

### 方案 A（推荐，当前即可执行）

1. A1 完成 review 后，以原始 evidence 重跑 `build-cases --evidence`，再运行
   `files/build_c2_benchmark.py` 的 canonical `assemble-benchmark` 路径。
2. 执行前冻结 `benchmark_manifest.json`、`summary.json`、manifest hash、P5+ DOI 列表、
   comparison family 和 primary endpoint。
3. C2 的统计/ Holm family 只纳入程序化 `data_fidelity` 与 `series_cohesion`；C5
   visual-form 不合并进 C2 headline。
4. 在 blind-to-outcome 的情况下报告 curation model identity 和局限：它们参与定义
   curated ground truth，但不在 scoring path。

### 方案 B（若必须有 C2 visual endpoint）

在看任何 C2 方法结果前，新增独立 evaluator protocol：

1. 选定不参与 curation 的第三模型（不同 provider/model family，尽量不同训练/服务
   路线）或预注册盲态 human panel；
2. 固定 served identity、endpoint class、prompt/rubric、image-only request policy、
   retry policy、版本/commit/hash；
3. 在请求中排除 method、case ID、source data、curation review、prior score；
4. 与程序化 endpoint 分开分析、单独做 multiplicity correction；不把 visual 分数解释成
   correctness；
5. 在 C5 的现有 CLI 仅允许 Claude/Gemini 的限制解除并完成新 evaluator qualification
   前，不声称 visual evaluator 独立性。

## 审计检查表

- [ ] `benchmark_manifest.json` 的 provenance、candidate/review/evidence/data hash 全部通过
      `--dataset-manifest` 的加载验证。
- [ ] 每个入选 P5+ multi case 的 `curation_status=verified`、
      `eligible_for_experiment=true`，且 `verification_evidence` 与 evidence JSON 一致。
- [ ] C2 matrix 仅由 `run_c2_v3.py --dataset-manifest ... --case-kind multi --min-panels 5`
      materialize；未用 `--input`、legacy manifest、normalizer 或 bypass。
- [ ] generation 与 curation/rejudge 请求日志不交换 review rationale、方法名或先验分数。
- [ ] C2 headline 不使用 C5 Claude/Gemini visual-form；若例外，采用方案 B 并明确标为
      independent-supporting endpoint。
- [ ] 报告中将“评分调用无泄漏”和“标签完全独立”分开表述，后者不得声称成立。

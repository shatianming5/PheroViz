# A2: Proposer chart_family 误分类根因 + integrity-safe 改进 (主 agent 接手, 已 python 交叉验证)

## 1. 根因 (proposals.py:481-568 analyze_table)
proposer 的 chart_family 是**纯基于 source-table 列 dtype 的确定性规则**, 完全不看已发表 figure 图像:
- run-order 列 → scatter
- 单一 temporal 列 → **line** (x_mode=temporal)
- 单一 categorical(非数值)列 → bar (x_mode=categorical)
- 否则单一单调数值列 → **line** (x_mode=linear)
- 否则 reject (x-column-not-found / ambiguous)

判官(claude-sonnet-4.6+gemini-3.5-flash)看 **figure 图像** 判断真实图型。两者系统性错配:
source table 是**宽表**(列=实验条件/基因型/重复), 一个"单调数值列"往往不是真 x 轴。

## 2. chart_family 分布 (distribution.json, 已交叉验证)
strict/200 单面板: line 84 (56 linear + 28 temporal), bar 84 (categorical), scatter 0
c2_full/116 单面板: line 48 (34 linear + 14 temporal), bar 52, scatter 0
高危误判源 = **line/linear (单调数值 x) = 56+34 = 90 单面板**; temporal-line (42) 较安全。

## 3. integrity-safe 分类器原型 (独立于判官, 无泄漏) — 对 90 个 line/linear 实测
读每个 case 真实 x 列数值, 结构化判据(不调用任何 LLM/判官):
- **consecutive 小整数序号** (x∈{1..k}, k≤10, 起于0/1, gap_cv=0) → 独立重复/类别编号 → **应为 bar**
- 否则 → 保留 line 判定

实测结果 (90 例, 6 例 xlsx 读失败):
- **likely_bar: 19** — 如 `x='Mice'`(1-8), `x='Animal#'`(1-7): 独立动物重复被误判 line, 实为 bar。判官正是因此拒绝("line inappropriate for independent animal replicates")。这 19 例是**可 integrity-safe 修复**的明确 line→bar 翻转。
- **likely_line: 65** — 但抽检暴露**更深的 x/y 绑定错误**: `x='m2-/-'`(基因型表头)值却是 1616-1858(测量值), `x='w+;w1118'`值 0.68-1.0。宽表把"实验条件列名"当 x 轴。真 line 仅剂量响应类(`[RPA2] 0-50nM`, branchpoint distance mm)。

## 4. 诚实结论 (对主线关键)
- 误判**不只是 line↔bar 翻转**, 而是**宽表源数据无法干净映射到单一 (chart_family, x, y) 三元组**。
- 简单结构化修复(consecutive-index→bar)只能救回明确的重复-序号类(~19/90 line-linear)。
- 相当比例的拒绝是**深层绑定错误**(条件/基因型列被当 x), 无法靠 chart_family 翻转救回——需要重新做 sheet→(x,y) 绑定, 或从池中排除这些 panel。
- 推论: **可验证池可能显著小于 155 raw P5+ DOI**。真实 fixable/excluded 比例待 A1 判官输出(rejection_analysis.json)校准: 判官若对某 panel 一致给出有效 line/bar/scatter 替代=可修复; 若判官说 heatmap/image/box 或 x/y 无法确定=真排除。

## 5. integrity 红线
分类器**绝不用 claude/gemini 判官**给 proposer 定 chart_family (否则 proposer⊥judge 泄漏, 基准失效)。
本原型仅用 source-table 数值结构, 完全独立于判官。若要用 figure 图像提升, 须用第3组独立模型或传统 CV, 不得复用两个判官。

## 6. 待办 (A1 完成后)
用 A1 的 rejection_analysis.json 校准 likely_bar/likely_line 与判官真值的一致率; 
量化 (a)consecutive-index→bar 修复的真实回收率, (b)深层绑定错误的真排除比例。

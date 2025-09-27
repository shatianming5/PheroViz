# ESS-Pro Ultra 单圈闭环工作流

ESS-Pro Ultra 将意图 → 规格 → 插槽 → 沙箱 → Judge → FEEDBACK 串成单圈闭环。仓库内提供完整的 JSON Schema、插槽注册表、Ultra 模板、沙箱执行器、Judge++、FEEDBACK 生成器、单圈 Runner、few-shot 片库与示例 slots，可直接驱动 Zhizengzeng Responses API 或其他兼容接口。

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows；或 source .venv/bin/activate
pip install -r requirements.txt
python run_chain.py data/sales_demo.csv "季度对比" bar --rounds 2 --intent '{"x":"月份","y":"销量","group":"品类","aesthetics":{"palette":"ColorBlindSafe"}}'
```

运行完毕后，`runs/` 下会生成时间戳目录（spec、slots、代码、chart.png、judge.json、迭代日志等）。

## 直接注入插槽

```bash
python scripts/run_slots.py data/sales_demo.csv examples/slots_bar_scatter_rightlog_ybreaks.json
```

脚本会装配 Ultra 模板、执行沙箱并输出评估分数，方便验证单独的 slots JSON。

## 目录速览

- `contracts/spec_schema.json`：Spec JSON Schema。
- `app/services/spec_deriver.py` / `spec_validator.py`：意图派生与校验。
- `app/services/slot_registry.py`：插槽全集、执行 DAG、层级白名单。
- `app/runtime/scaffold_elements_pro.py.j2`：Ultra 模板（所有插槽位）。
- `app/services/sandbox_runner.py`：临时目录 + shim + Matplotlib/Agg 沙箱。
- `configs/judge_rules.yml` / `diagnostics_map.yml`：Judge++ 规则与诊断键映射。
- `app/services/judge.py`：图像粗评分 + 诊断收敛信号。
- `app/services/feedback_builder.py`：结构化 FEEDBACK 文本。
- `app/services/single_chain_runner.py`：单圈 orchestrator（L1→L4→沙箱→Judge→FEEDBACK）。
- `app/snippets/slots_library.json`：常用插槽片库。
- `examples/slots_bar_scatter_rightlog_ybreaks.json`：组合图 slots 示例。

## FEEDBACK → Prompt 链

单圈 Runner 每轮执行：
1. derive_spec → validate_spec 得到基础 Spec。
2. 依次请求 L1/L2/L3/L4（预留 `_llm_generate_slots` 接口对接 Zhizengzeng Responses API）。
3. assemble_with_slots → sandbox_runner.execute_script。
4. Judge++ 打分，compose_feedback 汇总诊断与层级守卫。
5. 分数达标 (VisualForm ≥ 0.75 且 DataFidelity ≥ 0.75) 即收敛，否则进入下一轮直至 round 上限。

层级守卫示例（内置）：
- L2：`allow=data.*; deny=ax/plt/text/legend/grid/theme`
- L3：`allow=marks.*,scales.*,colorbar.apply; deny=axes.*,legend.*,grid.*,annot.*,theme.*`
- L4：`allow=axes.*,legend.*,grid.*,annot.*,theme.*; deny=data.*,marks.*`

## CLI

- CLI：`python run_chain.py <excel_or_csv> <user_goal> <chart_family> [--rounds N] [--sheet NAME] [--intent JSON]`

## few-shot 片库

`app/snippets/slots_library.json` 覆盖 bar.grouped、scatter.main、heatmap+colorbar、axes/legend/grid/annot/scales 等 17 组函数体，均为“只包含函数体”的合法插槽代码，可直接塞入 Ultra 模板。

## 测试

```bash
pytest -q
```

包含两个基础冒烟测试：
- Spec 派生→校验→基本字段验证。
- 空插槽装配可成功生成 `run` 函数。

## 示例数据

`data/sales_demo.csv` 提供月度销量与转化率示例，可直接用于运行组合图 slots。

---

如需重新生成 Ultra 模板或扩展 slots，请遵守 `task.txt` 中的统一输出契约，并通过 `scripts/run_slots.py` / `pytest` 验证后再接入单圈 Runner。

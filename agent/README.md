# PheroViz Agent

PheroViz Agent 是一个由四个阶段（L1-L4）组成的 Slot Pipeline，可自动生成基于 Matplotlib 的可视化。每一轮流程都会推导基础规格、填充各阶段的 slot 函数、在沙箱内渲染图像，并让 Judge++ 根据视觉表现与数据忠实度给出评分，从而决定是否继续迭代。

## 项目结构

- `app/`
  - `services/`
    - `default_slots_v2.py`：默认 slot 集（v2），负责多 overlay 推断、数据准备、绘制与版式控制。
    - `single_chain_runner.py`：主 orchestrator，负责组织 prompt、调用沙箱、记录产物并驱动多轮迭代。
    - `sandbox_runner.py`：在临时目录落盘 scaffold/shim，调用 Matplotlib/Agg 渲染，并把更新后的上下文写回。
    - `code_assembler.py`：根据 slot 片段生成最终可执行的 scaffold 脚本。
    - `judge.py` / `feedback_builder.py`：Judge 评分与诊断、下一轮反馈生成。
  - `runtime/scaffold_elements_pro.py.j2`：Jinja 模板，提供 slot stub 与运行时代码支撑。
- `configs/`：Judge 规则与诊断映射配置。
- `data/`：示例数据集（如 `sales_demo.csv`、`channel_share_dual.csv`、`actual_target_plan.csv`）。
- `runs/`：每次运行生成的目录，包含 `iteration_*.json`、`figure_round_*.png`、`inputs.json` 等。
- `scripts/`：用于 slot 回放或批量实验的辅助脚本。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

> 请在 `.env` 中配置 `LLM_API_KEY`（或兼容的 `OPENAI_API_KEY` / `LLM_API_BASE` / `LLM_MODEL`），以便在默认 slot 之外需要调用 Zhizengzeng Responses API 时能够正常工作。

## 运行方式

```powershell
python run_chain.py <data_path> <user_goal> <chart_family> [--rounds N] [--sheet SHEET] [--intent JSON]
```

- `data_path`：CSV 或 Excel 文件路径；配合 `--sheet` 可指定 Excel 工作表。
- `user_goal`：业务/分析目标描述，会写入标题与上下文。
- `chart_family`：初始图形类型（如 `bar`、`line`、`area`、`scatter`）。
- `--intent`：JSON 字符串，声明 x / y / group 及其它意图；在 PowerShell 中推荐配合 `--%` 或单引号避免转义问题。
- `--rounds`：最大迭代次数；若 Judge 评分（`visual_form` 与 `data_fidelity`）均达到 0.75，将提前停止。

运行结束后，可在 `runs/<timestamp>/` 中查看：
- `figure_round_*.png`：各轮渲染出的图像。
- `iteration_*.json`：记录当轮 spec、接受的 slot、诊断及评分。
- `inputs.json`：本次任务的数据画像、意图与初始 spec 快照。

## Default Slots v2 摘要

默认 slot 覆盖 L1-L4 全链路：

- **L1（spec.compose / theme_defaults）**
  - 自动识别时间列、数值列、比例列，推断 x/y/group。
  - 根据 `chart_family` 和数据特征选择合适的 mark，并为多 overlay 打基础。
  - 通过关键词（`target`、`plan`、`baseline`、`trend` 等）识别目标、计划、基准、趋势字段，自动添加不同线型样式的参考层。
  - 比例与绝对值混合时会分配左右轴，右轴系列数量默认上限为 2，并在 `_v2_meta` 中保留角色信息。
- **L2（data.prepare / aggregate / encode）**
  - 统一做类型转换、类目排序，保留所有 overlay 所需的列。
  - `ratio_flag` 针对 `secondary_ratio`、`peer_ratio_metric` 等角色按均值聚合，并为后续百分比格式化提供依据。
- **L3（marks.* / scales.*）**
  - 默认实现面积、柱状、折线、散点等 mark，并维护调色板缓存、类别 x jitter、z-order 等细节。
  - 对比例轴的安全检查、对数轴保护等逻辑均在此层完成。
- **L4（axes.* / legend.apply / grid.apply / annot.*）**
  - 统一字体缩放、网格策略，legend 优先外置并区分左右轴标签。
  - 自动根据比例角色切换百分比刻度，必要时提示断轴信息。

所有阶段都会把推断结果写入 `ctx['_v2_meta']`，供后续 slot 共享（例如 legend 策略、聚合方式、字体缩放等）。

## 多 Overlay 示例

```powershell
# 面积图 + 右轴折线
python --% run_chain.py data/channel_share_dual.csv "渠道占比与收入对比" area --rounds 1 --intent "{"x":"month","y":"share","group":"channel"}"

# 散点图（按品类着色）
python --% run_chain.py data/product_scatter.csv "价格 vs 销量" scatter --rounds 1 --intent "{"x":"price","y":"units","group":"category"}"

# 实际 vs 目标 / 计划 / 基准线
python --% run_chain.py data/actual_target_plan.csv "实际与目标对比" line --rounds 1 --intent "{"x":"month","y":"actual"}"
```

查看 `runs/<timestamp>/iteration_1.json` 可确认多 overlay 已写入 spec，并与图像保持一致。

## 调试建议

- `debug_default_slots.py`：快速渲染当前默认 slot 组合，适用于排查语法或缩进问题。
- 若 Judge 诊断 `empty.plot`，优先检查 `data.prepare` / `data.aggregate` 是否过滤掉所有行。
- 每次运行的临时 scaffold 会拷贝到对应的 `runs/<timestamp>/` 目录中，便于分析生成代码。
- `run.txt` 记录了近期 CLI 命令，方便复现任务。

## 测试

```powershell
pytest -q
```

现有测试覆盖 spec 推导、验证器、Judge 合约等逻辑；图像层面仍建议通过示例运行进行回归。

## 后续工作建议

1. 为新增的 `target` / `forecast` / `baseline` 等角色补充单元测试或回归脚本，确保聚合与 legend 逻辑稳定。
2. 清理无用调试产物（如 `debug_*.png`），并视情况更新 `.gitignore`。
3. 若计划继续扩展 overlay 规则，可在 L1 中向 `_v2_meta` 写入更多标记，并在 L2/L3/L4 中消费这些元数据。

---

更多细节可直接阅读 `app/services/default_slots_v2.py` 及其 `_v2_meta` 注释，或查看近期 `runs/<timestamp>/iteration_*.json` 获取完整上下文。

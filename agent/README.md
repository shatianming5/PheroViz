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

> 请在被 Git 忽略的 `.env` 中配置模型网关。统一客户端支持
> `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`，也兼容
> `LLM_API_BASE` + `LLM_API_KEY`。模型名由 `LLM_MODEL` 指定，视觉
> judge 可单独使用 `VLM_MODEL`。密钥不得写入实验 spec 或提交到 Git。

## 运行方式

```powershell
python run_chain.py <data_path> <user_goal> <chart_family> \
  [--rounds N] [--sheet SHEET] [--intent JSON] \
  [--initial-generation defaults|model_spec|model] \
  [--memory-mode none|ephemeral|untyped|constraints|patches|full] \
  [--seed N] [--temperature T] \
  [--expectation expectation.json] [--metric-config metrics.json]
```

- `data_path`：CSV 或 Excel 文件路径；配合 `--sheet` 可指定 Excel 工作表。
- `user_goal`：业务/分析目标描述，会写入标题与上下文。
- `chart_family`：初始图形类型（如 `bar`、`line`、`area`、`scatter`）。
- `--intent`：JSON 字符串，声明 x / y / group 及其它意图；在 PowerShell 中推荐配合 `--%` 或单引号避免转义问题。
- `--rounds`：最大迭代次数；若 Judge 评分（`visual_form` 与 `data_fidelity`）均达到 0.75，将提前停止。
- `--initial-generation model_spec`：首轮由模型产生完整 L1 spec，L2--L4
  使用所有方法共享的确定性 renderer；这是推荐的公平实验模式。
  `model` 会让首轮 L1--L4 全部调用模型，用于严格的全生成诊断。
- `--memory-mode`：选择无记忆、轮间清空的 typed memory、untyped log、
  constraint-only、patch-only 或完整双层 memory。每种模式控制真实读写，
  不是结果标签。
- `--expectation`：启用基于 Matplotlib Figure/Axes/Artist 的程序化
  fidelity；有该输入时，主 `data_fidelity` 来自 source-table 对拍，而不是
  VLM。

运行结束后，可在 `runs/<timestamp>/` 中查看：
- `figure_round_*.png`：各轮渲染出的图像。
- `iteration_*.json`：记录当轮 spec、接受的 slot、诊断及评分。
- `inputs.json`：本次任务的数据画像、意图与初始 spec 快照。
- `programmatic_evaluation_round_*.json`：版本化 fidelity/cohesion
  计数、比例与定位到 panel/series 的 mismatch。
- `memory_snapshot*.json` / `memory_trace.json`：typed constraint/patch
  memory 与冲突、复用、清理事件。

## Multi-panel 运行

`run_multi_panel.py` 使用确定性 round-robin 调度，各 panel 独立渲染，
高作用域 constraint/patch memory 共享；最终写出组合图、共享 memory、
调度 trace 和跨 panel cohesion：

```bash
python run_multi_panel.py multi_panel_case.json \
  --initial-generation model --memory-mode full --rounds 3 --seed 0
```

manifest 至少包含 `panels`；正式评测还应提供
`evaluation_expectation`（schema 见
`app/evaluation/schemas/expectation.schema.json`）。每个 panel 必须给唯一
`id`、`data_path`、`user_goal` 和 `chart_family`。输出目录包含
`combined_figure.png`、`programmatic_evaluation.json`、
`shared_memory_snapshot.json` 与 `schedule_trace.json`。

每个完成的 global round 还会冻结到
`checkpoints/round_NNNN/`：当轮组合图、programmatic evaluation、共享
memory snapshot/trace、schedule trace 及各 panel 的源 artifact 都不会被
后续轮次覆盖。实验 provider 使用以下统一矩阵契约（组合图只是 archive
artifact，不额外计为一次 render）：

| schedule | provider 调用数 | 每次返回 | 每个 candidate 的 `render_count` | 总 render budget |
| --- | ---: | --- | ---: | ---: |
| `iterative` | 1 | 含 R 个 candidate 的 `ProviderBatch` | P | P × R |
| `best_of_n` | R | 1 个独立完整 candidate | P | P × R |

其中 P 是 panel 数，R 是完整 global round/candidate 数；例如 P=2 时，
R=1/2/3 分别要求 budget=2/4/6。render scheduler 采用 exact-fill：budget
必须能被 P 整除，partial panel round、少用或超用 budget、或用最终结果
冒充缺失的中间 checkpoint 都会失败。harness 只归档 provider 分配目录内的
artifact，并为每个 candidate 的组合图、指标、memory 与 schedule 记录
独立路径和 SHA-256。

## 可追溯实验

实验矩阵由 `python -m experiments run <matrix.yaml>` 执行。每个
case/method/backbone/seed/budget 都有独立 run 名和冻结的数据 manifest；
聚合器拒绝 dirty worktree、旧 run、test-only provider、缺 artifact hash
或 case 集不配对的比较。外部 ChartCoder、MatPlotAgent、nvAgent 只从固定
commit 的外部 checkout 运行，审计记录位于
`experiments/baseline_audits/`。

## Open-weight Transformers 文本服务

`experiments.transformers_server` 提供仅文本的 OpenAI-compatible
`/v1/chat/completions`，用于固定本地 checkpoint 的 open-weight 实验。
它不支持图片、音频、tools、URL 或任意文件输入。模型目录必须已经存在于
本机；服务以 `local_files_only=True`、`trust_remote_code=False` 加载。

```bash
export OPEN_MODEL_API_KEY='仅保存在环境变量中的随机密钥'
python -m experiments.transformers_server \
  --model-path /absolute/path/to/Qwen-checkpoint \
  --served-model-name Qwen/Qwen3.5 \
  --host 127.0.0.1 --port 8000 \
  --device cuda --dtype bfloat16 \
  --api-key-env OPEN_MODEL_API_KEY \
  --state-file runs/open_weight_server.json
```

默认只绑定 `127.0.0.1`；绑定 `0.0.0.0` 或其它非 loopback 地址必须额外传
`--allow-remote`。state JSON 权限为 `0600`，包含实际 PID、端口和一次性
shutdown token，但不包含 API key。实验结束后由控制进程读取 state 文件并
调用受 token 保护的 `/shutdown`：

```bash
python - <<'PY'
import http.client, json
from pathlib import Path

state = json.loads(Path("runs/open_weight_server.json").read_text())
conn = http.client.HTTPConnection(state["host"], state["port"], timeout=10)
conn.request(
    "POST",
    "/shutdown",
    headers={state["shutdown_header"]: state["shutdown_token"]},
)
response = conn.getresponse()
print(response.status, response.read().decode())
PY
```

随后等待 state 中的 PID 退出，并用 `nvidia-smi`（或对应设备工具）确认该
PID 已不再占用 GPU；进程退出前会删除模型引用并调用 CUDA cache cleanup。

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

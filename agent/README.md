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

MatPlotAgent 与 nvAgent 的单 panel case 若声明 `evaluation_expectation`，
会在隔离的 agent evaluator Python 环境中经静态安全过滤后重新执行已生成
代码，以原始 source table/sheet 调用同一 `evaluate_figure`，并归档
`programmatic_evaluation.json`、subprocess 证据和 nvAgent 可逆 alias 映射。
没有 expectation 的工程 manifest 仍只记录 `execution_success`；该流程不是
OS sandbox，危险/不支持代码、非唯一 Figure、越界 artifact 或评测错误均失败
关闭，且不声称 nvAgent 支持 multi-panel cohesion。
MatPlotAgent mainworkflow 正常返回但缺少 code/PNG 时，driver 会写入带一次性
invocation ID 的结构化 output-contract marker；只有该 marker 严格验签后才按
method failure 记零行。API/auth/import/dependency、畸形 marker、adapter safety
timeout，以及无法由 hash-bound static rejection 证明来源的 evaluator/config
错误仍为 blocking failure；分类从不依赖 stderr/stdout/model 文本匹配。

### C2 frozen-input runner

`run_c2_v3.sh` 是 C2 的安全入口，委托给 `run_c2_v3.py`。它先验证冻结文章
目录的 provenance，再依次执行 `build-cases`、`propose-cases`、matrix
materialization，以及可选的 agent/render/Judge++ execution。按 panel 数分片
matrix，因此每个 render budget 都是 `panel_count × rounds_per_case`，不会出现
partial global round。

`--input` 生成的 `proposed.jsonl` 永远是未审核的 legacy/exploratory input，
不能声明为 sealed benchmark。生产运行必须先用 corpus 的两模型 review 和
`assemble-benchmark` 生成 `benchmark_manifest.json`，再使用
`--dataset-manifest`：

```bash
# Build strict proposals only; the output directory must be new.
PYTHON=/path/to/python bash agent/run_c2_v3.sh \
  --input data/c2_p5_4k \
  --output nature_download/outputs/c2_proposal_materialization \
  --profile benchmark --rounds-per-case 1

# Run an already sealed manifest. MODEL_API_BASE and MODEL_API_KEY stay in env;
# --model becomes the per-run LLM_MODEL value.
export MODEL_API_BASE="https://model-gateway.example/v1"
export MODEL_API_KEY="..."
PYTHON=/path/to/python bash agent/run_c2_v3.sh \
  --dataset-manifest nature_download/outputs/verified_c2/benchmark_manifest.json \
  --manifest-data-root "$PWD" \
  --output nature_download/outputs/c2_benchmark_run \
  --profile benchmark --rounds-per-case 1 --model gpt-5.6-sol --execute
```

For a local no-key smoke test only, combine `--normalizer-exploratory`,
`--offline-defaults`, `--profile smoke`, `--rounds-per-case 1`, and `--execute`.
Those artifacts are explicitly `test_only` and are rejected by production
aggregation.

MatPlotAgent 与 nvAgent 的 paper-ready 单 panel 子轨分别由
`experiments/baseline_specs/matplotagent-single-test-renderable-v1.json` 和
`nvagent-single-test-renderable-v1.json` 声明。两者仅按公开接口和 parent case
metadata，从 SHA-256 为
`6c6c5cb50603d899a9be0f41c9e562517c92ae15477610e9111f6504cd1757b3`
的 sealed renderable benchmark 选择兼容 test singles：MatPlotAgent 固定 venv
没有 XLSX engine，故仅纳入 2 个 CSV cases；nvAgent adapter 会先把 16 个
CSV/XLSX cases 规范化为 CSV。每个子轨内的 PheroViz comparator 使用完全相同
case 集。tracked spec/matrix 不保存任何机器绝对路径；portable builder 在目标
机器上验证 parent/audit/evaluator/checkout/venv/dependency hash 后，按
repo-relative suffix 重映射 source/provenance 路径。materialized manifest
只新增各自的 `input_track` 和显式 runtime-root remap，并逐 case 绑定
parent/source/instruction/expectation hash。矩阵模板为
`experiments/matrices/baseline_matplotagent_single_renderable_v1.yaml` 与
`baseline_nvagent_single_renderable_v1.yaml`：external 与 PheroViz 均使用
`gpt-4o-mini`、render budget 1、seeds `0/1/2` 和相同 case/input track。
主指标为 programmatic `data_fidelity`，`execution_success` 由 exact-budget
完成状态报告；single-only 因而 cohesion `C=NA`。两项外部基线 license 均为
`not_declared`，三次 seed 是 paired replicate 标识，公开 adapter 不保证向
上游模型注入 seed。ChartCoder 仍因 checkpoint/license provenance 阻塞，不在
任何 method 或结果行中。fresh Linux venv 位于工作区的 `venvs/`，而 checkout 位于
PheroViz checkout 内的 `baseline_repos/`。冻结的 v1 portable spec 保留旧的
workspace-relative names；cluster 上以 `baseline_repos -> repo/baseline_repos` 和
`.baseline_envs/{matplotagent,nvagent} -> ../venvs/{matplotagent-linux,nvagent-linux}`
兼容，不复制环境。执行前还必须提供 gateway 环境变量并保持 PheroViz worktree clean。

C2 external multi-panel 比较使用 hash-pinned 的同一 P5+ frozen corpus。由于 harness
要求每个 render budget 能被 panel count 整除，21 个 case 被分为
`c2_baseline_{matplotagent,nvagent}_multipanel_p{5..11}_v1.yaml` 七个 shard；每个
external provider fan-out 到单 panel，再以同一 `evaluate_cohesion` 与
`combine_figure_manifests` 聚合。MatPlotAgent 使用 direct mode，nvAgent 显式使用
`openai_compatible: true`。不把 credential 写入 YAML；在 serialized execution 前将
`MATPLOTAGENT_API_KEY`/`MATPLOTAGENT_BASE_URL` 和
`NVAGENT_AZURE_OPENAI_API_KEY`/`NVAGENT_AZURE_OPENAI_ENDPOINT`/
`NVAGENT_OPENAI_API_VERSION` 从 gateway environment 映射到 adapter 所需的
OpenAI-compatible endpoint（`$ANTHROPIC_BASE_URL/v1`）。

只做无模型 materialize + dry-run：

```bash
cd /path/to/PheroViz
python agent/experiments/baseline_specs/build_portable_subtracks.py \
  --repo-root . --workspace-root .. --require-clean \
  --out agent/experiments/runs/preflight/external_baseline_subtracks
cd agent
python -m experiments run \
  experiments/runs/preflight/external_baseline_subtracks/matplotagent.matrix.yaml \
  --dry-run
python -m experiments run \
  experiments/runs/preflight/external_baseline_subtracks/nvagent.matrix.yaml \
  --dry-run
```

MatPlotAgent v2 是与已完成 v1 完全隔离的新子轨。它由
`experiments/baseline_specs/matplotagent-single-test-normalized-v2.json`
预声明，并使用 `build_matplotagent_v2.py` 将全部 16 个兼容 test singles 的
CSV 或指定 XLSX sheet fail-closed 地规范化为一 case 一份 canonical UTF-8
CSV。MatPlotAgent 与 PheroViz 共享 materialized manifest 中完全相同的 CSV
路径和 SHA-256；derivation 同时绑定 parent source hash/sheet、normalizer
代码/hash、normalized CSV hash、instruction 和 expectation，并保留 parent
renderability policy/audit seal。该子轨覆盖 6 个 DOI、3 个 paired seeds、
render budget 1、`gpt-4o-mini`，且仍为 P=1、`C=NA`、license
`not_declared`。v2 使用新的
`baseline_matplotagent_single_normalized_v2_gpt4omini_r1` root，绝不覆盖或
合并 v1。仅做无模型 materialize + dry-run：

```bash
cd /path/to/PheroViz
python agent/experiments/baseline_specs/build_matplotagent_v2.py \
  --repo-root . --workspace-root .. --require-clean \
  --out agent/experiments/runs/preflight/matplotagent_v2
cd agent
python -m experiments run \
  experiments/runs/preflight/matplotagent_v2/matplotagent_v2.matrix.yaml \
  --dry-run
```

C1/C3 的首个 production 矩阵位于
`experiments/matrices/c1_c3_final_benchmark_v2_seed0.yaml`。它只选择
`final_benchmark_renderable_v1_seed0` 的 test split，固定 `B_R=6`、backbone
`gpt-5.6-sol` 和 seeds `0/1/2`，并声明三个统一 provider 方法：

- `best_of_n`：`schedule=best_of_n`、`memory_mode=none`，进行
  `R=6/P` 次独立 provider 调用，每次返回一个完整 checkpoint candidate；
- `flat_iterative`：`schedule=iterative`、`memory_mode=none`，一次调用返回
  `R=6/P` 个 round checkpoint candidates；
- `pheroviz_full`：与 flat iterative 使用同一 checkpoint 轨迹契约，但
  `memory_mode=full`。

三者都从 `initial_generation=model_spec` 开始，并指向统一的
`experiments.providers:UnifiedBenchmarkProvider`。矩阵展开会在创建任何
run 目录前检查 render budget 是正整数且能被每个已选 case 的 panel 数整除；
因此 P=1/2/3/6 均可 exact-fill `B_R=6`，partial panel checkpoint 会直接
拒绝。每个方法还在 immutable spec 中固定
`render_timeout_seconds=120`；这是基础设施 watchdog，不是 render budget，
不得通过进程环境静默改变。只验证展开而不启动付费模型调用：

```bash
python -m experiments run \
  experiments/matrices/c1_c3_final_benchmark_v2_seed0.yaml --dry-run
```

C3 memory-mode matrix 的 tracked template 是
`experiments/matrices/c3_memory_modes_final_benchmark_v2_seed0_br6_gpt56sol.yaml`。
它固定 renderable derivative manifest SHA-256、P=2/3/6 三个 test cases、
六种 memory mode、seeds 0/1/2、`gpt-5.6-sol` 和 `B_R=6`，共 54 specs；
tracked 文件不含绝对路径。正式运行前，必须在 clean relocated checkout 中用
`experiments.c3_runtime_materializer` 生成 Git-ignored runtime matrix、spec
hash manifest 和 summary。materializer 会重映射 `manifest_data_root`、绑定
runtime commit/代码/manifest hashes，并拒绝未被 ignore policy 覆盖的 artifact
root。F=1、C=1、renders=6 是 outcome-independent prospective threshold，
可用于 confirmatory render-horizon attainment 与 restricted mean
renders-to-threshold。wall-clock tau 仍为 `NA_BLOCKED`；不得计算 wall-clock
attainment/RMST，也不得据此声称 latency 或 speed。

### C4 backbone tiers

C4 复用上述 frontier 矩阵，并新增
`c4_mid_final_benchmark_v2_seed0.yaml`（`gpt-4o-mini`）和
`c4_open_final_benchmark_v2_seed0.yaml`
（`Qwen/Qwen2.5-Coder-7B-Instruct`）。三个 tier 的 19 个 test cases、
方法、seeds、`B_R=6`、metric 和 dataset hash 完全一致，但必须在不同
进程/环境中运行，并写入各自被 Git 忽略的 artifact root。每个矩阵展开
171 个互异 spec，三者 run name 的并集为 513。

Mid tier 按 registry 固定 `temperature=0.2`。Open tier 的 registry 尚未
声明可复现的 temperature/seed 契约，因此矩阵不设置 temperature；正式启动
前必须验证服务返回的模型身份恰为上述名称及 revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`（remote model audit commit
`39adeb5`），并关闭任何不能证明的确定性声明。服务 endpoint、host 和
credentials 只由该 tier 的运行环境提供，不得写入矩阵。

后续 combined C4 summary 必须显式读取并验证三个独立 per-tier
`summary.json` 后再合并；不得扫描 `runs/production` 或其它 mixed root
来隐式聚合。

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

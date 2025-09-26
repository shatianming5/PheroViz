# PheroViz 单链可视化原型

本仓库实现论文中的 L1→L4 单链流程：Excel 画像 → 规划 → 模板出码 → 沙箱渲染 → Judge 与信息素记录。代码面向**论文复现**与实验，非工业 API 服务。

## 1. 准备环境

- Python 3.10+（当前在 Python 3.13 验证）
- 建议创建虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell / CMD
# source .venv/bin/activate     # Git Bash / WSL
```

## 2. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

依赖仅包含 Excel 解析、模板渲染、Matplotlib 绘图、Requests 等论文级组件。

## 3. 配置智增增 Zhizengzeng API

脚本直接对接智增增的 Responses API（OpenAI 兼容）。在根目录创建 `.env`，示例：

```
LLM_API_BASE=https://api.zhizengzeng.com/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4.1-mini
LLM_TIMEOUT=180               # 读超时时间，秒
LLM_MAX_TOKENS=700            # 限制模型输出 token，避免过长
LLM_FORCE_JSON=0              # 允许模型附带说明文本，内部会提取 JSON
```

如需强制 JSON（模型原生支持时更稳定），把 `LLM_FORCE_JSON` 设为 `1`。

## 4. 运行单链流程

命令行入口 `run_chain.py`：

```bash
python run_chain.py data.xlsx "分析各地区销售表现" bar --rounds 1
```

参数说明：

- `excel`：输入 Excel 路径
- `user_goal`：自然语言目标
- `chart_family`：图形家族（line / bar / stacked_bar / area / pie / scatter / hist / box / heatmap / auto）
- `--rounds`：链式迭代次数（默认为环境变量 `CHAIN_DEFAULT_ROUNDS`，gz 环境建议 1）
- `--sheet`：仅加载指定表，可重复
- `--storage`：结果输出目录（默认 `runs/`）

运行完成后，脚本会在 `runs/YYYYMMDDTHHMMSSxxxxxxZ/` 中写出：

- `inputs.json`：入参
- `profile.json`：Excel 画像
- `final_spec.json`：合并后的 L1-L4 规范
- `code.py`：生成的 Python 绘图代码
- `chart.png`：渲染图像（如需中文字体，建议安装黑体并在模板中设置）
- `judge.json`：VisualForm / DataFidelity 占位分
- `iterations.json`：每轮计划与反馈
- `pheromones.json`：信息素日志（类型、增量、时间戳）

## 5. 自测

```bash
python -m pytest -q
```

包含 Excel 画像、模板出码、沙箱安全的三项单测（警告关于 Matplotlib 图例与 pandas 日期解析，可忽略）。

## 6. 模块速览

- `app/services/excel_loader.py`：检测表头、推断列类型
- `app/services/data_profile.py`：生成列画像与白名单
- `app/services/prompts_chain.py`：L1/L2/L3/L4 Prompt 模板（强制 JSON 逻辑由 `LLM_FORCE_JSON` 控制）
- `app/services/chain_runner.py`：单链 Sense→Plan→Code/Patch→Render→Judge + 信息素写入
- `app/services/code_templates.py`：将规范转换为 Matplotlib 代码
- `app/services/sandbox.py`：AST 审计 + 安全执行绘图
- `app/services/judge.py`：占位 Judge 分
- `app/utils/audit.py`：落盘输入、输出、信息素

通过 `.env` 配置好智增增 API Key 后，直接执行 `run_chain.py` 即可复现论文中的单链流程，并得到图片、代码与信息素日志等产物。

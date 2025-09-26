# PheroViz 单链可视化原型

本仓库给出论文中“树先不实现，只做一条链（Chain）”的参考实现：读取 Excel → L1/L2/L3/L4 计划 → 模板化生成 Python 绘图代码 → 沙箱渲染 → 轻量 Judge，并把信息素增量落盘。实现目标是**验证论文流程**，而不是工业级 API 服务。

## 一、准备环境

- Python 3.10+（示例在 Python 3.13 上验证）
- 建议创建虚拟环境隔离依赖：

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell / CMD
# source .venv/bin/activate     # Git Bash / WSL
```

## 二、安装依赖

依赖仅保留 Excel 解析、LLM 调用与可视化所需库：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 三、配置智增增（Zhizengzeng）兼容 API

实现通过 `requests` 直接调用 OpenAI/智增增兼容的 Responses API。运行前请设置：

```
set LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4      # 举例：智增增开放平台
set LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx                 # 你的 API Key
set LLM_MODEL=gpt-4.1-mini                                  # 兼容模型名称
```

可选变量：

- `PHEROVIZ_STORAGE_ROOT`（默认 `runs/`）：结果与信息素日志保存目录
- `CHAIN_DEFAULT_ROUNDS` / `CHAIN_MAX_ROUNDS`：链式循环默认轮次与最大轮次

## 四、运行单链流程

使用命令行脚本 `run_chain.py`：

```bash
python run_chain.py data.xlsx "分析各地区销售表现" bar --rounds 3
```

常用参数：

- `excel`：输入 Excel 路径
- `user_goal`：自然语言目标
- `chart_family`：图形家族（line/bar/stacked_bar/area/pie/scatter/hist/box/heatmap/auto）
- `--rounds`：循环次数（可选）
- `--sheet`：仅加载指定工作表，可重复
- `--storage`：结果输出目录

脚本会生成一个时间戳目录，包含：

- `inputs.json`：运行入参
- `profile.json`：Excel 画像
- `final_spec.json`：合并后的 L1-L4 规范
- `code.py`：生成的 Python 绘图脚本
- `chart.png`：渲染图像
- `judge.json`：轻量三指标占位分
- `iterations.json`：每一轮的 plan/emit/feedback
- `pheromones.json`：信息素日志（类型、增量、时间戳）

## 五、快速自测

项目提供两组 PyTest 用例验证 Excel 解析、画像输出、代码模板与沙箱安全：

```bash
python -m pytest -q
```

日志中可能出现 matplotlib 图例与 pandas `infer_datetime_format` 的警告，不影响功能。

## 六、模块概览

- `app/services/excel_loader.py`：自动定位表头、推断列类型
- `app/services/data_profile.py`：生成列画像与白名单
- `app/services/prompts_chain.py`：L1/L2/L3/L4 Prompt 模板（强制 JSON）
- `app/services/chain_runner.py`：线性 Sense→Plan→Code/Patch→Render→Judge 循环与信息素写入
- `app/services/code_templates.py`：将规格转换为 Matplotlib 代码
- `app/services/sandbox.py`：AST 审计 + 安全执行绘图
- `app/services/judge.py`：轻量指标占位
- `app/utils/audit.py`：落盘输入、输出、信息素

遵循上述步骤即可复现论文中的单链流程，并可直接对接智增增 API 进行实验。

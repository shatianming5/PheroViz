# PheroViz 链式可视化服务

本项目实现 `task.txt` 中描述的单链可视化工作流：读取 Excel，交由 L1-L4 计划链生成结构化规范，模板化产出 Python 绘图代码，在沙箱中渲染出图，并记录信息素日志。

## 一、环境准备

- Python 3.10 及以上（当前在 Python 3.13 上验证）
- 建议使用虚拟环境隔离依赖：

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell / CMD
# source .venv/bin/activate     # Git Bash / WSL
```

## 二、安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 三、环境变量配置

链式规划调用兼容 OpenAI Responses API 的模型，需要在启动前设置：

```
set LLM_API_BASE=https://api.openai.com/v1
set LLM_API_KEY=sk-...
set LLM_MODEL=gpt-4.1-mini
```

可选参数：

- `PHEROVIZ_STORAGE_ROOT`（默认 `runs/`）：运行日志与产物存储目录
- `CHAIN_DEFAULT_ROUNDS` / `CHAIN_MAX_ROUNDS`：链式循环的默认轮数与上限

## 四、运行测试

```bash
python -m pytest -q
```

当前项目包含 3 个用例，全部应通过。日志中的 matplotlib 图例与 pandas `infer_datetime_format` 警告对功能无影响。

## 五、启动接口服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

接口列表：

- `GET  /health`：健康探针
- `POST /chain/profile_excel`：上传 Excel 获取画像 JSON
- `POST /chain/plan_emit_run`：执行 L1→L4 链路并返回最终规格、代码、PNG、评审与信息素日志

## 六、示例调用

```bash
curl -X POST http://localhost:8000/chain/plan_emit_run   -F "file=@sample.xlsx"   -F "user_goal=对各地区销售进行对比"   -F "chart_family=bar"   -F "rounds=3"
```

响应将包含：

- `final_spec`：合并后的 L1-L4 规格
- `code`：可执行的 Python 绘图脚本
- `png_base64`：渲染图像（Base64）
- `judge`：三项指标得分与诊断
- `iterations`：每轮的计划、执行与反馈
- `pheromones`：记录增量、类型与时间戳的信息素日志

所有产物会保存在 `PHEROVIZ_STORAGE_ROOT` 指定目录下的时间戳子目录，可用于离线回放。

## 七、流程速览

1. `/chain/profile_excel` 使用 `ExcelLoader` 和 `DataProfiler` 构建工作簿画像与列白名单。
2. `ChainRunner` 按 L1 Composition → L2 Orchestration → L3 Calibration → L4 Refinement 顺序调用大模型，并写入信息素。
3. `code_templates.render_code_from_spec` 将最终规格转换为安全的 Matplotlib 代码。
4. `sandbox.run_render_chart` 进行 AST 审计、限制内置并输出 PNG 图像。
5. `judge.simple_judge` 计算可读性与数据契合度占位分，若未达阈值则串联下一轮反馈。
6. `AuditLogger` 将输入、规格、代码、图片、评审与信息素以 JSON/PNG 形式落盘。

按上述步骤即可完成部署与调用。
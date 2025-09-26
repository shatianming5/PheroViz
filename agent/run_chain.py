from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.services import ChainRunner, DataProfiler, ExcelLoader, LLMClient
from app.services.code_templates import render_code_from_spec
from app.services.sandbox import run_render_chart
from app.utils.audit import AuditLogger


def load_env_file(path: Path = Path('.env')) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单链可视化规划并保存结果")
    parser.add_argument("excel", type=Path, help="输入的 Excel 文件路径")
    parser.add_argument("user_goal", help="业务目标/自然语言描述")
    parser.add_argument("chart_family", help="图形家族，例如 line/bar/auto")
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="链式迭代次数（默认 3 次，可被环境变量覆盖）",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path(os.getenv("PHEROVIZ_STORAGE_ROOT", "runs")),
        help="运行产物输出目录，默认 ./runs",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        default=None,
        help="只加载指定工作表，可重复使用",
    )
    return parser.parse_args()


def build_runner(rounds: Optional[int]) -> tuple[ChainRunner, int]:
    base = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    if not base or not key:
        raise RuntimeError("请先设置 LLM_API_BASE 与 LLM_API_KEY 环境变量（支持智增增兼容接口）")

    default_rounds = int(os.getenv("CHAIN_DEFAULT_ROUNDS", "3"))
    max_rounds = int(os.getenv("CHAIN_MAX_ROUNDS", "4"))
    requested_rounds = rounds or default_rounds
    capped_rounds = max(1, min(requested_rounds, max_rounds))

    llm = LLMClient(api_base=base, api_key=key, model=model)
    runner = ChainRunner(llm=llm, code_renderer=render_code_from_spec, sandbox_runner=run_render_chart)
    return runner, capped_rounds


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env_file()
    args = parse_args()
    excel_path: Path = args.excel
    if not excel_path.exists():
        raise FileNotFoundError(f"未找到 Excel 文件：{excel_path}")

    runner, rounds = build_runner(args.rounds)

    loader = ExcelLoader()
    data = excel_path.read_bytes()
    tables = loader.load_workbook(data, sheet_names=args.sheet)
    if not tables:
        raise RuntimeError("Excel 未包含任何可解析的数据表")

    profiler = DataProfiler()
    profile = profiler.build_profile(tables)
    profile_json = json.dumps(profile, ensure_ascii=False)

    results = runner.run_chain(
        profile_json=profile_json,
        user_goal=args.user_goal,
        chart_family=args.chart_family,
        tables=tables,
        rounds=rounds,
    )

    output = results.get("output") or {}

    storage_root = args.storage.resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(storage_root)
    run_dir = logger.persist(
        run_inputs={
            "excel": str(excel_path.resolve()),
            "user_goal": args.user_goal,
            "chart_family": args.chart_family,
            "rounds": rounds,
        },
        profile=profile,
        output=output,
        pheromones=results.get("pheromones"),
        iterations=results.get("iterations"),
    )

    judge = output.get("judge", {})
    form = judge.get("VisualForm")
    fidelity = judge.get("DataFidelity")
    print("运行完成！")
    print(f"输出目录：{run_dir}")
    if form is not None and fidelity is not None:
        print(f"得分：VisualForm={form} DataFidelity={fidelity}")
    else:
        print("未生成完整评估指标，请查看 iterations.json 了解详情。")


if __name__ == "__main__":
    main()

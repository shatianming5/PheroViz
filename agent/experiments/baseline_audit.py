from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .baseline_registry import (
    BASELINE_REGISTRY,
    BaselineDefinition,
    preflight_baseline,
)
from .models import utc_now, write_json_atomic


_AUDIT_DETAILS: Dict[str, Dict[str, Any]] = {
    "chartcoder": {
        "source_evidence": [
            "inference.py:12-56",
            "README.md:44-61",
            "README.md:85-86",
            "pyproject.toml:8-62",
        ],
        "actual_interface": {
            "kind": "Python class; no runnable CLI",
            "entrypoint": "inference.py:ChartCoder",
            "inputs": ["instruction", "reference chart image"],
            "outputs": ["model response containing plotting code"],
            "checkpoint": (
                "inference.py hard-codes /mnt/afs/chartcoder; adapter replaces only "
                "the loader argument from CHARTCODER_CHECKPOINT"
            ),
            "api_configuration": "local checkpoint; no hosted API",
        },
        "adapter_track": "image-to-code reconstruction only",
        "adapter_outputs": ["generated code", "executed PNG", "execution log"],
        "dependency_manifest": "requirements.txt and pyproject.toml",
        "limitations": [
            "Reference image and local checkpoint are mandatory.",
            "The upstream class assumes CUDA via .cuda().",
            "The upstream repository root declares no license file.",
        ],
    },
    "matplotagent": {
        "source_evidence": [
            "one_time_generate.py:14-73",
            "workflow.py:15-153",
            "agents/plot_agent/agent.py:12-197",
            "agents/openai_chatComplete.py:17-71",
            "agents/config/openai.py:1-3",
            "models/model_config.py:1-32",
        ],
        "actual_interface": {
            "kind": "CLI scripts plus importable mainworkflow functions",
            "entrypoints": {
                "direct": "one_time_generate.py:mainworkflow",
                "workflow": "workflow.py:mainworkflow",
            },
            "inputs": [
                "workspace containing table files",
                "simple instruction",
                "expert instruction for workflow",
                "model_type",
            ],
            "outputs": [
                "code_action_*.py",
                "PNG requested by the prompt",
                "workflow log",
            ],
            "api_configuration": (
                "agents/config/openai.py constants or OpenAI-compatible local "
                "servers from models/model_config.py"
            ),
        },
        "adapter_track": "table plus instruction",
        "adapter_outputs": ["generated code", "PNG", "subprocess logs"],
        "dependency_manifest": "requirements.txt",
        "limitations": [
            "Top-level CLIs hard-code the upstream benchmark data directory.",
            "Adapter invokes the real mainworkflow in an external work directory.",
            "The upstream repository root declares no license file.",
        ],
    },
    "nvagent": {
        "source_evidence": [
            "run_evaluate.py:8-43",
            "core/chat_manager.py:39-103",
            "core/agents.py:167-264",
            "core/agents.py:1053-1130",
            "core/llm.py:33-109",
            "core/api_config.py:1-17",
            "viseval/dataset.py:8-58",
        ],
        "actual_interface": {
            "kind": "Importable ChatManager; evaluation CLI has hard-coded paths",
            "entrypoint": "core.chat_manager:ChatManager",
            "inputs": [
                "natural-language query",
                "db_id",
                "list of CSV table paths",
                "matplotlib library selector",
            ],
            "multi_table": (
                "supported: Processor iterates table paths and Validator reads every "
                "CSV in databases/<db_id>"
            ),
            "outputs": ["generated Python code", "SVG via execute_to_svg", "API log"],
            "api_configuration": (
                "AzureOpenAI environment consumed by core/llm.py; upstream "
                "core/api_config.py contains placeholders"
            ),
        },
        "adapter_track": "table(s) plus natural-language instruction",
        "adapter_outputs": ["generated code", "SVG", "subprocess/API logs"],
        "dependency_manifest": "requirements.txt",
        "limitations": [
            "run_evaluate.py hard-codes dataset and webdriver paths.",
            "The adapter uses ChatManager directly and preserves its multi-table list.",
            "The upstream repository root declares no license file.",
        ],
    },
}


def _checkpoint_from_environment(
    definition: BaselineDefinition,
    environment: Mapping[str, str],
) -> Path | None:
    if definition.checkpoint_env is None:
        return None
    value = environment.get(definition.checkpoint_env)
    return Path(value).expanduser() if value else None


def audit_baseline(
    key: str,
    checkout: Path,
    *,
    python_executable: str,
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    definition = BASELINE_REGISTRY[key]
    report = preflight_baseline(
        definition,
        checkout,
        python_executable=python_executable,
        checkpoint=_checkpoint_from_environment(definition, environment),
        check_dependencies=True,
        environ=environment,
    )
    return {
        "audit_version": "1.0",
        "generated_at": utc_now(),
        "registry": definition.to_dict(),
        "preflight": report.to_dict(),
        "inventory": _AUDIT_DETAILS[key],
    }


def write_real_baseline_audits(
    checkout_root: Path,
    output_dir: Path,
    *,
    python_executable: str = "python",
    environment: Mapping[str, str] | None = None,
) -> list[Path]:
    environment = os.environ if environment is None else environment
    checkout_names = {
        "chartcoder": "ChartCoder",
        "matplotagent": "MatPlotAgent",
        "nvagent": "nvAgent",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    index_entries = []
    for key, checkout_name in checkout_names.items():
        payload = audit_baseline(
            key,
            checkout_root / checkout_name,
            python_executable=python_executable,
            environment=environment,
        )
        destination = output_dir / f"{key}.json"
        write_json_atomic(destination, payload)
        outputs.append(destination)
        index_entries.append(
            {
                "baseline": payload["registry"]["name"],
                "repo_url": payload["registry"]["repo_url"],
                "commit": payload["registry"]["commit"],
                "license_status": payload["registry"]["license_status"],
                "ready": payload["preflight"]["ready"],
                "audit_file": destination.name,
            }
        )
    index_path = output_dir / "index.json"
    write_json_atomic(
        index_path,
        {
            "audit_version": "1.0",
            "generated_at": utc_now(),
            "audits": index_entries,
        },
    )
    outputs.append(index_path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--python", default="python")
    args = parser.parse_args(argv)
    write_real_baseline_audits(
        args.checkout_root,
        args.output_dir,
        python_executable=args.python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.harness import execute_experiment
from experiments.matrix import expand_matrix
from experiments.models import sha256_path, write_json_atomic


COMMIT = "17990af8188b9994d88233ba226159d2b498a2d7"
CASE_ID = "case-s41467-023-42692-7-figure1-panelb-5ac7fb84108f"
PARENT = Path("/Users/tommy/Downloads/mayi/PheroViz")
ARTIFACT_ROOT = (
    PARENT
    / "agent/experiments/runs/calibration/"
    "blocker_null_category_17990af_seed0_br6"
)
MANIFEST = (
    PARENT
    / "nature_download/outputs/combined_verified_400/"
    "final_benchmark_v2_seed0/benchmark_manifest.json"
)
SUMMARY = (
    PARENT
    / "agent/experiments/preflight/"
    "blocker_calibration_17990af_seed0_br6_summary.json"
)
SPECS = (
    PARENT
    / "agent/experiments/preflight/"
    "blocker_calibration_17990af_seed0_br6_expanded_specs.json"
)
LOCK = ARTIFACT_ROOT / "_calibration.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metadata(
    run_dir: Path,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text: list[dict[str, Any]] = []
    vision: list[dict[str, Any]] = []
    for candidate in candidates:
        for label, relative in candidate["artifact_paths"].items():
            if label != "iteration" and not label.endswith(".iteration"):
                continue
            payload = json.loads((run_dir / relative).read_text(encoding="utf-8"))
            for stage in (payload.get("stages") or {}).values():
                item = (stage or {}).get("model_metadata")
                if isinstance(item, dict) and item:
                    text.append(dict(item))
            judge = payload.get("judge_model_metadata")
            if isinstance(judge, dict) and judge:
                vision.append(dict(judge))
    return text, vision


def usage(items: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in items:
        for key, value in (item.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value)
    return dict(sorted(totals.items()))


def main() -> int:
    if not os.getenv("MODEL_API_BASE") or not os.getenv("MODEL_API_KEY"):
        raise RuntimeError("Mapped model environment is missing")
    if os.getenv("VLM_MODEL") != "" or os.getenv("VLM_REQUIRED") != "":
        raise RuntimeError("VLM variables must be blank")

    repo = Path.cwd().parent.resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != COMMIT or dirty:
        raise RuntimeError("Calibration worktree is not the clean fixed commit")
    if ARTIFACT_ROOT.exists() or SUMMARY.exists() or SPECS.exists():
        raise RuntimeError("Fresh calibration paths are required")

    matrix_path = (
        repo / "agent/experiments/matrices/c1_c3_final_benchmark_v2_seed0.yaml"
    )
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix["experiment_name"] = "blocker-calibration-17990af-seed0-br6"
    matrix["dataset_manifest"] = str(MANIFEST)
    matrix["artifact_root"] = str(ARTIFACT_ROOT)
    matrix["case_ids"] = [CASE_ID]
    matrix["seeds"] = [0]
    matrix["budgets"] = [{"type": "renders", "value": 6}]
    specs = expand_matrix(matrix, base_dir=matrix_path.parent, repo_root=repo)
    if len(specs) != 3:
        raise RuntimeError("Calibration did not expand to three runs")
    if Counter(spec.method for spec in specs) != {
        "best_of_n": 1,
        "flat_iterative": 1,
        "pheroviz_full": 1,
    }:
        raise RuntimeError("Calibration method set changed")
    if {spec.git_commit for spec in specs} != {COMMIT}:
        raise RuntimeError("Calibration commit mismatch")
    if {spec.git_dirty for spec in specs} != {False}:
        raise RuntimeError("Calibration expanded dirty specs")
    order = {"best_of_n": 0, "flat_iterative": 1, "pheroviz_full": 2}
    specs.sort(key=lambda spec: order[spec.method])

    ARTIFACT_ROOT.mkdir(parents=True)
    descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}) + "\n")
    write_json_atomic(
        SPECS,
        {
            "schema_version": "1.0",
            "run_count": len(specs),
            "specs": [
                {**spec.to_dict(), "spec_hash": spec.spec_hash}
                for spec in specs
            ],
        },
    )
    report: dict[str, Any] = {
        "calibration_version": "1.0",
        "status": "running",
        "source_commit": COMMIT,
        "source_clean_at_expansion": True,
        "artifact_root": str(ARTIFACT_ROOT),
        "case_id": CASE_ID,
        "panel_count": 1,
        "methods": [spec.method for spec in specs],
        "seed": 0,
        "render_budget": 6,
        "backbone": "gpt-5.6-sol",
        "vlm_model": "",
        "vlm_required": "",
        "started_at": utc_now(),
        "finished_at": None,
        "expanded_specs": str(SPECS),
        "expanded_specs_sha256": sha256_path(SPECS),
        "runs": [],
    }
    write_json_atomic(SUMMARY, report)
    started = time.monotonic()
    try:
        for spec in specs:
            run_started = time.monotonic()
            outcome = execute_experiment(spec)
            record = outcome.record
            run_dir = ARTIFACT_ROOT / spec.run_name
            text, vision = metadata(run_dir, record.candidates)
            row = {
                "run_name": spec.run_name,
                "method": spec.method,
                "status": record.status,
                "error": record.error,
                "record_hash": record.record_hash,
                "run_record": str(run_dir / "run_record.json"),
                "run_record_file_sha256": sha256_path(run_dir / "run_record.json"),
                "provider_invocations": len(
                    {candidate["call_index"] for candidate in record.candidates}
                ),
                "text_model_calls": len(text),
                "vlm_calls": len(vision),
                "render_count": record.render_count,
                "candidate_count": len(record.candidates),
                "usage_totals": usage(text),
                "text_model_latency_seconds": sum(
                    float(item.get("latency_seconds") or 0.0) for item in text
                ),
                "wall_clock_seconds": record.wall_clock_seconds,
                "observed_seconds": time.monotonic() - run_started,
                "run_dir": str(run_dir),
            }
            report["runs"].append(row)
            report["elapsed_seconds"] = time.monotonic() - started
            if record.status != "completed" or len(vision) != 0:
                report["status"] = "failed"
                report["finished_at"] = utc_now()
                write_json_atomic(SUMMARY, report)
                return 1
            write_json_atomic(SUMMARY, report)
            print(
                json.dumps(
                    {
                        "run_name": spec.run_name,
                        "status": record.status,
                        "text_calls": len(text),
                        "renders": record.render_count,
                        "wall_clock_seconds": record.wall_clock_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        report["status"] = "passed"
        report["finished_at"] = utc_now()
        report["elapsed_seconds"] = time.monotonic() - started
        report["actual"] = {
            "completed_runs": len(report["runs"]),
            "provider_invocations": sum(
                row["provider_invocations"] for row in report["runs"]
            ),
            "text_model_calls": sum(
                row["text_model_calls"] for row in report["runs"]
            ),
            "vlm_calls": sum(row["vlm_calls"] for row in report["runs"]),
            "panel_renders": sum(
                row["render_count"] for row in report["runs"]
            ),
            "text_model_latency_seconds": sum(
                row["text_model_latency_seconds"] for row in report["runs"]
            ),
            "recorded_wall_clock_seconds": sum(
                row["wall_clock_seconds"] for row in report["runs"]
            ),
            "usage_totals": {
                key: sum(
                    row["usage_totals"].get(key, 0.0)
                    for row in report["runs"]
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        }
        write_json_atomic(SUMMARY, report)
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

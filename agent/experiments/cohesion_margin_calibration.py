from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.evaluation import evaluate_figure
from app.evaluation.schema import validate_expectation
from experiments.models import (
    canonical_json,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


CALIBRATION_SCHEMA_VERSION = "1.0"
REPORT_FILENAME = "cohesion_margin_report.json"
AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
DEFAULT_FIXTURE_BANK = (
    Path(__file__).resolve().parent
    / "calibration"
    / "cohesion_margin_fixture_bank_v1.json"
)

EXPECTED_FIXTURE_BANK_FILE_SHA256 = (
    "219c53b2bba85a2589a324247cae3d1c901f557ae8892df0059ff429298be334"
)
EXPECTED_BANK_SPEC_HASH = (
    "9c48098a2a838f7a29bfbc46b6c91dba530d7bb7e61e8c8f0c85b3e3d7b49182"
)
EXPECTED_SOURCE_SHA256 = (
    "fcea3980d2b630f7556ce1a09f235f52b5d1c57ad8b97c3f00ad4f380490bb17"
)
EXPECTED_EXPECTATION_SHA256 = (
    "fefa6b3402a41a4509e9a831ecb191214cfc8a668f3cf3722053e84a6eef2350"
)
EXPECTED_EVALUATOR_CONFIG_SHA256 = (
    "5a3039a6b238e5ca01ada69335ed7d1602b17a05223a884119355f66a752e555"
)
EXPECTED_FIXTURE_SPEC_HASHES = {
    "C0": "c44160a5726923a49bf06e4fa8a1c00ca2b8729b5be27bd0fae0f261c941df3b",
    "C1-scale": "eded69b94d3804a46ebca66bfdbf343b573d58d841897084d66418b5209b1d71",
    "C2-unit": "779bdc693eb16d424d803570653c583012b6d3b74918211ae68893e95d1b7f8c",
    "C3-legend": "474e3c038099e8e5e4dfa251ebb7a694f37c21aee16994a5e09483bf606dea79",
    "C4-palette": "4a8c604c0737f6c4cd6dc9e0a441c7959584f78964e6313685308315f44e1452",
}
EXPECTED_RUN_ORDER = [
    f"r{replicate}:{fixture_id}"
    for replicate in range(1, 4)
    for fixture_id in EXPECTED_FIXTURE_SPEC_HASHES
]
EXPECTED_LEAKAGE_DECLARATION = {
    "uses_human_ratings": False,
    "uses_method_outputs": False,
    "uses_model_outputs": False,
    "uses_network": False,
    "uses_randomness": False,
    "uses_test_inputs": False,
    "uses_test_metrics": False,
}
EXPECTED_CANDIDATE_RULE = {
    "class_aggregation": "median",
    "defect_classes": [
        "shared_scale",
        "shared_unit",
        "legend_deduplication",
        "palette_mapping",
    ],
    "expected_candidate_delta_C": 0.25,
    "final_aggregation": "minimum",
    "replicates": 3,
}
EXPECTED_RESULTS = {
    "C0": {
        "cohesion": [4, 4],
        "shared_scale": [1, 1],
        "shared_unit": [1, 1],
        "legend_deduplication": [1, 1],
        "palette_mapping": [1, 1],
        "ratio": 1.0,
        "mismatch_codes": [],
    },
    "C1-scale": {
        "cohesion": [3, 4],
        "shared_scale": [0, 1],
        "shared_unit": [1, 1],
        "legend_deduplication": [1, 1],
        "palette_mapping": [1, 1],
        "ratio": 0.75,
        "mismatch_codes": ["shared_scale_mismatch"],
    },
    "C2-unit": {
        "cohesion": [3, 4],
        "shared_scale": [1, 1],
        "shared_unit": [0, 1],
        "legend_deduplication": [1, 1],
        "palette_mapping": [1, 1],
        "ratio": 0.75,
        "mismatch_codes": ["shared_unit_mismatch"],
    },
    "C3-legend": {
        "cohesion": [3, 4],
        "shared_scale": [1, 1],
        "shared_unit": [1, 1],
        "legend_deduplication": [0, 1],
        "palette_mapping": [1, 1],
        "ratio": 0.75,
        "mismatch_codes": ["legend_not_deduplicated"],
    },
    "C4-palette": {
        "cohesion": [3, 4],
        "shared_scale": [1, 1],
        "shared_unit": [1, 1],
        "legend_deduplication": [1, 1],
        "palette_mapping": [0, 1],
        "ratio": 0.75,
        "mismatch_codes": ["palette_mapping_mismatch"],
    },
}
FIXTURE_CORE_KEYS = (
    "schema_version",
    "fixture_id",
    "defect",
    "panel_a",
    "panel_b",
    "legend_mode",
)
DEPENDENCY_PATHS = (
    "agent/app/evaluation/__init__.py",
    "agent/app/evaluation/evaluator.py",
    "agent/app/evaluation/manifest.py",
    "agent/app/evaluation/models.py",
    "agent/app/evaluation/schema.py",
    "agent/app/evaluation/schemas/expectation.schema.json",
    "agent/app/evaluation/schemas/figure_manifest.schema.json",
    "agent/app/evaluation/schemas/metric_config.schema.json",
    "agent/experiments/models.py",
)
FORBIDDEN_INPUT_MARKERS = (
    "best_of_n",
    "flat_iterative",
    "pheroviz_full",
    "final_benchmark",
    "run_record",
    "record_hash",
    "judge_score",
    "10.1038/",
)


class CalibrationError(RuntimeError):
    """Raised when the structural calibration cannot fail closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{name} must be an object")
    return dict(value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                CalibrationError(f"Non-finite JSON constant: {item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Cannot load fixture bank {path}: {exc}") from exc
    return _required_mapping(value, "fixture bank")


def _fixture_core(fixture: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in FIXTURE_CORE_KEYS if key not in fixture]
    if missing:
        raise CalibrationError(f"Fixture is missing core keys: {missing}")
    return {key: fixture[key] for key in FIXTURE_CORE_KEYS}


def _validate_fixture_bank(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved == DEFAULT_FIXTURE_BANK.resolve():
        actual_file_hash = sha256_file(resolved)
        if actual_file_hash != EXPECTED_FIXTURE_BANK_FILE_SHA256:
            raise CalibrationError(
                "Frozen fixture-bank file SHA-256 mismatch: "
                f"{actual_file_hash}"
            )
    bank = _load_json(resolved)
    if bank.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError("Unsupported fixture-bank schema")
    if bank.get("bank_id") != "cohesion-margin-v1":
        raise CalibrationError("Unexpected fixture-bank ID")

    source = bank.get("source_csv")
    if not isinstance(source, str):
        raise CalibrationError("source_csv must be a string")
    source_hash = _sha256_bytes(source.encode("utf-8"))
    if (
        source_hash != EXPECTED_SOURCE_SHA256
        or bank.get("source_sha256") != source_hash
    ):
        raise CalibrationError("Source CSV hash mismatch")

    expectation = _required_mapping(bank.get("expectation"), "expectation")
    validate_expectation(expectation)
    expectation_hash = sha256_json(expectation)
    if (
        expectation_hash != EXPECTED_EXPECTATION_SHA256
        or bank.get("expectation_sha256") != expectation_hash
    ):
        raise CalibrationError("Expectation hash mismatch")

    evaluator_config = _required_mapping(
        bank.get("evaluator_config"),
        "evaluator_config",
    )
    evaluator_hash = sha256_json(evaluator_config)
    if (
        evaluator_hash != EXPECTED_EVALUATOR_CONFIG_SHA256
        or bank.get("evaluator_config_sha256") != evaluator_hash
    ):
        raise CalibrationError("Evaluator-config hash mismatch")

    fixtures = bank.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 5:
        raise CalibrationError("Fixture bank must contain exactly five fixtures")
    seen: set[str] = set()
    fixture_hashes: dict[str, str] = {}
    for raw_fixture in fixtures:
        fixture = _required_mapping(raw_fixture, "fixture")
        fixture_id = fixture.get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id not in EXPECTED_RESULTS:
            raise CalibrationError(f"Unexpected fixture_id: {fixture_id!r}")
        if fixture_id in seen:
            raise CalibrationError(f"Duplicate fixture_id: {fixture_id}")
        seen.add(fixture_id)
        core_hash = sha256_json(_fixture_core(fixture))
        expected_hash = EXPECTED_FIXTURE_SPEC_HASHES[fixture_id]
        if core_hash != expected_hash or fixture.get("spec_sha256") != expected_hash:
            raise CalibrationError(f"Fixture-spec hash mismatch: {fixture_id}")
        if fixture.get("expected") != EXPECTED_RESULTS[fixture_id]:
            raise CalibrationError(f"Expected-count table changed: {fixture_id}")
        fixture_hashes[fixture_id] = core_hash
    if list(seen) and set(seen) != set(EXPECTED_FIXTURE_SPEC_HASHES):
        raise CalibrationError("Fixture IDs do not exactly match the frozen bank")

    if bank.get("candidate_rule") != EXPECTED_CANDIDATE_RULE:
        raise CalibrationError("Candidate rule changed")
    if bank.get("run_order") != EXPECTED_RUN_ORDER:
        raise CalibrationError("Run order changed")
    if bank.get("leakage_declaration") != EXPECTED_LEAKAGE_DECLARATION:
        raise CalibrationError("Leakage declaration changed")

    bank_spec = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "bank_id": "cohesion-margin-v1",
        "source_sha256": source_hash,
        "expectation_sha256": expectation_hash,
        "evaluator_config_sha256": evaluator_hash,
        "fixture_spec_sha256": fixture_hashes,
    }
    bank_spec_hash = sha256_json(bank_spec)
    if (
        bank_spec_hash != EXPECTED_BANK_SPEC_HASH
        or bank.get("bank_spec_hash") != bank_spec_hash
    ):
        raise CalibrationError("Canonical bank-spec hash mismatch")

    input_text = canonical_json(
        {
            "source_csv": source,
            "expectation": expectation,
            "evaluator_config": evaluator_config,
            "fixtures": fixtures,
            "candidate_rule": bank["candidate_rule"],
            "run_order": bank["run_order"],
        }
    ).casefold()
    leaked = [marker for marker in FORBIDDEN_INPUT_MARKERS if marker in input_text]
    if leaked:
        raise CalibrationError(f"Forbidden production/test markers: {leaked}")

    return {
        **bank,
        "_path": str(resolved),
        "_file_sha256": sha256_file(resolved),
        "_bank_spec_hash": bank_spec_hash,
    }


@contextmanager
def _network_disabled() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise CalibrationError("Network access is forbidden during calibration")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


def _build_figure(fixture: Mapping[str, Any]) -> Any:
    panel_a = _required_mapping(fixture.get("panel_a"), "panel_a")
    panel_b = _required_mapping(fixture.get("panel_b"), "panel_b")
    legend_mode = fixture.get("legend_mode")
    if legend_mode not in {"figure_single", "two_axis_legends"}:
        raise CalibrationError(f"Invalid legend mode: {legend_mode!r}")

    rc = {
        "font.family": "DejaVu Sans",
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=100)
        lines = []
        for axis, panel_id, panel in zip(
            axes,
            ("panel-a", "panel-b"),
            (panel_a, panel_b),
        ):
            axis.set_gid(panel_id)
            (line,) = axis.plot(
                [1, 2, 3],
                [10, 20, 30],
                label="signal",
                color=panel["color"],
            )
            axis.set_yscale(panel["scale"])
            axis.set_ylabel(panel["ylabel"])
            lines.append(line)
        if legend_mode == "figure_single":
            fig.legend([lines[0]], ["signal"])
        else:
            for axis in axes:
                axis.legend()
        fig.canvas.draw()
        return fig


def _check_counts(result: Any) -> dict[str, Any]:
    cohesion = result.cohesion
    checks = {
        name: [
            cohesion.checks[name].numerator,
            cohesion.checks[name].denominator,
        ]
        for name in (
            "shared_scale",
            "shared_unit",
            "legend_deduplication",
            "palette_mapping",
        )
    }
    return {
        "cohesion": [cohesion.numerator, cohesion.denominator],
        **checks,
        "ratio": cohesion.ratio,
        "mismatch_codes": [item.code for item in cohesion.mismatches],
    }


def _write_run_artifacts(
    run_dir: Path,
    *,
    fig: Any,
    evaluation: Mapping[str, Any],
    fixture_result: Mapping[str, Any],
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=False)
    render_path = run_dir / "render.png"
    fig.savefig(
        render_path,
        dpi=100,
        metadata={"Software": "PheroViz cohesion-margin calibration"},
    )
    paths = {
        "figure_manifest": run_dir / "figure_manifest.json",
        "cohesion_result": run_dir / "cohesion_result.json",
        "evaluation": run_dir / "evaluation.json",
        "fixture_result": run_dir / "fixture_result.json",
    }
    write_json_atomic(paths["figure_manifest"], evaluation["figure_manifest"])
    write_json_atomic(paths["cohesion_result"], evaluation["cohesion"])
    write_json_atomic(paths["evaluation"], evaluation)
    write_json_atomic(paths["fixture_result"], fixture_result)
    return {
        "render.png": sha256_file(render_path),
        **{
            f"{name}.json": sha256_file(path)
            for name, path in paths.items()
        },
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        dependency_diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *DEPENDENCY_PATHS],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CalibrationError(f"Cannot record git provenance: {exc}") from exc
    if dependency_diff not in {0, 1}:
        raise CalibrationError("Cannot inspect evaluator dependency diff")
    return {
        "commit": commit,
        "dirty": dirty,
        "dependency_tree_clean": dependency_diff == 0,
    }


def _dependency_hashes() -> dict[str, str]:
    output = {}
    for relative in DEPENDENCY_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise CalibrationError(f"Calibration dependency is missing: {relative}")
        output[relative] = sha256_file(path)
    return output


def _environment_payload() -> dict[str, Any]:
    git_state = _git_state()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "backend": matplotlib.get_backend(),
        "git": git_state,
        "dependency_hashes": _dependency_hashes(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def _semantic_results(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[float]], float]:
    baseline = {
        int(item["replicate"]): float(item["ratio"])
        for item in results
        if item["fixture_id"] == "C0"
    }
    class_drops: dict[str, list[float]] = {
        name: [] for name in EXPECTED_CANDIDATE_RULE["defect_classes"]
    }
    for item in results:
        defect = str(item["defect"])
        if defect == "none":
            continue
        replicate = int(item["replicate"])
        class_drops[defect].append(
            baseline[replicate] - float(item["ratio"])
        )
    class_medians = {
        name: float(median(values))
        for name, values in class_drops.items()
    }
    return class_drops, min(class_medians.values())


def _repeatability_check(results: Sequence[Mapping[str, Any]]) -> bool:
    by_fixture: dict[str, list[str]] = {}
    for item in results:
        stable = {
            key: value
            for key, value in item.items()
            if key not in {"run_index", "replicate"}
        }
        by_fixture.setdefault(str(item["fixture_id"]), []).append(
            sha256_json(stable)
        )
    return all(len(set(hashes)) == 1 for hashes in by_fixture.values())


def _write_report(
    output_dir: Path,
    semantic_payload: Mapping[str, Any],
    *,
    generated_at: str,
) -> dict[str, Any]:
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "semantic_hash_scope": (
            "sha256_json(semantic_payload); generated_at and report_hash excluded"
        ),
        "semantic_payload": dict(semantic_payload),
        "semantic_hash": sha256_json(semantic_payload),
    }
    report["report_hash"] = sha256_json(report)
    write_json_atomic(output_dir / REPORT_FILENAME, report)
    return report


def run_calibration(
    output_dir: Path,
    *,
    fixture_bank_path: Path = DEFAULT_FIXTURE_BANK,
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CalibrationError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at or _utc_now()
    results: list[dict[str, Any]] = []
    base_payload: dict[str, Any] = {
        "calibration_id": "cohesion-margin-v1",
        "status": "blocked",
        "run_count": 0,
        "candidate_delta_C": None,
        "final_delta_C": None,
        "delta_C_status": "UNSET",
        "leakage_declaration": EXPECTED_LEAKAGE_DECLARATION,
    }

    try:
        bank = _validate_fixture_bank(fixture_bank_path)
        environment = _environment_payload()
        if not environment["git"]["dependency_tree_clean"]:
            raise CalibrationError("Evaluator dependency tree is dirty")

        source_path = output_dir / "source.csv"
        source_path.write_text(bank["source_csv"], encoding="utf-8", newline="")
        write_json_atomic(output_dir / "expectation.json", bank["expectation"])
        write_json_atomic(
            output_dir / "evaluator_config.json",
            bank["evaluator_config"],
        )
        write_json_atomic(
            output_dir / "fixture_bank_snapshot.json",
            {
                key: value
                for key, value in bank.items()
                if not key.startswith("_")
            },
        )
        write_json_atomic(output_dir / "environment.json", environment)
        write_json_atomic(
            output_dir / "run_order.json",
            {"run_order": bank["run_order"]},
        )

        source_df = pd.read_csv(source_path)
        fixtures = {
            item["fixture_id"]: item for item in bank["fixtures"]
        }
        with _network_disabled():
            for run_index, run_id in enumerate(bank["run_order"], 1):
                replicate_text, fixture_id = str(run_id).split(":", 1)
                replicate = int(replicate_text.removeprefix("r"))
                fixture = fixtures[fixture_id]
                fig = _build_figure(fixture)
                try:
                    evaluation_result = evaluate_figure(
                        fig,
                        source_df,
                        bank["expectation"],
                        bank["evaluator_config"],
                    )
                    evaluation = evaluation_result.to_dict()
                    observed = _check_counts(evaluation_result)
                    expected = EXPECTED_RESULTS[fixture_id]
                    if observed != expected:
                        raise CalibrationError(
                            f"Unexpected cohesion result for {run_id}: "
                            f"expected={expected}, observed={observed}"
                        )
                    fixture_result = {
                        "fixture_id": fixture_id,
                        "defect": fixture["defect"],
                        "fixture_spec_sha256": fixture["spec_sha256"],
                        **observed,
                    }
                    artifact_hashes = _write_run_artifacts(
                        output_dir
                        / f"run_{run_index:02d}_{replicate_text}_{fixture_id}",
                        fig=fig,
                        evaluation=evaluation,
                        fixture_result=fixture_result,
                    )
                finally:
                    plt.close(fig)
                results.append(
                    {
                        "run_index": run_index,
                        "replicate": replicate,
                        **fixture_result,
                        "artifact_hashes": artifact_hashes,
                    }
                )

        class_drops, candidate_delta = _semantic_results(results)
        class_medians = {
            name: float(median(values))
            for name, values in class_drops.items()
        }
        checks = {
            "exact_run_count": len(results) == 15,
            "exact_fixture_counts": all(
                item["fixture_id"] in EXPECTED_RESULTS for item in results
            ),
            "candidate_rule_exact": candidate_delta == 0.25,
            "all_class_drops_exact": all(
                values == [0.25, 0.25, 0.25]
                for values in class_drops.values()
            ),
            "within_run_repeatability": _repeatability_check(results),
            "leakage_declaration_clean": (
                bank["leakage_declaration"]
                == EXPECTED_LEAKAGE_DECLARATION
            ),
            "dependency_tree_clean": environment["git"][
                "dependency_tree_clean"
            ],
            "input_hashes_exact": (
                sha256_file(source_path) == EXPECTED_SOURCE_SHA256
                and sha256_json(bank["expectation"])
                == EXPECTED_EXPECTATION_SHA256
                and sha256_json(bank["evaluator_config"])
                == EXPECTED_EVALUATOR_CONFIG_SHA256
                and bank["_bank_spec_hash"] == EXPECTED_BANK_SPEC_HASH
            ),
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise CalibrationError(f"Acceptance checks failed: {failed}")

        semantic_payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_id": "cohesion-margin-v1",
            "status": "passed",
            "run_count": len(results),
            "input_hashes": {
                "fixture_bank_file_sha256": bank["_file_sha256"],
                "bank_spec_sha256": bank["_bank_spec_hash"],
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "expectation_sha256": EXPECTED_EXPECTATION_SHA256,
                "evaluator_config_sha256": EXPECTED_EVALUATOR_CONFIG_SHA256,
                "script_sha256": environment["script_sha256"],
                "dependency_hashes": environment["dependency_hashes"],
            },
            "environment": environment,
            "run_order": bank["run_order"],
            "results": results,
            "candidate_rule": EXPECTED_CANDIDATE_RULE,
            "class_drops": class_drops,
            "class_medians": class_medians,
            "candidate_delta_C": candidate_delta,
            "final_delta_C": candidate_delta,
            "delta_C_status": "FROZEN",
            "acceptance_checks": checks,
            "leakage_declaration": EXPECTED_LEAKAGE_DECLARATION,
        }
        return _write_report(
            output_dir,
            semantic_payload,
            generated_at=timestamp,
        )
    except Exception as exc:
        semantic_payload = {
            **base_payload,
            "run_count": len(results),
            "results": results,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        return _write_report(
            output_dir,
            semantic_payload,
            generated_at=timestamp,
        )


def _verify_report(path: Path) -> dict[str, Any]:
    report = _load_json(path.expanduser().resolve(strict=True))
    report_hash = report.get("report_hash")
    unhashed = dict(report)
    unhashed.pop("report_hash", None)
    if report_hash != sha256_json(unhashed):
        raise CalibrationError(f"Report hash mismatch: {path}")
    semantic_payload = _required_mapping(
        report.get("semantic_payload"),
        "semantic_payload",
    )
    if report.get("semantic_hash") != sha256_json(semantic_payload):
        raise CalibrationError(f"Semantic hash mismatch: {path}")
    return report


def compare_reports(first: Path, second: Path) -> dict[str, Any]:
    left = _verify_report(first)
    right = _verify_report(second)
    stable = (
        left["semantic_hash"] == right["semantic_hash"]
        and canonical_json(left["semantic_payload"])
        == canonical_json(right["semantic_payload"])
    )
    return {
        "semantic_payload_stable": stable,
        "first_semantic_hash": left["semantic_hash"],
        "second_semantic_hash": right["semantic_hash"],
        "generated_at_excluded": True,
        "first_report_hash": left["report_hash"],
        "second_report_hash": right["report_hash"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic structural cohesion-margin calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument(
        "--fixture-bank",
        type=Path,
        default=DEFAULT_FIXTURE_BANK,
    )
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        report = run_calibration(
            args.out,
            fixture_bank_path=args.fixture_bank,
        )
        payload = report["semantic_payload"]
        print(
            json.dumps(
                {
                    "report": str(args.out.resolve() / REPORT_FILENAME),
                    "report_hash": report["report_hash"],
                    "semantic_hash": report["semantic_hash"],
                    "status": payload["status"],
                    "run_count": payload["run_count"],
                    "candidate_delta_C": payload.get("candidate_delta_C"),
                    "final_delta_C": payload.get("final_delta_C"),
                    "delta_C_status": payload["delta_C_status"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["status"] == "passed" else 1
    comparison = compare_reports(args.first, args.second)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if comparison["semantic_payload_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

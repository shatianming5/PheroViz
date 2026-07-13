from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pytest
from jsonschema import validate

import experiments.production_statistics as statistics
from experiments.cli import main
from experiments.models import (
    ExperimentSpec,
    RunRecord,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from experiments.production_statistics import (
    StatisticsError,
    analyze_summary,
    load_provenance_summary,
    normalize_trajectory_threshold_config,
)


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".trajectory_test_work"
    path = parent / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _threshold() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "fidelity": {
            "metric": "data_fidelity",
            "threshold": 0.8,
        },
        "cohesion": {
            "metric": "series_cohesion",
            "threshold": 0.75,
        },
        "restriction": {
            "renders": 6,
            "wall_clock_seconds": 10.0,
        },
    }


def _run_name(method: str, case_id: str, seed: int) -> str:
    return (
        f"trajectory__case-{case_id}__method-{method}"
        f"__backbone-model-a__seed-{seed}__budget-renders-6"
    )


def _write_completed_record(
    run_root: Path,
    *,
    method: str,
    case_id: str,
    seed: int,
    panel_count: int = 2,
    schedule: str = "iterative",
    fidelity: Sequence[float] = (0.4, 0.8, 1.0),
    cohesion: Sequence[float | None] = (0.5, 0.8, 1.0),
    cumulative_time: Sequence[float] | None = (1.0, 4.0, 6.0),
    cumulative_render: Sequence[int] | None = None,
) -> RunRecord:
    manifest_source = run_root / "dataset_manifest.json"
    if not manifest_source.exists():
        manifest_source.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": "case-a",
                            "doi": "10.1234/a",
                            "panel_count": 2,
                            "split": "test",
                        },
                        {
                            "case_id": "case-b",
                            "doi": "10.1234/b",
                            "panel_count": 2,
                            "split": "test",
                        },
                    ]
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    metric_config = {
        "selection": {
            "metric": "data_fidelity",
            "direction": "maximize",
        }
    }
    spec = ExperimentSpec(
        run_name=_run_name(method, case_id, seed),
        method=method,
        schedule=schedule,
        backbone="model-a",
        case_id=case_id,
        panel_count=panel_count,
        split="test",
        seed=seed,
        budget_type="renders",
        budget_value=6.0,
        dataset_manifest_path=str(manifest_source),
        dataset_manifest_hash=sha256_file(manifest_source),
        git_commit="c" * 40,
        git_dirty=False,
        provider="provider",
        artifact_root=str(run_root),
        repo_root=str(run_root),
        metric_config=metric_config,
        metric_config_hash=sha256_json(metric_config),
        metric_version="metrics-v1",
        method_config={"memory_mode": "full"},
        provider_options={},
    )
    run_dir = run_root / spec.run_name
    run_dir.mkdir(parents=True)
    frozen = run_dir / "dataset_manifest.frozen.json"
    frozen.write_bytes(manifest_source.read_bytes())
    record = RunRecord.start(spec, attempt=1)
    record.provider_name = "provider"
    record.artifact_paths["dataset_manifest"] = frozen.name
    record.artifact_hashes["dataset_manifest"] = sha256_file(frozen)
    expected_candidates = 6 // panel_count
    assert len(fidelity) == len(cohesion) == expected_candidates
    times = list(cumulative_time or ())
    renders = list(
        cumulative_render
        or [
            panel_count * index for index in range(1, expected_candidates + 1)
        ]
    )
    candidates = []
    best_score = -1.0
    for index, (fidelity_value, cohesion_value) in enumerate(
        zip(fidelity, cohesion),
        1,
    ):
        candidate_id = f"candidate_{index:04d}"
        artifact_dir = run_dir / "artifacts" / candidate_id
        artifact_dir.mkdir(parents=True)
        panel_fidelity = {
            panel: {
                "numerator": int(round(fidelity_value * 1000)),
                "denominator": 1000,
            }
            for panel in ("left", "right")
        }
        programmatic = {
            "panel_fidelity": panel_fidelity,
            "cohesion": (
                {"applicable": True, "ratio": cohesion_value}
                if cohesion_value is not None
                else {"applicable": False, "ratio": None}
            ),
        }
        programmatic_path = artifact_dir / "programmatic_evaluation.json"
        write_json_atomic(programmatic_path, programmatic)
        relative_programmatic = programmatic_path.relative_to(run_dir).as_posix()
        metrics = {
            "data_fidelity": float(fidelity_value),
            "execution_success": 1.0,
        }
        if cohesion_value is not None:
            metrics["series_cohesion"] = float(cohesion_value)
        provider_metadata: dict[str, Any] = {
            "panel_count": panel_count,
            "cumulative_render_count": renders[index - 1],
        }
        if cumulative_time is not None:
            provider_metadata["cumulative_wall_clock_seconds"] = times[index - 1]
        candidate = {
            "candidate_id": candidate_id,
            "call_index": index if schedule == "best_of_n" else 1,
            "metrics": metrics,
            "selection_score": float(fidelity_value),
            "selection_direction": "maximize",
            "render_count": panel_count,
            "artifact_paths": {
                "programmatic_evaluation": relative_programmatic,
            },
            "artifact_hashes": {
                "programmatic_evaluation": sha256_file(programmatic_path),
            },
            "provider_metadata": provider_metadata,
            "test_only": False,
        }
        sidecar = run_dir / "attempt_001" / "candidates" / f"{candidate_id}.json"
        write_json_atomic(sidecar, candidate)
        relative_sidecar = sidecar.relative_to(run_dir).as_posix()
        record.artifact_paths[
            f"{candidate_id}.programmatic_evaluation"
        ] = relative_programmatic
        record.artifact_hashes[
            f"{candidate_id}.programmatic_evaluation"
        ] = sha256_file(programmatic_path)
        record.artifact_paths[f"{candidate_id}.metadata"] = relative_sidecar
        record.artifact_hashes[f"{candidate_id}.metadata"] = sha256_file(
            sidecar
        )
        candidates.append(candidate)
        if fidelity_value > best_score:
            best_score = float(fidelity_value)
            record.best_candidate_id = candidate_id
            record.metrics = dict(metrics)
        record.best_history.append(
            {
                "after_candidate_id": candidate_id,
                "best_candidate_id": record.best_candidate_id,
                "best_selection_score": best_score,
            }
        )
    record.candidates = candidates
    record.status = "completed"
    record.finished_at = record.started_at
    record.render_count = 6
    record.wall_clock_seconds = 8.0
    best_path = run_dir / "best_so_far.json"
    write_json_atomic(
        best_path,
        {
            "best_candidate_id": record.best_candidate_id,
            "metrics": record.metrics,
        },
    )
    record.artifact_paths["best_so_far"] = best_path.name
    record.artifact_hashes["best_so_far"] = sha256_file(best_path)
    record.write(run_dir / "run_record.json")
    return record


def _write_failed_record(
    run_root: Path,
    *,
    method: str,
    case_id: str,
    seed: int,
) -> RunRecord:
    record = _write_completed_record(
        run_root,
        method=method,
        case_id=case_id,
        seed=seed,
    )
    run_dir = run_root / record.run_name
    record.status = "failed"
    record.error = {
        "type": "MethodFailure",
        "message": "synthetic method failure",
        "attribution": "method",
    }
    record.render_count = 0
    record.wall_clock_seconds = 1.0
    record.candidates = []
    record.best_candidate_id = None
    record.best_history = []
    record.metrics = {}
    for key in list(record.artifact_paths):
        if key != "dataset_manifest":
            record.artifact_paths.pop(key)
            record.artifact_hashes.pop(key)
    record.write(run_dir / "run_record.json")
    return record


def _write_late_failed_record(
    run_root: Path,
    *,
    method: str,
    case_id: str,
    seed: int,
    candidates_to_keep: int,
    fidelity: Sequence[float] = (0.4, 0.8, 1.0),
    cohesion: Sequence[float | None] = (0.5, 0.8, 1.0),
) -> RunRecord:
    record = _write_completed_record(
        run_root,
        method=method,
        case_id=case_id,
        seed=seed,
        fidelity=fidelity,
        cohesion=cohesion,
    )
    run_dir = run_root / record.run_name
    kept = record.candidates[:candidates_to_keep]
    kept_ids = {candidate["candidate_id"] for candidate in kept}
    record.status = "failed"
    record.error = {
        "type": "MethodFailure",
        "message": "synthetic late method failure",
        "attribution": "method",
    }
    record.candidates = kept
    record.render_count = candidates_to_keep * int(record.panel_count or 0)
    record.best_history = record.best_history[:candidates_to_keep]
    if kept:
        best = max(kept, key=lambda item: item["selection_score"])
        record.best_candidate_id = str(best["candidate_id"])
        record.metrics = dict(best["metrics"])
    else:
        record.best_candidate_id = None
        record.metrics = {}
    for key in list(record.artifact_paths):
        if key == "dataset_manifest":
            continue
        candidate_prefix = key.split(".", 1)[0]
        if candidate_prefix not in kept_ids:
            record.artifact_paths.pop(key)
            record.artifact_hashes.pop(key)
    record.write(run_dir / "run_record.json")
    return record


def _summary_row(record: RunRecord, doi: str) -> dict[str, Any]:
    completed = record.status == "completed"
    metrics = dict(record.metrics) if completed else {
        "data_fidelity": 0.0,
        "series_cohesion": 0.0,
    }
    metrics["execution_success"] = 1.0 if completed else 0.0
    return {
        "run_name": record.run_name,
        "method": record.method,
        "backbone": record.backbone,
        "case_id": record.case_id,
        "doi": doi,
        "panel_count": record.panel_count,
        "split": record.split,
        "seed": record.seed,
        "budget_type": record.budget_type,
        "budget_value": record.budget_value,
        "dataset_manifest_hash": record.dataset_manifest_hash,
        "git_commit": record.git_commit,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "status": record.status,
        "execution_success": 1.0 if completed else 0.0,
        "failure_attribution": "" if completed else "method",
        "failure_type": "" if completed else "MethodFailure",
        "test_only": False,
        "render_count": record.render_count,
        "wall_clock_seconds": record.wall_clock_seconds,
        "metric_version": record.metric_version,
        "metric_config_hash": record.metric_config_hash,
        "provider": record.provider,
        "provider_name": record.provider_name,
        "best_candidate_id": record.best_candidate_id,
        "spec_hash": record.spec_hash,
        "record_hash": record.record_hash,
        **{f"metric.{name}": value for name, value in metrics.items()},
    }


def _write_summary(
    workspace: Path,
    records: Sequence[tuple[RunRecord, str]],
) -> Path:
    rows = [_summary_row(record, doi) for record, doi in records]
    payload = {
        "schema_version": "2.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "source_root": str(workspace / "runs"),
        "run_count": len(rows),
        "columns": sorted({key for row in rows for key in row}),
        "runs": rows,
        "input_record_hashes": {
            row["run_name"]: row["record_hash"] for row in rows
        },
    }
    payload["summary_hash"] = sha256_json(payload)
    path = workspace / "summary.json"
    write_json_atomic(path, payload)
    return path


def _analysis(
    summary_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(
        statistics,
        "_git_provenance",
        lambda: ("9" * 40, False),
    )
    monkeypatch.setattr(
        statistics,
        "utc_now",
        lambda: "2026-01-01T00:00:00Z",
    )
    return analyze_summary(
        load_provenance_summary(summary_path),
        reference="flat",
        methods=["full"],
        metric="metric.data_fidelity",
        panel_scope="multi_panel",
        trajectory_threshold=threshold or _threshold(),
        bootstrap_resamples=20,
        monte_carlo_permutations=20,
    )


def test_observed_censored_and_seed_task_doi_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("observed-censored") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                for seed in (0, 1):
                    if method == "flat" and case_id == "case-a" and seed == 1:
                        fidelity = (0.4, 0.6, 0.7)
                        cohesion = (0.4, 0.6, 0.7)
                    elif method == "flat" and case_id == "case-a":
                        fidelity = (0.4, 0.8, 1.0)
                        cohesion = (0.4, 0.8, 1.0)
                    else:
                        fidelity = (0.9, 0.9, 1.0)
                        cohesion = (0.8, 0.8, 1.0)
                    record = _write_completed_record(
                        run_root,
                        method=method,
                        case_id=case_id,
                        seed=seed,
                        fidelity=fidelity,
                        cohesion=cohesion,
                    )
                    records.append((record, doi))
        summary_path = _write_summary(workspace, records)

        first = _analysis(summary_path, monkeypatch)
        second = _analysis(summary_path, monkeypatch)

        assert first == second
        assert first["analysis_hash"] == second["analysis_hash"]
        trajectory = first["right_censored_rmst"]
        assert trajectory["status"] == "ok"
        assert trajectory["threshold_config"] == normalize_trajectory_threshold_config(
            _threshold()
        )
        assert trajectory["threshold_config_hash"] == sha256_json(
            trajectory["threshold_config"]
        )
        result = trajectory["slices"][0]
        flat = next(item for item in result["methods"] if item["method"] == "flat")
        full = next(item for item in result["methods"] if item["method"] == "full")
        assert flat["attainment_rate"] == pytest.approx(0.75)
        assert flat["restricted_mean_renders_to_threshold"] == pytest.approx(3.5)
        assert flat[
            "restricted_mean_wall_clock_seconds_to_threshold"
        ] == pytest.approx(4.0)
        assert full["attainment_rate"] == 1.0
        assert full["restricted_mean_renders_to_threshold"] == 2.0
        assert full[
            "restricted_mean_wall_clock_seconds_to_threshold"
        ] == 1.0
        censored = next(
            item
            for item in result["run_observations"]
            if item["method"] == "flat"
            and item["case_id"] == "case-a"
            and item["seed"] == 1
        )
        assert not censored["joint_attainment"]
        assert censored["restricted_renders_to_threshold"] == 6.0
        observed = next(
            item
            for item in result["run_observations"]
            if item["method"] == "flat"
            and item["case_id"] == "case-a"
            and item["seed"] == 0
        )
        assert observed["threshold_crossing_render"] == 4
        assert observed["threshold_crossing_wall_clock_seconds"] == 4.0


def test_best_of_n_accumulates_complete_p2_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("best-of-n") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                record = _write_completed_record(
                    run_root,
                    method=method,
                    case_id=case_id,
                    seed=0,
                    schedule="best_of_n",
                    cumulative_time=(1.0, 3.0, 6.0),
                )
                records.append((record, doi))
        result = _analysis(_write_summary(workspace, records), monkeypatch)
        observation = result["right_censored_rmst"]["slices"][0][
            "run_observations"
        ][0]
        assert [
            item["cumulative_render_count"]
            for item in observation["trajectory_points"]
        ] == [2, 4, 6]
        assert [
            item["cumulative_wall_clock_seconds"]
            for item in observation["trajectory_points"]
        ] == [1.0, 3.0, 6.0]


def test_single_panel_trajectory_is_explicitly_inapplicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("single-panel") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                record = _write_completed_record(
                    run_root,
                    method=method,
                    case_id=case_id,
                    seed=0,
                    panel_count=1,
                    fidelity=(0.4, 0.8, 1.0, 1.0, 1.0, 1.0),
                    cohesion=(0.4, 0.8, 1.0, 1.0, 1.0, 1.0),
                    cumulative_time=(1, 2, 3, 4, 5, 6),
                )
                records.append((record, doi))
        summary = load_provenance_summary(_write_summary(workspace, records))
        monkeypatch.setattr(
            statistics,
            "_git_provenance",
            lambda: ("9" * 40, False),
        )
        with pytest.raises(StatisticsError, match="single-panel.*inapplicable"):
            analyze_summary(
                summary,
                reference="flat",
                methods=["full"],
                metric="metric.data_fidelity",
                panel_scope="single_panel",
                trajectory_threshold=_threshold(),
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_explicit_method_failures_are_right_censored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("method-failures") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for case_id, doi in (
            ("case-a", "10.1234/a"),
            ("case-b", "10.1234/b"),
        ):
            for seed in (0, 1):
                records.append(
                    (
                        _write_completed_record(
                            run_root,
                            method="flat",
                            case_id=case_id,
                            seed=seed,
                        ),
                        doi,
                    )
                )
                records.append(
                    (
                        _write_failed_record(
                            run_root,
                            method="full",
                            case_id=case_id,
                            seed=seed,
                        ),
                        doi,
                    )
                )
        analysis = _analysis(_write_summary(workspace, records), monkeypatch)
        full = next(
            item
            for item in analysis["right_censored_rmst"]["slices"][0]["methods"]
            if item["method"] == "full"
        )
        assert full["attainment_rate"] == 0.0
        assert full["restricted_mean_renders_to_threshold"] == 6.0
        assert full["restricted_mean_wall_clock_seconds_to_threshold"] == 10.0
        observations = analysis["right_censored_rmst"]["slices"][0][
            "run_observations"
        ]
        assert {
            item["status"]
            for item in observations
            if item["method"] == "full"
        } == {"method_failure_censored"}


def test_wall_clock_restriction_censors_valid_late_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("wall-restriction") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                records.append(
                    (
                        _write_completed_record(
                            run_root,
                            method=method,
                            case_id=case_id,
                            seed=0,
                            fidelity=(0.4, 0.6, 0.9),
                            cohesion=(0.4, 0.6, 0.9),
                            cumulative_time=(1.0, 4.0, 6.0),
                        ),
                        doi,
                    )
                )
        threshold = _threshold()
        threshold["restriction"]["wall_clock_seconds"] = 5.0
        analysis = _analysis(
            _write_summary(workspace, records),
            monkeypatch,
            threshold=threshold,
        )
        observations = analysis["right_censored_rmst"]["slices"][0][
            "run_observations"
        ]
        assert all(item["render_event_observed"] for item in observations)
        assert all(not item["time_event_observed"] for item in observations)
        assert all(not item["joint_attainment"] for item in observations)
        assert {
            item["censor_reason"] for item in observations
        } == {"wall_clock_restriction_exceeded"}
        assert {
            item["restricted_wall_clock_seconds_to_threshold"]
            for item in observations
        } == {5.0}


def test_late_method_failure_preserves_observed_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("late-failure-observed") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for case_id, doi in (
            ("case-a", "10.1234/a"),
            ("case-b", "10.1234/b"),
        ):
            records.append(
                (
                    _write_completed_record(
                        run_root,
                        method="flat",
                        case_id=case_id,
                        seed=0,
                    ),
                    doi,
                )
            )
            records.append(
                (
                    _write_late_failed_record(
                        run_root,
                        method="full",
                        case_id=case_id,
                        seed=0,
                        candidates_to_keep=2,
                    ),
                    doi,
                )
            )
        analysis = _analysis(_write_summary(workspace, records), monkeypatch)
        observations = [
            item
            for item in analysis["right_censored_rmst"]["slices"][0][
                "run_observations"
            ]
            if item["method"] == "full"
        ]
        assert {item["status"] for item in observations} == {
            "observed_before_method_failure"
        }
        assert {item["threshold_crossing_render"] for item in observations} == {
            4
        }
        assert all(item["joint_attainment"] for item in observations)


def test_method_failure_before_crossing_remains_censored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("late-failure-censored") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for case_id, doi in (
            ("case-a", "10.1234/a"),
            ("case-b", "10.1234/b"),
        ):
            records.append(
                (
                    _write_completed_record(
                        run_root,
                        method="flat",
                        case_id=case_id,
                        seed=0,
                    ),
                    doi,
                )
            )
            records.append(
                (
                    _write_late_failed_record(
                        run_root,
                        method="full",
                        case_id=case_id,
                        seed=0,
                        candidates_to_keep=1,
                    ),
                    doi,
                )
            )
        analysis = _analysis(_write_summary(workspace, records), monkeypatch)
        observations = [
            item
            for item in analysis["right_censored_rmst"]["slices"][0][
                "run_observations"
            ]
            if item["method"] == "full"
        ]
        assert {item["status"] for item in observations} == {
            "method_failure_censored"
        }
        assert {item["censor_reason"] for item in observations} == {
            "explicit_method_failure_before_threshold"
        }
        assert all(not item["joint_attainment"] for item in observations)
        assert {
            item["restricted_renders_to_threshold"] for item in observations
        } == {6.0}


@pytest.mark.parametrize(
    ("field", "forged_value", "message"),
    [
        ("doi", "10.9999/forged", "DOI disagrees with frozen manifest"),
        (
            "metric.data_fidelity",
            0.123,
            "summary metric disagrees with record",
        ),
    ],
)
def test_rehashed_summary_cannot_forge_trajectory_doi_or_metrics(
    field: str,
    forged_value: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace(f"forged-{field}") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                records.append(
                    (
                        _write_completed_record(
                            run_root,
                            method=method,
                            case_id=case_id,
                            seed=0,
                        ),
                        doi,
                    )
                )
        summary_path = _write_summary(workspace, records)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if field == "doi":
            for row in payload["runs"]:
                if row["case_id"] == "case-a":
                    row[field] = forged_value
        else:
            payload["runs"][0][field] = forged_value
        payload.pop("summary_hash")
        payload["summary_hash"] = sha256_json(payload)
        write_json_atomic(summary_path, payload)

        with pytest.raises(StatisticsError, match=message):
            _analysis(summary_path, monkeypatch)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("tamper", "Invalid trajectory provenance"),
        ("nonmonotonic_render", "Nonmonotonic iterative trajectory metadata"),
        ("nonmonotonic_time", "Nonmonotonic iterative trajectory metadata"),
        ("missing_time", "must be numeric"),
        ("missing_cohesion", "cohesion is unavailable"),
    ],
)
def test_trajectory_fail_closed_artifact_and_metadata_guards(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace(f"guard-{mutation}") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                kwargs: dict[str, Any] = {}
                if method == "flat" and case_id == "case-a":
                    if mutation == "nonmonotonic_render":
                        kwargs["cumulative_render"] = (2, 2, 6)
                    elif mutation == "nonmonotonic_time":
                        kwargs["cumulative_time"] = (1.0, 0.5, 2.0)
                    elif mutation == "missing_time":
                        kwargs["cumulative_time"] = None
                    elif mutation == "missing_cohesion":
                        kwargs["cohesion"] = (None, None, None)
                record = _write_completed_record(
                    run_root,
                    method=method,
                    case_id=case_id,
                    seed=0,
                    **kwargs,
                )
                records.append((record, doi))
        if mutation == "tamper":
            record = records[0][0]
            run_dir = run_root / record.run_name
            path = (
                run_dir
                / record.candidates[0]["artifact_paths"][
                    "programmatic_evaluation"
                ]
            )
            path.write_text('{"tampered":true}\n', encoding="utf-8")
        with pytest.raises(StatisticsError, match=message):
            _analysis(_write_summary(workspace, records), monkeypatch)


def test_trajectory_rejects_dirty_code_and_incomplete_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("dirty-pairing") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                records.append(
                    (
                        _write_completed_record(
                            run_root,
                            method=method,
                            case_id=case_id,
                            seed=0,
                        ),
                        doi,
                    )
                )
        summary_path = _write_summary(workspace, records)
        monkeypatch.setattr(
            statistics,
            "_git_provenance",
            lambda: ("9" * 40, True),
        )
        with pytest.raises(StatisticsError, match="dirty analysis worktree"):
            analyze_summary(
                load_provenance_summary(summary_path),
                reference="flat",
                methods=["full"],
                metric="metric.data_fidelity",
                panel_scope="multi_panel",
                trajectory_threshold=_threshold(),
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )

        incomplete = [
            item
            for item in records
            if not (item[0].method == "full" and item[0].case_id == "case-b")
        ]
        monkeypatch.setattr(
            statistics,
            "_git_provenance",
            lambda: ("9" * 40, False),
        )
        with pytest.raises(StatisticsError, match="Paired case mismatch"):
            _analysis(_write_summary(workspace, incomplete), monkeypatch)


def test_threshold_config_has_no_defaults_and_a_deterministic_hash() -> None:
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas/c3_trajectory_threshold.schema.json"
        ).read_text(encoding="utf-8")
    )
    validate(instance=_threshold(), schema=schema)
    first = normalize_trajectory_threshold_config(_threshold())
    second = normalize_trajectory_threshold_config(
        json.loads(json.dumps(_threshold(), sort_keys=True))
    )
    assert first == second
    assert sha256_json(first) == sha256_json(second)
    with pytest.raises(StatisticsError, match="exactly"):
        normalize_trajectory_threshold_config(
            {
                "schema_version": "1.0",
                "fidelity": {
                    "metric": "data_fidelity",
                    "threshold": 0.8,
                },
            }
        )


def test_cli_requires_an_explicit_threshold_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("cli-threshold") as workspace:
        run_root = workspace / "runs"
        run_root.mkdir()
        records = []
        for method in ("flat", "full"):
            for case_id, doi in (
                ("case-a", "10.1234/a"),
                ("case-b", "10.1234/b"),
            ):
                records.append(
                    (
                        _write_completed_record(
                            run_root,
                            method=method,
                            case_id=case_id,
                            seed=0,
                        ),
                        doi,
                    )
                )
        summary = _write_summary(workspace, records)
        threshold_path = workspace / "threshold.json"
        threshold_path.write_text(
            json.dumps(_threshold(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            statistics,
            "_git_provenance",
            lambda: ("9" * 40, False),
        )
        monkeypatch.setattr(
            statistics,
            "utc_now",
            lambda: "2026-01-01T00:00:00Z",
        )
        output = workspace / "analysis"

        exit_code = main(
            [
                "analyze",
                str(summary),
                "--reference",
                "flat",
                "--methods",
                "full",
                "--metric",
                "metric.data_fidelity",
                "--panel-scope",
                "multi_panel",
                "--trajectory-threshold-config",
                str(threshold_path),
                "--bootstrap-resamples",
                "20",
                "--permutations",
                "20",
                "--out",
                str(output),
            ]
        )

        assert exit_code == 0
        cli_result = json.loads(capsys.readouterr().out)
        persisted = json.loads(
            Path(cli_result["analysis_json"]).read_text(encoding="utf-8")
        )
        assert persisted["right_censored_rmst"]["status"] == "ok"
        assert persisted["analysis_config"]["trajectory_threshold"] == (
            normalize_trajectory_threshold_config(_threshold())
        )
        reloaded = statistics._load_analysis_artifact(
            Path(cli_result["analysis_json"])
        )
        assert reloaded["right_censored_rmst"] == persisted[
            "right_censored_rmst"
        ]

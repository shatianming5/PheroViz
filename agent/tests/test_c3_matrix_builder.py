from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

import experiments.matrix as matrix_module
from experiments.c3_matrix_builder import (
    AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256,
    C3MatrixError,
    MEMORY_MODES,
    materialize_c3_matrix,
    validate_selected_case_payloads,
    verify_c3_matrix,
    write_spec_hash_manifest,
)
from experiments.matrix import load_and_expand_matrix
from experiments.production_statistics import (
    load_trajectory_threshold_config,
)
from experiments.c3_runtime_materializer import (
    C3RuntimeMaterializationError,
    materialize_runtime_c3_matrix,
    verify_runtime_c3_matrix,
)
from tests.test_experiment_matrix import _sealed_source_binding


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "nature_download/outputs/combined_verified_400/"
    "final_benchmark_renderable_v1_seed0/benchmark_manifest.json"
)
TRACKED_TEMPLATE = (
    REPO_ROOT
    / "agent/experiments/matrices/"
    "c3_memory_modes_final_benchmark_v2_seed0_br6_gpt56sol.yaml"
)


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".c3_matrix_test_work"
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


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_repo(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=C3 Test",
        "-c",
        "user.email=c3@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _synthetic_portable_repositories(
    workspace: Path,
) -> tuple[Path, Path, Path, str]:
    source_repo = workspace / "source-repo"
    source_repo.mkdir()
    (source_repo / ".gitignore").write_text(
        "agent/experiments/runs/\n",
        encoding="utf-8",
    )
    for relative in (
        "agent/experiments/c3_matrix_builder.py",
        "agent/experiments/c3_runtime_materializer.py",
        "agent/experiments/matrix.py",
        "agent/experiments/manifest.py",
        "agent/experiments/providers.py",
    ):
        target = source_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    candidate_ids = ["multi-p2", "multi-p3", "multi-p6"]
    source_binding, source_binding_hash, verifications = (
        _sealed_source_binding(source_repo, candidate_ids)
    )
    cases = []
    for candidate_id, panel_count, doi in (
        ("multi-p2", 2, "10.1234/p2"),
        ("multi-p3", 3, "10.1234/p3"),
        ("multi-p6", 6, "10.1234/p6"),
    ):
        data = source_repo / "nature_download/data" / f"{candidate_id}.csv"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("x,y\n0,1\n", encoding="utf-8")
        data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
        panel_ids = [
            f"panel-{index}" for index in range(1, panel_count + 1)
        ]
        cases.append(
            {
                "case_id": candidate_id,
                "candidate_id": candidate_id,
                "doi": doi,
                "panel_count": panel_count,
                "split": "test",
                "data_path": str(data),
                "data_sha256": data_sha256,
                "panels": [
                    {
                        "id": panel_id,
                        "data_path": str(data),
                        "data_sha256": data_sha256,
                        "user_goal": panel_id,
                        "chart_family": "line",
                        "intent": {"x": "x", "y": "y"},
                    }
                    for panel_id in panel_ids
                ],
                "evaluation_expectation": {
                    "schema_version": "1.1.0",
                    "panels": [
                        {
                            "panel_id": panel_id,
                            "axis_index": index,
                            "series": [
                                {
                                    "series_id": "y",
                                    "kind": "line",
                                    "x": "x",
                                    "y": "y",
                                }
                            ],
                        }
                        for index, panel_id in enumerate(panel_ids)
                    ],
                    "panel_groups": [],
                },
                "curation_status": "verified",
                "eligible_for_experiment": True,
                "verification_evidence": verifications[candidate_id],
            }
        )
    manifest = (
        source_repo
        / "nature_download/outputs/benchmark/benchmark_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provenance": {
                    "code_commit": "b" * 40,
                    "code_dirty": False,
                    "source_binding": source_binding,
                    "source_binding_hash": source_binding_hash,
                },
                "cases": cases,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _git(source_repo, "init", "-q")
    _commit_repo(source_repo, "source")
    template = (
        source_repo
        / "agent/experiments/matrices/c3-template.yaml"
    )
    materialize_c3_matrix(
        manifest_path=manifest,
        output_path=template,
        repo_root=source_repo,
        expected_manifest_sha256=manifest_sha256,
    )
    _commit_repo(source_repo, "template")

    runtime_repo = workspace / "runtime-repo"
    shutil.copytree(
        source_repo,
        runtime_repo,
        ignore=shutil.ignore_patterns(".git"),
    )
    _git(runtime_repo, "init", "-q")
    _commit_repo(runtime_repo, "runtime")
    runtime_template = (
        runtime_repo / template.relative_to(source_repo)
    )
    shutil.rmtree(source_repo)
    assert not source_repo.exists()
    return source_repo, runtime_repo, runtime_template, manifest_sha256


def test_materialized_matrix_has_exact_modes_cases_and_dry_run_count() -> None:
    with _workspace("exact") as workspace:
        path = workspace / "matrix.yaml"
        matrix = materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=path,
            repo_root=REPO_ROOT,
        )
        verified = verify_c3_matrix(path, repo_root=REPO_ROOT)
        specs = load_and_expand_matrix(path)

        assert verified["case_count"] == 3
        assert verified["doi_count"] == 3
        assert verified["run_count"] == 6 * 3 * 3 == 54
        assert len(specs) == len({spec.run_name for spec in specs}) == 54
        assert {
            item["name"]: item["memory_mode"]
            for item in matrix["methods"]
        } == dict(MEMORY_MODES)
        assert {spec.schedule for spec in specs} == {"iterative"}
        assert {
            spec.method_config["initial_generation"] for spec in specs
        } == {"model_spec"}
        assert {
            spec.method_config["render_timeout_seconds"] for spec in specs
        } == {120}
        assert {spec.backbone for spec in specs} == {"gpt-5.6-sol"}
        assert {spec.seed for spec in specs} == {0, 1, 2}
        assert {spec.budget_value for spec in specs} == {6.0}
        assert all(int(spec.budget_value) % int(spec.panel_count or 1) == 0 for spec in specs)
        assert matrix["materialization"]["outcome_inputs_read"] is False
        assert (
            matrix["dataset_manifest_sha256"]
            == AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
        )
        assert "trajectory_threshold" not in matrix


def test_tracked_template_is_hash_bound_and_superseded_parent_is_rejected() -> None:
    verified = verify_c3_matrix(TRACKED_TEMPLATE, repo_root=REPO_ROOT)
    assert verified["run_count"] == 54
    assert (
        hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        == AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
    )

    superseded_parent = (
        REPO_ROOT
        / "nature_download/outputs/combined_verified_400/"
        "final_benchmark_v2_seed0/benchmark_manifest.json"
    )
    with _workspace("superseded-parent") as workspace:
        with pytest.raises(C3MatrixError, match="authoritative renderable derivative"):
            materialize_c3_matrix(
                manifest_path=superseded_parent,
                output_path=workspace / "matrix.yaml",
                repo_root=REPO_ROOT,
            )

    with pytest.raises(C3MatrixError, match="manifest_data_root must be an absolute"):
        verify_c3_matrix(
            TRACKED_TEMPLATE,
            repo_root=REPO_ROOT,
            manifest_data_root=Path("relative/original-root"),
        )


def test_spec_hash_manifest_requires_and_binds_clean_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("spec-hashes") as workspace:
        matrix_path = workspace / "matrix.yaml"
        matrix = materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=matrix_path,
            repo_root=REPO_ROOT,
        )
        commit = matrix["materialization"]["repository_commit"]
        monkeypatch.setattr(
            matrix_module,
            "_git_provenance",
            lambda repo_root: (commit, False),
        )
        output = workspace / "specs.json"

        payload = write_spec_hash_manifest(
            matrix_path,
            output_path=output,
            repo_root=REPO_ROOT,
        )

        assert payload["source_commit"] == commit
        assert payload["source_dirty"] is False
        assert payload["run_count"] == 54
        assert len({item["run_name"] for item in payload["specs"]}) == 54
        assert len({item["spec_hash"] for item in payload["specs"]}) == 54
        unhashed = dict(payload)
        unhashed.pop("manifest_hash")
        assert payload["manifest_hash"] == matrix_module.sha256_json(unhashed)


def test_materialization_is_outcome_independent_and_deterministic() -> None:
    with _workspace("outcome-independent") as workspace:
        first = workspace / "first.yaml"
        second = workspace / "second.yaml"
        materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=first,
            repo_root=REPO_ROOT,
        )
        fake_run = workspace / "runs/fake/run_record.json"
        fake_run.parent.mkdir(parents=True)
        fake_run.write_text(
            '{"status":"completed","metric.data_fidelity":0.999}\n',
            encoding="utf-8",
        )
        materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=second,
            repo_root=REPO_ROOT,
        )

        assert first.read_bytes() == second.read_bytes()
        assert json.loads(
            json.dumps(yaml.safe_load(first.read_text()), sort_keys=True)
        ) == yaml.safe_load(second.read_text())


def test_runtime_materialization_relocates_sealed_paths_and_pairs_54_specs() -> None:
    with _workspace("runtime-relocation") as workspace:
        source_repo, runtime_repo, template, manifest_sha256 = (
            _synthetic_portable_repositories(workspace)
        )
        output_root = runtime_repo / "agent/experiments/runs/materialized/c3"
        result = materialize_runtime_c3_matrix(
            template_path=template,
            runtime_repo_root=runtime_repo,
            original_manifest_data_root=source_repo,
            output_root=output_root,
            expected_manifest_sha256=manifest_sha256,
        )
        assert not source_repo.exists()
        runtime_matrix = yaml.safe_load(
            Path(result["matrix_path"]).read_text(encoding="utf-8")
        )
        spec_manifest = json.loads(
            Path(result["spec_manifest_path"]).read_text(encoding="utf-8")
        )
        summary = json.loads(
            Path(result["summary_path"]).read_text(encoding="utf-8")
        )

        assert result["run_count"] == result["unique_run_names"] == 54
        assert runtime_matrix["repo_root"] == str(runtime_repo.resolve())
        assert runtime_matrix["provider_options"] == {
            "manifest_data_root": str(source_repo.resolve())
        }
        assert Path(runtime_matrix["dataset_manifest"]).is_absolute()
        assert Path(runtime_matrix["artifact_root"]).is_absolute()
        assert spec_manifest["runtime_clean"] is True
        assert spec_manifest["run_count"] == 54
        assert summary["runtime_clean"] is True
        assert summary["run_count"] == 54
        unhashed_summary = dict(summary)
        summary_hash = unhashed_summary.pop("summary_hash")
        assert summary_hash == matrix_module.sha256_json(unhashed_summary)
        assert (
            summary["runtime_spec_manifest_sha256"]
            == hashlib.sha256(
                Path(result["spec_manifest_path"]).read_bytes()
            ).hexdigest()
        )
        pairings = {
            (item["case_id"], item["seed"]): set()
            for item in spec_manifest["specs"]
        }
        for item in spec_manifest["specs"]:
            pairings[(item["case_id"], item["seed"])].add(item["method"])
        assert set(map(frozenset, pairings.values())) == {
            frozenset(name for name, _ in MEMORY_MODES)
        }
        assert not _git(runtime_repo, "status", "--porcelain")


@pytest.mark.parametrize(
    "original_root",
    [
        Path("relative/original-root"),
        Path("/recorded/source/../source"),
    ],
)
def test_runtime_materialization_rejects_nonabsolute_or_unnormalized_prefix(
    original_root: Path,
) -> None:
    with _workspace("runtime-invalid-prefix") as workspace:
        _, runtime_repo, template, manifest_sha256 = (
            _synthetic_portable_repositories(workspace)
        )
        output_root = runtime_repo / "agent/experiments/runs/materialized/c3"

        with pytest.raises(
            C3RuntimeMaterializationError,
            match="original_manifest_data_root",
        ):
            materialize_runtime_c3_matrix(
                template_path=template,
                runtime_repo_root=runtime_repo,
                original_manifest_data_root=original_root,
                output_root=output_root,
                expected_manifest_sha256=manifest_sha256,
            )


@pytest.mark.parametrize(
    "mutation",
    ["manifest_data_root", "runtime_manifest_hash", "template_hash"],
)
def test_runtime_materialization_tampering_fails_closed(
    mutation: str,
) -> None:
    with _workspace(f"runtime-tamper-{mutation}") as workspace:
        source_repo, runtime_repo, template, manifest_sha256 = (
            _synthetic_portable_repositories(workspace)
        )
        output_root = runtime_repo / "agent/experiments/runs/materialized/c3"
        result = materialize_runtime_c3_matrix(
            template_path=template,
            runtime_repo_root=runtime_repo,
            original_manifest_data_root=source_repo,
            output_root=output_root,
            expected_manifest_sha256=manifest_sha256,
        )
        path = Path(result["matrix_path"])
        runtime = yaml.safe_load(path.read_text(encoding="utf-8"))
        if mutation == "manifest_data_root":
            runtime["provider_options"]["manifest_data_root"] = "/wrong/root"
        elif mutation == "runtime_manifest_hash":
            runtime["runtime_materialization"][
                "runtime_manifest_sha256"
            ] = "0" * 64
        else:
            runtime["runtime_materialization"][
                "portable_template_sha256"
            ] = "0" * 64
        path.write_text(
            yaml.safe_dump(runtime, sort_keys=False),
            encoding="utf-8",
        )

        with pytest.raises(C3RuntimeMaterializationError):
            verify_runtime_c3_matrix(
                path,
                template_path=template,
                runtime_repo_root=runtime_repo,
                original_manifest_data_root=source_repo,
                expected_manifest_sha256=manifest_sha256,
            )


def test_matrix_contains_no_absolute_paths_or_secret_fields() -> None:
    with _workspace("portable") as workspace:
        path = workspace / "matrix.yaml"
        matrix = materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=path,
            repo_root=REPO_ROOT,
        )
        for key, value in _walk(matrix):
            lowered = key.casefold()
            assert not any(
                token in lowered
                for token in ("api_key", "apikey", "password", "secret", "auth_token")
            )
            if isinstance(value, str):
                assert not value.startswith("/")
                assert not (
                    len(value) > 2
                    and value[1] == ":"
                    and value[2] in {"\\", "/"}
                )


@pytest.mark.parametrize(
    "mutation",
    [
        "case_set",
        "mode",
        "manifest_hash",
        "builder_hash",
        "absolute_path",
        "secret",
    ],
)
def test_matrix_tampering_fails_closed(mutation: str) -> None:
    with _workspace(f"tamper-{mutation}") as workspace:
        path = workspace / "matrix.yaml"
        materialize_c3_matrix(
            manifest_path=MANIFEST,
            output_path=path,
            repo_root=REPO_ROOT,
        )
        matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
        if mutation == "case_set":
            matrix["case_ids"].pop()
        elif mutation == "mode":
            matrix["methods"][0]["memory_mode"] = "full"
        elif mutation == "manifest_hash":
            matrix["materialization"][
                "parent_manifest_semantic_sha256"
            ] = "0" * 64
        elif mutation == "builder_hash":
            matrix["materialization"]["builder_sha256"] = "0" * 64
        elif mutation == "absolute_path":
            matrix["artifact_root"] = "/forbidden/absolute/root"
        else:
            matrix["provider_options"] = {"api_key": "forbidden"}
        path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

        with pytest.raises(C3MatrixError):
            verify_c3_matrix(path, repo_root=REPO_ROOT)


def test_manifest_tampering_and_selection_requirements_fail_closed() -> None:
    with _workspace("manifest-tamper") as workspace:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["cases"][0]["panel_count"] = 5
        tampered = workspace / "manifest.json"
        tampered.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        with pytest.raises(C3MatrixError):
            materialize_c3_matrix(
                manifest_path=tampered,
                output_path=workspace / "matrix.yaml",
                repo_root=REPO_ROOT,
            )

    with pytest.raises(C3MatrixError, match="does not divide"):
        validate_selected_case_payloads(
            [
                {
                    "case_id": "case-a",
                    "doi": "10.1234/a",
                    "panel_count": 5,
                    "split": "test",
                },
                {
                    "case_id": "case-b",
                    "doi": "10.1234/b",
                    "panel_count": 2,
                    "split": "test",
                },
            ]
        )
    with pytest.raises(C3MatrixError, match="two independent DOI"):
        validate_selected_case_payloads(
            [
                {
                    "case_id": "case-a",
                    "doi": "10.1234/a",
                    "panel_count": 2,
                    "split": "test",
                },
                {
                    "case_id": "case-b",
                    "doi": "10.1234/a",
                    "panel_count": 3,
                    "split": "test",
                },
                {
                    "case_id": "case-c",
                    "doi": "10.1234/a",
                    "panel_count": 6,
                    "split": "test",
                },
            ]
        )


def test_render_threshold_evidence_is_prospective_and_hash_bound() -> None:
    quality_path = (
        REPO_ROOT
        / "agent/experiments/thresholds/c3_joint_quality_threshold_v1.json"
    )
    status_path = (
        REPO_ROOT
        / "agent/experiments/thresholds/c3_joint_threshold_status_v1.json"
    )
    report_path = (
        REPO_ROOT
        / "agent/experiments/thresholds/c3_joint_threshold_provenance_v1.json"
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == report_hash
    assert quality["fidelity"]["threshold"] == 1.0
    assert quality["cohesion"]["threshold"] == 1.0
    assert quality["restriction"] == {"renders": 6}
    assert status["status"] == "RENDER_ONLY_READY_WALL_CLOCK_BLOCKED"
    assert status["analysis_ready"] is True
    assert status["render_analysis_ready"] is True
    assert status["wall_clock_analysis_ready"] is False
    assert status["wall_clock_seconds"] is None
    assert status["missing_analysis_field"] == "restriction.wall_clock_seconds"
    assert report["production_or_partial_outcomes_read"] is False
    assert (
        report["quality_threshold_file_sha256"]
        == hashlib.sha256(quality_path.read_bytes()).hexdigest()
    )
    assert report["execution_policy"] == {
        "matrix_can_run": True,
        "render_only_analysis_can_run": True,
        "wall_clock_analysis_can_run": False,
        "rmst_analysis_can_run": False,
        "required_unblock": status["required_unblock"],
    }
    assert load_trajectory_threshold_config(quality_path) == quality

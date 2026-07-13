from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
from jsonschema import validate

import experiments.decision_report as decision_report
import experiments.production_statistics as production_statistics
from experiments.cli import main
from experiments.decision_report import DecisionReportError, build_decision_report
from experiments.models import SCHEMA_VERSION, sha256_json, write_json_atomic
from experiments.production_statistics import (
    analyze_summary,
    build_holm_family,
    load_provenance_summary,
)


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".decision_report_test_work"
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _patch_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        production_statistics,
        "_git_provenance",
        lambda: ("9" * 40, False),
    )
    monkeypatch.setattr(
        decision_report,
        "_git_provenance",
        lambda: ("d" * 40, False),
    )


def _build_inputs(
    workspace: Path,
    *,
    doi_count: int = 8,
    mid_cohesion_gap: float = 0.30,
    cross_zero_ci: bool = False,
    include_best_analysis: bool = True,
    render_budget: float = 6.0,
    experiment_commits: dict[str, str] | None = None,
) -> tuple[Path, list[Path]]:
    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    analysis_paths: dict[tuple[str, str], Path] = {}
    for backbone in ("frontier", "mid", "open"):
        rows: list[dict[str, Any]] = []
        for case_index in range(doi_count):
            case_id = f"case-{case_index}"
            for method in ("flat_iterative", "best_of_n", "pheroviz_full"):
                fidelity_gap = {
                    "flat_iterative": 0.0,
                    "best_of_n": 0.015,
                    "pheroviz_full": 0.03,
                }[method]
                cohesion_gap = {
                    "flat_iterative": 0.0,
                    "best_of_n": 0.20,
                    "pheroviz_full": (
                        mid_cohesion_gap if backbone == "mid" else 0.30
                    ),
                }[method]
                if (
                    cross_zero_ci
                    and backbone == "mid"
                    and method == "pheroviz_full"
                ):
                    fidelity_gap = 0.10 if case_index < 7 else -0.40
                run_name = f"{backbone}-{method}-{case_id}-7"
                rows.append(
                    {
                        "run_name": run_name,
                        "method": method,
                        "backbone": backbone,
                        "case_id": case_id,
                        "doi": f"10.1234/{case_id}",
                        "panel_count": 2,
                        "split": "test",
                        "seed": 7,
                        "budget_type": "renders",
                        "budget_value": render_budget,
                        "dataset_manifest_hash": "a" * 64,
                        "git_commit": (experiment_commits or {}).get(
                            backbone,
                            decision_report.D90_EXPERIMENT_COMMIT,
                        ),
                        "started_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:01:00Z",
                        "status": "completed",
                        "execution_success": 1.0,
                        "failure_attribution": "",
                        "failure_type": "",
                        "test_only": False,
                        "render_count": int(render_budget),
                        "wall_clock_seconds": 1.0,
                        "metric_version": "metrics-v1",
                        "metric_config_hash": "b" * 64,
                        "provider": "provider",
                        "provider_name": "provider",
                        "best_candidate_id": "candidate_0001",
                        "spec_hash": _digest(f"spec-{run_name}"),
                        "record_hash": _digest(f"record-{run_name}"),
                        "metric.data_fidelity": 0.50 + fidelity_gap,
                        "metric.series_cohesion": 1.00 + cohesion_gap,
                        "metric.execution_success": 1.0,
                    }
                )
        summary_payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-01-01T00:02:00Z",
            "source_root": f"synthetic-{backbone}",
            "run_count": len(rows),
            "columns": sorted({key for row in rows for key in row}),
            "runs": rows,
            "input_record_hashes": {
                row["run_name"]: row["record_hash"] for row in rows
            },
        }
        summary_payload["summary_hash"] = sha256_json(summary_payload)
        summary_path = workspace / f"summary-{backbone}.json"
        write_json_atomic(summary_path, summary_payload)
        summary = load_provenance_summary(summary_path)

        for label, metric, panel_scope in (
            ("fidelity", "metric.data_fidelity", "all"),
            ("cohesion", "metric.series_cohesion", "multi_panel"),
        ):
            analysis = analyze_summary(
                summary,
                reference="flat_iterative",
                methods=(
                    ["best_of_n", "pheroviz_full"]
                    if include_best_analysis
                    else ["pheroviz_full"]
                ),
                metric=metric,
                panel_scope=panel_scope,
                bootstrap_resamples=200,
                monte_carlo_permutations=100,
            )
            analysis["code_git_dirty"] = False
            analysis.pop("analysis_hash")
            analysis["analysis_hash"] = sha256_json(analysis)
            path = workspace / f"{backbone}-{label}-analysis.json"
            write_json_atomic(path, analysis)
            analyses[(backbone, label)] = analysis
            analysis_paths[(backbone, label)] = path

    members = []
    for backbone in ("frontier", "mid", "open"):
        for label, metric, panel_scope in (
            ("fidelity", "metric.data_fidelity", "all"),
            ("cohesion", "metric.series_cohesion", "multi_panel"),
        ):
            members.append(
                {
                    "name": f"{backbone}.{label}",
                    "analysis_path": analysis_paths[(backbone, label)].name,
                    "analysis_hash": analyses[(backbone, label)]["analysis_hash"],
                    "backbone": backbone,
                    "budget_type": "renders",
                    "budget_value": render_budget,
                    "split": "test",
                    "reference": "flat_iterative",
                    "method": "pheroviz_full",
                    "metric": metric,
                    "panel_scope": panel_scope,
                }
            )
    manifest_path = workspace / "family-manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": "1.0",
            "family_name": "c4-six-member-family",
            "members": members,
        },
    )
    family = build_holm_family(manifest_path)
    family_path = workspace / "holm-family.json"
    write_json_atomic(family_path, family)
    return family_path, list(analysis_paths.values())


def test_report_passes_all_six_and_validates_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean(monkeypatch)
    with _workspace("pass") as workspace:
        family_path, analyses = _build_inputs(workspace)
        report = build_decision_report(analyses, family_path)

        assert report["c4"]["status"] == "pass"
        assert report["c4"]["cross_tier_claim_permitted"] is True
        assert len(report["c4"]["cells"]) == 6
        assert len(report["provenance"]["analysis_artifacts"]) == 6
        assert len(report["provenance"]["input_summary_hashes"]) == 3
        assert len(report["provenance"]["tier_summary_hashes"]) == 3
        assert set(
            report["provenance"]["tier_experiment_git_commits"].values()
        ) == {decision_report.D90_EXPERIMENT_COMMIT}
        assert all(cell["status"] == "pass" for cell in report["c4"]["cells"])
        assert {cell["practical_margin"] for cell in report["c4"]["cells"]} == {
            0.02,
            0.25,
        }
        assert len(report["c1"]["comparisons"]) == 12
        assert {
            row["method"] for row in report["c1"]["comparisons"]
        } == {"best_of_n", "pheroviz_full"}
        assert report["c1"]["wall_clock"]["status"] == "unavailable"
        assert report["c1"]["wall_clock"]["claim_permitted"] is False
        assert report["decision_report_hash"] == sha256_json(
            {
                key: value
                for key, value in report.items()
                if key != "decision_report_hash"
            }
        )

        schema_path = (
            Path(decision_report.__file__).resolve().parent
            / "schemas/c1_c4_decision_report.schema.json"
        )
        validate(
            report,
            json.loads(schema_path.read_text(encoding="utf-8")),
        )


def test_report_uses_tier_fallback_for_each_failed_criterion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean(monkeypatch)
    with _workspace("margin") as workspace:
        family_path, analyses = _build_inputs(
            workspace,
            mid_cohesion_gap=0.20,
        )
        report = build_decision_report(analyses, family_path)
        failed = [
            cell
            for cell in report["c4"]["cells"]
            if cell["backbone"] == "mid"
            and cell["metric"] == "metric.series_cohesion"
        ][0]
        assert failed["passes_practical_margin"] is False
        assert report["c4"]["status"] == "tier_scoped_fallback"
        assert {
            row["backbone"]: row["status"]
            for row in report["c4"]["tier_decisions"]
        }["mid"] == "report_effects_only"

    with _workspace("holm") as workspace:
        family_path, analyses = _build_inputs(workspace, doi_count=7)
        report = build_decision_report(analyses, family_path)
        assert all(
            cell["passes_practical_margin"]
            and cell["passes_positive_ci"]
            and not cell["passes_cross_family_holm"]
            for cell in report["c4"]["cells"]
        )
        assert report["c4"]["status"] == "tier_scoped_fallback"

    with _workspace("ci") as workspace:
        family_path, analyses = _build_inputs(workspace, cross_zero_ci=True)
        report = build_decision_report(analyses, family_path)
        failed = [
            cell
            for cell in report["c4"]["cells"]
            if cell["backbone"] == "mid"
            and cell["metric"] == "metric.data_fidelity"
        ][0]
        assert failed["passes_practical_margin"] is True
        assert failed["passes_positive_ci"] is False
        assert failed["status"] == "fail"


def test_report_fails_closed_on_duplicate_missing_forged_mixed_or_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean(monkeypatch)
    with _workspace("fail-closed") as workspace:
        family_path, analyses = _build_inputs(workspace)

        with pytest.raises(DecisionReportError, match="Duplicate analysis"):
            build_decision_report([*analyses, analyses[0]], family_path)
        with pytest.raises(
            DecisionReportError,
            match="exactly six",
        ):
            build_decision_report(analyses[:1], family_path)

        family = json.loads(family_path.read_text(encoding="utf-8"))
        dirty_family = dict(family)
        dirty_family["code_git_dirty"] = True
        dirty_family.pop("family_hash")
        dirty_family["family_hash"] = sha256_json(dirty_family)
        dirty_family_path = workspace / "dirty-family.json"
        write_json_atomic(dirty_family_path, dirty_family)
        with pytest.raises(DecisionReportError, match="dirty Holm family"):
            build_decision_report(analyses, dirty_family_path)

        family["members"][0]["adjusted_p_value"] = 0.99
        family.pop("family_hash")
        family["family_hash"] = sha256_json(family)
        forged_family = workspace / "forged-family.json"
        write_json_atomic(forged_family, family)
        with pytest.raises(DecisionReportError, match="recomputation"):
            build_decision_report(analyses, forged_family)

        analysis = json.loads(analyses[0].read_text(encoding="utf-8"))
        analysis["slices"][0]["comparisons"][0]["overall"]["mean_gap"] += 1.0
        analysis.pop("analysis_hash")
        analysis["analysis_hash"] = sha256_json(analysis)
        write_json_atomic(analyses[0], analysis)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="recomputation",
        ):
            build_decision_report(analyses, family_path)

    with _workspace("stale-summary") as workspace:
        family_path, analyses = _build_inputs(workspace)
        summary_path = workspace / "summary-frontier.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["runs"][0]["metric.data_fidelity"] += 0.01
        summary.pop("summary_hash")
        summary["summary_hash"] = sha256_json(summary)
        write_json_atomic(summary_path, summary)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="input_summary_hash disagrees",
        ):
            build_decision_report(analyses, family_path)

    with _workspace("wrong-contract") as workspace:
        family_path, analyses = _build_inputs(
            workspace,
            include_best_analysis=False,
        )
        with pytest.raises(DecisionReportError, match="frozen .* contract"):
            build_decision_report(analyses, family_path)

    with _workspace("wrong-budget") as workspace:
        family_path, analyses = _build_inputs(workspace, render_budget=3.0)
        with pytest.raises(DecisionReportError, match="budget B_R=6"):
            build_decision_report(analyses, family_path)

    with _workspace("mixed") as workspace:
        family_path, analyses = _build_inputs(workspace)
        analysis = json.loads(analyses[0].read_text(encoding="utf-8"))
        analysis["code_git_commit"] = "e" * 40
        analysis.pop("analysis_hash")
        analysis["analysis_hash"] = sha256_json(analysis)
        write_json_atomic(analyses[0], analysis)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="analysis_hash mismatch",
        ):
            build_decision_report(analyses, family_path)

    with _workspace("mixed-experiment-commits") as workspace:
        with pytest.raises(
            production_statistics.StatisticsError,
            match="mix experiment provenance",
        ):
            _build_inputs(
                workspace,
                experiment_commits={"open": "e" * 40},
            )

    monkeypatch.setattr(
        decision_report,
        "_git_provenance",
        lambda: ("d" * 40, True),
    )
    with pytest.raises(DecisionReportError, match="clean worktree"):
        build_decision_report([Path("unused.json")], Path("unused-family.json"))


def test_family_rejects_mismatched_tier_summaries_and_duplicate_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean(monkeypatch)
    with _workspace("tier-summary-mismatch") as workspace:
        _, _ = _build_inputs(workspace)
        original_summary_path = workspace / "summary-mid.json"
        alternate_summary = json.loads(
            original_summary_path.read_text(encoding="utf-8")
        )
        alternate_summary["source_root"] = "synthetic-mid-alternate"
        alternate_summary.pop("summary_hash")
        alternate_summary["summary_hash"] = sha256_json(alternate_summary)
        alternate_summary_path = workspace / "summary-mid-alternate.json"
        write_json_atomic(alternate_summary_path, alternate_summary)

        analysis_path = workspace / "mid-cohesion-analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["input_summary_path"] = str(alternate_summary_path.resolve())
        analysis["input_summary_hash"] = alternate_summary["summary_hash"]
        analysis.pop("analysis_hash")
        analysis["analysis_hash"] = sha256_json(analysis)
        write_json_atomic(analysis_path, analysis)

        manifest_path = workspace / "family-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        member = next(
            value
            for value in manifest["members"]
            if value["name"] == "mid.cohesion"
        )
        member["analysis_hash"] = analysis["analysis_hash"]
        write_json_atomic(manifest_path, manifest)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="one backbone with multiple input summaries",
        ):
            build_holm_family(manifest_path)

    with _workspace("duplicate-metric") as workspace:
        _, _ = _build_inputs(workspace)
        manifest_path = workspace / "family-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fidelity = json.loads(
            (workspace / "mid-fidelity-analysis.json").read_text(
                encoding="utf-8"
            )
        )
        member = next(
            value
            for value in manifest["members"]
            if value["name"] == "mid.cohesion"
        )
        member.update(
            {
                "analysis_path": "mid-fidelity-analysis.json",
                "analysis_hash": fidelity["analysis_hash"],
                "metric": "metric.data_fidelity",
                "panel_scope": "all",
            }
        )
        write_json_atomic(manifest_path, manifest)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="Duplicate Holm hypothesis selector",
        ):
            build_holm_family(manifest_path)

    with _workspace("mixed-analysis-commits") as workspace:
        _, _ = _build_inputs(workspace)
        analysis_path = workspace / "open-cohesion-analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["code_git_commit"] = "e" * 40
        analysis.pop("analysis_hash")
        analysis["analysis_hash"] = sha256_json(analysis)
        write_json_atomic(analysis_path, analysis)
        manifest_path = workspace / "family-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        member = next(
            value
            for value in manifest["members"]
            if value["name"] == "open.cohesion"
        )
        member["analysis_hash"] = analysis["analysis_hash"]
        write_json_atomic(manifest_path, manifest)
        with pytest.raises(
            production_statistics.StatisticsError,
            match="mixes analysis code commits",
        ):
            build_holm_family(manifest_path)


def test_decision_report_cli_writes_hash_bound_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean(monkeypatch)
    with _workspace("cli") as workspace:
        family_path, analyses = _build_inputs(workspace)
        out = workspace / "decision.json"
        exit_code = main(
            [
                "decision-report",
                str(family_path),
                *(str(path) for path in analyses),
                "--out",
                str(out),
            ]
        )
        assert exit_code == 0
        response = json.loads(capsys.readouterr().out)
        persisted = json.loads(out.read_text(encoding="utf-8"))
        assert response["decision_report"] == str(out.resolve())
        assert response["decision_report_hash"] == persisted[
            "decision_report_hash"
        ]
        assert response["c4_status"] == "pass"

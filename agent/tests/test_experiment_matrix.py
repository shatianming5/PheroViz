from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.cli import main
from experiments.matrix import MatrixError, load_and_expand_matrix
from tests.test_experiment_support import experiment_workspace


def _minimal_matrix(manifest: Path, artifact_root: Path) -> dict:
    return {
        "experiment_name": "case-protocol",
        "dataset_manifest": str(manifest),
        "artifact_root": str(artifact_root),
        "provider": "",
        "methods": ["iterative"],
        "backbones": ["model-a"],
        "seeds": [1],
        "budgets": [{"type": "renders", "value": 1}],
        "metric_version": "v1",
        "metric_config": {
            "selection": {
                "metric": "score",
                "direction": "maximize",
            }
        },
    }


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_matrix_expands_method_backbone_seed_budget_case_product(
    suffix: str,
) -> None:
    with experiment_workspace("matrix") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": "paper-A",
                            "panel_count": 1,
                            "split": "test",
                        },
                        {
                            "case_id": "paper/B",
                            "panel_count": 4,
                            "split": "train",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        artifact_root = workspace / "experiment_runs"
        matrix = {
            "experiment_name": "claim-c1",
            "dataset_manifest": str(manifest),
            "artifact_root": str(artifact_root),
            "provider": "experiments.providers:SingleChainProvider",
            "methods": [
                {"name": "best_of_n", "schedule": "best_of_n"},
                {"name": "phero_viz", "schedule": "iterative"},
            ],
            "backbones": ["model-a", "model-b"],
            "seeds": [11, 29],
            "budgets": [
                {"type": "renders", "value": 3},
                {"type": "wall_clock_seconds", "value": 10},
            ],
            "metric": {
                "version": "metrics-v1",
                "config": {
                    "selection": {
                        "metric": "quality",
                        "direction": "maximize",
                    }
                },
            },
        }
        spec_path = workspace / f"matrix{suffix}"
        if suffix == ".json":
            spec_path.write_text(json.dumps(matrix), encoding="utf-8")
        else:
            spec_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")

        specs = load_and_expand_matrix(spec_path)

        assert len(specs) == 32
        assert len({spec.run_name for spec in specs}) == 32
        for spec in specs:
            assert f"case-{spec.case_id.replace('/', '-')}" in spec.run_name
            assert f"method-{spec.method}" in spec.run_name
            assert f"backbone-{spec.backbone}" in spec.run_name
            assert f"seed-{spec.seed}" in spec.run_name
            assert "budget-" in spec.run_name
            assert len(spec.dataset_manifest_hash) == 64
            assert len(spec.metric_config_hash) == 64
        assert {
            (spec.case_id, spec.panel_count, spec.split) for spec in specs
        } == {
            ("paper-A", 1, "test"),
            ("paper/B", 4, "train"),
        }


def test_dry_run_does_not_create_artifact_root(capsys: pytest.CaptureFixture[str]) -> None:
    with experiment_workspace("dry-run") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps({"cases": [{"case_id": "dry-case"}]}),
            encoding="utf-8",
        )
        artifact_root = workspace / "must-not-exist"
        matrix = {
            "experiment_name": "dry-run",
            "dataset_manifest": str(manifest),
            "artifact_root": str(artifact_root),
            "provider": "",
            "methods": ["iterative"],
            "backbones": ["model-a"],
            "seeds": [1],
            "budgets": [{"type": "renders", "value": 1}],
            "metric_version": "v1",
            "metric_config": {
                "selection": {
                    "metric": "score",
                    "direction": "maximize",
                }
            },
        }
        spec_path = workspace / "matrix.json"
        spec_path.write_text(json.dumps(matrix), encoding="utf-8")

        assert main(["run", str(spec_path), "--dry-run"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 1
        assert "spec_hash" in payload[0]
        assert payload[0]["case_id"] == "dry-case"
        assert not artifact_root.exists()


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"case_ids": ["case-2"]}, {"case-2"}),
        ({"splits": ["test"]}, {"case-1", "case-3"}),
        (
            {"case_ids": ["case-1", "case-2"], "splits": ["test"]},
            {"case-1"},
        ),
    ],
)
def test_matrix_case_filters(
    filters: dict,
    expected: set[str],
) -> None:
    with experiment_workspace("filters") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {"case_id": "case-1", "split": "test"},
                        {"case_id": "case-2", "split": "train"},
                        {"case_id": "case-3", "split": "test"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix.update(filters)
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")

        specs = load_and_expand_matrix(path)

        assert {spec.case_id for spec in specs} == expected


@pytest.mark.parametrize(
    "manifest_payload",
    [
        {},
        {"cases": []},
        {"cases": [{"split": "test"}]},
        {"cases": [{"case_id": "dup"}, {"case_id": "dup"}]},
        {"cases": [{"case_id": "bad-panels", "panel_count": 0}]},
    ],
)
def test_invalid_or_legacy_manifests_fail_closed(
    manifest_payload: dict,
) -> None:
    with experiment_workspace("invalid-manifest") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        matrix = _minimal_matrix(manifest, workspace / "runs")
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")

        with pytest.raises(MatrixError):
            load_and_expand_matrix(path)


def test_combined_case_filters_cannot_select_empty_set() -> None:
    with experiment_workspace("empty-filter") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {"case_id": "case-1", "split": "test"},
                        {"case_id": "case-2", "split": "train"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix.update({"case_ids": ["case-1"], "splits": ["train"]})
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")

        with pytest.raises(MatrixError, match="selected no dataset cases"):
            load_and_expand_matrix(path)

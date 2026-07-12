from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.cli import main
from experiments.matrix import load_and_expand_matrix
from tests.test_experiment_support import experiment_workspace


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_matrix_expands_method_backbone_seed_budget_product(
    suffix: str,
) -> None:
    with experiment_workspace("matrix") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")
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

        assert len(specs) == 16
        assert len({spec.run_name for spec in specs}) == 16
        for spec in specs:
            assert f"method-{spec.method}" in spec.run_name
            assert f"backbone-{spec.backbone}" in spec.run_name
            assert f"seed-{spec.seed}" in spec.run_name
            assert "budget-" in spec.run_name
            assert len(spec.dataset_manifest_hash) == 64
            assert len(spec.metric_config_hash) == 64


def test_dry_run_does_not_create_artifact_root(capsys: pytest.CaptureFixture[str]) -> None:
    with experiment_workspace("dry-run") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")
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
        assert not artifact_root.exists()

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from experiments.cli import main
from experiments.matrix import (
    MatrixError,
    expand_matrix,
    load_and_expand_matrix,
)
from experiments.providers import (
    PHEROVIZ_PROVIDER_IMPORT_PATH,
    UnifiedBenchmarkProvider,
    load_provider,
)
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


def _seal(value: dict, field: str) -> dict:
    sealed = dict(value)
    sealed[field] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return sealed


def _sealed_source_binding(
    workspace: Path,
    candidate_ids: list[str],
) -> tuple[dict, str, dict[str, dict]]:
    proposed_path = workspace / "proposed.jsonl"
    proposed_path.write_text(
        "".join(
            json.dumps({"candidate_id": candidate_id}) + "\n"
            for candidate_id in candidate_ids
        ),
        encoding="utf-8",
    )
    proposed_hash = hashlib.sha256(proposed_path.read_bytes()).hexdigest()
    reviews = []
    verifications = []
    verification_summaries = {}
    for candidate_id in candidate_ids:
        review = _seal(
            {
                "candidate_id": candidate_id,
                "proposal_type": "single_panel",
                "status": "accepted",
                "rejection_reasons": [],
                "binding": {},
                "model_reviews": [],
            },
            "review_hash",
        )
        reviews.append(review)
        verification = {
            "candidate_id": candidate_id,
            "status": "verified",
            "curation_status": "verified",
            "evidence_type": "external_validation",
            "evidence_ref": f"review:{review['review_hash']}",
            "reviewer_or_source": "judge-a+judge-b",
            "review_hash": review["review_hash"],
            "review_models": ["judge-a", "judge-b"],
            "experiment_case": {},
        }
        verifications.append(verification)
        verification_summaries[candidate_id] = {
            key: verification[key]
            for key in (
                "evidence_type",
                "evidence_ref",
                "review_hash",
                "review_models",
            )
        }
    reviews_path = workspace / "reviews.jsonl"
    reviews_path.write_text(
        "".join(json.dumps(review, sort_keys=True) + "\n" for review in reviews),
        encoding="utf-8",
    )
    evidence = _seal(
        {
            "schema_version": "1.0",
            "evidence_type": "external_validation",
            "human_claims": 0,
            "input_proposed_sha256": proposed_hash,
            "rubric_hash": "a" * 64,
            "code_commit": "b" * 40,
            "code_dirty": False,
            "judge_models": ["judge-a", "judge-b"],
            "verifications": verifications,
        },
        "evidence_hash",
    )
    evidence_path = workspace / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True),
        encoding="utf-8",
    )
    review_summary = _seal(
        {
            "code_dirty": False,
            "evidence_hash": evidence["evidence_hash"],
            "review_hashes": {
                review["candidate_id"]: review["review_hash"]
                for review in reviews
            },
        },
        "summary_hash",
    )
    review_summary_path = workspace / "review_summary.json"
    review_summary_path.write_text(
        json.dumps(review_summary, sort_keys=True),
        encoding="utf-8",
    )
    candidates_path = workspace / "candidates.jsonl"
    candidates_path.write_text(
        "".join(
            json.dumps(
                {
                    "candidate_id": verification["candidate_id"],
                    "curation_status": "verified",
                    "eligible_for_experiment": True,
                    "verification_evidence": verification,
                },
                sort_keys=True,
            )
            + "\n"
            for verification in verifications
        ),
        encoding="utf-8",
    )
    corpus_path = workspace / "corpus.jsonl"
    corpus_path.write_text('{"download_eligible":true}\n', encoding="utf-8")
    candidates_hash = hashlib.sha256(candidates_path.read_bytes()).hexdigest()
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    case_summary = _seal(
        {
            "candidates_sha256": candidates_hash,
            "corpus_manifest_sha256": corpus_hash,
            "code_dirty": False,
            "eligible_for_experiment": len(candidate_ids),
        },
        "summary_hash",
    )
    case_summary_path = workspace / "case_summary.json"
    case_summary_path.write_text(
        json.dumps(case_summary, sort_keys=True),
        encoding="utf-8",
    )
    binding = {
        "candidate_inputs": [
            {
                "candidates_path": str(candidates_path),
                "candidates_sha256": candidates_hash,
                "summary_path": str(case_summary_path),
                "summary_sha256": hashlib.sha256(
                    case_summary_path.read_bytes()
                ).hexdigest(),
                "summary_hash": case_summary["summary_hash"],
                "code_commit": "b" * 40,
                "corpus_manifest": str(corpus_path),
                "corpus_manifest_sha256": corpus_hash,
                "content_root": str(workspace),
                "output_root": str(workspace),
            }
        ],
        "evidence": {
            "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "evidence_hash": evidence["evidence_hash"],
        },
        "proposed": {
            "path": str(proposed_path),
            "sha256": proposed_hash,
        },
        "reviews": {
            "path": str(reviews_path),
            "sha256": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
            "summary_path": str(review_summary_path),
            "summary_sha256": hashlib.sha256(
                review_summary_path.read_bytes()
            ).hexdigest(),
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return binding, digest, verification_summaries


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
                {"type": "renders", "value": 4},
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


def test_render_budget_preflight_rejects_partial_panel_checkpoint() -> None:
    with experiment_workspace("matrix-panel-budget") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": "single",
                            "panel_count": 1,
                            "split": "test",
                        },
                        {
                            "case_id": "triple",
                            "panel_count": 3,
                            "split": "test",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        artifact_root = workspace / "must-not-exist"
        matrix = _minimal_matrix(manifest, artifact_root)
        matrix["budgets"] = [{"type": "renders", "value": 4}]
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")

        with pytest.raises(
            MatrixError,
            match=r"triple\(P=3\)",
        ):
            load_and_expand_matrix(path)

        assert not artifact_root.exists()


@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf")])
def test_budget_preflight_rejects_nonpositive_or_fractional_render_budget(
    value: float,
) -> None:
    with experiment_workspace("matrix-invalid-budget") as workspace:
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": "single",
                            "panel_count": 1,
                            "split": "test",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix["budgets"] = [{"type": "renders", "value": value}]

        with pytest.raises(MatrixError):
            expand_matrix(matrix, base_dir=workspace)


def test_production_c1_c3_matrix_contract_expands_cleanly() -> None:
    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "matrices"
        / "c1_c3_final_benchmark_v2_seed0.yaml"
    )
    production = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    artifact_root = (
        matrix_path.parent / production["artifact_root"]
    ).resolve()

    assert production["dataset_mode"] == "sealed_benchmark"
    assert production["dataset_manifest"].endswith(
        "final_benchmark_v2_seed0/benchmark_manifest.json"
    )
    assert production["dataset_manifest_sha256"] == (
        "6059edf04d9b2c142af74f561fc0ed29b1b00d85068e32bdb35943d61d36a66b"
    )
    assert production["splits"] == ["test"]
    assert production["backbones"] == ["gpt-5.6-sol"]
    assert production["seeds"] == [0, 1, 2]
    assert production["budgets"] == [{"type": "renders", "value": 6}]
    assert artifact_root == (
        matrix_path.parents[1]
        / "runs"
        / "production"
        / "c1_c3_final_benchmark_v2_seed0_br6_gpt56sol"
    )
    assert subprocess.run(
        ["git", "check-ignore", "--quiet", str(artifact_root)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    ).returncode == 0
    assert (
        production["provider"]
        == PHEROVIZ_PROVIDER_IMPORT_PATH
    )
    assert production["provider_options"] == {
        "manifest_data_root": "/Users/tommy/Downloads/mayi/PheroViz"
    }
    assert isinstance(
        load_provider(production["provider"], {}),
        UnifiedBenchmarkProvider,
    )

    with experiment_workspace("production-matrix") as workspace:
        manifest = workspace / "mixed-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": f"case-p{panel_count}",
                            "panel_count": panel_count,
                            "split": "test",
                        }
                        for panel_count in (1, 2, 3, 6)
                    ]
                    + [
                        {
                            "case_id": "ignored-val",
                            "panel_count": 2,
                            "split": "val",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        artifact_root = workspace / "production-runs"
        matrix = dict(production)
        matrix["dataset_manifest"] = str(manifest)
        matrix["dataset_mode"] = "legacy"
        matrix.pop("dataset_manifest_sha256")
        matrix["artifact_root"] = str(artifact_root)
        path = workspace / "production-matrix.yaml"
        path.write_text(yaml.safe_dump(matrix), encoding="utf-8")

        specs = load_and_expand_matrix(path)

        assert len(specs) == 4 * 3 * 3
        assert {spec.split for spec in specs} == {"test"}
        assert {spec.panel_count for spec in specs} == {1, 2, 3, 6}
        assert {spec.budget_value for spec in specs} == {6.0}
        assert all(
            int(spec.budget_value) % int(spec.panel_count or 1) == 0
            for spec in specs
        )
        by_method = {
            method: [spec for spec in specs if spec.method == method]
            for method in {
                "best_of_n",
                "flat_iterative",
                "pheroviz_full",
            }
        }
        assert {spec.schedule for spec in by_method["best_of_n"]} == {
            "best_of_n"
        }
        assert {spec.schedule for spec in by_method["flat_iterative"]} == {
            "iterative"
        }
        assert {spec.schedule for spec in by_method["pheroviz_full"]} == {
            "iterative"
        }
        assert {
            spec.method_config["memory_mode"]
            for spec in by_method["best_of_n"]
        } == {"none"}
        assert {
            spec.method_config["memory_mode"]
            for spec in by_method["flat_iterative"]
        } == {"none"}
        assert {
            spec.method_config["memory_mode"]
            for spec in by_method["pheroviz_full"]
        } == {"full"}
        assert {
            spec.method_config["initial_generation"] for spec in specs
        } == {"model_spec"}
        assert {
            spec.method_config["render_timeout_seconds"] for spec in specs
        } == {120}
        assert {spec.provider for spec in specs} == {
            PHEROVIZ_PROVIDER_IMPORT_PATH
        }
        assert {
            spec.provider_options["manifest_data_root"] for spec in specs
        } == {"/Users/tommy/Downloads/mayi/PheroViz"}
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


def test_sealed_benchmark_requires_expected_manifest_hash() -> None:
    with experiment_workspace("sealed-benchmark") as workspace:
        data = workspace / "case.csv"
        data.write_text("x,y\n0,1\n", encoding="utf-8")
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        source_binding, source_binding_hash, verifications = (
            _sealed_source_binding(workspace, ["verified-case"])
        )
        manifest = workspace / "manifest.json"
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
                    "cases": [
                        {
                            "case_id": "verified-case",
                            "candidate_id": "verified-case",
                            "doi": "10.1038/example",
                            "panel_count": 1,
                            "split": "test",
                            "data_path": str(data),
                            "data_sha256": data_hash,
                            "curation_status": "verified",
                            "eligible_for_experiment": True,
                            "verification_evidence": verifications[
                                "verified-case"
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix["dataset_mode"] = "sealed_benchmark"
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")
        with pytest.raises(MatrixError, match="dataset_manifest_sha256"):
            load_and_expand_matrix(path)

        matrix["dataset_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(matrix), encoding="utf-8")
        specs = load_and_expand_matrix(path)
        assert len(specs) == 1
        assert specs[0].case_id == "verified-case"


def test_sealed_benchmark_rejects_same_doi_across_splits() -> None:
    with experiment_workspace("doi-overlap") as workspace:
        source_binding, source_binding_hash, verifications = (
            _sealed_source_binding(
                workspace,
                ["case-train", "case-test"],
            )
        )
        cases = [
            {
                "case_id": f"case-{split}",
                "candidate_id": f"case-{split}",
                "doi": doi,
                "panel_count": 1,
                "split": split,
                "data_path": str(workspace / f"{split}.csv"),
                "data_sha256": "a" * 64,
                "curation_status": "verified",
                "eligible_for_experiment": True,
                "verification_evidence": verifications[f"case-{split}"],
            }
            for split, doi in (
                ("train", "10.1038/same-paper"),
                ("test", "https://doi.org/10.1038/same-paper"),
            )
        ]
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "provenance": {
                        "code_commit": "b" * 40,
                        "code_dirty": False,
                        "source_binding": source_binding,
                        "source_binding_hash": source_binding_hash,
                    },
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix["dataset_mode"] = "sealed_benchmark"
        matrix["dataset_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")
        with pytest.raises(MatrixError, match="multiple benchmark splits"):
            load_and_expand_matrix(path)


def test_sealed_benchmark_accepts_multiple_review_bundles() -> None:
    with experiment_workspace("multiple-review-bundles") as workspace:
        source_bindings = []
        verifications = {}
        for suffix in ("a", "b"):
            root = workspace / suffix
            root.mkdir()
            binding, _, bundle_verifications = _sealed_source_binding(
                root,
                [f"case-{suffix}"],
            )
            source_bindings.append(binding)
            verifications.update(bundle_verifications)
        source_binding = {
            "candidate_inputs": [
                item
                for binding in source_bindings
                for item in binding["candidate_inputs"]
            ],
            "review_bundles": [
                {
                    "evidence": binding["evidence"],
                    "proposed": binding["proposed"],
                    "reviews": binding["reviews"],
                }
                for binding in source_bindings
            ],
        }
        source_binding_hash = hashlib.sha256(
            json.dumps(
                source_binding,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cases = []
        for suffix, split in (("a", "train"), ("b", "test")):
            data = workspace / f"{suffix}.csv"
            data.write_text("x,y\n0,1\n", encoding="utf-8")
            cases.append(
                {
                    "case_id": f"case-{suffix}",
                    "candidate_id": f"case-{suffix}",
                    "doi": f"10.1038/article-{suffix}",
                    "panel_count": 1,
                    "split": split,
                    "data_path": str(data),
                    "data_sha256": hashlib.sha256(
                        data.read_bytes()
                    ).hexdigest(),
                    "curation_status": "verified",
                    "eligible_for_experiment": True,
                    "verification_evidence": verifications[f"case-{suffix}"],
                }
            )
        manifest = workspace / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "provenance": {
                        "code_commit": "b" * 40,
                        "code_dirty": False,
                        "source_binding": source_binding,
                        "source_binding_hash": source_binding_hash,
                    },
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        matrix = _minimal_matrix(manifest, workspace / "runs")
        matrix["dataset_mode"] = "sealed_benchmark"
        matrix["dataset_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        path = workspace / "matrix.json"
        path.write_text(json.dumps(matrix), encoding="utf-8")

        specs = load_and_expand_matrix(path)
        assert {spec.case_id for spec in specs} == {"case-a", "case-b"}
        assert {spec.dataset_mode for spec in specs} == {"sealed_benchmark"}

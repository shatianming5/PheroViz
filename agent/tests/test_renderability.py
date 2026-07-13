from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

import pytest

from experiments.manifest import load_dataset_manifest
from experiments.models import sha256_file
from experiments.renderability import (
    MAX_BAR_CATEGORICAL_X,
    EXPECTED_POLICY_HASH,
    POLICY_HASH,
    RENDERABILITY_POLICY,
    RenderabilityError,
    build_renderability_derivative,
    main,
    validate_renderability_outputs,
)
from tests.test_experiment_support import experiment_workspace


RECORDED_ROOT = Path("/recorded/renderability-fixture")


def test_policy_hash_is_frozen() -> None:
    assert POLICY_HASH == EXPECTED_POLICY_HASH
    assert MAX_BAR_CATEGORICAL_X == 200


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = deepcopy(value)
    sealed[field] = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return sealed


def _recorded(repo: Path, path: Path) -> str:
    return str(RECORDED_ROOT / path.resolve().relative_to(repo.resolve()))


def _write_csv(path: Path, rows: int, *, two_series: bool = False) -> None:
    header = "category,value_a,value_b\n" if two_series else "category,value\n"
    body = "".join(
        (
            f"category-{index},{index},{index + 1}\n"
            if two_series
            else f"category-{index},{index}\n"
        )
        for index in range(rows)
    )
    path.write_text(header + body, encoding="utf-8")


def _expectation(
    panel_id: str,
    *,
    kind: str,
    two_series: bool = False,
) -> dict[str, Any]:
    series = [
        {
            "series_id": "value_a" if two_series else "value",
            "kind": kind,
            "x": "category",
            "y": "value_a" if two_series else "value",
        }
    ]
    if two_series:
        series.append(
            {
                "series_id": "value_b",
                "kind": kind,
                "x": "category",
                "y": "value_b",
            }
        )
    return {
        "panel_id": panel_id,
        "axis_index": 0,
        "x_scale": "categorical" if kind == "bar" else "linear",
        "series": series,
    }


def _verification_subset(verification: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(verification[key])
        for key in (
            "evidence_type",
            "evidence_ref",
            "review_hash",
            "review_models",
        )
    }


def _source_binding(
    repo: Path,
    candidate_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    bindings = repo / "bindings"
    bindings.mkdir()
    proposed_path = bindings / "proposed.jsonl"
    proposed_path.write_text(
        "".join(
            json.dumps({"candidate_id": candidate_id}) + "\n"
            for candidate_id in candidate_ids
        ),
        encoding="utf-8",
    )
    proposed_hash = sha256_file(proposed_path)

    reviews: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        review = _seal(
            {
                "candidate_id": candidate_id,
                "proposal_type": (
                    "multi_panel"
                    if candidate_id.startswith("multi-")
                    else "single_panel"
                ),
                "status": "accepted",
                "rejection_reasons": [],
                "binding": {},
                "model_reviews": [],
            },
            "review_hash",
        )
        reviews.append(review)
        verifications.append(
            {
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
        )
    reviews_path = bindings / "reviews.jsonl"
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
    evidence_path = bindings / "evidence.json"
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
                for review in sorted(
                    reviews,
                    key=lambda item: item["candidate_id"],
                )
            },
        },
        "summary_hash",
    )
    review_summary_path = bindings / "review_summary.json"
    review_summary_path.write_text(
        json.dumps(review_summary, sort_keys=True),
        encoding="utf-8",
    )

    candidates_path = bindings / "candidates.jsonl"
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
    corpus_path = bindings / "corpus.jsonl"
    corpus_path.write_text('{"download_eligible":true}\n', encoding="utf-8")
    case_summary = _seal(
        {
            "candidates_sha256": sha256_file(candidates_path),
            "corpus_manifest_sha256": sha256_file(corpus_path),
            "code_dirty": False,
            "eligible_for_experiment": len(candidate_ids),
        },
        "summary_hash",
    )
    case_summary_path = bindings / "case_summary.json"
    case_summary_path.write_text(
        json.dumps(case_summary, sort_keys=True),
        encoding="utf-8",
    )
    source_binding = {
        "candidate_inputs": [
            {
                "candidates_path": _recorded(repo, candidates_path),
                "candidates_sha256": sha256_file(candidates_path),
                "summary_path": _recorded(repo, case_summary_path),
                "summary_sha256": sha256_file(case_summary_path),
                "summary_hash": case_summary["summary_hash"],
                "code_commit": "b" * 40,
                "corpus_manifest": _recorded(repo, corpus_path),
                "corpus_manifest_sha256": sha256_file(corpus_path),
                "content_root": str(RECORDED_ROOT),
                "output_root": _recorded(repo, bindings),
            }
        ],
        "evidence": {
            "path": _recorded(repo, evidence_path),
            "sha256": sha256_file(evidence_path),
            "evidence_hash": evidence["evidence_hash"],
        },
        "proposed": {
            "path": _recorded(repo, proposed_path),
            "sha256": proposed_hash,
        },
        "reviews": {
            "path": _recorded(repo, reviews_path),
            "sha256": sha256_file(reviews_path),
            "summary_path": _recorded(repo, review_summary_path),
            "summary_sha256": sha256_file(review_summary_path),
        },
    }
    return source_binding, {
        verification["candidate_id"]: verification for verification in verifications
    }


def _single_case(
    repo: Path,
    *,
    case_id: str,
    doi: str,
    split: str,
    data_path: Path,
    rows: int,
    verification: dict[str, Any],
    kind: str,
    two_series: bool = False,
) -> dict[str, Any]:
    del rows
    panel_id = case_id.rsplit("-", 1)[-1]
    return {
        "candidate_id": case_id,
        "case_id": case_id,
        "doi": doi,
        "split": split,
        "panel_count": 1,
        "panel_id": panel_id,
        "chart_family": kind,
        "data_path": _recorded(repo, data_path),
        "data_sha256": sha256_file(data_path),
        "sheet": None,
        "user_goal": f"Render {case_id}",
        "intent": {},
        "evaluation_expectation": {
            "schema_version": "1.1.0",
            "panels": [
                _expectation(
                    panel_id,
                    kind=kind,
                    two_series=two_series,
                )
            ],
            "panel_groups": [],
        },
        "curation_status": "verified",
        "eligible_for_experiment": True,
        "eligibility_reasons": [],
        "verification_evidence": _verification_subset(verification),
    }


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@contextmanager
def _sealed_fixture(label: str) -> Iterator[dict[str, Any]]:
    with experiment_workspace(label) as workspace:
        repo = workspace / "repo"
        data = repo / "data"
        manifests = repo / "manifests"
        data.mkdir(parents=True)
        manifests.mkdir()
        bar_201 = data / "bar_201.csv"
        bar_200 = data / "bar_200.csv"
        line_10000 = data / "line_10000.csv"
        _write_csv(bar_201, 201, two_series=True)
        _write_csv(bar_200, 200)
        _write_csv(line_10000, 10_000)

        candidate_ids = [
            "bar-201-train",
            "bar-201-val",
            "bar-201-test",
            "bar-200-test",
            "line-10000-val",
            "multi-bad-train",
        ]
        source_binding, verifications = _source_binding(repo, candidate_ids)
        cases = [
            _single_case(
                repo,
                case_id=f"bar-201-{split}",
                doi=f"10.1000/bar-201-{split}",
                split=split,
                data_path=bar_201,
                rows=201,
                verification=verifications[f"bar-201-{split}"],
                kind="bar",
                two_series=True,
            )
            for split in ("test", "train", "val")
        ]
        cases.extend(
            [
                _single_case(
                    repo,
                    case_id="bar-200-test",
                    doi="10.1000/bar-200",
                    split="test",
                    data_path=bar_200,
                    rows=200,
                    verification=verifications["bar-200-test"],
                    kind="bar",
                ),
                _single_case(
                    repo,
                    case_id="line-10000-val",
                    doi="10.1000/line-10000",
                    split="val",
                    data_path=line_10000,
                    rows=10_000,
                    verification=verifications["line-10000-val"],
                    kind="line",
                ),
            ]
        )
        multi_verification = verifications["multi-bad-train"]
        cases.append(
            {
                "candidate_id": "multi-bad-train",
                "case_id": "multi-bad-train",
                "doi": "10.1000/multi-bad",
                "split": "train",
                "panel_count": 2,
                "chart_family": "multi_panel",
                "panels": [
                    {
                        "id": "bad",
                        "chart_family": "bar",
                        "data_path": _recorded(repo, bar_201),
                        "data_sha256": sha256_file(bar_201),
                        "sheet": None,
                        "user_goal": "Render bad bar",
                        "intent": {},
                    },
                    {
                        "id": "good",
                        "chart_family": "line",
                        "data_path": _recorded(repo, line_10000),
                        "data_sha256": sha256_file(line_10000),
                        "sheet": None,
                        "user_goal": "Render good line",
                        "intent": {},
                    },
                ],
                "user_goal": "Render two panels",
                "intent": {},
                "evaluation_expectation": {
                    "schema_version": "1.1.0",
                    "panels": [
                        _expectation("bad", kind="bar", two_series=True),
                        _expectation("good", kind="line"),
                    ],
                    "panel_groups": [],
                },
                "curation_status": "verified",
                "eligible_for_experiment": True,
                "eligibility_reasons": [],
                "verification_evidence": _verification_subset(multi_verification),
            }
        )
        parent = {
            "schema_version": "1.0",
            "provenance": {
                "code_commit": "b" * 40,
                "code_dirty": False,
                "source_binding": source_binding,
                "source_binding_hash": hashlib.sha256(
                    _canonical_json(source_binding).encode("utf-8")
                ).hexdigest(),
                "split_seed": 17,
                "split_ratios": {
                    "train": 0.5,
                    "val": 0.25,
                    "test": 0.25,
                },
            },
            "cases": cases,
        }
        parent_path = manifests / "benchmark_manifest.json"
        parent_path.write_text(
            json.dumps(parent, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (repo / ".gitignore").write_text("derived-*/\n", encoding="utf-8")
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Renderability Test")
        _commit_all(repo, "synthetic sealed fixture")
        yield {
            "repo": repo,
            "parent": parent_path,
            "parent_payload": parent,
            "bar_201": bar_201,
            "recorded_root": RECORDED_ROOT,
        }


def _build(fixture: dict[str, Any], output_name: str) -> dict[str, Any]:
    repo = fixture["repo"]
    return build_renderability_derivative(
        fixture["parent"],
        repo / output_name,
        manifest_data_root=fixture["recorded_root"],
        runtime_repo_root=repo,
        repo_root=repo,
    )


def test_static_policy_builds_a_deterministic_sealed_derivative() -> None:
    with _sealed_fixture("renderability-policy") as fixture:
        repo = fixture["repo"]
        parent_bytes = fixture["parent"].read_bytes()
        first = _build(fixture, "derived-a")
        second = _build(fixture, "derived-b")

        assert MAX_BAR_CATEGORICAL_X == 200
        assert RENDERABILITY_POLICY["split_scope"] == [
            "train",
            "val",
            "test",
        ]
        assert RENDERABILITY_POLICY["non_bar_cardinality_action"] == "no_cap"
        assert first["input_cases"] == 6
        assert first["accepted_cases"] == 2
        assert first["rejected_cases"] == 4
        assert first["rejected_case_ids"] == [
            "bar-201-test",
            "bar-201-train",
            "bar-201-val",
            "multi-bad-train",
        ]
        assert fixture["parent"].read_bytes() == parent_bytes

        first_root = repo / "derived-a"
        second_root = repo / "derived-b"
        for name in (
            "benchmark_manifest.json",
            "renderability_audit.json",
            "summary.json",
        ):
            assert (first_root / name).read_bytes() == (second_root / name).read_bytes()
        assert first["summary_hash"] == second["summary_hash"]
        assert first["policy_hash"] == second["policy_hash"] == POLICY_HASH

        audit = json.loads(
            (first_root / "renderability_audit.json").read_text(encoding="utf-8")
        )
        case_rows = {row["case_id"]: row for row in audit["case_rows"]}
        panel_rows = {
            (row["case_id"], row["panel_id"]): row for row in audit["panel_rows"]
        }
        assert {
            case_rows[f"bar-201-{split}"]["decision"]
            for split in ("train", "val", "test")
        } == {"rejected"}
        assert {
            case_rows[f"bar-201-{split}"]["reason"]
            for split in ("train", "val", "test")
        } == {
            f"one_or_more_panels_failed:{split}" for split in ("train", "val", "test")
        }
        assert panel_rows[("bar-200-test", "test")]["unique_categorical_x"] == 200
        assert panel_rows[("bar-200-test", "test")]["decision"] == "accepted"
        assert panel_rows[("bar-201-test", "test")]["unique_categorical_x"] == 201
        assert panel_rows[("bar-201-test", "test")]["expected_points"] == 402
        line = panel_rows[("line-10000-val", "val")]
        assert line["expected_points"] == 10_000
        assert line["unique_categorical_x"] is None
        assert line["decision"] == "accepted"
        assert case_rows["multi-bad-train"]["decision"] == "rejected"
        assert panel_rows[("multi-bad-train", "bad")]["decision"] == "rejected"
        assert panel_rows[("multi-bad-train", "good")]["decision"] == "accepted"
        assert panel_rows[("multi-bad-train", "good")]["case_decision"] == "rejected"

        parent = fixture["parent_payload"]
        derived = json.loads(
            (first_root / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        assert [case["case_id"] for case in derived["cases"]] == [
            "bar-200-test",
            "line-10000-val",
        ]
        assert _canonical_json(
            derived["provenance"]["source_binding"]
        ) == _canonical_json(parent["provenance"]["source_binding"])
        assert (
            derived["provenance"]["source_binding_hash"]
            == parent["provenance"]["source_binding_hash"]
        )
        parent_evidence = {
            case["case_id"]: case["verification_evidence"] for case in parent["cases"]
        }
        assert all(
            case["verification_evidence"] == parent_evidence[case["case_id"]]
            for case in derived["cases"]
        )
        assert derived["provenance"]["parent_manifest_sha256"] == sha256_file(
            fixture["parent"]
        )
        assert derived["provenance"]["renderability_audit_sha256"] == sha256_file(
            first_root / "renderability_audit.json"
        )

        validated = validate_renderability_outputs(
            first_root,
            manifest_data_root=RECORDED_ROOT,
            runtime_repo_root=repo,
        )
        assert validated == first
        sealed_cases = load_dataset_manifest(
            first_root / "benchmark_manifest.json",
            dataset_mode="sealed_benchmark",
            manifest_data_root=RECORDED_ROOT,
            runtime_repo_root=repo,
        )
        assert len(sealed_cases) == 2


def test_renderability_cli_writes_only_the_three_sealed_outputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _sealed_fixture("renderability-cli") as fixture:
        repo = fixture["repo"]
        output = repo / "derived-cli"
        assert (
            main(
                [
                    "--parent-manifest",
                    str(fixture["parent"]),
                    "--output-root",
                    str(output),
                    "--manifest-data-root",
                    str(RECORDED_ROOT),
                    "--runtime-repo-root",
                    str(repo),
                    "--repo-root",
                    str(repo),
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["accepted_cases"] == 2
        assert sorted(path.name for path in output.iterdir()) == [
            "benchmark_manifest.json",
            "renderability_audit.json",
            "summary.json",
        ]


def test_builder_fails_closed_on_dirty_or_changed_inputs() -> None:
    with _sealed_fixture("renderability-dirty") as fixture:
        dirty = fixture["repo"] / "untracked.txt"
        dirty.write_text("dirty\n", encoding="utf-8")
        with pytest.raises(RenderabilityError, match="git-worktree-dirty"):
            _build(fixture, "derived-dirty")

    mutations = {
        "hash": lambda fixture: fixture["bar_201"].write_text(
            "category,value_a,value_b\nchanged,1,2\n",
            encoding="utf-8",
        ),
        "missing": lambda fixture: fixture["bar_201"].unlink(),
        "schema": lambda fixture: fixture["parent"].write_text(
            json.dumps(
                {
                    **fixture["parent_payload"],
                    "schema_version": "unexpected",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        ),
    }
    expected = {
        "hash": "data SHA-256 changed",
        "missing": "data file is missing",
        "schema": "schema-unexpected",
    }
    for name, mutate in mutations.items():
        with _sealed_fixture(f"renderability-{name}") as fixture:
            mutate(fixture)
            _commit_all(fixture["repo"], f"mutate {name}")
            with pytest.raises(RenderabilityError, match=expected[name]):
                _build(fixture, f"derived-{name}")

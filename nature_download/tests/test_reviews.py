from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import nature_download.corpus.reviews as reviews_module
from nature_download.corpus.cases import _load_evidence
from nature_download.corpus.provenance import sha256_file
from nature_download.corpus.proposals import (
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    PROPOSAL_RULE_V2,
    PROPOSAL_RULE_V3,
    PROPOSAL_RULE_V4,
    _multi_panel_proposals,
    propose_single_candidate,
)
from nature_download.corpus.reviews import (
    ReviewError,
    review_proposals,
    validate_review_artifacts,
)


MODELS = ("judge-a", "judge-b")
CLEAN_GIT = {"commit": "abcdef1", "dirty": False}


@dataclass
class FakeResponse:
    value: dict[str, Any]
    model: str
    usage: dict[str, Any]
    stop_reason: str | None = "end_turn"


class FakeClient:
    def __init__(
        self,
        model: str,
        output: Callable[[str, str], dict[str, Any]],
        calls: list[dict[str, Any]],
        *,
        stop_reason: str | None = "end_turn",
        error: Exception | None = None,
    ) -> None:
        self.model = model
        self.output = output
        self.calls = calls
        self.stop_reason = stop_reason
        self.error = error

    def evaluate_image_json(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        model: str,
        max_tokens: int | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "image_path": str(image_path),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        if self.error:
            raise self.error
        return FakeResponse(
            value=self.output(prompt, model),
            model=self.model,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "api_key": "must-not-persist",
            },
            stop_reason=self.stop_reason,
        )


def descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def make_single(
    workdir: Path,
    *,
    candidate_id: str = "panel-a",
    panel_id: str = "a",
    figure_no: int = 1,
) -> dict[str, Any]:
    table = workdir / f"{candidate_id}.csv"
    figure = workdir / f"{candidate_id}.png"
    caption = workdir / f"{candidate_id}.txt"
    table.write_text("Category,Value\nA,1\nB,2\n", encoding="utf-8")
    figure.write_bytes(b"image fixture")
    caption.write_text("caption fixture", encoding="utf-8")
    expectation = {
        "schema_version": "1.1.0",
        "panels": [
            {
                "panel_id": panel_id,
                "axis_index": 0,
                "x_scale": "linear",
                "series": [
                    {
                        "series_id": "Value",
                        "kind": "bar",
                        "x": "Category",
                        "y": "Value",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "proposal_type": "single_panel",
        "doi": "10.1038/example",
        "figure_no": figure_no,
        "panel_ids": [panel_id],
        "curation_status": "proposed",
        "eligible_for_experiment": False,
        "eligibility_reasons": ["external-validation-required"],
        "source_table": {**descriptor(table), "sheet_name": None},
        "figure": descriptor(figure),
        "caption": descriptor(caption),
        "experiment_case": {
            "case_id": candidate_id,
            "panel_count": 1,
            "split": None,
            "data_path": str(table.resolve()),
            "sheet": None,
            "panel_id": panel_id,
            "user_goal": "Create a bar chart.",
            "chart_family": "bar",
            "intent": {
                "x": "Category",
                "y": "Value",
                "series": ["Value"],
                "x_scale": "categorical",
                "units": {},
            },
            "evaluation_expectation": expectation,
        },
    }


def make_multi(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return _multi_panel_proposals(
        [first, second],
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
    )[0]


def canonical_single(candidate: dict[str, Any]) -> dict[str, Any]:
    return propose_single_candidate(
        candidate,
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
    )


def write_proposed(path: Path, proposals: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(proposal, sort_keys=True) + "\n"
            for proposal in proposals
        ),
        encoding="utf-8",
    )


def matching_output(prompt: str, model: str) -> dict[str, Any]:
    return {
        "valid": True,
        "chart_family": "bar",
        "x": "Category",
        "y": ["Value"],
        "reason": "Exact match.",
    }


def factory_for(
    output: Callable[[str, str], dict[str, Any]] = matching_output,
    *,
    stop_reason: str | None = "end_turn",
    error: Exception | None = None,
) -> tuple[Callable[[str], FakeClient], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def factory(model: str) -> FakeClient:
        return FakeClient(
            model,
            output,
            calls,
            stop_reason=stop_reason,
            error=error,
        )

    return factory, calls


def test_all_models_agree_generates_case_builder_evidence(workdir: Path) -> None:
    proposal = propose_single_candidate(
        make_single(workdir),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
    )
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])
    factory, calls = factory_for()
    result = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert len(calls) == 2
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert result["summary"]["single_accepted"] == 1
    assert result["summary"]["human_claims"] == 0
    assert result["rejected"] == []
    verification = result["evidence"]["verifications"][0]
    assert verification["evidence_type"] == "external_validation"
    assert verification["experiment_case"] == proposal["experiment_case"]
    accepted, invalid = _load_evidence(output / "evidence.json")
    assert invalid == 0
    assert "panel-a" in accepted
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.is_file()
    )
    assert "must-not-persist" not in persisted
    review = result["reviews"][0]
    assert review["binding"]["rubric_hash"]
    assert review["binding"]["prompt_hash"]
    assert review["binding"]["source"]["actual_sha256"]
    assert review["model_reviews"][0]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
    }
    validated = validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )
    assert validated["evidence"] == result["evidence"]
    assert validated["summary"] == result["summary"]


def test_v2_scatter_proposal_is_reviewed_and_validated(
    workdir: Path,
) -> None:
    candidate = make_single(workdir, candidate_id="scatter-a")
    table = Path(candidate["source_table"]["path"])
    table.write_text(
        "Run order,Value\n1,3\n2,1\n3,2\n",
        encoding="utf-8",
    )
    candidate["source_table"] = {
        **descriptor(table),
        "sheet_name": None,
    }
    proposal = propose_single_candidate(
        candidate,
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V2,
    )
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])

    def scatter_output(prompt: str, model: str) -> dict[str, Any]:
        return {
            "valid": True,
            "chart_family": "scatter",
            "x": "Run order",
            "y": ["Value"],
            "reason": "Exact match.",
        }

    factory, calls = factory_for(scatter_output)
    result = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )

    assert result["summary"]["single_accepted"] == 1
    assert (
        result["summary"]["rubric_hash"]
        == reviews_module.REVIEW_RUBRIC_V2_HASH
    )
    assert all("line, bar, or scatter" in call["prompt"] for call in calls)
    validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )


def test_v3_wide_melt_proposal_is_reviewable_without_source_mutation(
    workdir: Path,
) -> None:
    candidate = make_single(workdir, candidate_id="wide-a")
    table = Path(candidate["source_table"]["path"])
    table.write_text("WT,KO\n1.0,2.0\n1.1,2.1\n", encoding="utf-8")
    candidate["source_table"] = {**descriptor(table), "sheet_name": None}
    proposal = propose_single_candidate(
        candidate,
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V3,
    )
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])

    def wide_output(prompt: str, model: str) -> dict[str, Any]:
        return {
            "valid": True,
            "chart_family": "scatter",
            "x": "__wide_group__",
            "y": ["__wide_value__"],
            "reason": "Exact deterministic wide-to-long binding.",
        }

    factory, _ = factory_for(wide_output)
    result = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert result["summary"]["single_accepted"] == 1
    assert result["summary"]["rubric_hash"] == reviews_module.REVIEW_RUBRIC_V3_HASH
    validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )


def test_v4_proposal_selects_v4_review_rubric(workdir: Path) -> None:
    candidate = make_single(workdir, candidate_id="wide-v4")
    table = Path(candidate["source_table"]["path"])
    table.write_text(
        "WT,KO\n1.0,2.0\n1.1,2.1\n1.2,2.2\n1.3,2.3\n1.4,2.4\n",
        encoding="utf-8",
    )
    candidate["source_table"] = {**descriptor(table), "sheet_name": None}
    proposal = propose_single_candidate(
        candidate,
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V4,
    )
    source = workdir / "proposed-v4.jsonl"
    output = workdir / "reviews-v4"
    write_proposed(source, [proposal])

    def wide_output(prompt: str, model: str) -> dict[str, Any]:
        return {
            "valid": True,
            "chart_family": "scatter",
            "x": "__wide_group__",
            "y": ["__wide_value__"],
            "reason": "Exact deterministic wide-to-long binding.",
        }

    factory, _ = factory_for(wide_output)
    result = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert result["summary"]["single_accepted"] == 1
    assert result["summary"]["rubric_hash"] == reviews_module.REVIEW_RUBRIC_V4_HASH
    validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )


@pytest.mark.parametrize("chart_family", ["line", "bar", "scatter"])
def test_v2_rubric_accepts_all_supported_chart_families(
    chart_family: str,
) -> None:
    allowed = reviews_module.REVIEW_RUBRICS[
        reviews_module.REVIEW_RUBRIC_V2_HASH
    ][1]
    output = reviews_module._strict_output(
        {
            "valid": True,
            "chart_family": chart_family,
            "x": "x",
            "y": ["y"],
            "reason": "supported",
        },
        allowed_chart_families=allowed,
    )
    assert output["chart_family"] == chart_family


def test_v1_rubric_rejects_scatter_model_output() -> None:
    allowed = reviews_module.REVIEW_RUBRICS[
        reviews_module.REVIEW_RUBRIC_V1_HASH
    ][1]
    with pytest.raises(ReviewError, match="model-output-schema-invalid"):
        reviews_module._strict_output(
            {
                "valid": True,
                "chart_family": "scatter",
                "x": "x",
                "y": ["y"],
                "reason": "unsupported in v1",
            },
            allowed_chart_families=allowed,
        )


def test_disagreement_and_correction_are_recorded_not_adopted(
    workdir: Path,
) -> None:
    proposal = canonical_single(make_single(workdir))
    original_case = json.loads(json.dumps(proposal["experiment_case"]))
    source = workdir / "proposed.jsonl"
    write_proposed(source, [proposal])

    def correction(prompt: str, model: str) -> dict[str, Any]:
        value = matching_output(prompt, model)
        if model == "judge-b":
            value = {
                **value,
                "x": "Value",
                "y": ["Category"],
                "reason": "Suggested correction.",
            }
        return value

    factory, _ = factory_for(correction)
    result = review_proposals(
        proposed_path=source,
        output_root=workdir / "reviews",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert result["evidence"]["verifications"] == []
    assert result["rejected"][0]["rejection_reasons"] == [
        "model-proposed-correction"
    ]
    correction_output = result["rejected"][0]["model_outputs"][1]["output"]
    assert correction_output["x"] == "Value"
    assert proposal["experiment_case"] == original_case


def test_asset_tamper_rejects_before_model_call(workdir: Path) -> None:
    proposal = make_single(workdir)
    source = workdir / "proposed.jsonl"
    write_proposed(source, [proposal])
    Path(proposal["source_table"]["path"]).write_text(
        "Category,Value\nA,999\n",
        encoding="utf-8",
    )
    factory, calls = factory_for()
    result = review_proposals(
        proposed_path=source,
        output_root=workdir / "reviews",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert calls == []
    assert "source-hash-mismatch" in result["rejected"][0]["rejection_reasons"]
    assert result["evidence"]["verifications"] == []


def test_resume_reuses_bound_reviews_and_rejects_tamper(workdir: Path) -> None:
    proposal = make_single(workdir)
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])
    factory, _ = factory_for()
    review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )

    def forbidden_factory(model: str):
        raise AssertionError("resume must not call a model")

    resumed = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=forbidden_factory,
        git_state=CLEAN_GIT,
        resume=True,
    )
    assert resumed["summary"]["resumed"] == 1

    Path(proposal["figure"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ReviewError, match="resume-binding-mismatch"):
        review_proposals(
            proposed_path=source,
            output_root=output,
            judge_models=MODELS,
            client_factory=forbidden_factory,
            git_state=CLEAN_GIT,
            resume=True,
        )


def test_resume_preserves_legacy_v1_rubric(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = canonical_single(make_single(workdir))
    proposal.pop("proposal_rule_version")
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])
    factory, calls = factory_for()
    monkeypatch.setattr(
        reviews_module,
        "REVIEW_RUBRIC_HASH",
        reviews_module.REVIEW_RUBRIC_V1_HASH,
    )
    first = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert (
        first["summary"]["rubric_hash"]
        == reviews_module.REVIEW_RUBRIC_V1_HASH
    )
    validated = validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )
    assert (
        validated["evidence"]["rubric_hash"]
        == reviews_module.REVIEW_RUBRIC_V1_HASH
    )

    monkeypatch.setattr(
        reviews_module,
        "REVIEW_RUBRIC_HASH",
        reviews_module.REVIEW_RUBRIC_V2_HASH,
    )
    calls.clear()
    resumed = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
        resume=True,
    )

    assert resumed["summary"]["resumed"] == 1
    assert (
        resumed["summary"]["rubric_hash"]
        == reviews_module.REVIEW_RUBRIC_V1_HASH
    )
    assert calls == []
    validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )


def test_resume_malformed_binding_fails_closed(workdir: Path) -> None:
    proposal = make_single(workdir)
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [proposal])
    factory, _ = factory_for()
    review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    reviews_path = output / "reviews.jsonl"
    record = json.loads(reviews_path.read_text(encoding="utf-8"))
    record["binding"] = []
    unhashed = dict(record)
    unhashed.pop("review_hash")
    record["review_hash"] = reviews_module._sha256_json(unhashed)
    reviews_path.write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="resume-review-binding-invalid"):
        review_proposals(
            proposed_path=source,
            output_root=output,
            judge_models=MODELS,
            client_factory=factory,
            git_state=CLEAN_GIT,
            resume=True,
        )


def test_unknown_proposal_rule_fails_before_review(workdir: Path) -> None:
    proposal = make_single(workdir)
    proposal["proposal_rule_version"] = "simple-2d-v999"
    source = workdir / "proposed.jsonl"
    write_proposed(source, [proposal])
    factory, calls = factory_for()

    with pytest.raises(ReviewError, match="proposal-rule-version-unsupported"):
        review_proposals(
            proposed_path=source,
            output_root=workdir / "reviews",
            judge_models=MODELS,
            client_factory=factory,
            git_state=CLEAN_GIT,
        )
    assert calls == []


def test_noncanonical_multi_is_rejected_without_evidence(
    workdir: Path,
) -> None:
    first = canonical_single(
        make_single(workdir, candidate_id="panel-a", panel_id="a")
    )
    second = canonical_single(
        make_single(workdir, candidate_id="panel-b", panel_id="b")
    )
    multi = make_multi(first, second)
    multi["experiment_case"]["user_goal"] = "Tampered multi-panel goal."
    source = workdir / "proposed.jsonl"
    output = workdir / "reviews"
    write_proposed(source, [first, second, multi])
    factory, _ = factory_for()

    result = review_proposals(
        proposed_path=source,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )

    rejected = next(
        item
        for item in result["rejected"]
        if item["candidate_id"] == multi["candidate_id"]
    )
    assert rejected["rejection_reasons"] == [
        "multi-proposal-not-canonical"
    ]
    assert {
        item["candidate_id"]
        for item in result["evidence"]["verifications"]
    } == {"panel-a", "panel-b"}
    validate_review_artifacts(
        proposed_path=source,
        reviews_path=output / "reviews.jsonl",
        evidence_path=output / "evidence.json",
    )

def test_dirty_code_is_rejected_by_default(workdir: Path) -> None:
    proposal = make_single(workdir)
    source = workdir / "proposed.jsonl"
    write_proposed(source, [proposal])
    factory, calls = factory_for()
    with pytest.raises(ReviewError, match="code-worktree-dirty"):
        review_proposals(
            proposed_path=source,
            output_root=workdir / "reviews",
            judge_models=MODELS,
            client_factory=factory,
            git_state={"commit": "abcdef1", "dirty": True},
        )
    assert calls == []


def test_multi_evidence_requires_all_source_candidates(workdir: Path) -> None:
    first = make_single(workdir, candidate_id="panel-a", panel_id="a")
    second = make_single(workdir, candidate_id="panel-b", panel_id="b")
    multi = make_multi(first, second)
    source = workdir / "proposed.jsonl"
    write_proposed(source, [first, second, multi])
    factory, _ = factory_for()
    accepted = review_proposals(
        proposed_path=source,
        output_root=workdir / "accepted",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert accepted["summary"]["single_accepted"] == 2
    assert accepted["summary"]["multi_accepted"] == 1
    assert {
        item["candidate_id"] for item in accepted["evidence"]["verifications"]
    } == {"panel-a", "panel-b", multi["candidate_id"]}

    def reject_second(prompt: str, model: str) -> dict[str, Any]:
        value = matching_output(prompt, model)
        if "panel-b" in prompt and model == "judge-b":
            value["valid"] = False
            value["reason"] = "Not supported."
        return value

    factory, _ = factory_for(reject_second)
    rejected = review_proposals(
        proposed_path=source,
        output_root=workdir / "rejected",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert {
        item["candidate_id"] for item in rejected["evidence"]["verifications"]
    } == {"panel-a"}
    multi_rejection = next(
        item
        for item in rejected["rejected"]
        if item["candidate_id"] == multi["candidate_id"]
    )
    assert multi_rejection["rejection_reasons"] == [
        "multi-source-not-all-validated"
    ]


def test_truncation_model_failure_and_duplicate_models_fail_closed(
    workdir: Path,
) -> None:
    proposal = make_single(workdir)
    source = workdir / "proposed.jsonl"
    write_proposed(source, [proposal])
    factory, _ = factory_for(stop_reason="max_tokens")
    truncated = review_proposals(
        proposed_path=source,
        output_root=workdir / "truncated",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert truncated["evidence"]["verifications"] == []
    assert truncated["rejected"][0]["rejection_reasons"] == [
        "model-response-truncated"
    ]

    factory, _ = factory_for(lambda prompt, model: {})
    empty = review_proposals(
        proposed_path=source,
        output_root=workdir / "empty",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert empty["evidence"]["verifications"] == []
    assert empty["rejected"][0]["rejection_reasons"] == [
        "model-output-schema-invalid"
    ]

    factory, _ = factory_for(error=RuntimeError("sk-secret-must-not-persist"))
    failed = review_proposals(
        proposed_path=source,
        output_root=workdir / "failed",
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    assert failed["evidence"]["verifications"] == []
    persisted = (workdir / "failed" / "reviews.jsonl").read_text(encoding="utf-8")
    assert "sk-secret" not in persisted
    with pytest.raises(ReviewError, match="judge-models-must-be-distinct"):
        review_proposals(
            proposed_path=source,
            output_root=workdir / "duplicate",
            judge_models=("same", "same"),
            client_factory=factory,
            git_state=CLEAN_GIT,
        )

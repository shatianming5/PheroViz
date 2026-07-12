from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from nature_download.corpus.cases import _load_evidence
from nature_download.corpus.provenance import sha256_file
from nature_download.corpus.reviews import ReviewError, review_proposals


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
    return {
        "schema_version": "1.0",
        "candidate_id": "multi-figure1",
        "proposal_type": "multi_panel",
        "source_candidate_ids": [
            first["candidate_id"],
            second["candidate_id"],
        ],
        "doi": first["doi"],
        "figure_no": 1,
        "panel_ids": ["a", "b"],
        "curation_status": "proposed",
        "eligible_for_experiment": False,
        "eligibility_reasons": ["external-validation-required"],
        "experiment_case": {
            "case_id": "multi-figure1",
            "panel_count": 2,
            "split": None,
            "panels": [
                {
                    "id": "a",
                    "data_path": first["source_table"]["path"],
                    "sheet": None,
                    "user_goal": "Panel A.",
                    "chart_family": "bar",
                    "intent": {"x": "Category", "y": "Value"},
                },
                {
                    "id": "b",
                    "data_path": second["source_table"]["path"],
                    "sheet": None,
                    "user_goal": "Panel B.",
                    "chart_family": "bar",
                    "intent": {"x": "Category", "y": "Value"},
                },
            ],
            "user_goal": "Create a two-panel figure.",
            "chart_family": "multi_panel",
            "intent": {"panels": []},
            "evaluation_expectation": {
                "schema_version": "1.1.0",
                "panels": [
                    first["experiment_case"]["evaluation_expectation"]["panels"][0],
                    second["experiment_case"]["evaluation_expectation"]["panels"][0],
                ],
                "panel_groups": [],
            },
        },
    }


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
    proposal = make_single(workdir)
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


def test_disagreement_and_correction_are_recorded_not_adopted(
    workdir: Path,
) -> None:
    proposal = make_single(workdir)
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
    } == {"panel-a", "panel-b", "multi-figure1"}

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
        if item["candidate_id"] == "multi-figure1"
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

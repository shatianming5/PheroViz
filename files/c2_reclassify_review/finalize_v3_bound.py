#!/usr/bin/env python3
"""Finalize V3 checkpoint reviews with the V3 code bound at review time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "files/c2_reclassify_review/v3_bound_runtime/nature_download"
sys.path.insert(0, str(RUNTIME))

from corpus.provenance import sha256_file
from corpus.reviews import (
    GitState,
    REVIEW_RUBRICS,
    REVIEW_RUBRIC_V3_HASH,
    _read_jsonl,
    _resolve_git_state,
    _single_binding,
    review_proposals,
)


def common_binding(records: list[dict]) -> tuple[GitState, tuple[str, ...], str]:
    bindings = [record.get("binding") or {} for record in records]
    commits = {binding.get("code_commit") for binding in bindings}
    dirties = {binding.get("code_dirty") for binding in bindings}
    model_tuples = {tuple(binding.get("request_models") or []) for binding in bindings}
    rubrics = {binding.get("rubric_hash") for binding in bindings}
    if (
        len(commits) != 1
        or len(dirties) != 1
        or len(model_tuples) != 1
        or len(rubrics) != 1
    ):
        raise SystemExit("checkpoint bindings are not homogeneous")
    commit = next(iter(commits))
    dirty = next(iter(dirties))
    models = next(iter(model_tuples))
    rubric_hash = next(iter(rubrics))
    if (
        not isinstance(commit, str)
        or not isinstance(dirty, bool)
        or len(models) < 2
        or rubric_hash != REVIEW_RUBRIC_V3_HASH
    ):
        raise SystemExit("checkpoint does not carry a V3 two-judge binding")
    return GitState(commit=commit, dirty=dirty), models, rubric_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    proposed = Path(args.proposed).resolve()
    out = Path(args.out).resolve()
    reviews = _read_jsonl(out / "reviews.jsonl")
    state, models, rubric_hash = common_binding(reviews)
    if args.verify_only:
        proposals = {
            str(proposal.get("candidate_id") or ""): proposal
            for proposal in _read_jsonl(proposed)
        }
        first = next(record for record in reviews if record.get("proposal_type") == "single_panel")
        rubric, families = REVIEW_RUBRICS[rubric_hash]
        binding, _, _, _, _ = _single_binding(
            proposals[str(first["candidate_id"])],
            input_hash=sha256_file(proposed),
            models=models,
            git_state=state,
            rubric=rubric,
            rubric_hash=rubric_hash,
            allowed_chart_families=families,
        )
        if binding["resume_binding_hash"] != first["binding"]["resume_binding_hash"]:
            raise SystemExit("bound V3 runtime does not reproduce checkpoint binding")
        print(
            json.dumps(
                {
                    "binding_reproduced": True,
                    "code_commit": state.commit,
                    "code_dirty": state.dirty,
                    "judge_models": list(models),
                    "rubric_hash": rubric_hash,
                },
                sort_keys=True,
            )
        )
        return
    result = review_proposals(
        proposed_path=proposed,
        output_root=out,
        judge_models=models,
        resume=True,
        allow_dirty=True,
        git_state=state,
    )
    print(
        json.dumps(
            {
                "bound_code_commit": state.commit,
                "bound_code_dirty": state.dirty,
                "judge_models": list(models),
                "summary": result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

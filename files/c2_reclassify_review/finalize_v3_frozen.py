#!/usr/bin/env python3
"""Finalize frozen-V3 checkpoint reviews without weakening their provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from frozen_v3_binding import decorate_binding, frozen_v3_input_binding


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "files/c2_reclassify_review/v3_bound_runtime/nature_download"
sys.path.insert(0, str(RUNTIME))

import corpus.reviews as reviews
from corpus.provenance import sha256_file


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def common_binding(
    records: list[dict[str, Any]], expected_freeze_binding: dict[str, Any]
) -> tuple[reviews.GitState, tuple[str, ...], str]:
    bindings = [record.get("binding") or {} for record in records]
    commits = {binding.get("code_commit") for binding in bindings}
    dirties = {binding.get("code_dirty") for binding in bindings}
    model_tuples = {tuple(binding.get("request_models") or []) for binding in bindings}
    rubrics = {binding.get("rubric_hash") for binding in bindings}
    freeze_bindings = {json.dumps(binding.get("frozen_v3_input_binding"), sort_keys=True) for binding in bindings}
    if (
        len(commits) != 1
        or len(dirties) != 1
        or len(model_tuples) != 1
        or len(rubrics) != 1
        or freeze_bindings != {json.dumps(expected_freeze_binding, sort_keys=True)}
    ):
        raise SystemExit("checkpoint bindings are not homogeneous or freeze-bound")
    commit = next(iter(commits))
    dirty = next(iter(dirties))
    models = next(iter(model_tuples))
    rubric_hash = next(iter(rubrics))
    if (
        not isinstance(commit, str)
        or not isinstance(dirty, bool)
        or len(models) < 2
        or rubric_hash != reviews.REVIEW_RUBRIC_V3_HASH
    ):
        raise SystemExit("checkpoint does not carry a V3 two-judge binding")
    return reviews.GitState(commit=commit, dirty=dirty), models, rubric_hash


def decorate_review_functions(freeze_binding: dict[str, Any]) -> None:
    original_single_binding = reviews._single_binding
    original_multi_binding = reviews._multi_binding

    def single_binding(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        binding, reasons, prompt, prompt_hash, expected = original_single_binding(
            *args, **kwargs
        )
        return (
            decorate_binding(
                binding,
                frozen_input_binding=freeze_binding,
                sha256_json=reviews._sha256_json,
            ),
            reasons,
            prompt,
            prompt_hash,
            expected,
        )

    def multi_binding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return decorate_binding(
            original_multi_binding(*args, **kwargs),
            frozen_input_binding=freeze_binding,
            sha256_json=reviews._sha256_json,
        )

    reviews._single_binding = single_binding
    reviews._multi_binding = multi_binding


def seal_top_level_outputs(out: Path, freeze_binding: dict[str, Any]) -> dict[str, Any]:
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("evidence_hash", None)
    evidence["frozen_v3_input_binding"] = freeze_binding
    evidence = reviews._seal(evidence, "evidence_hash")
    write_json(evidence_path, evidence)

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("summary_hash", None)
    summary["frozen_v3_input_binding"] = freeze_binding
    summary = reviews._seal(summary, "summary_hash")
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    parser.add_argument("--freeze-input-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    proposed = Path(args.proposed).resolve(strict=True)
    out = Path(args.out).resolve()
    freeze_binding = frozen_v3_input_binding(
        freeze_manifest=Path(args.freeze_manifest),
        expected_manifest_internal_sha256=args.freeze_manifest_internal_sha256,
        frozen_input=proposed,
        input_label=args.freeze_input_label,
    )
    records = reviews._read_jsonl(out / "reviews.jsonl")
    if not records:
        raise SystemExit("checkpoint has no reviews")
    state, models, rubric_hash = common_binding(records, freeze_binding)
    decorate_review_functions(freeze_binding)
    proposals = {
        str(proposal.get("candidate_id") or ""): proposal
        for proposal in reviews._read_jsonl(proposed)
    }
    single_ids = {
        candidate_id
        for candidate_id, proposal in proposals.items()
        if proposal.get("proposal_type") == "single_panel"
    }
    reviewed_single_ids = {
        str(record.get("candidate_id") or "")
        for record in records
        if record.get("proposal_type") == "single_panel"
    }
    if args.verify_only:
        for record in records:
            reviews._validate_review_hash(record)
            if record.get("proposal_type") == "single_panel":
                rubric, families = reviews.REVIEW_RUBRICS[rubric_hash]
                binding, _, _, _, _ = reviews._single_binding(
                    proposals[str(record["candidate_id"])],
                    input_hash=sha256_file(proposed),
                    models=models,
                    git_state=state,
                    rubric=rubric,
                    rubric_hash=rubric_hash,
                    allowed_chart_families=families,
                )
                if binding != record.get("binding"):
                    raise SystemExit("bound V3 runtime does not reproduce checkpoint binding")
        print(
            json.dumps(
                {
                    "binding_reproduced": True,
                    "frozen_v3_input_binding": freeze_binding,
                    "reviewed_single_count": len(reviewed_single_ids),
                },
                sort_keys=True,
            )
        )
        return
    if reviewed_single_ids != single_ids:
        raise SystemExit("checkpoint is incomplete for frozen proposal input")
    result = reviews.review_proposals(
        proposed_path=proposed,
        output_root=out,
        judge_models=models,
        resume=True,
        allow_dirty=True,
        git_state=state,
    )
    summary = seal_top_level_outputs(out, freeze_binding)
    print(
        json.dumps(
            {
                "bound_code_commit": state.commit,
                "bound_code_dirty": state.dirty,
                "judge_models": list(models),
                "frozen_v3_input_binding": freeze_binding,
                "summary": summary,
                "review_count": len(result["reviews"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

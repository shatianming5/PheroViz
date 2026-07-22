#!/usr/bin/env python3
"""Serial, resumable V4 review runner bound to an immutable frozen input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from frozen_v4_binding import decorate_binding, frozen_v4_input_binding


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "files/c2_reclassify_review/v4_bound_runtime/nature_download"
sys.path.insert(0, str(RUNTIME))

from corpus.proposals import PROPOSAL_RULE_V4, _resolve_proposal_rule_version
from corpus.provenance import sha256_file
from corpus.reviews import (
    REVIEW_RUBRIC_V4_HASH,
    REVIEW_RUBRICS,
    ReviewError,
    _read_jsonl,
    _resolve_git_state,
    _sha256_json,
    _single_binding,
    _single_review,
    _validate_models,
    _validate_review_hash,
)


class GatewayUnavailable(RuntimeError):
    """A retryable gateway failure exhausted the bounded retry budget."""


def checkpoint_write(path: Path, records: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate_id in sorted(records):
            handle.write(
                json.dumps(records[candidate_id], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ReviewError("checkpoint-run-binding-mismatch")
        return
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_transient_gateway_failure(review: dict[str, Any]) -> bool:
    return any(
        item.get("failure_code") == "model-call-failed"
        for item in review.get("model_reviews", [])
        if isinstance(item, dict)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    parser.add_argument("--freeze-input-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-delay", type=int, default=10)
    parser.add_argument("--max-transient-retries", type=int, default=8)
    args = parser.parse_args()
    if args.retry_delay < 1 or args.max_transient_retries < 1:
        raise SystemExit("retry settings must be positive")

    proposed_path = Path(args.proposed).resolve(strict=True)
    freeze_binding = frozen_v4_input_binding(
        freeze_manifest=Path(args.freeze_manifest),
        expected_manifest_internal_sha256=args.freeze_manifest_internal_sha256,
        frozen_input=proposed_path,
        input_label=args.freeze_input_label,
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reviews_path = out / "reviews.jsonl"
    log_path = out / "checkpoint.log"
    models = _validate_models(args.judge_model)
    state = _resolve_git_state(None, allow_dirty=True)
    input_hash = sha256_file(proposed_path)
    if input_hash != freeze_binding["frozen_proposal_input"]["sha256"]:
        raise ReviewError("checkpoint-frozen-input-hash-mismatch")
    proposals = _read_jsonl(proposed_path)
    if any(
        proposal.get("proposal_type") in {"single_panel", "multi_panel"}
        and _resolve_proposal_rule_version(proposal) != PROPOSAL_RULE_V4
        for proposal in proposals
    ):
        raise ReviewError("checkpoint-proposal-not-v4")
    by_id: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        candidate_id = proposal.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_id:
            raise ReviewError("checkpoint-proposal-candidate-invalid")
        by_id[candidate_id] = proposal
    selected_ids = (
        set(args.candidate_id)
        if args.candidate_id
        else {
            candidate_id
            for candidate_id, proposal in by_id.items()
            if proposal.get("proposal_type") == "single_panel"
        }
    )
    if not selected_ids or any(
        candidate_id not in by_id
        or by_id[candidate_id].get("proposal_type") != "single_panel"
        for candidate_id in selected_ids
    ):
        raise ReviewError("checkpoint-selected-single-invalid")
    run_binding = {
        "schema_version": "c2-v4-frozen-review-run-v1",
        "frozen_v4_input_binding": freeze_binding,
        "review_rubric_hash": REVIEW_RUBRIC_V4_HASH,
        "judge_models": list(models),
        "code_commit": state.commit,
        "code_dirty": state.dirty,
        "selected_single_candidate_ids": sorted(selected_ids),
    }
    write_json_once(out / "run_binding.json", run_binding)

    existing: dict[str, dict[str, Any]] = {}
    if reviews_path.exists():
        if not args.resume:
            raise ReviewError("checkpoint-output-exists-use-resume")
        for record in _read_jsonl(reviews_path):
            _validate_review_hash(record)
            candidate_id = record.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in selected_ids
                or candidate_id in existing
                or record.get("proposal_type") != "single_panel"
                or record.get("binding", {}).get("frozen_v4_input_binding")
                != freeze_binding
                or record.get("binding", {}).get("rubric_hash")
                != REVIEW_RUBRIC_V4_HASH
            ):
                raise ReviewError("checkpoint-resume-review-invalid")
            existing[candidate_id] = record

    rubric, allowed_chart_families = REVIEW_RUBRICS[REVIEW_RUBRIC_V4_HASH]
    clients: dict[str, Any] = {}

    def get_client(model: str) -> Any:
        if model not in clients:
            agent_path = str(ROOT / "agent")
            if agent_path not in sys.path:
                sys.path.insert(0, agent_path)
            from app.services.model_client import ModelClient

            clients[model] = ModelClient.from_env(model=model)
        return clients[model]

    ordered_ids = sorted(selected_ids)
    log(
        log_path,
        "checkpoint-start "
        f"total_singles={len(ordered_ids)} durable_singles={len(existing)} "
        f"rubric_hash={REVIEW_RUBRIC_V4_HASH}",
    )
    for index, candidate_id in enumerate(ordered_ids, 1):
        proposal = by_id[candidate_id]
        binding, reasons, prompt, _, expected = _single_binding(
            proposal,
            input_hash=input_hash,
            models=models,
            git_state=state,
            rubric=rubric,
            rubric_hash=REVIEW_RUBRIC_V4_HASH,
            allowed_chart_families=allowed_chart_families,
        )
        binding = decorate_binding(
            binding,
            frozen_input_binding=freeze_binding,
            sha256_json=_sha256_json,
        )
        if candidate_id in existing:
            if (
                existing[candidate_id].get("binding", {}).get("resume_binding_hash")
                != binding["resume_binding_hash"]
            ):
                raise ReviewError("checkpoint-resume-binding-mismatch")
            continue
        for retry_attempt in range(args.max_transient_retries + 1):
            record = _single_review(
                proposal,
                binding=binding,
                preflight_reasons=reasons,
                prompt=prompt,
                expected=expected,
                models=models,
                get_client=get_client,
                allowed_chart_families=allowed_chart_families,
            )
            if not is_transient_gateway_failure(record):
                break
            if retry_attempt == args.max_transient_retries:
                log(
                    log_path,
                    "gateway-retry-budget-exhausted "
                    f"candidate={candidate_id} attempts={retry_attempt + 1}",
                )
                raise GatewayUnavailable(
                    f"gateway retry budget exhausted for {candidate_id}; "
                    "no incomplete review was checkpointed"
                )
            delay = min(args.retry_delay * (2**min(retry_attempt, 5)), 300)
            log(
                log_path,
                "gateway-call-failed "
                f"candidate={candidate_id} retry={retry_attempt + 1} "
                f"sleep_seconds={delay}",
            )
            time.sleep(delay)
        existing[candidate_id] = record
        checkpoint_write(reviews_path, existing)
        log(
            log_path,
            "checkpointed "
            f"index={index}/{len(ordered_ids)} durable_singles={len(existing)} "
            f"candidate={candidate_id} status={record.get('status')}",
        )
    log(log_path, f"checkpoint-complete durable_singles={len(existing)}")


if __name__ == "__main__":
    main()

"""CLI handlers for CC-BY corpus discovery, validation, manifests, and splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .benchmark import (
    assemble_verified_benchmark,
    derive_multi_review_batch,
    write_benchmark_outputs,
    write_derived_proposal_outputs,
)
from .cases import (
    DEFAULT_MAX_ZIP_FILES,
    DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
    DEFAULT_MAX_XLSX_SHEETS,
    build_cases,
    write_case_outputs,
)
from .discovery import (
    crossref_discover,
    crossref_lookup,
    evaluate_discovery,
    write_discovery_artifacts,
    write_jsonl,
)
from .policy import evaluate_crossref_item, evaluate_record, normalize_doi
from .proposals import (
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    propose_cases,
    write_proposal_outputs,
)
from .reviews import _resolve_git_state, review_proposals
from .provenance import (
    ProvenanceError,
    build_article_manifest,
    build_corpus_manifest,
    sha256_file,
    validate_article_manifest,
)
from .splits import generate_split_bundle, write_split_bundle


class LicenseGateError(RuntimeError):
    """Raised before a download path can enqueue an ineligible article."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def cmd_discover(args: argparse.Namespace) -> None:
    items = crossref_discover(
        args.query,
        rows=args.max,
        journal=args.journal,
        from_date=args.from_date,
        until_date=args.until_date,
        mailto=args.mailto,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep=args.sleep,
    )
    records = evaluate_discovery(
        items,
        require_cc_by=args.require_cc_by,
    )
    summary = write_discovery_artifacts(records, args.out, query=args.query)
    print(
        "[done] Crossref-only discovery: "
        f"accepted={summary['accepted']} rejected={summary['rejected']} "
        f"total={summary['total']} out={args.out}"
    )


def cmd_validate(args: argparse.Namespace) -> None:
    source = read_jsonl(args.jsonl)
    records: list[dict[str, Any]] = []
    for record in source:
        if args.refresh_crossref:
            doi = normalize_doi(record.get("doi") or record.get("DOI"))
            if not doi:
                decision = evaluate_record(
                    record,
                    require_cc_by=args.require_cc_by,
                )
            else:
                item = crossref_lookup(
                    doi,
                    mailto=args.mailto,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    sleep=args.sleep,
                )
                decision = evaluate_crossref_item(
                    item,
                    require_cc_by=args.require_cc_by,
                )
        else:
            decision = evaluate_record(
                record,
                require_cc_by=args.require_cc_by,
            )
        records.append(decision)
    summary = write_discovery_artifacts(records, args.out, query=None)
    print(
        "[done] Validation: "
        f"accepted={summary['accepted']} rejected={summary['rejected']} "
        f"total={summary['total']} out={args.out}"
    )


def cmd_build_manifest(args: argparse.Namespace) -> None:
    records = read_jsonl(args.jsonl)
    try:
        manifests = build_corpus_manifest(
            records,
            args.content_root,
            require_cc_by=args.require_cc_by,
            source_data_origin=args.source_data_origin,
            reconstructed_verification=args.reconstructed_verification,
            reconstructed_evidence=args.reconstructed_evidence,
        )
    except ProvenanceError as exc:
        raise SystemExit(f"provenance build refused: {exc}") from exc

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "corpus_manifest.jsonl"
    write_jsonl(manifest_path, manifests)
    validation_rows = []
    for manifest in manifests:
        errors = validate_article_manifest(
            manifest,
            content_root=args.content_root,
        )
        validation_rows.append(
            {
                "doi": manifest.get("doi"),
                "valid": not errors,
                "errors": errors,
            }
        )
    write_jsonl(output / "manifest_validation.jsonl", validation_rows)
    invalid = [row for row in validation_rows if not row["valid"]]
    summary = {
        "articles": len(manifests),
        "valid": len(manifests) - len(invalid),
        "invalid": len(invalid),
        "accepted": sum(
            bool(manifest.get("download_eligible")) for manifest in manifests
        ),
        "rejected": sum(
            not bool(manifest.get("download_eligible")) for manifest in manifests
        ),
        "download_statuses": dict(
            sorted(
                Counter(
                    str(manifest.get("download_status")) for manifest in manifests
                ).items()
            )
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] Manifest: articles={summary['articles']} "
        f"valid={summary['valid']} invalid={summary['invalid']} "
        f"sha256={summary['manifest_sha256']} out={manifest_path}"
    )
    if invalid:
        raise SystemExit(2)


def cmd_split(args: argparse.Namespace) -> None:
    records = read_jsonl(args.manifest)
    bundle = generate_split_bundle(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        source_manifest_sha256=sha256_file(args.manifest),
        model_cutoff_date=args.model_cutoff_date,
    )
    write_split_bundle(bundle, args.out)
    print(
        f"[done] Splits: {bundle['counts']} seed={bundle['seed']} "
        f"manifest_sha256={bundle['source_manifest_sha256']} out={args.out}"
    )


def cmd_build_cases(args: argparse.Namespace) -> None:
    code_state = _resolve_git_state(None, allow_dirty=False)
    candidates, ambiguous, summary = build_cases(
        corpus_manifest=args.corpus_manifest,
        content_root=args.content_root,
        output_root=args.out,
        evidence_file=args.evidence,
        max_zip_files=args.max_zip_files,
        max_zip_uncompressed_bytes=args.max_zip_uncompressed_bytes,
        max_xlsx_sheets=args.max_xlsx_sheets,
    )
    summary = write_case_outputs(
        args.out,
        candidates,
        ambiguous,
        summary,
        code_commit=code_state.commit,
        code_dirty=code_state.dirty,
    )
    print(
        f"[done] Cases: candidates={summary['candidates']} "
        f"ambiguous={summary['ambiguous']} verified={summary['verified']} "
        f"eligible={summary['eligible_for_experiment']} out={args.out}"
    )


def cmd_propose_cases(args: argparse.Namespace) -> None:
    candidates_path = Path(args.candidates).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    if output_path == candidates_path.parent:
        raise ValueError(
            "proposal output must differ from the candidates directory "
            "to avoid overwriting case-builder provenance"
        )
    proposed, rejected, summary = propose_cases(
        candidates_path=candidates_path,
        max_file_bytes=args.max_file_bytes,
        max_rows=args.max_rows,
        max_columns=args.max_columns,
    )
    write_proposal_outputs(output_path, proposed, rejected, summary)
    print(
        f"[done] Proposals: single={summary['single_proposals']} "
        f"multi={summary['multi_panel_proposals']} "
        f"rejected={summary['rejected']} eligible=0 out={output_path}"
    )


def cmd_review_proposals(args: argparse.Namespace) -> None:
    result = review_proposals(
        proposed_path=args.proposed,
        output_root=args.out,
        judge_models=args.judge_model,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
    )
    summary = result["summary"]
    print(
        f"[done] Reviews: single={summary['single_accepted']}/"
        f"{summary['single_reviewed']} multi={summary['multi_accepted']}/"
        f"{summary['multi_reviewed']} rejected={summary['rejected']} "
        f"evidence={summary['evidence_records']} out={args.out}"
    )


def cmd_assemble_benchmark(args: argparse.Namespace) -> None:
    code_state = _resolve_git_state(None, allow_dirty=False)
    if not (
        len(args.proposed) == len(args.reviews) == len(args.evidence)
    ):
        raise ValueError(
            "--proposed, --reviews, and --evidence counts must match"
        )
    result = assemble_verified_benchmark(
        candidate_paths=args.candidates,
        review_bundles=zip(
            args.proposed,
            args.reviews,
            args.evidence,
            strict=True,
        ),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        code_commit=code_state.commit,
        code_dirty=code_state.dirty,
    )
    summary = write_benchmark_outputs(args.out, result)
    print(
        f"[done] Benchmark: cases={summary['cases']} "
        f"single={summary['single_cases']} "
        f"multi={summary['multi_panel_cases']} "
        f"dois={summary['unique_dois']} "
        f"sha256={summary['benchmark_manifest_sha256']} out={args.out}"
    )


def cmd_derive_multi_proposals(args: argparse.Namespace) -> None:
    if not (
        len(args.proposed) == len(args.reviews) == len(args.evidence)
    ):
        raise ValueError(
            "--proposed, --reviews, and --evidence counts must match"
        )
    code_state = _resolve_git_state(None, allow_dirty=False)
    result = derive_multi_review_batch(
        review_bundles=zip(
            args.proposed,
            args.reviews,
            args.evidence,
            strict=True,
        ),
        code_commit=code_state.commit,
        code_dirty=code_state.dirty,
    )
    summary = write_derived_proposal_outputs(args.out, result)
    print(
        f"[done] Derived proposals: single="
        f"{summary['accepted_single_proposals']} multi="
        f"{summary['derived_multi_panel_proposals']} "
        f"eligible=0 sha256={summary['proposed_sha256']} out={args.out}"
    )


def _doi_from_nature_url(url: str) -> str | None:
    match = re.search(r"/articles/([^/?#]+)", url)
    return f"10.1038/{match.group(1)}" if match else None


def authorize_direct_download(
    *,
    url: str,
    doi: str | None,
    require_cc_by: bool,
    mailto: str | None,
    timeout: float,
    max_retries: int,
    sleep: float,
) -> dict[str, Any]:
    if not require_cc_by:
        raise LicenseGateError(
            "download refused: pass --require-cc-by and provide verifiable "
            "CC BY metadata"
        )
    resolved_doi = normalize_doi(doi) or _doi_from_nature_url(url)
    if not resolved_doi:
        raise LicenseGateError("download refused: DOI cannot be determined")
    item = crossref_lookup(
        resolved_doi,
        mailto=mailto,
        timeout=timeout,
        max_retries=max_retries,
        sleep=sleep,
    )
    decision = evaluate_crossref_item(item, require_cc_by=True)
    if not decision.get("download_eligible"):
        reasons = ",".join(decision.get("reject_reasons") or ["unknown"])
        raise LicenseGateError(
            f"download refused for {resolved_doi}: {reasons}"
        )
    return decision


def require_record_download_eligibility(
    record: dict[str, Any],
    *,
    require_cc_by: bool,
) -> dict[str, Any]:
    if not require_cc_by:
        raise LicenseGateError(
            "download queue refused: --require-cc-by is mandatory"
        )
    decision = evaluate_record(record, require_cc_by=True)
    if not decision.get("download_eligible"):
        reasons = ",".join(decision.get("reject_reasons") or ["unknown"])
        raise LicenseGateError(
            f"download queue refused for {decision.get('doi')}: {reasons}"
        )
    return decision


def write_article_provenance(
    record: dict[str, Any],
    content_root: str | Path,
    *,
    download_status: str,
    rejection_reason: str | None = None,
) -> Path | None:
    manifest = build_article_manifest(
        record,
        content_root,
        require_cc_by=True,
        download_status=download_status,
        rejection_reason=rejection_reason,
    )
    article_url = str(manifest.get("article_url") or "")
    match = re.search(r"/articles/([^/?#]+)", article_url)
    if not match:
        return None
    article_dir = Path(content_root) / match.group(1)
    output = (
        article_dir / "meta" / "provenance.json"
        if article_dir.exists()
        else Path(content_root) / "_provenance" / f"{match.group(1)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def add_corpus_subcommands(subparsers: argparse._SubParsersAction) -> None:
    boolean_optional = argparse.BooleanOptionalAction

    discover = subparsers.add_parser(
        "discover",
        help="Crossref-only discovery with the CC-BY gate enabled by default",
    )
    discover.add_argument(
        "--query",
        default="",
        help="Optional topical query; omit for journal-wide discovery",
    )
    discover.add_argument("--max", type=int, default=10)
    discover.add_argument(
        "--journal",
        default=None,
        help=(
            "Optional Crossref journal query, e.g. 'Nature Communications', "
            "'Scientific Reports', or 'npj'; strict allowlist filtering still applies"
        ),
    )
    discover.add_argument("--from-date", default=None, help="YYYY-MM-DD")
    discover.add_argument("--until-date", default=None, help="YYYY-MM-DD")
    discover.add_argument("--out", default="outputs/corpus_discovery")
    discover.add_argument("--mailto", default=None)
    discover.add_argument("--sleep", type=float, default=1.0)
    discover.add_argument("--timeout", type=float, default=30)
    discover.add_argument("--max-retries", type=int, default=3)
    discover.add_argument(
        "--require-cc-by",
        action=boolean_optional,
        default=True,
        help="Require verified CC BY 3.0/4.0 (default: enabled)",
    )
    discover.set_defaults(func=cmd_discover)

    validate = subparsers.add_parser(
        "validate",
        help="Revalidate discovery JSONL and emit accepted/rejected queues",
    )
    validate.add_argument("--jsonl", required=True)
    validate.add_argument("--out", default="outputs/corpus_validation")
    validate.add_argument("--refresh-crossref", action="store_true")
    validate.add_argument("--mailto", default=None)
    validate.add_argument("--sleep", type=float, default=1.0)
    validate.add_argument("--timeout", type=float, default=30)
    validate.add_argument("--max-retries", type=int, default=3)
    validate.add_argument(
        "--require-cc-by",
        action=boolean_optional,
        default=True,
    )
    validate.set_defaults(func=cmd_validate)

    manifest = subparsers.add_parser(
        "build-manifest",
        help="Build checksummed article provenance manifests",
    )
    manifest.add_argument("--jsonl", required=True)
    manifest.add_argument("--content-root", required=True)
    manifest.add_argument("--out", default="outputs/corpus_manifest")
    manifest.add_argument(
        "--source-data-origin",
        choices=["supplementary_information", "reconstructed"],
        default=None,
        help="Required for untracked source-data files; never inferred",
    )
    manifest.add_argument(
        "--reconstructed-verification",
        choices=["unverified", "verified"],
        default="unverified",
    )
    manifest.add_argument("--reconstructed-evidence", default=None)
    manifest.add_argument(
        "--require-cc-by",
        action=boolean_optional,
        default=True,
    )
    manifest.set_defaults(func=cmd_build_manifest)

    split = subparsers.add_parser(
        "split",
        help="Generate deterministic DOI-disjoint paper-level splits",
    )
    split.add_argument("--manifest", required=True)
    split.add_argument("--out", default="outputs/corpus_splits")
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument(
        "--model-cutoff-date",
        default=None,
        help="Explicit YYYY-MM-DD cutoff; omitted means unconfigured strata",
    )
    split.set_defaults(func=cmd_split)

    cases = subparsers.add_parser(
        "build-cases",
        help="Build fail-closed benchmark candidates from Source Data",
    )
    cases.add_argument("--corpus-manifest", required=True)
    cases.add_argument("--content-root", required=True)
    cases.add_argument("--out", required=True)
    cases.add_argument(
        "--evidence",
        default=None,
        help="Explicit human/external verification evidence JSON",
    )
    cases.add_argument(
        "--max-zip-files",
        type=int,
        default=DEFAULT_MAX_ZIP_FILES,
    )
    cases.add_argument(
        "--max-zip-uncompressed-bytes",
        type=int,
        default=DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
    )
    cases.add_argument(
        "--max-xlsx-sheets",
        type=int,
        default=DEFAULT_MAX_XLSX_SHEETS,
    )
    cases.set_defaults(func=cmd_build_cases)

    proposals = subparsers.add_parser(
        "propose-cases",
        help="Deterministically propose experiment cases without verification",
    )
    proposals.add_argument("--candidates", required=True)
    proposals.add_argument("--out", required=True)
    proposals.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
    )
    proposals.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
    )
    proposals.add_argument(
        "--max-columns",
        type=int,
        default=DEFAULT_MAX_COLUMNS,
    )
    proposals.set_defaults(func=cmd_propose_cases)

    reviews = subparsers.add_parser(
        "review-proposals",
        help="Externally validate proposals with at least two distinct models",
    )
    reviews.add_argument("--proposed", required=True)
    reviews.add_argument("--out", required=True)
    reviews.add_argument(
        "--judge-model",
        action="append",
        required=True,
        help="Repeat for at least two distinct judge models",
    )
    reviews.add_argument("--resume", action="store_true")
    reviews.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit dirty code only when explicitly requested",
    )
    reviews.set_defaults(func=cmd_review_proposals)

    benchmark = subparsers.add_parser(
        "assemble-benchmark",
        help="Assemble verified cases into a DOI-disjoint experiment manifest",
    )
    benchmark.add_argument(
        "--candidates",
        action="append",
        required=True,
        help="Repeat for each rebuilt case-builder candidates.jsonl",
    )
    benchmark.add_argument("--evidence", action="append", required=True)
    benchmark.add_argument("--proposed", action="append", required=True)
    benchmark.add_argument("--reviews", action="append", required=True)
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--seed", type=int, required=True)
    benchmark.add_argument("--train-ratio", type=float, default=0.8)
    benchmark.add_argument("--val-ratio", type=float, default=0.1)
    benchmark.add_argument("--test-ratio", type=float, default=0.1)
    benchmark.set_defaults(func=cmd_assemble_benchmark)

    derived = subparsers.add_parser(
        "derive-multi-proposals",
        help="Derive canonical multi-panel proposals from reviewed singles",
    )
    derived.add_argument("--proposed", action="append", required=True)
    derived.add_argument("--reviews", action="append", required=True)
    derived.add_argument("--evidence", action="append", required=True)
    derived.add_argument("--out", required=True)
    derived.set_defaults(func=cmd_derive_multi_proposals)

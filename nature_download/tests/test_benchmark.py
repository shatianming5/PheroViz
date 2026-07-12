from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from nature_download.corpus.benchmark import (
    BenchmarkBuildError,
    assemble_verified_benchmark,
    derive_multi_review_batch,
    write_benchmark_outputs,
    write_derived_proposal_outputs,
)
from nature_download.corpus.cases import write_case_outputs
from nature_download.corpus.provenance import sha256_file
from nature_download.corpus.proposals import (
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    _multi_panel_proposals,
    propose_single_candidate,
)
from nature_download.corpus.reviews import review_proposals
from .test_reviews import (
    CLEAN_GIT,
    MODELS,
    factory_for,
    make_single,
    write_proposed,
)


def _review_bundle(
    root: Path,
    proposals: list[dict],
) -> tuple[Path, Path, Path, dict[str, dict]]:
    for proposal in proposals:
        if proposal.get("proposal_type") != "single_panel":
            continue
        source = proposal["source_table"]
        source.update(
            {
                "format": "csv",
                "path_root": "content_root",
                "relative_path": Path(source["path"]).name,
                "archive": None,
            }
        )
    proposed = root / "proposed.jsonl"
    write_proposed(proposed, proposals)
    factory, _ = factory_for()
    output = root / "reviews"
    result = review_proposals(
        proposed_path=proposed,
        output_root=output,
        judge_models=MODELS,
        client_factory=factory,
        git_state=CLEAN_GIT,
    )
    evidence = {
        item["candidate_id"]: item
        for item in result["evidence"]["verifications"]
    }
    return proposed, output / "reviews.jsonl", output / "evidence.json", evidence


def _canonical_single(
    root: Path,
    *,
    candidate_id: str,
    panel_id: str,
    doi: str,
    corpus_manifest_sha256: str,
) -> dict:
    candidate = make_single(
        root,
        candidate_id=candidate_id,
        panel_id=panel_id,
    )
    candidate["doi"] = doi
    candidate["corpus_manifest_sha256"] = corpus_manifest_sha256
    source = candidate["source_table"]
    source.update(
        {
            "format": "csv",
            "path_root": "content_root",
            "relative_path": Path(source["path"]).name,
            "archive": None,
        }
    )
    return propose_single_candidate(
        candidate,
        input_candidates_sha256="f" * 64,
        code_commit="2" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
    )


def _candidate_from_proposal(
    proposal: dict,
    *,
    verification: dict,
    evidence_sha256: str,
) -> dict:
    source = deepcopy(proposal["source_table"])
    case = deepcopy(proposal["experiment_case"])
    case.update(
        {
            "data_path": source["path"],
            "sheet": source.get("sheet_name"),
            "data_sha256": source["sha256"],
            "curation_status": "verified",
            "eligible_for_experiment": True,
            "eligibility_reasons": [],
        }
    )
    return {
        key: deepcopy(proposal.get(key))
        for key in (
            "candidate_id",
            "doi",
            "figure_no",
            "panel_ids",
            "panel_qualifier",
            "source_table",
            "figure",
            "caption",
            "license_evidence_sha256",
            "corpus_manifest_sha256",
            "provenance_manifest_sha256",
        )
    } | {
        "curation_status": "verified",
        "eligible_for_experiment": True,
        "verification_evidence": verification,
        "verification_evidence_file_sha256": evidence_sha256,
        "experiment_case": case,
    }


def _write_case_batch(
    root: Path,
    *,
    records: list[dict],
    evidence_sha256: str,
    corpus_manifest: Path,
) -> Path:
    summary = {
        "schema_version": "1.0",
        "corpus_manifest": str(corpus_manifest.resolve()),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "content_root": str(corpus_manifest.parent.resolve()),
        "output_root": str(root.resolve()),
        "evidence_file_sha256": evidence_sha256,
        "candidates": len(records),
        "verified": len(records),
        "eligible_for_experiment": len(records),
    }
    write_case_outputs(
        root,
        records,
        [],
        summary,
        code_commit="1" * 40,
        code_dirty=False,
    )
    return root / "candidates.jsonl"


def _manifest_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "corpus_manifest.jsonl"
    path.write_text('{"download_eligible":true}\n', encoding="utf-8")
    return path


def test_verified_benchmark_is_doi_disjoint_and_sealed(tmp_path: Path) -> None:
    corpus_manifest = _manifest_fixture(tmp_path)
    corpus_hash = sha256_file(corpus_manifest)
    specifications = [
        ("case-a1", "10.1038/article-a", "a"),
        ("case-a2", "10.1038/article-a", "b"),
        ("case-b1", "10.1038/article-b", "a"),
        ("case-c1", "10.1038/article-c", "a"),
    ]
    proposals = []
    for candidate_id, doi, panel_id in specifications:
        proposal = _canonical_single(
            tmp_path,
            candidate_id=candidate_id,
            panel_id=panel_id,
            doi=doi,
            corpus_manifest_sha256=corpus_hash,
        )
        proposals.append(proposal)
    proposed, reviews, evidence_path, evidence = _review_bundle(
        tmp_path,
        proposals,
    )
    evidence_sha256 = sha256_file(evidence_path)
    records = [
        _candidate_from_proposal(
            proposal,
            verification=evidence[proposal["candidate_id"]],
            evidence_sha256=evidence_sha256,
        )
        for proposal in proposals
    ]
    first = _write_case_batch(
        tmp_path / "batch-a",
        records=records[:2],
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )
    second = _write_case_batch(
        tmp_path / "batch-b",
        records=records[2:],
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )

    result = assemble_verified_benchmark(
        candidate_paths=[first, second],
        evidence_path=evidence_path,
        proposed_path=proposed,
        reviews_path=reviews,
        seed=41,
        code_commit="a" * 40,
        code_dirty=False,
    )
    manifest_cases = result["manifest"]["cases"]
    splits_by_id = {case["case_id"]: case["split"] for case in manifest_cases}
    assert splits_by_id["case-a1"] == splits_by_id["case-a2"]
    assert result["summary"]["cases"] == 4
    assert result["summary"]["unique_dois"] == 3
    assert all(
        result["summary"]["paper_split_counts"][split_name] == 1
        for split_name in ("train", "val", "test")
    )

    output = tmp_path / "benchmark"
    summary = write_benchmark_outputs(output, result)
    assert sha256_file(output / "benchmark_manifest.json") == summary[
        "benchmark_manifest_sha256"
    ]
    with pytest.raises(BenchmarkBuildError, match="output-not-empty"):
        write_benchmark_outputs(output, result)


def test_verified_benchmark_rejects_data_and_doi_tampering(
    tmp_path: Path,
) -> None:
    corpus_manifest = _manifest_fixture(tmp_path)
    proposal = _canonical_single(
        tmp_path,
        candidate_id="case-a",
        panel_id="a",
        doi="10.1038/example",
        corpus_manifest_sha256=sha256_file(corpus_manifest),
    )
    proposed, reviews, evidence_path, evidence = _review_bundle(
        tmp_path,
        [proposal],
    )
    evidence_sha256 = sha256_file(evidence_path)
    record = _candidate_from_proposal(
        proposal,
        verification=evidence["case-a"],
        evidence_sha256=evidence_sha256,
    )
    candidates = _write_case_batch(
        tmp_path / "batch",
        records=[record],
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )
    Path(record["experiment_case"]["data_path"]).write_text(
        "Category,Value\nA,999\n",
        encoding="utf-8",
    )
    with pytest.raises(
        BenchmarkBuildError,
        match="validation-single-binding-mismatch",
    ):
        assemble_verified_benchmark(
            candidate_paths=[candidates],
            evidence_path=evidence_path,
            proposed_path=proposed,
            reviews_path=reviews,
            seed=1,
            code_commit="b" * 40,
            code_dirty=False,
        )

    Path(record["experiment_case"]["data_path"]).write_text(
        "Category,Value\nA,1\nB,2\n",
        encoding="utf-8",
    )
    record["doi"] = "10.1038/attacker-controlled"
    _write_case_batch(
        tmp_path / "batch",
        records=[record],
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )
    with pytest.raises(
        BenchmarkBuildError,
        match="candidate-proposal-identity-mismatch",
    ):
        assemble_verified_benchmark(
            candidate_paths=[candidates],
            evidence_path=evidence_path,
            proposed_path=proposed,
            reviews_path=reviews,
            seed=1,
            code_commit="b" * 40,
            code_dirty=False,
        )


def test_multi_case_requires_all_sources_and_rebinds_checked_paths(
    tmp_path: Path,
) -> None:
    corpus_manifest = _manifest_fixture(tmp_path)
    corpus_hash = sha256_file(corpus_manifest)
    first = _canonical_single(
        tmp_path,
        candidate_id="case-a",
        panel_id="a",
        doi="10.1038/example",
        corpus_manifest_sha256=corpus_hash,
    )
    second = _canonical_single(
        tmp_path,
        candidate_id="case-b",
        panel_id="b",
        doi="10.1038/example",
        corpus_manifest_sha256=corpus_hash,
    )
    multi = _multi_panel_proposals(
        [first, second],
        input_candidates_sha256="f" * 64,
        code_commit="2" * 40,
    )[0]
    proposed, reviews, evidence_path, evidence = _review_bundle(
        tmp_path,
        [first, second, multi],
    )
    evidence_sha256 = sha256_file(evidence_path)
    records = [
        _candidate_from_proposal(
            proposal,
            verification=evidence[proposal["candidate_id"]],
            evidence_sha256=evidence_sha256,
        )
        for proposal in (first, second)
    ]
    candidates = _write_case_batch(
        tmp_path / "batch",
        records=records,
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )

    result = assemble_verified_benchmark(
        candidate_paths=[candidates],
        evidence_path=evidence_path,
        proposed_path=proposed,
        reviews_path=reviews,
        seed=3,
        code_commit="c" * 40,
        code_dirty=False,
    )
    assert result["summary"]["multi_panel_cases"] == 1
    multi_case = next(
        case
        for case in result["manifest"]["cases"]
        if case["case_id"] == multi["candidate_id"]
    )
    assert [panel["data_path"] for panel in multi_case["panels"]] == [
        records[0]["experiment_case"]["data_path"],
        records[1]["experiment_case"]["data_path"],
    ]
    assert all(panel["data_sha256"] for panel in multi_case["panels"])

    candidates = _write_case_batch(
        tmp_path / "batch",
        records=records[:1],
        evidence_sha256=evidence_sha256,
        corpus_manifest=corpus_manifest,
    )
    with pytest.raises(
        BenchmarkBuildError,
        match="multi-proposal-not-canonical",
    ):
        assemble_verified_benchmark(
            candidate_paths=[candidates],
            evidence_path=evidence_path,
            proposed_path=proposed,
            reviews_path=reviews,
            seed=3,
            code_commit="c" * 40,
            code_dirty=False,
        )


def test_multiple_review_bundles_merge_without_rejudging(
    tmp_path: Path,
) -> None:
    candidate_paths = []
    review_bundles = []
    for suffix in ("a", "b"):
        root = tmp_path / f"bundle-{suffix}"
        root.mkdir()
        corpus_manifest = _manifest_fixture(root)
        proposal = _canonical_single(
            root,
            candidate_id=f"case-{suffix}",
            panel_id="a",
            doi=f"10.1038/article-{suffix}",
            corpus_manifest_sha256=sha256_file(corpus_manifest),
        )
        proposed, reviews, evidence_path, evidence = _review_bundle(
            root,
            [proposal],
        )
        evidence_sha256 = sha256_file(evidence_path)
        record = _candidate_from_proposal(
            proposal,
            verification=evidence[proposal["candidate_id"]],
            evidence_sha256=evidence_sha256,
        )
        candidate_paths.append(
            _write_case_batch(
                root / "cases",
                records=[record],
                evidence_sha256=evidence_sha256,
                corpus_manifest=corpus_manifest,
            )
        )
        review_bundles.append((proposed, reviews, evidence_path))

    result = assemble_verified_benchmark(
        candidate_paths=candidate_paths,
        review_bundles=review_bundles,
        seed=19,
        code_commit="d" * 40,
        code_dirty=False,
    )
    assert result["summary"]["cases"] == 2
    assert result["summary"]["unique_dois"] == 2
    assert len(
        result["manifest"]["provenance"]["source_binding"]["review_bundles"]
    ) == 2


def test_derive_multi_batch_uses_only_reviewed_singles(
    tmp_path: Path,
) -> None:
    corpus_manifest = _manifest_fixture(tmp_path)
    corpus_hash = sha256_file(corpus_manifest)
    singles = [
        _canonical_single(
            tmp_path,
            candidate_id=f"case-{panel_id}",
            panel_id=panel_id,
            doi="10.1038/shared-article",
            corpus_manifest_sha256=corpus_hash,
        )
        for panel_id in ("a", "b")
    ]
    proposed, reviews, evidence_path, _ = _review_bundle(
        tmp_path,
        singles,
    )
    result = derive_multi_review_batch(
        review_bundles=[(proposed, reviews, evidence_path)],
        code_commit="e" * 40,
        code_dirty=False,
    )
    assert result["summary"]["accepted_single_proposals"] == 2
    assert result["summary"]["derived_multi_panel_proposals"] == 1
    multi = [
        proposal
        for proposal in result["proposals"]
        if proposal["proposal_type"] == "multi_panel"
    ][0]
    assert multi["source_candidate_ids"] == ["case-a", "case-b"]
    assert multi["experiment_case"]["panel_count"] == 2

    summary = write_derived_proposal_outputs(
        tmp_path / "derived",
        result,
    )
    assert summary["proposals_total"] == 3
    assert sha256_file(tmp_path / "derived" / "proposed.jsonl") == summary[
        "proposed_sha256"
    ]

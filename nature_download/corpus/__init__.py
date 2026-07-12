"""Fail-closed corpus licensing, provenance, and split utilities."""

from .policy import (
    ALLOWED_JOURNALS,
    evaluate_crossref_item,
    evaluate_record,
    is_allowed_journal,
    normalize_cc_by_url,
)
from .cases import build_cases, parse_figure_panel, safe_extract_tables
from .benchmark import assemble_verified_benchmark, write_benchmark_outputs
from .provenance import (
    build_article_manifest,
    build_corpus_manifest,
    sha256_file,
    validate_article_manifest,
)
from .proposals import propose_cases, write_proposal_outputs
from .reviews import review_proposals, validate_review_artifacts
from .splits import generate_split_bundle

__all__ = [
    "ALLOWED_JOURNALS",
    "assemble_verified_benchmark",
    "build_article_manifest",
    "build_cases",
    "build_corpus_manifest",
    "evaluate_crossref_item",
    "evaluate_record",
    "generate_split_bundle",
    "is_allowed_journal",
    "normalize_cc_by_url",
    "parse_figure_panel",
    "propose_cases",
    "review_proposals",
    "safe_extract_tables",
    "sha256_file",
    "validate_article_manifest",
    "validate_review_artifacts",
    "write_proposal_outputs",
    "write_benchmark_outputs",
]

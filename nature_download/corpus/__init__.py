"""Fail-closed corpus licensing, provenance, and split utilities."""

from .policy import (
    ALLOWED_JOURNALS,
    evaluate_crossref_item,
    evaluate_record,
    is_allowed_journal,
    normalize_cc_by_url,
)
from .provenance import (
    build_article_manifest,
    build_corpus_manifest,
    sha256_file,
    validate_article_manifest,
)
from .splits import generate_split_bundle

__all__ = [
    "ALLOWED_JOURNALS",
    "build_article_manifest",
    "build_corpus_manifest",
    "evaluate_crossref_item",
    "evaluate_record",
    "generate_split_bundle",
    "is_allowed_journal",
    "normalize_cc_by_url",
    "sha256_file",
    "validate_article_manifest",
]

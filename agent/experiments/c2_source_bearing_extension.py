"""Fail-closed V2 source-bearing evidence construction for C2 remediation.

This module is intentionally self contained.  It consumes only descriptor-bound
source bytes already opened by the raw-evidence reader and writes only through a
retained private-staging descriptor supplied by the remediation finalizer.  It
does not call models, download data, inspect a directory by pathname, or infer a
case from an unbound filename.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import stat
import struct
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlsplit


class SourceBearingExtensionError(ValueError):
    """Raised when V2 source/canonical evidence is incomplete or tampered."""


class _TargetRoot(Protocol):
    """The descriptor-only subset of the remediation staging root we require."""

    def write_bytes(self, relative: str, payload: bytes) -> str: ...

    def read_bytes(self, relative: str) -> bytes: ...

    def sha256(self, relative: str) -> str: ...


SCHEMA_VERSION = "c2-source-bearing-remediation-v2"
REVIEW_MODE = "C2_V2_STRUCTURAL_REVIEW_V1"
FORMAT_CLASSIFIER_ID = "c2_v2_fd_format_classifier_v1"
ZIP_PARSER_ID = "c2_v2_zip_v1_central_directory_parser"
XLSX_PROFILE_PARSER_ID = "c2_v2_xlsx_profile_parser_v1"
CANDIDATE_RULE_ID = "c2_v2_candidate_builder_v1"
CANONICAL_RULE_ID = "c2_v2_canonical_builder_v1"
P_RULE_ID = "c2_v2_panel_stratum_v1"
MAX_CONTAINER_DEPTH = 4
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_CONTAINER_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_CONTAINER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_RUN_COMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_RUN_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024

_EXTENSION_RELATIVE_PATH = "agent/experiments/c2_source_bearing_extension.py"
_FINALIZER_RELATIVE_PATH = "agent/experiments/c2_remediation_root_finalizer.py"
_CLI_RELATIVE_PATH = "agent/experiments/cli.py"
_MODELS_RELATIVE_PATH = "agent/experiments/models.py"
_CODE_ATTESTATION_RELATIVE_PATH = (
    "agent/experiments/c2_source_bearing_extension_code_attestation.json"
)
_REQUIRED_SCHEMA_NAMES = (
    "c2_v2_fd_format_classifier_config_v1.schema.json",
    "c2_v2_detected_format_v1.schema.json",
    "c2_v2_container_accounting_index_v1.schema.json",
    "c2_v2_consumable_source_unit_v1.schema.json",
    "c2_v2_downstream_consumption_v1.schema.json",
    "c2_v2_candidate_set_input_v1.schema.json",
    "c2_v2_consumption_bijection_validation_v1.schema.json",
)
_REQUIRED_ATTESTED_CODE_PATHS = frozenset(
    {
        _EXTENSION_RELATIVE_PATH,
        _FINALIZER_RELATIVE_PATH,
        _CLI_RELATIVE_PATH,
        _MODELS_RELATIVE_PATH,
        *(
            f"agent/experiments/schemas/{name}" for name in _REQUIRED_SCHEMA_NAMES
        ),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP64_MARKER = 0xFFFFFFFF

_PROHIBITED_LEGACY_FIELDS = frozenset(
    {
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "created",
        "updated",
        "absolute_path",
        "path",
        "root",
        "worktree",
        "supersedes",
        "superseded_by",
        "preferred_version",
        "resume",
        "selection",
        "selected",
        "fallback",
        "model",
        "model_result",
        "model_results",
        "benchmark_split",
        "metric",
        "metrics",
        "trend",
        "trends",
        "equivalence",
        "coverage",
        "download_status",
        "asserted_panel_count",
        "asserted_panel_ids",
        "asserted_public_stratum",
        "asserted_code_label",
    }
)

_TERMINAL_STATUS_ADAPTER = {
    "downloaded": "DOWNLOADED",
    "no-source-data": "NO_SOURCE_DATA",
    "no-figures": "NO_FIGURES",
    "no-usable-content": "NO_USABLE_CONTENT",
    "policy-rejected": "POLICY_REJECTED",
    "fetch-error": "DOWNLOAD_FAILED",
    "download-failed": "DOWNLOAD_FAILED",
    "retry-exhausted": "RETRY_EXHAUSTED",
}
_FINAL_DISPOSITION_BY_STATUS = {
    "NO_SOURCE_DATA": "NON_STRATIFIED_NO_SOURCE_DATA",
    "NO_FIGURES": "NON_STRATIFIED_NO_FIGURES",
    "NO_USABLE_CONTENT": "NON_STRATIFIED_NO_USABLE_CONTENT",
    "POLICY_REJECTED": "NON_STRATIFIED_POLICY_REJECTED",
    "DOWNLOAD_FAILED": "NON_STRATIFIED_DOWNLOAD_FAILED",
    "RETRY_EXHAUSTED": "NON_STRATIFIED_RETRY_EXHAUSTED",
}
_STRUCTURAL_OUTCOMES = frozenset(
    {"ACCEPTED_STRUCTURAL", "REJECTED_DUPLICATE_PANEL_GROUP"}
)
_CONSUMPTION_DISPOSITIONS = frozenset(
    {"CANDIDATE_SET_INPUT", "SOURCE_ONLY_EXCLUSION", "REJECTION"}
)

_DECLARED_TUPLES = {
    ("NONE", "CSV_V1"),
    ("NONE", "OTHER_REGISTERED_V1"),
    ("ZIP_V1", "XLSX_V1"),
    ("ZIP_V1", "GENERIC_ZIP_V1"),
}
_DECLARED_KIND_TUPLES = {
    "source_data": {
        ("NONE", "CSV_V1"),
        ("ZIP_V1", "XLSX_V1"),
    },
    "source_archive": {("ZIP_V1", "GENERIC_ZIP_V1")},
    "figure": {("NONE", "OTHER_REGISTERED_V1")},
    "caption": {("NONE", "OTHER_REGISTERED_V1")},
}
_SOURCE_ONLY_REASONS = frozenset(
    {
        "CONTAINER_EXPANDED_RECURSIVELY",
        "UNREGISTERED_MEMBER_TYPE",
        "NO_BOUND_FIGURE_CAPTION",
        "AMBIGUOUS_SOURCE_MAPPING",
        "TABLE_PARSE_FAILURE",
        "XLSX_PACKAGE_COMPONENT",
        "DIRECTORY_ENTRY",
        "RESOURCE_FORK",
    }
)

# Every generated semantic object is closed before it is hashed.  The static
# JSON schemas in ``schemas/`` mirror the externally consumed core records;
# this in-process registry keeps the remaining nested records equally closed.
_CLOSED_FIELDS: dict[str, frozenset[str]] = {
    "c2_v2_fd_format_classifier_config_v1": frozenset(
        {
            "schema_version",
            "format_classifier_id",
            "format_classifier_version",
            "format_classifier_code_sha256",
            "zip_v1_parser_id",
            "zip_v1_parser_version",
            "zip_v1_parser_code_sha256",
            "xlsx_profile_parser_id",
            "xlsx_profile_parser_version",
            "xlsx_profile_parser_code_sha256",
            "noncontainer_parser_registry_id",
            "noncontainer_parser_registry_version",
            "noncontainer_parser_registry_hash",
            "format_detection_rule_id",
            "format_detection_rule_version",
            "format_detection_rule_hash",
            "format_error_enum_version",
            "format_error_enum_hash",
            "maximum_container_depth",
            "schema_hashes",
            "closed_schema_registry_hash",
            "approved_implementation_commit_full",
            "attestation_commit_full",
            "code_attestation_manifest_sha256",
            "attested_code_blobs_sha256",
            "review_mode",
            "review_protocol_hash",
            "config_hash",
        }
    ),
    "c2_v2_raw_source_asset_v1": frozenset(
        {
            "schema_version",
            "parent_doi_id",
            "frozen_input_ordinal",
            "raw_article_id",
            "asset_id",
            "declared_asset_kind",
            "declared_format_tuple",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "source_descriptor_relative_path",
            "source_descriptor_sha256",
            "provenance_relative_path",
            "asset_record_hash",
        }
    ),
    "c2_v2_detected_format_v1": frozenset(
        {
            "schema_version",
            "origin_record_hash",
            "parent_doi_id",
            "frozen_input_ordinal",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "container_format",
            "content_profile",
            "container_node_id_or_null",
            "format_classifier_id",
            "format_classifier_version",
            "format_classifier_code_sha256",
            "format_detection_rule_hash",
            "format_hash",
        }
    ),
    "c2_v2_archive_central_entry_v1": frozenset(
        {
            "schema_version",
            "central_index",
            "raw_name_sha256",
            "selector",
            "selector_nfc",
            "selector_casefold",
            "central_header_sha256",
            "compression_method",
            "flag_bits",
            "crc32",
            "compressed_bytes",
            "uncompressed_bytes",
            "external_attributes",
            "local_header_offset",
            "derived_member_record_hash_or_null",
            "source_only_exclusion_reason_or_null",
            "archive_source_only_disposition_record_hash_or_null",
            "child_container_node_id_or_null",
            "disposition",
            "entry_hash",
        }
    ),
    "c2_v2_container_index_entry_v1": frozenset(
        {
            "schema_version",
            "container_node_id",
            "parent_container_node_id_or_null",
            "origin_record_hash",
            "parent_doi_id",
            "frozen_input_ordinal",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "depth",
            "content_profile",
            "account_relative_path",
            "account_sha256",
            "archive_accounting_hash",
            "entry_hash",
        }
    ),
    "c2_v2_container_accounting_index_v1": frozenset(
        {
            "schema_version",
            "format_classifier_config_hash",
            "container_nodes",
            "index_hash",
        }
    ),
    "c2_v2_archive_accounting_manifest_v1": frozenset(
        {
            "schema_version",
            "container_node_id",
            "origin_record_hash",
            "parent_doi_id",
            "frozen_input_ordinal",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "container_format",
            "content_profile",
            "format_classifier_id",
            "format_classifier_version",
            "format_classifier_code_sha256",
            "zip_v1_parser_id",
            "zip_v1_parser_version",
            "zip_v1_parser_code_sha256",
            "central_directory_sha256",
            "eocd_sha256",
            "physical_entry_count",
            "central_entry_uncompressed_bytes",
            "regular_member_uncompressed_bytes",
            "derived_member_uncompressed_bytes",
            "source_only_excluded_uncompressed_bytes",
            "entries",
            "archive_accounting_hash",
        }
    ),
    "c2_v2_derived_archive_member_v1": frozenset(
        {
            "schema_version",
            "derived_member_id",
            "container_node_id",
            "central_index",
            "central_header_sha256",
            "parent_doi_id",
            "frozen_input_ordinal",
            "raw_article_id",
            "raw_asset_id",
            "member_selector",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "detected_format_tuple",
            "record_hash",
        }
    ),
    "c2_v2_archive_source_only_disposition_v1": frozenset(
        {
            "schema_version",
            "container_node_id",
            "central_index",
            "central_header_sha256",
            "parent_doi_id",
            "reason_code",
            "record_hash",
        }
    ),
    "c2_v2_consumable_source_unit_v1": frozenset(
        {
            "schema_version",
            "unit_id",
            "parent_doi_id",
            "frozen_input_ordinal",
            "origin_kind",
            "origin_record_hash",
            "container_node_id_or_null",
            "verified_relative_path",
            "verified_file_sha256",
            "verified_bytes",
            "detected_format_tuple",
            "format_classifier_id",
            "format_classifier_version",
            "format_classifier_code_sha256",
            "format_detection_rule_hash",
            "raw_article_id",
            "raw_asset_id",
            "member_selector_or_null",
            "candidate_capable",
            "unit_hash",
        }
    ),
    "c2_v2_source_only_exclusion_v1": frozenset(
        {
            "schema_version",
            "unit_id",
            "unit_hash",
            "parent_doi_id",
            "reason_code",
            "record_hash",
        }
    ),
    "c2_v2_downstream_consumption_v1": frozenset(
        {
            "schema_version",
            "unit_id",
            "unit_hash",
            "parent_doi_id",
            "origin_record_hash",
            "container_node_id_or_null",
            "consumption_link_hash",
            "candidate_input_id_or_null",
            "candidate_input_record_hash_or_null",
            "consumption_disposition",
            "reason_code",
            "source_only_exclusion_record_hash_or_null",
            "record_hash",
        }
    ),
    "c2_v2_candidate_set_input_v1": frozenset(
        {
            "schema_version",
            "candidate_input_id",
            "unit_id",
            "unit_hash",
            "parent_doi_id",
            "source_inventory_collection_hash",
            "container_accounting_index_hash",
            "consumption_link_hash",
            "candidate_builder_rule_id",
            "candidate_builder_rule_version",
            "candidate_builder_rule_hash",
            "candidate_outcome",
            "candidate_ids",
            "candidate_record_hashes",
            "source_only_exclusion_record_hash_or_null",
            "record_hash",
        }
    ),
    "c2_v2_candidate_v1": frozenset(
        {
            "schema_version",
            "candidate_id",
            "candidate_input_id",
            "parent_doi_id",
            "unit_id",
            "unit_hash",
            "consumption_link_hash",
            "panel_id",
            "case_group_or_null",
            "figure_asset_record_hash",
            "caption_asset_record_hash",
            "source_table_sha256",
            "source_table_bytes",
            "record_hash",
        }
    ),
    "c2_v2_proposal_v1": frozenset(
        {
            "schema_version",
            "proposal_id",
            "candidate_id",
            "candidate_record_hash",
            "candidate_input_record_hash",
            "consumption_record_hash",
            "unit_hash",
            "parent_doi_id",
            "record_hash",
        }
    ),
    "c2_v2_proposal_input_manifest_v1": frozenset(
        {
            "schema_version",
            "candidate_collection_hash",
            "candidate_set_input_collection_hash",
            "consumption_collection_hash",
            "consumption_bijection_hash",
            "source_inventory_collection_hash",
            "container_accounting_index_hash",
            "format_classifier_config_hash",
            "proposal_rule_hash",
            "review_protocol_hash",
            "manifest_hash",
        }
    ),
    "c2_v2_structural_review_protocol_v1": frozenset(
        {
            "schema_version",
            "review_mode",
            "review_protocol_id",
            "review_protocol_version",
            "review_protocol_hash",
            "protocol_hash",
        }
    ),
    "c2_v2_structural_review_outcome_v1": frozenset(
        {
            "schema_version",
            "proposal_id",
            "proposal_record_hash",
            "candidate_id",
            "candidate_record_hash",
            "review_mode",
            "review_protocol_hash",
            "outcome",
            "record_hash",
        }
    ),
    "c2_v2_canonical_case_v1": frozenset(
        {
            "schema_version",
            "case_id",
            "parent_doi_id",
            "case_group",
            "case_kind",
            "source_candidate_ids",
            "source_candidate_record_hashes",
            "candidate_input_record_hashes",
            "consumption_link_hashes",
            "unit_hashes",
            "verified_panel_ids",
            "qualified_panel_count",
            "canonical_builder_rule_hash",
            "record_hash",
        }
    ),
    "c2_v2_canonical_summary_v1": frozenset(
        {
            "schema_version",
            "case_count",
            "canonical_case_collection_hash",
            "proposal_collection_hash",
            "review_collection_hash",
            "canonical_builder_rule_hash",
            "all_cases_retained",
            "summary_hash",
        }
    ),
    "c2_v2_canonical_case_set_manifest_v1": frozenset(
        {
            "schema_version",
            "case_sets",
            "canonical_case_collection_hash",
            "case_ordering_rule_id",
            "doi_case_aggregation_rule_id",
            "doi_case_aggregation_rule_hash",
            "case_set_manifest_hash",
        }
    ),
    "c2_v2_doi_source_inventory_v1": frozenset(
        {
            "schema_version",
            "parent_doi_id",
            "input_ordinal",
            "raw_article_id",
            "terminal_status_raw",
            "source_descriptor_relative_path_or_null",
            "source_descriptor_sha256_or_null",
            "raw_asset_record_hashes",
            "inventory_status",
            "inventory_hash",
        }
    ),
    "c2_v2_source_classification_v1": frozenset(
        {
            "schema_version",
            "doi_id",
            "parent_doi_id",
            "cluster_id",
            "canonical_case_set_hash",
            "canonical_case_ids",
            "canonical_case_record_hashes",
            "verified_panel_descriptors_sha256",
            "qualified_panel_counts",
            "derived_public_stratum",
            "derived_code_label",
            "source_inventory_collection_hash",
            "candidate_collection_hash",
            "proposal_collection_hash",
            "review_collection_hash",
            "canonical_collection_hash",
            "consumption_bijection_hash",
            "panel_rule_hash",
            "record_hash",
        }
    ),
    "c2_v2_acquisition_disposition_v1": frozenset(
        {
            "schema_version",
            "doi_id",
            "parent_doi_id",
            "input_ordinal",
            "terminal_status_raw",
            "terminal_status",
            "terminal_evidence_binding",
            "final_disposition",
            "source_inventory_binding_or_null",
            "canonical_builder_binding_or_null",
            "all_eligible_case_ids",
            "classification_reason",
            "source_classification_record_hash_or_null",
            "record_hash",
        }
    ),
    "c2_v2_p_evidence_summary_v1": frozenset(
        {
            "schema_version",
            "input_total",
            "source_classification_count",
            "source_only_disposition_count",
            "classification_collection_hash",
            "disposition_collection_hash",
            "canonical_case_set_manifest_sha256",
            "acquisition_disposition_doi_ids_sha256",
            "stratified_source_doi_ids_sha256",
            "non_stratified_doi_ids_sha256",
            "per_doi_case_set_hashes_sha256",
            "terminal_status_counts",
            "disposition_counts",
            "stratum_source_doi_counts",
            "stratum_independent_cluster_counts",
            "all_case_equal_weighting_rule_hash",
            "summary_hash",
        }
    ),
    "c2_v2_consumption_bijection_validation_v1": frozenset(
        {
            "schema_version",
            "derived_member_ids",
            "derived_unit_ids",
            "derived_member_unit_pairs",
            "unit_collection_hash",
            "consumption_collection_hash",
            "candidate_input_collection_hash",
            "candidate_collection_hash",
            "source_only_exclusion_collection_hash",
            "consumption_bijection_hash",
        }
    ),
    "c2_v2_source_bearing_extension_validation_v1": frozenset(
        {
            "schema_version",
            "status",
            "source_chunk_sha256",
            "input_total",
            "raw_asset_count",
            "detected_format_count",
            "container_count",
            "derived_member_count",
            "source_unit_count",
            "consumption_count",
            "candidate_input_count",
            "candidate_count",
            "proposal_count",
            "review_count",
            "canonical_case_count",
            "source_classification_count",
            "acquisition_disposition_count",
            "format_config_hash",
            "consumption_bijection_hash",
            "canonical_case_set_manifest_hash",
            "approved_implementation_commit_full",
            "attestation_commit_full",
            "code_attestation_manifest_sha256",
            "attested_code_blobs_sha256",
            "validation_hash",
        }
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise SourceBearingExtensionError(f"self-hash field already exists: {field}")
    schema_version = value.get("schema_version")
    expected_fields = _CLOSED_FIELDS.get(schema_version) if isinstance(schema_version, str) else None
    if expected_fields is None:
        raise SourceBearingExtensionError(
            f"generated V2 object has an unregistered schema: {schema_version!r}"
        )
    _require(
        set(value) | {field} == expected_fields,
        f"{schema_version} fields are not closed",
    )
    _assert_no_legacy_fields(value)
    value[field] = _sha256_json(value)
    return value


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    schema_version = value.get("schema_version")
    expected_fields = _CLOSED_FIELDS.get(schema_version) if isinstance(schema_version, str) else None
    _require(
        expected_fields is not None and set(value) == expected_fields,
        f"{label} fields are not closed",
    )
    _assert_no_legacy_fields(_without(value, field))
    digest = value.get(field)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise SourceBearingExtensionError(f"{label} has no valid {field}")
    if _sha256_json(_without(value, field)) != digest:
        raise SourceBearingExtensionError(f"{label} semantic hash mismatch")


def _assert_no_legacy_fields(value: Any) -> None:
    """Reject legacy semantic fields recursively, but not raw copied evidence."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SourceBearingExtensionError("generated V2 object has a non-string key")
            lowered = key.casefold()
            if (
                lowered in _PROHIBITED_LEGACY_FIELDS
                or lowered.startswith("asserted_")
                or lowered.startswith("legacy_")
                or lowered.startswith(
                    (
                        "absolute_",
                        "supersed",
                        "preferred_",
                        "resume",
                        "fallback",
                        "model",
                        "metric",
                        "trend",
                        "coverage",
                        "selection",
                    )
                )
                or lowered.endswith(("_timestamp", "_timestamps", "_at"))
            ):
                raise SourceBearingExtensionError(
                    f"generated V2 object contains prohibited legacy field: {key}"
                )
            _assert_no_legacy_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_legacy_fields(child)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceBearingExtensionError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, label)
    return value


def _require_identifier(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None,
        label,
    )
    return value


def _normalise_doi(value: Any, label: str) -> str:
    """Return the contract's canonical DOI identifier or fail closed."""

    _require(isinstance(value, str), label)
    raw = unquote(value).strip()
    raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE).strip()
    parsed = urlsplit(raw)
    if parsed.scheme:
        _require(
            parsed.scheme.casefold() in {"http", "https"}
            and parsed.hostname is not None
            and parsed.hostname.casefold() in {"doi.org", "dx.doi.org"},
            label,
        )
        raw = unquote(parsed.path)
    else:
        raw = raw.split("?", 1)[0].split("#", 1)[0]
    normalized = raw.strip().casefold()
    _require(
        normalized.startswith("10.")
        and "/" in normalized
        and "?" not in normalized
        and "#" not in normalized
        and not any(character.isspace() for character in normalized),
        label,
    )
    return normalized


def _require_doi(value: Any, label: str) -> str:
    normalized = _normalise_doi(value, label)
    _require(
        isinstance(value, str)
        and value == normalized,
        label,
    )
    return normalized


def _require_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value and "\x00" not in value, label)
    _require(
        not value.startswith("/")
        and "\\" not in value
        and not _DRIVE_RE.match(value)
        and all(part not in {"", ".", ".."} for part in value.split("/")),
        label,
    )
    return value


def _stratum_for_panel_count(count: Any) -> str:
    _require(
        isinstance(count, int) and not isinstance(count, bool) and count >= 1,
        "qualified panel count is invalid",
    )
    if count == 1:
        return "P1"
    if count == 2:
        return "P2"
    if count in {3, 4}:
        return "P3_4"
    return "P5PLUS"


def _code_label_for_stratum(stratum: str) -> str:
    labels = {
        "P1": "P=1",
        "P2": "P=2",
        "P3_4": "P=3-4",
        "P5PLUS": "P=5+",
    }
    _require(stratum in labels, "derived public stratum is invalid")
    return labels[stratum]


def _counts_by_value(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBearingExtensionError(f"{label} is not UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBearingExtensionError(f"{label} is not JSONL") from exc
    _require(all(line for line in lines), f"{label} has blank JSONL lines")
    _require(all(isinstance(value, dict) for value in values), f"{label} has nonobjects")
    return values


def _write_json(root: _TargetRoot, relative: str, value: Mapping[str, Any]) -> str:
    _assert_no_legacy_fields(value)
    root.write_bytes(relative, _canonical_bytes(value) + b"\n")
    return relative


def _write_jsonl(
    root: _TargetRoot,
    relative: str,
    values: Iterable[Mapping[str, Any]],
) -> str:
    materialized = list(values)
    for value in materialized:
        _assert_no_legacy_fields(value)
    root.write_bytes(
        relative,
        b"".join(_canonical_bytes(value) + b"\n" for value in materialized),
    )
    return relative


def _file_binding(root: _TargetRoot, relative: str) -> dict[str, Any]:
    payload = root.read_bytes(relative)
    return {
        "relative_path": relative,
        "sha256": _sha256(payload),
        "byte_count": len(payload),
    }


def _rule_hash(rule_id: str, version: str) -> str:
    return _sha256_json({"rule_id": rule_id, "version": version})


@dataclass(frozen=True)
class _AttestedCodeBlob:
    relative_path: str
    git_blob_object_id: str
    sha256: str


@dataclass(frozen=True)
class SourceExtensionCodeAttestation:
    """A reviewed, non-circular Git/blob binding for the extension runtime."""

    worktree: Path
    attestation_commit_full: str
    approved_implementation_commit_full: str
    manifest_sha256: str
    code_blobs: tuple[_AttestedCodeBlob, ...]

    @property
    def code_blob_set_sha256(self) -> str:
        return _sha256_json(
            [
                {
                    "relative_path": blob.relative_path,
                    "git_blob_object_id": blob.git_blob_object_id,
                    "sha256": blob.sha256,
                }
                for blob in self.code_blobs
            ]
        )

    def sha256_for(self, relative_path: str) -> str:
        for blob in self.code_blobs:
            if blob.relative_path == relative_path:
                return blob.sha256
        raise SourceBearingExtensionError(
            f"code attestation has no binding for {relative_path}"
        )

    def verify_runtime(self, *, loaded_finalizer_path: Path | None = None) -> None:
        observed = verify_source_extension_code_attestation(
            self.worktree,
            loaded_finalizer_path=loaded_finalizer_path,
        )
        _require(
            observed == self,
            "source extension code attestation changed during execution",
        )


def _git_text(worktree: Path, arguments: Sequence[str], label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceBearingExtensionError(
            f"cannot verify source extension {label}"
        ) from exc
    return completed.stdout.strip()


def _git_bytes(worktree: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceBearingExtensionError(
            f"cannot read source extension {label}"
        ) from exc
    return completed.stdout


def _attested_runtime_paths(
    *,
    worktree: Path,
    loaded_extension_path: Path | None,
    loaded_finalizer_path: Path | None,
) -> dict[str, Path]:
    runtime_paths = {
        _EXTENSION_RELATIVE_PATH: (
            Path(__file__) if loaded_extension_path is None else loaded_extension_path
        )
    }
    if loaded_finalizer_path is not None:
        runtime_paths[_FINALIZER_RELATIVE_PATH] = loaded_finalizer_path
    else:
        finalizer_module = sys.modules.get(
            "experiments.c2_remediation_root_finalizer"
        )
        module_path = getattr(finalizer_module, "__file__", None)
        if isinstance(module_path, str):
            runtime_paths[_FINALIZER_RELATIVE_PATH] = Path(module_path)
    for relative_path, runtime_path in runtime_paths.items():
        expected_path = worktree / relative_path
        _require(
            runtime_path.is_file()
            and not runtime_path.is_symlink()
            and runtime_path.resolve() == expected_path.resolve(),
            f"loaded runtime path is not the attested {relative_path}",
        )
    return runtime_paths


def _module_worktree() -> Path:
    module_path = Path(__file__).resolve()
    _require(
        module_path.as_posix().endswith(_EXTENSION_RELATIVE_PATH),
        "source extension module path is not repository-relative",
    )
    return module_path.parents[2]


def verify_source_extension_code_attestation(
    worktree: Path | None = None,
    *,
    loaded_extension_path: Path | None = None,
    loaded_finalizer_path: Path | None = None,
) -> SourceExtensionCodeAttestation:
    """Verify the attestation commit and every loaded/declared extension blob."""

    candidate = _module_worktree() if worktree is None else Path(worktree)
    _require(
        candidate.is_absolute()
        and candidate.is_dir()
        and not candidate.is_symlink(),
        "source extension worktree is unsafe",
    )
    resolved_worktree = candidate.resolve()
    repository_root = Path(
        _git_text(candidate, ("rev-parse", "--show-toplevel"), "worktree root")
    ).resolve()
    _require(
        repository_root == resolved_worktree,
        "source extension worktree is not the Git repository root",
    )
    status = _git_text(
        resolved_worktree,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "worktree status",
    )
    _require(not status, "source extension worktree is dirty")
    attestation_commit = _git_text(
        resolved_worktree, ("rev-parse", "HEAD"), "attestation commit"
    )
    _require(
        _GIT_OBJECT_RE.fullmatch(attestation_commit) is not None,
        "source extension attestation commit is invalid",
    )
    implementation_commit = _git_text(
        resolved_worktree, ("rev-parse", "HEAD^"), "implementation parent commit"
    )
    _require(
        _GIT_OBJECT_RE.fullmatch(implementation_commit) is not None,
        "source extension implementation commit is invalid",
    )
    changed = _git_text(
        resolved_worktree,
        ("diff", "--name-status", "--no-renames", implementation_commit, attestation_commit),
        "attestation commit contents",
    )
    _require(
        changed == f"A\t{_CODE_ATTESTATION_RELATIVE_PATH}",
        "source extension attestation commit must add only its manifest",
    )
    manifest_path = resolved_worktree / _CODE_ATTESTATION_RELATIVE_PATH
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "source extension attestation manifest is unavailable",
    )
    manifest_payload = manifest_path.read_bytes()
    _require(
        manifest_payload
        == _git_bytes(
            resolved_worktree,
            ("show", f"{attestation_commit}:{_CODE_ATTESTATION_RELATIVE_PATH}"),
            "attestation manifest blob",
        ),
        "source extension attestation manifest runtime bytes changed",
    )
    manifest = _json_object(manifest_payload, "source extension code attestation")
    _require(
        set(manifest)
        == {
            "schema_version",
            "approved_implementation_commit_full",
            "attested_paths",
        }
        and manifest.get("schema_version")
        == "c2_source_bearing_extension_code_attestation_v1"
        and manifest.get("approved_implementation_commit_full")
        == implementation_commit
        and manifest_payload == _canonical_bytes(manifest) + b"\n",
        "source extension code attestation manifest is invalid",
    )
    raw_blobs = manifest.get("attested_paths")
    _require(
        isinstance(raw_blobs, list) and raw_blobs,
        "source extension code attestation has no paths",
    )
    code_blobs: list[_AttestedCodeBlob] = []
    seen_paths: set[str] = set()
    for raw_blob in raw_blobs:
        _require(
            isinstance(raw_blob, Mapping)
            and set(raw_blob)
            == {"relative_path", "git_blob_object_id", "sha256"},
            "source extension code attestation blob fields are invalid",
        )
        relative_path = _require_relative(
            raw_blob.get("relative_path"),
            "source extension code attestation path is invalid",
        )
        blob_object_id = raw_blob.get("git_blob_object_id")
        digest = raw_blob.get("sha256")
        _require(
            relative_path in _REQUIRED_ATTESTED_CODE_PATHS
            and relative_path not in seen_paths
            and isinstance(blob_object_id, str)
            and _GIT_OBJECT_RE.fullmatch(blob_object_id) is not None
            and isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest) is not None,
            "source extension code attestation blob is invalid",
        )
        implementation_blob_object_id = _git_text(
            resolved_worktree,
            ("rev-parse", f"{implementation_commit}:{relative_path}"),
            f"implementation blob {relative_path}",
        )
        implementation_payload = _git_bytes(
            resolved_worktree,
            ("show", f"{implementation_commit}:{relative_path}"),
            f"implementation bytes {relative_path}",
        )
        attestation_payload = _git_bytes(
            resolved_worktree,
            ("show", f"{attestation_commit}:{relative_path}"),
            f"attestation bytes {relative_path}",
        )
        runtime_path = resolved_worktree / relative_path
        _require(
            runtime_path.is_file()
            and not runtime_path.is_symlink()
            and implementation_blob_object_id == blob_object_id
            and attestation_payload == implementation_payload
            and runtime_path.read_bytes() == implementation_payload
            and _sha256(implementation_payload) == digest,
            f"source extension attested blob mismatch: {relative_path}",
        )
        seen_paths.add(relative_path)
        code_blobs.append(
            _AttestedCodeBlob(relative_path, blob_object_id, digest)
        )
    _require(
        seen_paths == _REQUIRED_ATTESTED_CODE_PATHS
        and [blob.relative_path for blob in code_blobs]
        == sorted(blob.relative_path for blob in code_blobs),
        "source extension code attestation path coverage is invalid",
    )
    _attested_runtime_paths(
        worktree=resolved_worktree,
        loaded_extension_path=loaded_extension_path,
        loaded_finalizer_path=loaded_finalizer_path,
    )
    return SourceExtensionCodeAttestation(
        worktree=resolved_worktree,
        attestation_commit_full=attestation_commit,
        approved_implementation_commit_full=implementation_commit,
        manifest_sha256=_sha256(manifest_payload),
        code_blobs=tuple(code_blobs),
    )


def _schema_hashes(attestation: SourceExtensionCodeAttestation) -> dict[str, str]:
    attestation.verify_runtime()
    return {
        name: attestation.sha256_for(f"agent/experiments/schemas/{name}")
        for name in _REQUIRED_SCHEMA_NAMES
    }


def _closed_schema_registry_hash() -> str:
    return _sha256_json(
        {
            schema_version: sorted(fields)
            for schema_version, fields in sorted(_CLOSED_FIELDS.items())
        }
    )


def _format_config(attestation: SourceExtensionCodeAttestation) -> dict[str, Any]:
    attestation.verify_runtime()
    source_sha = attestation.sha256_for(_EXTENSION_RELATIVE_PATH)
    value: dict[str, Any] = {
        "schema_version": "c2_v2_fd_format_classifier_config_v1",
        "format_classifier_id": FORMAT_CLASSIFIER_ID,
        "format_classifier_version": "1",
        "format_classifier_code_sha256": source_sha,
        "zip_v1_parser_id": ZIP_PARSER_ID,
        "zip_v1_parser_version": "1",
        "zip_v1_parser_code_sha256": source_sha,
        "xlsx_profile_parser_id": XLSX_PROFILE_PARSER_ID,
        "xlsx_profile_parser_version": "1",
        "xlsx_profile_parser_code_sha256": source_sha,
        "noncontainer_parser_registry_id": "c2_v2_noncontainer_registry_v1",
        "noncontainer_parser_registry_version": "1",
        "noncontainer_parser_registry_hash": _rule_hash(
            "c2_v2_noncontainer_registry_v1", "1"
        ),
        "format_detection_rule_id": "c2_v2_fd_format_detection_rule_v1",
        "format_detection_rule_version": "1",
        "format_detection_rule_hash": _rule_hash(
            "c2_v2_fd_format_detection_rule_v1", "1"
        ),
        "format_error_enum_version": "1",
        "format_error_enum_hash": _rule_hash("c2_v2_format_error_enum", "1"),
        "maximum_container_depth": MAX_CONTAINER_DEPTH,
        "schema_hashes": _schema_hashes(attestation),
        "closed_schema_registry_hash": _closed_schema_registry_hash(),
        "approved_implementation_commit_full": (
            attestation.approved_implementation_commit_full
        ),
        "attestation_commit_full": attestation.attestation_commit_full,
        "code_attestation_manifest_sha256": attestation.manifest_sha256,
        "attested_code_blobs_sha256": attestation.code_blob_set_sha256,
        "review_mode": REVIEW_MODE,
        "review_protocol_hash": _rule_hash(REVIEW_MODE, "1"),
    }
    return _seal(value, "config_hash")


def _zip_like(payload: bytes) -> bool:
    return payload.startswith(_ZIP_SIGNATURES) or payload.rfind(_EOCD_SIGNATURE) >= 0


@dataclass(frozen=True)
class _CentralEntry:
    index: int
    raw_name: bytes
    selector: str
    selector_casefold: str
    header_sha256: str
    compression_method: int
    flag_bits: int
    crc32: int
    compressed_bytes: int
    uncompressed_bytes: int
    external_attributes: int
    local_header_offset: int
    is_directory: bool
    is_resource_fork: bool
    zip_info: zipfile.ZipInfo


@dataclass(frozen=True)
class _ZipInfo:
    entries: tuple[_CentralEntry, ...]
    central_directory_sha256: str
    eocd_sha256: str
    total_compressed_bytes: int
    total_uncompressed_bytes: int


@dataclass
class _ArchiveRunBudget:
    """Bound every accepted container before member extraction begins."""

    compressed_bytes: int = 0
    uncompressed_bytes: int = 0

    def reserve(self, payload: bytes, archive: _ZipInfo) -> None:
        next_compressed = self.compressed_bytes + len(payload)
        next_uncompressed = self.uncompressed_bytes + archive.total_uncompressed_bytes
        _require(
            next_compressed <= MAX_ARCHIVE_RUN_COMPRESSED_BYTES,
            "REJECT_ZIP_RUN_COMPRESSED_LIMIT",
        )
        _require(
            next_uncompressed <= MAX_ARCHIVE_RUN_UNCOMPRESSED_BYTES,
            "REJECT_ZIP_RUN_UNCOMPRESSED_LIMIT",
        )
        self.compressed_bytes = next_compressed
        self.uncompressed_bytes = next_uncompressed


def _find_eocd(payload: bytes) -> tuple[int, tuple[int, ...]]:
    start = max(0, len(payload) - (65535 + 22))
    offset = payload.rfind(_EOCD_SIGNATURE, start)
    while offset >= start:
        if offset + 22 <= len(payload):
            fields = struct.unpack_from("<IHHHHIIH", payload, offset)
            comment_size = fields[-1]
            if offset + 22 + comment_size == len(payload):
                return offset, fields
        offset = payload.rfind(_EOCD_SIGNATURE, start, offset)
    raise SourceBearingExtensionError("REJECT_FORMAT_ZIP_LIKE_INVALID")


def _normalise_selector(raw_name: bytes) -> tuple[str, str]:
    try:
        selector = raw_name.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceBearingExtensionError("REJECT_ZIP_SELECTOR_ENCODING") from exc
    _require(selector == unicodedata.normalize("NFC", selector), "REJECT_ZIP_SELECTOR_NFC")
    _require(
        selector
        and "\x00" not in selector
        and "\\" not in selector
        and not selector.startswith("/")
        and not _DRIVE_RE.match(selector)
        and not any(ord(character) < 32 or ord(character) == 127 for character in selector),
        "REJECT_ZIP_SELECTOR_UNSAFE",
    )
    is_directory = selector.endswith("/")
    body = selector[:-1] if is_directory else selector
    pieces = body.split("/")
    _require(
        body and all(piece not in {"", ".", ".."} for piece in pieces),
        "REJECT_ZIP_SELECTOR_TRAVERSAL",
    )
    return selector, selector.casefold()


def _parse_zip_v1(payload: bytes) -> _ZipInfo:
    """Parse raw central bytes in physical order and bind them to ZipInfo objects."""

    _require(
        len(payload) <= MAX_ARCHIVE_CONTAINER_COMPRESSED_BYTES,
        "REJECT_ZIP_CONTAINER_COMPRESSED_LIMIT",
    )
    eocd_offset, eocd = _find_eocd(payload)
    _, disk_number, central_disk, disk_entries, total_entries, central_size, central_offset, _ = eocd
    _require(
        disk_number == 0
        and central_disk == 0
        and disk_entries == total_entries
        and total_entries != 0xFFFF
        and central_size != _ZIP64_MARKER
        and central_offset != _ZIP64_MARKER,
        "REJECT_ZIP_UNSUPPORTED_FEATURE",
    )
    _require(
        0 <= central_offset <= eocd_offset
        and central_size <= eocd_offset - central_offset
        and total_entries <= MAX_ARCHIVE_ENTRIES,
        "REJECT_ZIP_CENTRAL_BOUNDS",
    )
    central = payload[central_offset : central_offset + central_size]
    _require(
        central_offset + central_size == eocd_offset,
        "REJECT_ZIP_CENTRAL_BOUNDS",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceBearingExtensionError("REJECT_ZIP_PARSER_FAILURE") from exc
    _require(len(infos) == total_entries, "REJECT_ZIP_ENTRY_COUNT")
    entries: list[_CentralEntry] = []
    local_ranges: list[tuple[int, int]] = []
    raw_seen: set[bytes] = set()
    selector_seen: set[str] = set()
    casefold_seen: set[str] = set()
    cursor = 0
    for index in range(total_entries):
        _require(cursor + 46 <= len(central), "REJECT_ZIP_CENTRAL_BOUNDS")
        unpacked = struct.unpack_from("<IHHHHHHIIIHHHHHII", central, cursor)
        (
            signature,
            made_by,
            _needed,
            flags,
            compression,
            _mod_time,
            _mod_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            comment_size,
            disk_start,
            _internal_attributes,
            external_attributes,
            local_offset,
        ) = unpacked
        _require(signature == 0x02014B50, "REJECT_ZIP_CENTRAL_SIGNATURE")
        record_end = cursor + 46 + name_size + extra_size + comment_size
        _require(record_end <= len(central), "REJECT_ZIP_CENTRAL_BOUNDS")
        _require(
            disk_start == 0 and local_offset != _ZIP64_MARKER,
            "REJECT_ZIP_UNSUPPORTED_FEATURE",
        )
        _require(flags in {0, 0x800}, "REJECT_ZIP_UNSUPPORTED_FEATURE")
        raw_name = central[cursor + 46 : cursor + 46 + name_size]
        _require(
            all(byte < 128 for byte in raw_name) or bool(flags & 0x800),
            "REJECT_ZIP_SELECTOR_ENCODING",
        )
        selector, selector_casefold = _normalise_selector(raw_name)
        _require(raw_name not in raw_seen, "REJECT_ZIP_SELECTOR_DUPLICATE")
        _require(selector not in selector_seen, "REJECT_ZIP_SELECTOR_ALIAS")
        _require(selector_casefold not in casefold_seen, "REJECT_ZIP_SELECTOR_ALIAS")
        raw_seen.add(raw_name)
        selector_seen.add(selector)
        casefold_seen.add(selector_casefold)
        _require(not (flags & 0x1), "REJECT_ZIP_ENCRYPTED")
        _require(
            compression in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
            "REJECT_ZIP_UNSUPPORTED_FEATURE",
        )
        _require(
            0 <= local_offset < central_offset and local_offset + 30 <= central_offset,
            "REJECT_ZIP_LOCAL_BOUNDS",
        )
        (
            local_signature,
            _local_needed,
            local_flags,
            local_compression,
            _local_time,
            _local_date,
            local_crc32,
            local_compressed_bytes,
            local_uncompressed_bytes,
            local_name_size,
            local_extra_size,
        ) = struct.unpack_from("<IHHHHHIIIHH", payload, local_offset)
        _require(local_signature == 0x04034B50, "REJECT_ZIP_LOCAL_SIGNATURE")
        _require(
            local_flags == flags
            and local_compression == compression
            and local_crc32 == crc32
            and local_compressed_bytes == compressed_size
            and local_uncompressed_bytes == uncompressed_size,
            "REJECT_ZIP_LOCAL_HEADER_MISMATCH",
        )
        local_name_start = local_offset + 30
        local_name_end = local_name_start + local_name_size
        data_start = local_name_end + local_extra_size
        data_end = data_start + compressed_size
        _require(
            local_name_end <= central_offset
            and data_start <= central_offset
            and data_end <= central_offset
            and payload[local_name_start:local_name_end] == raw_name,
            "REJECT_ZIP_LOCAL_BOUNDS",
        )
        local_ranges.append((local_offset, data_end))
        mode = (external_attributes >> 16) & 0xFFFF
        is_directory = selector.endswith("/")
        if mode:
            kind = stat.S_IFMT(mode)
            _require(
                kind in {0, stat.S_IFREG, stat.S_IFDIR},
                "REJECT_ZIP_LINK_OR_SPECIAL",
            )
            if stat.S_ISDIR(mode):
                is_directory = True
        if is_directory:
            _require(
                selector.endswith("/")
                and compressed_size == 0
                and uncompressed_size == 0
                and crc32 == 0,
                "REJECT_ZIP_DIRECTORY_ENTRY",
            )
        info = infos[index]
        _require(
            info.flag_bits == flags
            and info.compress_type == compression
            and info.file_size == uncompressed_size
            and info.compress_size == compressed_size
            and info.header_offset == local_offset,
            "REJECT_ZIP_PARSER_DISAGREEMENT",
        )
        entries.append(
            _CentralEntry(
                index=index + 1,
                raw_name=raw_name,
                selector=selector,
                selector_casefold=selector_casefold,
                header_sha256=_sha256(central[cursor:record_end]),
                compression_method=compression,
                flag_bits=flags,
                crc32=crc32,
                compressed_bytes=compressed_size,
                uncompressed_bytes=uncompressed_size,
                external_attributes=external_attributes,
                local_header_offset=local_offset,
                is_directory=is_directory,
                is_resource_fork=selector.startswith("__MACOSX/")
                or any(part.startswith("._") for part in selector.split("/")),
                zip_info=info,
            )
        )
        cursor = record_end
    _require(cursor == len(central), "REJECT_ZIP_CENTRAL_BOUNDS")
    total_compressed_bytes = sum(entry.compressed_bytes for entry in entries)
    total_uncompressed_bytes = sum(entry.uncompressed_bytes for entry in entries)
    _require(
        total_compressed_bytes <= MAX_ARCHIVE_CONTAINER_COMPRESSED_BYTES,
        "REJECT_ZIP_CONTAINER_COMPRESSED_LIMIT",
    )
    _require(
        total_uncompressed_bytes <= MAX_ARCHIVE_CONTAINER_UNCOMPRESSED_BYTES,
        "REJECT_ZIP_CONTAINER_UNCOMPRESSED_LIMIT",
    )
    for (_, previous_end), (next_start, _) in zip(
        sorted(local_ranges), sorted(local_ranges)[1:], strict=False
    ):
        _require(previous_end <= next_start, "REJECT_ZIP_LOCAL_OVERLAP")
    return _ZipInfo(
        entries=tuple(entries),
        central_directory_sha256=_sha256(central),
        eocd_sha256=_sha256(payload[eocd_offset:]),
        total_compressed_bytes=total_compressed_bytes,
        total_uncompressed_bytes=total_uncompressed_bytes,
    )


def _read_zip_member(payload: bytes, entry: _CentralEntry) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            with archive.open(entry.zip_info, "r") as member:
                result = bytearray()
                total = 0
                while True:
                    block = member.read(64 * 1024)
                    if not block:
                        break
                    total += len(block)
                    _require(total <= MAX_ARCHIVE_MEMBER_BYTES, "REJECT_ZIP_MEMBER_TOO_LARGE")
                    result.extend(block)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SourceBearingExtensionError("REJECT_ZIP_MEMBER_STREAM_FAILURE") from exc
    payload = bytes(result)
    _require(
        len(payload) == entry.uncompressed_bytes,
        "REJECT_ZIP_MEMBER_SIZE_MISMATCH",
    )
    return payload


def _is_csv_v1(payload: bytes) -> bool:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text or "\x00" in text:
        return False
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error:
        return False
    if not rows or not rows[0] or any(not field.strip() for field in rows[0]):
        return False
    width = len(rows[0])
    # A one-column text document is an OTHER_REGISTERED text asset rather than
    # an ambiguous table.  V2 deliberately has no suffix/kind fallback.
    return width > 1 and all(len(row) == width for row in rows[1:])


def _is_other_registered(payload: bytes) -> bool:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") or payload.startswith(b"\xff\xd8\xff"):
        return True
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(text.strip()) and "\x00" not in text


_OOXML_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_OOXML_PACKAGE_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
_OOXML_SPREADSHEET_NS = (
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
)
_OOXML_DOCUMENT_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_OOXML_OFFICE_DOCUMENT_RELATIONSHIP = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
    "officeDocument"
)
_OOXML_WORKSHEET_RELATIONSHIP = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
)
_OOXML_WORKBOOK_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
)
_OOXML_WORKSHEET_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
)


def _parse_ooxml_xml(payload: bytes) -> ElementTree.Element | None:
    if b"<!DOCTYPE" in payload or b"<!ENTITY" in payload:
        return None
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return None


def _ooxml_relationships(
    root: ElementTree.Element,
) -> dict[str, tuple[str, str]] | None:
    if root.tag != f"{{{_OOXML_PACKAGE_RELATIONSHIPS_NS}}}Relationships":
        return None
    relationships: dict[str, tuple[str, str]] = {}
    for child in root:
        if child.tag != f"{{{_OOXML_PACKAGE_RELATIONSHIPS_NS}}}Relationship":
            return None
        relationship_id = child.get("Id")
        relationship_type = child.get("Type")
        target = child.get("Target")
        if (
            not relationship_id
            or not relationship_type
            or not target
            or relationship_id in relationships
            or child.get("TargetMode") not in {None, "Internal"}
        ):
            return None
        relationships[relationship_id] = (relationship_type, target)
    return relationships


def _xlsx_profile(payload: bytes, archive: _ZipInfo) -> bool:
    """Recognize only a structurally valid OOXML spreadsheet package."""

    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
    }
    selectors = {entry.selector for entry in archive.entries}
    if not required.issubset(selectors):
        return False
    by_selector = {entry.selector: entry for entry in archive.entries}
    try:
        parsed = {
            selector: _parse_ooxml_xml(
                _read_zip_member(payload, by_selector[selector])
            )
            for selector in required
        }
    except SourceBearingExtensionError:
        return False
    if any(root is None for root in parsed.values()):
        return False
    content_types = parsed["[Content_Types].xml"]
    root_relationships = _ooxml_relationships(parsed["_rels/.rels"])
    workbook = parsed["xl/workbook.xml"]
    workbook_relationships = _ooxml_relationships(
        parsed["xl/_rels/workbook.xml.rels"]
    )
    if (
        content_types is None
        or content_types.tag != f"{{{_OOXML_CONTENT_TYPES_NS}}}Types"
        or root_relationships is None
        or workbook is None
        or workbook.tag != f"{{{_OOXML_SPREADSHEET_NS}}}workbook"
        or workbook_relationships is None
    ):
        return False
    overrides: dict[str, str] = {}
    for child in content_types:
        if child.tag == f"{{{_OOXML_CONTENT_TYPES_NS}}}Default":
            if not child.get("Extension") or not child.get("ContentType"):
                return False
            continue
        if child.tag != f"{{{_OOXML_CONTENT_TYPES_NS}}}Override":
            return False
        part_name = child.get("PartName")
        content_type = child.get("ContentType")
        if (
            not isinstance(part_name, str)
            or not part_name.startswith("/")
            or not content_type
            or part_name in overrides
        ):
            return False
        overrides[part_name] = content_type
    if overrides.get("/xl/workbook.xml") != _OOXML_WORKBOOK_CONTENT_TYPE:
        return False
    office_document_targets = {
        target
        for relationship_type, target in root_relationships.values()
        if relationship_type == _OOXML_OFFICE_DOCUMENT_RELATIONSHIP
    }
    if office_document_targets != {"xl/workbook.xml"}:
        return False
    sheets = workbook.find(f"{{{_OOXML_SPREADSHEET_NS}}}sheets")
    if sheets is None:
        return False
    worksheet_selectors: set[str] = set()
    sheet_ids: set[int] = set()
    relationship_ids: set[str] = set()
    for sheet in sheets:
        if sheet.tag != f"{{{_OOXML_SPREADSHEET_NS}}}sheet":
            return False
        relationship_id = sheet.get(
            f"{{{_OOXML_DOCUMENT_RELATIONSHIPS_NS}}}id"
        )
        sheet_id = sheet.get("sheetId")
        if (
            not sheet.get("name")
            or not relationship_id
            or not sheet_id
            or not sheet_id.isdecimal()
            or int(sheet_id) < 1
            or int(sheet_id) in sheet_ids
            or relationship_id in relationship_ids
        ):
            return False
        relationship = workbook_relationships.get(relationship_id)
        if relationship is None or relationship[0] != _OOXML_WORKSHEET_RELATIONSHIP:
            return False
        target = relationship[1]
        if (
            target.startswith("/")
            or "\\" in target
            or not target.startswith("worksheets/")
            or any(part in {"", ".", ".."} for part in target.split("/"))
        ):
            return False
        selector = f"xl/{target}"
        if selector not in selectors or selector in worksheet_selectors:
            return False
        sheet_ids.add(int(sheet_id))
        relationship_ids.add(relationship_id)
        worksheet_selectors.add(selector)
    if not worksheet_selectors:
        return False
    worksheet_relationship_targets = {
        f"xl/{target}"
        for relationship_type, target in workbook_relationships.values()
        if relationship_type == _OOXML_WORKSHEET_RELATIONSHIP
        and not target.startswith("/")
        and "\\" not in target
        and target.startswith("worksheets/")
        and all(part not in {"", ".", ".."} for part in target.split("/"))
    }
    if worksheet_relationship_targets != worksheet_selectors:
        return False
    try:
        for selector in sorted(worksheet_selectors):
            worksheet = _parse_ooxml_xml(
                _read_zip_member(payload, by_selector[selector])
            )
            if (
                worksheet is None
                or worksheet.tag != f"{{{_OOXML_SPREADSHEET_NS}}}worksheet"
                or worksheet.find(f"{{{_OOXML_SPREADSHEET_NS}}}sheetData") is None
                or overrides.get(f"/{selector}") != _OOXML_WORKSHEET_CONTENT_TYPE
            ):
                return False
    except SourceBearingExtensionError:
        return False
    return True


@dataclass(frozen=True)
class _Detected:
    container_format: str
    content_profile: str
    zip_info: _ZipInfo | None

    @property
    def tuple(self) -> tuple[str, str]:
        return (self.container_format, self.content_profile)


def _detect_format(
    payload: bytes,
    *,
    archive_budget: _ArchiveRunBudget | None = None,
) -> _Detected:
    if _zip_like(payload):
        archive = _parse_zip_v1(payload)
        if archive_budget is not None:
            archive_budget.reserve(payload, archive)
        if _xlsx_profile(payload, archive):
            return _Detected("ZIP_V1", "XLSX_V1", archive)
        return _Detected("ZIP_V1", "GENERIC_ZIP_V1", archive)
    if _is_csv_v1(payload):
        return _Detected("NONE", "CSV_V1", None)
    if _is_other_registered(payload):
        return _Detected("NONE", "OTHER_REGISTERED_V1", None)
    raise SourceBearingExtensionError("REJECT_FORMAT_UNRECOGNIZED")


def _validate_declared_format(asset: Mapping[str, Any], detected: _Detected) -> None:
    declared_kind = asset.get("declared_asset_kind")
    declared_tuple = asset.get("declared_format_tuple")
    _require(
        declared_kind in _DECLARED_KIND_TUPLES
        and isinstance(declared_tuple, list)
        and len(declared_tuple) == 2
        and all(isinstance(item, str) for item in declared_tuple),
        "REJECT_DECLARED_KIND_FORMAT_MISMATCH",
    )
    declared = (declared_tuple[0], declared_tuple[1])
    _require(
        declared in _DECLARED_TUPLES
        and declared == detected.tuple
        and declared in _DECLARED_KIND_TUPLES[declared_kind],
        "REJECT_DECLARED_KIND_FORMAT_MISMATCH",
    )


def _asset_hints(asset: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    hints = asset.get("candidate_hints", [])
    _require(isinstance(hints, list), "source asset candidate_hints must be a list")
    seen: set[tuple[str | None, str]] = set()
    output: list[Mapping[str, Any]] = []
    for index, hint in enumerate(hints):
        _require(isinstance(hint, dict), "candidate hint is invalid")
        _require(
            set(hint)
            == {
                "member_selector_or_null",
                "panel_id",
                "case_group_or_null",
                "figure_asset_id",
                "caption_asset_id",
            },
            "candidate hint fields are not closed",
        )
        selector = hint["member_selector_or_null"]
        _require(selector is None or isinstance(selector, str), "candidate hint selector invalid")
        if selector is not None:
            _normalise_selector(selector.encode("utf-8"))
        panel_id = _require_identifier(hint["panel_id"], "candidate hint panel ID invalid")
        group = hint["case_group_or_null"]
        _require(group is None or _IDENTIFIER_RE.fullmatch(group) is not None, "case group invalid")
        figure = _require_identifier(hint["figure_asset_id"], "figure asset ID invalid")
        caption = _require_identifier(hint["caption_asset_id"], "caption asset ID invalid")
        key = (selector, panel_id)
        _require(key not in seen, f"duplicate candidate hint {index}")
        seen.add(key)
        output.append(
            {
                "member_selector_or_null": selector,
                "panel_id": panel_id,
                "case_group_or_null": group,
                "figure_asset_id": figure,
                "caption_asset_id": caption,
            }
        )
    return tuple(output)


def validate_source_evidence_descriptor_v2(
    descriptor: Mapping[str, Any],
    *,
    article_id: str,
    doi_id: str,
    provenance_relative_path: str,
) -> tuple[Mapping[str, Any], ...]:
    """Validate the typed V2 raw descriptor before the generic finalizer copies it."""

    canonical_doi = _require_doi(doi_id, "source evidence V2 parent DOI is invalid")
    _require(
        set(descriptor)
        == {
            "schema_version",
            "doi",
            "article_id",
            "provenance_relative_path",
            "assets",
            "descriptor_hash",
        },
        "source evidence V2 descriptor fields are not closed",
    )
    _require(
        descriptor["schema_version"] == "c2-source-evidence-v2"
        and _normalise_doi(descriptor["doi"], "source evidence V2 DOI is invalid")
        == canonical_doi
        and descriptor["article_id"] == article_id
        and descriptor["provenance_relative_path"] == provenance_relative_path,
        "source evidence V2 descriptor provenance linkage mismatch",
    )
    descriptor_hash = descriptor.get("descriptor_hash")
    _require(
        isinstance(descriptor_hash, str)
        and _SHA256_RE.fullmatch(descriptor_hash) is not None
        and _sha256_json(_without(descriptor, "descriptor_hash")) == descriptor_hash,
        "source evidence V2 descriptor semantic hash mismatch",
    )
    assets = descriptor["assets"]
    _require(isinstance(assets, list) and assets, "source evidence V2 has no assets")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    output: list[Mapping[str, Any]] = []
    prefix = f"content/_sources/{article_id}/"
    for index, asset in enumerate(assets):
        _require(isinstance(asset, dict), f"source asset {index} is not an object")
        _require(
            set(asset)
            == {
                "asset_id",
                "relative_path",
                "sha256",
                "bytes",
                "doi",
                "declared_asset_kind",
                "declared_format_tuple",
                "candidate_hints",
            },
            f"source asset {index} fields are not closed",
        )
        asset_id = _require_identifier(asset["asset_id"], f"source asset {index} ID invalid")
        relative = _require_relative(
            asset["relative_path"], f"source asset {index} relative path invalid"
        )
        _require(relative.startswith(prefix), f"source asset {index} path is outside DOI root")
        _require(asset_id not in seen_ids and relative not in seen_paths, "duplicate source asset")
        _require(
            _normalise_doi(asset["doi"], f"source asset {index} DOI is invalid")
            == canonical_doi
            and _SHA256_RE.fullmatch(str(asset["sha256"])) is not None
            and isinstance(asset["bytes"], int)
            and not isinstance(asset["bytes"], bool)
            and asset["bytes"] > 0,
            f"source asset {index} metadata invalid",
        )
        _asset_hints(asset)
        seen_ids.add(asset_id)
        seen_paths.add(relative)
        output.append(asset)
    return tuple(output)


@dataclass(frozen=True)
class _RawAsset:
    article_id: str
    doi_id: str
    input_ordinal: int
    asset: Mapping[str, Any]
    asset_record: Mapping[str, Any]
    payload: bytes


@dataclass(frozen=True)
class SourceBearingExtensionResult:
    """Bindings inserted into the remediation root's immutable sealed report."""

    status: str
    cases: Mapping[str, Any]
    proposals: Mapping[str, Any]
    review: Mapping[str, Any]
    canonical: Mapping[str, Any]
    p_evidence: Mapping[str, Any]
    extension_validation: Mapping[str, Any]


def _build_canonical_case_set_manifest(
    cases: Sequence[Mapping[str, Any]],
    *,
    applicable_doi_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Build the complete, ordered DOI case-set binding before P derivation."""

    cases_by_doi: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        doi_id = _require_doi(case.get("parent_doi_id"), "canonical case DOI is invalid")
        cases_by_doi[doi_id].append(case)

    case_sets: list[dict[str, Any]] = []
    by_doi: dict[str, Mapping[str, Any]] = {}
    ordered_doi_ids = [
        _require_doi(doi_id, "canonical case-set DOI is invalid")
        for doi_id in applicable_doi_ids
    ]
    _require(
        len(ordered_doi_ids) == len(set(ordered_doi_ids))
        and set(cases_by_doi).issubset(ordered_doi_ids),
        "canonical case-set DOI coverage is invalid",
    )
    for doi_id in ordered_doi_ids:
        descriptors: list[dict[str, Any]] = []
        for case in sorted(cases_by_doi[doi_id], key=lambda item: str(item["case_id"])):
            panel_count = case.get("qualified_panel_count")
            stratum = _stratum_for_panel_count(panel_count)
            descriptors.append(
                {
                    "case_id": case["case_id"],
                    "parent_doi_id": doi_id,
                    "source_candidate_ids": list(case["source_candidate_ids"]),
                    "source_candidate_record_hashes": list(
                        case["source_candidate_record_hashes"]
                    ),
                    "candidate_input_record_hashes": list(
                        case["candidate_input_record_hashes"]
                    ),
                    "consumption_link_hashes": list(case["consumption_link_hashes"]),
                    "unit_hashes": list(case["unit_hashes"]),
                    "verified_panel_ids": list(case["verified_panel_ids"]),
                    "qualified_panel_count": panel_count,
                    "derived_public_stratum": stratum,
                    "derived_code_label": _code_label_for_stratum(stratum),
                    "canonical_case_record_hash": case["record_hash"],
                }
            )
        case_set: dict[str, Any] = {
            "doi_id": doi_id,
            "cases": descriptors,
            "canonical_case_set_hash": _sha256_json(
                {"doi_id": doi_id, "cases": descriptors}
            ),
        }
        case_sets.append(case_set)
        by_doi[doi_id] = case_set

    manifest: dict[str, Any] = {
        "schema_version": "c2_v2_canonical_case_set_manifest_v1",
        "case_sets": case_sets,
        "canonical_case_collection_hash": _sha256_json(
            [item["record_hash"] for item in cases]
        ),
        "case_ordering_rule_id": "UNICODE_CODEPOINT_CASE_ID_V1",
        "doi_case_aggregation_rule_id": "DOI_CASE_AGGREGATION_V1",
        "doi_case_aggregation_rule_hash": _rule_hash("DOI_CASE_AGGREGATION_V1", "1"),
    }
    _seal(manifest, "case_set_manifest_hash")
    return manifest, by_doi


def _applicable_case_set_doi_ids(
    terminal_rows: Sequence[Mapping[str, Any]],
    source_by_article: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return frozen-order downloaded DOI with complete verified source inventory."""

    doi_ids: list[str] = []
    for terminal in sorted(
        terminal_rows, key=lambda item: int(item["input_index_1based"])
    ):
        terminal_status_raw = terminal.get("terminal_status")
        _require(
            isinstance(terminal_status_raw, str)
            and terminal_status_raw in _TERMINAL_STATUS_ADAPTER,
            "terminal status is not in the closed adapter",
        )
        if _TERMINAL_STATUS_ADAPTER[terminal_status_raw] != "DOWNLOADED":
            continue
        article_id = _require_identifier(
            terminal.get("article_id"), "terminal article ID is invalid"
        )
        if article_id not in source_by_article:
            continue
        doi_ids.append(
            _require_doi(terminal.get("doi"), "terminal DOI is invalid")
        )
    _require(
        len(doi_ids) == len(set(doi_ids)),
        "applicable canonical case-set DOI is duplicated",
    )
    return tuple(doi_ids)


def _panel_descriptors_for_case_set(case_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the manifest's canonical records into the sole P input descriptors."""

    _require(
        set(case_set) == {"doi_id", "cases", "canonical_case_set_hash"},
        "canonical case-set fields are not closed",
    )
    doi_id = _require_doi(case_set.get("doi_id"), "canonical case-set DOI is invalid")
    cases = case_set.get("cases")
    _require(isinstance(cases, list), "canonical case-set cases are invalid")
    descriptors: list[dict[str, Any]] = []
    expected_case_fields = {
        "case_id",
        "parent_doi_id",
        "source_candidate_ids",
        "source_candidate_record_hashes",
        "candidate_input_record_hashes",
        "consumption_link_hashes",
        "unit_hashes",
        "verified_panel_ids",
        "qualified_panel_count",
        "derived_public_stratum",
        "derived_code_label",
        "canonical_case_record_hash",
    }
    for descriptor in cases:
        _require(
            isinstance(descriptor, Mapping) and set(descriptor) == expected_case_fields,
            "canonical case descriptor fields are not closed",
        )
        _require(
            descriptor.get("parent_doi_id") == doi_id,
            "canonical case descriptor DOI mismatch",
        )
        vectors = (
            descriptor["verified_panel_ids"],
            descriptor["source_candidate_ids"],
            descriptor["source_candidate_record_hashes"],
            descriptor["candidate_input_record_hashes"],
            descriptor["consumption_link_hashes"],
            descriptor["unit_hashes"],
        )
        _require(
            all(isinstance(vector, list) for vector in vectors)
            and len({len(vector) for vector in vectors}) == 1
            and len(vectors[0]) == descriptor["qualified_panel_count"],
            "canonical case descriptor panel binding mismatch",
        )
        _require(
            descriptor["derived_public_stratum"]
            == _stratum_for_panel_count(descriptor["qualified_panel_count"])
            and descriptor["derived_code_label"]
            == _code_label_for_stratum(descriptor["derived_public_stratum"]),
            "canonical case descriptor P derivation mismatch",
        )
        descriptors.extend(
            {
                "case_id": descriptor["case_id"],
                "panel_id": panel_id,
                "source_candidate_id": candidate_id,
                "source_candidate_record_hash": candidate_hash,
                "candidate_input_record_hash": candidate_input_hash,
                "consumption_link_hash": consumption_hash,
                "unit_hash": unit_hash,
                "parent_doi_id": doi_id,
            }
            for panel_id, candidate_id, candidate_hash, candidate_input_hash, consumption_hash, unit_hash in zip(
                *vectors,
                strict=True,
            )
        )
    return descriptors


class _Builder:
    def __init__(
        self,
        *,
        root: _TargetRoot,
        raw_assets: Sequence[_RawAsset],
        terminal_rows: Sequence[Mapping[str, Any]],
        source_by_article: Mapping[str, Mapping[str, Any]],
        partition_records: int,
        source_chunk_sha256: str,
        code_attestation: SourceExtensionCodeAttestation,
    ) -> None:
        self.root = root
        self.raw_assets = tuple(raw_assets)
        self.terminal_rows = tuple(terminal_rows)
        self.source_by_article = source_by_article
        self.partition_records = partition_records
        self.source_chunk_sha256 = source_chunk_sha256
        self.code_attestation = code_attestation
        self.code_attestation.verify_runtime()
        self.config = _format_config(code_attestation)
        self.detected: list[dict[str, Any]] = []
        self.accounts: dict[str, dict[str, Any]] = {}
        self.account_paths: dict[str, str] = {}
        self.nodes: list[dict[str, Any]] = []
        self.derived: list[dict[str, Any]] = []
        self.archive_exclusions: list[dict[str, Any]] = []
        self.units: list[dict[str, Any]] = []
        self._unit_payloads: dict[str, bytes] = {}
        self._asset_by_id: dict[tuple[str, str], _RawAsset] = {
            (asset.article_id, str(asset.asset["asset_id"])): asset for asset in raw_assets
        }
        self._hints: dict[tuple[str, str, str | None], tuple[Mapping[str, Any], ...]] = {}
        self._next_node = 0
        self._archive_budget = _ArchiveRunBudget()
        self._load_hints()

    def _load_hints(self) -> None:
        for raw in self.raw_assets:
            asset_id = str(raw.asset["asset_id"])
            hints = _asset_hints(raw.asset)
            by_selector: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
            for hint in hints:
                by_selector[hint["member_selector_or_null"]].append(hint)
            for selector, values in by_selector.items():
                self._hints[(raw.article_id, asset_id, selector)] = tuple(values)

    def _format_record(
        self,
        *,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        detected: _Detected,
        node_id: str | None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": "c2_v2_detected_format_v1",
            "origin_record_hash": origin_record_hash,
            "parent_doi_id": doi_id,
            "frozen_input_ordinal": input_ordinal,
            "verified_relative_path": relative_path,
            "verified_file_sha256": _sha256(payload),
            "verified_bytes": len(payload),
            "container_format": detected.container_format,
            "content_profile": detected.content_profile,
            "container_node_id_or_null": node_id,
            "format_classifier_id": FORMAT_CLASSIFIER_ID,
            "format_classifier_version": "1",
            "format_classifier_code_sha256": self.config["format_classifier_code_sha256"],
            "format_detection_rule_hash": self.config["format_detection_rule_hash"],
        }
        return _seal(value, "format_hash")

    def _new_unit(
        self,
        *,
        origin_kind: str,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        detected: _Detected,
        node_id: str | None,
        raw_article_id: str,
        raw_asset_id: str,
        selector: str | None,
        candidate_capable: bool,
    ) -> dict[str, Any]:
        unit_id = f"unit-{len(self.units) + 1:06d}"
        value: dict[str, Any] = {
            "schema_version": "c2_v2_consumable_source_unit_v1",
            "unit_id": unit_id,
            "parent_doi_id": doi_id,
            "frozen_input_ordinal": input_ordinal,
            "origin_kind": origin_kind,
            "origin_record_hash": origin_record_hash,
            "container_node_id_or_null": node_id,
            "verified_relative_path": relative_path,
            "verified_file_sha256": _sha256(payload),
            "verified_bytes": len(payload),
            "detected_format_tuple": [detected.container_format, detected.content_profile],
            "format_classifier_id": FORMAT_CLASSIFIER_ID,
            "format_classifier_version": "1",
            "format_classifier_code_sha256": self.config["format_classifier_code_sha256"],
            "format_detection_rule_hash": self.config["format_detection_rule_hash"],
            "raw_article_id": raw_article_id,
            "raw_asset_id": raw_asset_id,
            "member_selector_or_null": selector,
            "candidate_capable": candidate_capable,
        }
        _seal(value, "unit_hash")
        self.units.append(value)
        self._unit_payloads[unit_id] = payload
        return value

    def _derived_path(self, node_id: str, index: int, payload: bytes) -> str:
        return f"source_inventory_v2/derived_members/{node_id}/{index:05d}-{_sha256(payload)}.bin"

    def _process_origin(
        self,
        *,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        raw_article_id: str,
        raw_asset_id: str,
        selector: str | None,
        parent_node_id: str | None,
        depth: int,
        origin_kind: str,
        pre_detected: _Detected | None = None,
    ) -> None:
        _require(depth <= MAX_CONTAINER_DEPTH, "REJECT_CONTAINER_DEPTH")
        detected = pre_detected or _detect_format(
            payload,
            archive_budget=self._archive_budget,
        )
        node_id: str | None = None
        if detected.container_format == "ZIP_V1":
            _require(
                detected.zip_info is not None,
                "REJECT_ZIP_PARSER_FAILURE",
            )
            self._next_node += 1
            node_id = f"container-{self._next_node:06d}"
        self.detected.append(
            self._format_record(
                origin_record_hash=origin_record_hash,
                doi_id=doi_id,
                input_ordinal=input_ordinal,
                relative_path=relative_path,
                payload=payload,
                detected=detected,
                node_id=node_id,
            )
        )
        if node_id is None:
            if detected.content_profile == "CSV_V1":
                self._new_unit(
                    origin_kind=origin_kind,
                    origin_record_hash=origin_record_hash,
                    doi_id=doi_id,
                    input_ordinal=input_ordinal,
                    relative_path=relative_path,
                    payload=payload,
                    detected=detected,
                    node_id=None,
                    raw_article_id=raw_article_id,
                    raw_asset_id=raw_asset_id,
                    selector=selector,
                    candidate_capable=True,
                )
            elif origin_kind == "DERIVED_ARCHIVE_MEMBER":
                self._new_unit(
                    origin_kind=origin_kind,
                    origin_record_hash=origin_record_hash,
                    doi_id=doi_id,
                    input_ordinal=input_ordinal,
                    relative_path=relative_path,
                    payload=payload,
                    detected=detected,
                    node_id=None,
                    raw_article_id=raw_article_id,
                    raw_asset_id=raw_asset_id,
                    selector=selector,
                    candidate_capable=False,
                )
            return

        self.nodes.append(
            {
                "container_node_id": node_id,
                "parent_container_node_id_or_null": parent_node_id,
                "origin_record_hash": origin_record_hash,
                "parent_doi_id": doi_id,
                "frozen_input_ordinal": input_ordinal,
                "verified_relative_path": relative_path,
                "verified_file_sha256": _sha256(payload),
                "verified_bytes": len(payload),
                "depth": depth,
                "content_profile": detected.content_profile,
            }
        )
        if detected.content_profile == "XLSX_V1":
            self._account_xlsx(
                node_id=node_id,
                origin_record_hash=origin_record_hash,
                doi_id=doi_id,
                input_ordinal=input_ordinal,
                relative_path=relative_path,
                payload=payload,
                archive=detected.zip_info,
            )
            self._new_unit(
                origin_kind=(
                    "RAW_XLSX_CONTAINER_SELF"
                    if origin_kind == "RAW_NONCONTAINER_SELF"
                    else "DERIVED_XLSX_CONTAINER_SELF"
                ),
                origin_record_hash=origin_record_hash,
                doi_id=doi_id,
                input_ordinal=input_ordinal,
                relative_path=relative_path,
                payload=payload,
                detected=detected,
                node_id=node_id,
                raw_article_id=raw_article_id,
                raw_asset_id=raw_asset_id,
                selector=selector,
                candidate_capable=True,
            )
            return
        self._account_generic_zip(
            node_id=node_id,
            origin_record_hash=origin_record_hash,
            doi_id=doi_id,
            input_ordinal=input_ordinal,
            relative_path=relative_path,
            payload=payload,
            archive=detected.zip_info,
            raw_article_id=raw_article_id,
            raw_asset_id=raw_asset_id,
            parent_selector=selector,
            depth=depth,
        )
        if origin_kind == "DERIVED_ARCHIVE_MEMBER":
            self._new_unit(
                origin_kind=origin_kind,
                origin_record_hash=origin_record_hash,
                doi_id=doi_id,
                input_ordinal=input_ordinal,
                relative_path=relative_path,
                payload=payload,
                detected=detected,
                node_id=node_id,
                raw_article_id=raw_article_id,
                raw_asset_id=raw_asset_id,
                selector=selector,
                candidate_capable=False,
            )

    def _base_entry(self, entry: _CentralEntry) -> dict[str, Any]:
        return {
            "schema_version": "c2_v2_archive_central_entry_v1",
            "central_index": entry.index,
            "raw_name_sha256": _sha256(entry.raw_name),
            "selector": entry.selector,
            "selector_nfc": entry.selector,
            "selector_casefold": entry.selector_casefold,
            "central_header_sha256": entry.header_sha256,
            "compression_method": entry.compression_method,
            "flag_bits": entry.flag_bits,
            "crc32": entry.crc32,
            "compressed_bytes": entry.compressed_bytes,
            "uncompressed_bytes": entry.uncompressed_bytes,
            "external_attributes": entry.external_attributes,
            "local_header_offset": entry.local_header_offset,
            "derived_member_record_hash_or_null": None,
            "source_only_exclusion_reason_or_null": None,
            "archive_source_only_disposition_record_hash_or_null": None,
            "child_container_node_id_or_null": None,
            "disposition": "",
        }

    def _seal_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return _seal(entry, "entry_hash")

    def _archive_exclusion(
        self,
        *,
        node_id: str,
        doi_id: str,
        entry: Mapping[str, Any],
        reason: str,
    ) -> str:
        _require(reason in _SOURCE_ONLY_REASONS, "archive exclusion reason is invalid")
        value: dict[str, Any] = {
            "schema_version": "c2_v2_archive_source_only_disposition_v1",
            "container_node_id": node_id,
            "central_index": entry["central_index"],
            "central_header_sha256": entry["central_header_sha256"],
            "parent_doi_id": doi_id,
            "reason_code": reason,
        }
        _seal(value, "record_hash")
        self.archive_exclusions.append(value)
        return value["record_hash"]

    def _write_account(
        self,
        *,
        node_id: str,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        archive: _ZipInfo,
        profile: str,
        entries: list[dict[str, Any]],
    ) -> None:
        for entry in entries:
            self._seal_entry(entry)
        value: dict[str, Any] = {
            "schema_version": "c2_v2_archive_accounting_manifest_v1",
            "container_node_id": node_id,
            "origin_record_hash": origin_record_hash,
            "parent_doi_id": doi_id,
            "frozen_input_ordinal": input_ordinal,
            "verified_relative_path": relative_path,
            "verified_file_sha256": _sha256(payload),
            "verified_bytes": len(payload),
            "container_format": "ZIP_V1",
            "content_profile": profile,
            "format_classifier_id": FORMAT_CLASSIFIER_ID,
            "format_classifier_version": "1",
            "format_classifier_code_sha256": self.config["format_classifier_code_sha256"],
            "zip_v1_parser_id": ZIP_PARSER_ID,
            "zip_v1_parser_version": "1",
            "zip_v1_parser_code_sha256": self.config["zip_v1_parser_code_sha256"],
            "central_directory_sha256": archive.central_directory_sha256,
            "eocd_sha256": archive.eocd_sha256,
            "physical_entry_count": len(entries),
            "central_entry_uncompressed_bytes": sum(
                int(entry["uncompressed_bytes"]) for entry in entries
            ),
            "regular_member_uncompressed_bytes": sum(
                int(entry["uncompressed_bytes"])
                for entry in entries
                if entry["disposition"] != "DIRECTORY_ENTRY"
            ),
            "derived_member_uncompressed_bytes": sum(
                int(entry["uncompressed_bytes"])
                for entry in entries
                if entry["derived_member_record_hash_or_null"] is not None
            ),
            "source_only_excluded_uncompressed_bytes": sum(
                int(entry["uncompressed_bytes"])
                for entry in entries
                if entry["archive_source_only_disposition_record_hash_or_null"]
                is not None
                and entry["disposition"] != "DIRECTORY_ENTRY"
            ),
            "entries": entries,
        }
        _seal(value, "archive_accounting_hash")
        relative = f"source_inventory_v2/container_accounts/{node_id}.json"
        _write_json(self.root, relative, value)
        self.accounts[node_id] = value
        self.account_paths[node_id] = relative

    def _account_xlsx(
        self,
        *,
        node_id: str,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        archive: _ZipInfo | None,
    ) -> None:
        _require(archive is not None, "REJECT_XLSX_PROFILE_FAILURE")
        entries: list[dict[str, Any]] = []
        for central in archive.entries:
            entry = self._base_entry(central)
            if central.is_directory:
                entry["disposition"] = "DIRECTORY_ENTRY"
                entry["source_only_exclusion_reason_or_null"] = "DIRECTORY_ENTRY"
                entry["archive_source_only_disposition_record_hash_or_null"] = (
                    self._archive_exclusion(
                        node_id=node_id,
                        doi_id=doi_id,
                        entry=entry,
                        reason="DIRECTORY_ENTRY",
                    )
                )
            elif central.is_resource_fork:
                entry["disposition"] = "SOURCE_ONLY_EXCLUSION"
                entry["source_only_exclusion_reason_or_null"] = "RESOURCE_FORK"
                entry["archive_source_only_disposition_record_hash_or_null"] = (
                    self._archive_exclusion(
                        node_id=node_id,
                        doi_id=doi_id,
                        entry=entry,
                        reason="RESOURCE_FORK",
                    )
                )
            else:
                _read_zip_member(payload, central)
                entry["disposition"] = "SOURCE_ONLY_EXCLUSION"
                entry["source_only_exclusion_reason_or_null"] = "XLSX_PACKAGE_COMPONENT"
                entry["archive_source_only_disposition_record_hash_or_null"] = (
                    self._archive_exclusion(
                        node_id=node_id,
                        doi_id=doi_id,
                        entry=entry,
                        reason="XLSX_PACKAGE_COMPONENT",
                    )
                )
            entries.append(entry)
        self._write_account(
            node_id=node_id,
            origin_record_hash=origin_record_hash,
            doi_id=doi_id,
            input_ordinal=input_ordinal,
            relative_path=relative_path,
            payload=payload,
            archive=archive,
            profile="XLSX_V1",
            entries=entries,
        )

    def _account_generic_zip(
        self,
        *,
        node_id: str,
        origin_record_hash: str,
        doi_id: str,
        input_ordinal: int,
        relative_path: str,
        payload: bytes,
        archive: _ZipInfo | None,
        raw_article_id: str,
        raw_asset_id: str,
        parent_selector: str | None,
        depth: int,
    ) -> None:
        _require(archive is not None, "REJECT_ZIP_PARSER_FAILURE")
        entries: list[dict[str, Any]] = []
        child_nodes: list[str] = []
        for central in archive.entries:
            entry = self._base_entry(central)
            if central.is_directory:
                entry["disposition"] = "DIRECTORY_ENTRY"
                entry["source_only_exclusion_reason_or_null"] = "DIRECTORY_ENTRY"
                entry["archive_source_only_disposition_record_hash_or_null"] = (
                    self._archive_exclusion(
                        node_id=node_id,
                        doi_id=doi_id,
                        entry=entry,
                        reason="DIRECTORY_ENTRY",
                    )
                )
                entries.append(entry)
                continue
            if central.is_resource_fork:
                _read_zip_member(payload, central)
                entry["disposition"] = "SOURCE_ONLY_EXCLUSION"
                entry["source_only_exclusion_reason_or_null"] = "RESOURCE_FORK"
                entry["archive_source_only_disposition_record_hash_or_null"] = (
                    self._archive_exclusion(
                        node_id=node_id,
                        doi_id=doi_id,
                        entry=entry,
                        reason="RESOURCE_FORK",
                    )
                )
                entries.append(entry)
                continue
            member_payload = _read_zip_member(payload, central)
            child_detected = _detect_format(
                member_payload,
                archive_budget=self._archive_budget,
            )
            selector = (
                central.selector
                if parent_selector is None
                else f"{parent_selector}!{central.selector}"
            )
            member_relative = self._derived_path(node_id, central.index, member_payload)
            self.root.write_bytes(member_relative, member_payload)
            derived: dict[str, Any] = {
                "schema_version": "c2_v2_derived_archive_member_v1",
                "derived_member_id": f"derived-{len(self.derived) + 1:06d}",
                "container_node_id": node_id,
                "central_index": central.index,
                "central_header_sha256": central.header_sha256,
                "parent_doi_id": doi_id,
                "frozen_input_ordinal": input_ordinal,
                "raw_article_id": raw_article_id,
                "raw_asset_id": raw_asset_id,
                "member_selector": selector,
                "verified_relative_path": member_relative,
                "verified_file_sha256": _sha256(member_payload),
                "verified_bytes": len(member_payload),
                "detected_format_tuple": [
                    child_detected.container_format,
                    child_detected.content_profile,
                ],
            }
            _seal(derived, "record_hash")
            self.derived.append(derived)
            entry["derived_member_record_hash_or_null"] = derived["record_hash"]
            entry["disposition"] = "DERIVED_MEMBER"
            nodes_before = len(self.nodes)
            self._process_origin(
                origin_record_hash=derived["record_hash"],
                doi_id=doi_id,
                input_ordinal=input_ordinal,
                relative_path=member_relative,
                payload=member_payload,
                raw_article_id=raw_article_id,
                raw_asset_id=raw_asset_id,
                selector=selector,
                parent_node_id=node_id,
                depth=depth + 1,
                origin_kind="DERIVED_ARCHIVE_MEMBER",
                pre_detected=child_detected,
            )
            if len(self.nodes) > nodes_before:
                child_nodes.extend(
                    item["container_node_id"] for item in self.nodes[nodes_before:]
                )
                entry["child_container_node_id_or_null"] = self.nodes[nodes_before][
                    "container_node_id"
                ]
            entries.append(entry)
        self._write_account(
            node_id=node_id,
            origin_record_hash=origin_record_hash,
            doi_id=doi_id,
            input_ordinal=input_ordinal,
            relative_path=relative_path,
            payload=payload,
            archive=archive,
            profile="GENERIC_ZIP_V1",
            entries=entries,
        )

    def classify_raw_assets(self) -> None:
        for raw in self.raw_assets:
            detected = _detect_format(
                raw.payload,
                archive_budget=self._archive_budget,
            )
            _validate_declared_format(raw.asset, detected)
            self._process_origin(
                origin_record_hash=raw.asset_record["asset_record_hash"],
                doi_id=raw.doi_id,
                input_ordinal=raw.input_ordinal,
                relative_path=str(raw.asset["relative_path"]),
                payload=raw.payload,
                raw_article_id=raw.article_id,
                raw_asset_id=str(raw.asset["asset_id"]),
                selector=None,
                parent_node_id=None,
                depth=0,
                origin_kind="RAW_NONCONTAINER_SELF",
                pre_detected=detected,
            )

    def _asset_supports_hint(
        self,
        *,
        article_id: str,
        asset_id: str,
        expected_kind: str,
    ) -> _RawAsset | None:
        raw = self._asset_by_id.get((article_id, asset_id))
        if raw is None or raw.asset["declared_asset_kind"] != expected_kind:
            return None
        return raw

    def _table_is_parseable(self, unit: Mapping[str, Any]) -> bool:
        payload = self._unit_payloads[str(unit["unit_id"])]
        profile = unit["detected_format_tuple"][1]
        if profile == "CSV_V1":
            return _is_csv_v1(payload)
        if profile == "XLSX_V1":
            try:
                archive = _parse_zip_v1(payload)
            except SourceBearingExtensionError:
                return False
            return _xlsx_profile(payload, archive)
        return False

    def _candidate_pipeline(self) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Return consumption, candidate inputs, candidates, source exclusions."""

        consumption: list[dict[str, Any]] = []
        candidate_inputs: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for unit in sorted(self.units, key=lambda item: item["unit_id"]):
            unit_id = str(unit["unit_id"])
            link_hash = _sha256_json(
                {
                    "unit_id": unit_id,
                    "unit_hash": unit["unit_hash"],
                    "parent_doi_id": unit["parent_doi_id"],
                }
            )
            if not unit["candidate_capable"]:
                reason = (
                    "CONTAINER_EXPANDED_RECURSIVELY"
                    if unit["detected_format_tuple"][0] == "ZIP_V1"
                    else "UNREGISTERED_MEMBER_TYPE"
                )
                exclusion: dict[str, Any] = {
                    "schema_version": "c2_v2_source_only_exclusion_v1",
                    "unit_id": unit_id,
                    "unit_hash": unit["unit_hash"],
                    "parent_doi_id": unit["parent_doi_id"],
                    "reason_code": reason,
                }
                _seal(exclusion, "record_hash")
                exclusions.append(exclusion)
                record: dict[str, Any] = {
                    "schema_version": "c2_v2_downstream_consumption_v1",
                    "unit_id": unit_id,
                    "unit_hash": unit["unit_hash"],
                    "parent_doi_id": unit["parent_doi_id"],
                    "origin_record_hash": unit["origin_record_hash"],
                    "container_node_id_or_null": unit["container_node_id_or_null"],
                    "consumption_link_hash": link_hash,
                    "candidate_input_id_or_null": None,
                    "candidate_input_record_hash_or_null": None,
                    "consumption_disposition": "SOURCE_ONLY_EXCLUSION",
                    "reason_code": reason,
                    "source_only_exclusion_record_hash_or_null": exclusion["record_hash"],
                }
                _seal(record, "record_hash")
                consumption.append(record)
                continue
            candidate_input_id = f"candidate-input-{len(candidate_inputs) + 1:06d}"
            raw_article_id = str(unit["raw_article_id"])
            raw_asset_id = str(unit["raw_asset_id"])
            selector = unit["member_selector_or_null"]
            hints = self._hints.get((raw_article_id, raw_asset_id, selector), ())
            valid_hints = [
                hint
                for hint in hints
                if self._asset_supports_hint(
                    article_id=raw_article_id,
                    asset_id=str(hint["figure_asset_id"]),
                    expected_kind="figure",
                )
                is not None
                and self._asset_supports_hint(
                    article_id=raw_article_id,
                    asset_id=str(hint["caption_asset_id"]),
                    expected_kind="caption",
                )
                is not None
            ]
            reason: str | None = None
            if not self._table_is_parseable(unit):
                reason = "TABLE_PARSE_FAILURE"
            elif not hints:
                reason = "NO_BOUND_FIGURE_CAPTION"
            elif len(valid_hints) != len(hints):
                reason = "NO_BOUND_FIGURE_CAPTION"
            elif len({(hint["panel_id"], hint["case_group_or_null"]) for hint in valid_hints}) != len(
                valid_hints
            ):
                reason = "AMBIGUOUS_SOURCE_MAPPING"
            candidate_ids: list[str] = []
            candidate_hashes: list[str] = []
            exclusion_hash: str | None = None
            if reason is not None:
                exclusion: dict[str, Any] = {
                    "schema_version": "c2_v2_source_only_exclusion_v1",
                    "unit_id": unit_id,
                    "unit_hash": unit["unit_hash"],
                    "parent_doi_id": unit["parent_doi_id"],
                    "reason_code": reason,
                }
                _seal(exclusion, "record_hash")
                exclusions.append(exclusion)
                exclusion_hash = exclusion["record_hash"]
            else:
                for hint in sorted(
                    valid_hints,
                    key=lambda value: (str(value["case_group_or_null"]), str(value["panel_id"])),
                ):
                    candidate_id = f"candidate-{len(candidates) + 1:06d}"
                    figure = self._asset_supports_hint(
                        article_id=raw_article_id,
                        asset_id=str(hint["figure_asset_id"]),
                        expected_kind="figure",
                    )
                    caption = self._asset_supports_hint(
                        article_id=raw_article_id,
                        asset_id=str(hint["caption_asset_id"]),
                        expected_kind="caption",
                    )
                    _require(figure is not None and caption is not None, "bound asset vanished")
                    candidate: dict[str, Any] = {
                        "schema_version": "c2_v2_candidate_v1",
                        "candidate_id": candidate_id,
                        "candidate_input_id": candidate_input_id,
                        "parent_doi_id": unit["parent_doi_id"],
                        "unit_id": unit_id,
                        "unit_hash": unit["unit_hash"],
                        "consumption_link_hash": link_hash,
                        "panel_id": hint["panel_id"],
                        "case_group_or_null": hint["case_group_or_null"],
                        "figure_asset_record_hash": figure.asset_record["asset_record_hash"],
                        "caption_asset_record_hash": caption.asset_record["asset_record_hash"],
                        "source_table_sha256": unit["verified_file_sha256"],
                        "source_table_bytes": unit["verified_bytes"],
                    }
                    _seal(candidate, "record_hash")
                    candidates.append(candidate)
                    candidate_ids.append(candidate_id)
                    candidate_hashes.append(candidate["record_hash"])
            candidate_input: dict[str, Any] = {
                "schema_version": "c2_v2_candidate_set_input_v1",
                "candidate_input_id": candidate_input_id,
                "unit_id": unit_id,
                "unit_hash": unit["unit_hash"],
                "parent_doi_id": unit["parent_doi_id"],
                "source_inventory_collection_hash": "",
                "container_accounting_index_hash": "",
                "consumption_link_hash": link_hash,
                "candidate_builder_rule_id": CANDIDATE_RULE_ID,
                "candidate_builder_rule_version": "1",
                "candidate_builder_rule_hash": _rule_hash(CANDIDATE_RULE_ID, "1"),
                "candidate_outcome": (
                    "SOURCE_ONLY_EXCLUSION" if reason is not None else "CANDIDATES_EMITTED"
                ),
                "candidate_ids": candidate_ids,
                "candidate_record_hashes": candidate_hashes,
                "source_only_exclusion_record_hash_or_null": exclusion_hash,
            }
            _seal(candidate_input, "record_hash")
            candidate_inputs.append(candidate_input)
            record = {
                "schema_version": "c2_v2_downstream_consumption_v1",
                "unit_id": unit_id,
                "unit_hash": unit["unit_hash"],
                "parent_doi_id": unit["parent_doi_id"],
                "origin_record_hash": unit["origin_record_hash"],
                "container_node_id_or_null": unit["container_node_id_or_null"],
                "consumption_link_hash": link_hash,
                "candidate_input_id_or_null": candidate_input_id,
                "candidate_input_record_hash_or_null": candidate_input["record_hash"],
                "consumption_disposition": "CANDIDATE_SET_INPUT",
                "reason_code": "BYTE_VALID_TABLE_UNIT",
                "source_only_exclusion_record_hash_or_null": None,
            }
            _seal(record, "record_hash")
            consumption.append(record)
        return consumption, candidate_inputs, candidates, exclusions

    def _proposal_pipeline(
        self,
        candidates: Sequence[Mapping[str, Any]],
        candidate_inputs: Sequence[Mapping[str, Any]],
        consumption: Sequence[Mapping[str, Any]],
        source_inventory_hash: str,
        container_index_hash: str,
        consumption_bijection_hash: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        input_by_id = {item["candidate_input_id"]: item for item in candidate_inputs}
        consumption_by_link = {item["consumption_link_hash"]: item for item in consumption}
        proposed: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
            candidate_input = input_by_id[candidate["candidate_input_id"]]
            consumption_record = consumption_by_link[candidate["consumption_link_hash"]]
            proposal: dict[str, Any] = {
                "schema_version": "c2_v2_proposal_v1",
                "proposal_id": f"proposal-{len(proposed) + 1:06d}",
                "candidate_id": candidate["candidate_id"],
                "candidate_record_hash": candidate["record_hash"],
                "candidate_input_record_hash": candidate_input["record_hash"],
                "consumption_record_hash": consumption_record["record_hash"],
                "unit_hash": candidate["unit_hash"],
                "parent_doi_id": candidate["parent_doi_id"],
            }
            _seal(proposal, "record_hash")
            proposed.append(proposal)
        proposal_manifest: dict[str, Any] = {
            "schema_version": "c2_v2_proposal_input_manifest_v1",
            "candidate_collection_hash": _sha256_json(
                [item["record_hash"] for item in candidates]
            ),
            "candidate_set_input_collection_hash": _sha256_json(
                [item["record_hash"] for item in candidate_inputs]
            ),
            "consumption_collection_hash": _sha256_json(
                [item["record_hash"] for item in consumption]
            ),
            "consumption_bijection_hash": consumption_bijection_hash,
            "source_inventory_collection_hash": source_inventory_hash,
            "container_accounting_index_hash": container_index_hash,
            "format_classifier_config_hash": self.config["config_hash"],
            "proposal_rule_hash": _rule_hash("c2_v2_proposal_builder_v1", "1"),
            "review_protocol_hash": self.config["review_protocol_hash"],
        }
        _seal(proposal_manifest, "manifest_hash")
        return proposed, [], proposal_manifest

    def _review_and_canonical(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        proposed: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        candidate_by_id = {item["candidate_id"]: item for item in candidates}
        seen_group_panels: set[tuple[str, str, str]] = set()
        reviews: list[dict[str, Any]] = []
        accepted_candidates: list[Mapping[str, Any]] = []
        protocol: dict[str, Any] = {
            "schema_version": "c2_v2_structural_review_protocol_v1",
            "review_mode": REVIEW_MODE,
            "review_protocol_id": REVIEW_MODE,
            "review_protocol_version": "1",
            "review_protocol_hash": self.config["review_protocol_hash"],
        }
        _seal(protocol, "protocol_hash")
        for proposal in sorted(proposed, key=lambda item: item["proposal_id"]):
            candidate = candidate_by_id[proposal["candidate_id"]]
            group = candidate["case_group_or_null"] or candidate["candidate_id"]
            panel_key = (candidate["parent_doi_id"], str(group), candidate["panel_id"])
            _require(
                panel_key not in seen_group_panels,
                "DUPLICATE_PANEL_MEMBERSHIP",
            )
            outcome = "ACCEPTED_STRUCTURAL"
            seen_group_panels.add(panel_key)
            accepted_candidates.append(candidate)
            review: dict[str, Any] = {
                "schema_version": "c2_v2_structural_review_outcome_v1",
                "proposal_id": proposal["proposal_id"],
                "proposal_record_hash": proposal["record_hash"],
                "candidate_id": candidate["candidate_id"],
                "candidate_record_hash": candidate["record_hash"],
                "review_mode": REVIEW_MODE,
                "review_protocol_hash": protocol["protocol_hash"],
                "outcome": outcome,
            }
            _seal(review, "record_hash")
            reviews.append(review)
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in accepted_candidates:
            group = candidate["case_group_or_null"] or candidate["candidate_id"]
            grouped[(candidate["parent_doi_id"], str(group))].append(candidate)
        cases: list[dict[str, Any]] = []
        for (doi_id, group), group_candidates in sorted(grouped.items()):
            ordered = sorted(group_candidates, key=lambda item: item["panel_id"])
            case: dict[str, Any] = {
                "schema_version": "c2_v2_canonical_case_v1",
                "case_id": f"case-{len(cases) + 1:06d}",
                "parent_doi_id": doi_id,
                "case_group": group,
                "case_kind": "single" if len(ordered) == 1 else "multi",
                "source_candidate_ids": [item["candidate_id"] for item in ordered],
                "source_candidate_record_hashes": [item["record_hash"] for item in ordered],
                "candidate_input_record_hashes": [
                    next(
                        input_item["record_hash"]
                        for input_item in self._candidate_inputs
                        if input_item["candidate_input_id"] == item["candidate_input_id"]
                    )
                    for item in ordered
                ],
                "consumption_link_hashes": [item["consumption_link_hash"] for item in ordered],
                "unit_hashes": [item["unit_hash"] for item in ordered],
                "verified_panel_ids": [item["panel_id"] for item in ordered],
                "qualified_panel_count": len(ordered),
                "canonical_builder_rule_hash": _rule_hash(CANONICAL_RULE_ID, "1"),
            }
            _seal(case, "record_hash")
            cases.append(case)
        return reviews, cases, protocol

    def _p_evidence(
        self,
        *,
        cases: Sequence[Mapping[str, Any]],
        case_sets_by_doi: Mapping[str, Mapping[str, Any]],
        canonical_case_set_manifest_sha256: str,
        source_inventory_hash: str,
        candidate_hash: str,
        proposal_hash: str,
        review_hash: str,
        canonical_hash: str,
        consumption_bijection_hash: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        cases_by_doi: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for case in cases:
            doi_id = _require_doi(
                case.get("parent_doi_id"), "canonical case DOI is invalid"
            )
            cases_by_doi[doi_id].append(case)
        terminal_by_doi: dict[str, Mapping[str, Any]] = {}
        for terminal in self.terminal_rows:
            doi_id = _require_doi(terminal.get("doi"), "terminal DOI is invalid")
            _require(doi_id not in terminal_by_doi, "duplicate terminal DOI")
            terminal_status_raw = terminal.get("terminal_status")
            _require(
                isinstance(terminal_status_raw, str)
                and terminal_status_raw in _TERMINAL_STATUS_ADAPTER,
                "terminal status is not in the closed adapter",
            )
            terminal_by_doi[doi_id] = terminal
        classifications: list[dict[str, Any]] = []
        case_stratum: dict[str, str] = {}
        for doi_id in sorted(cases_by_doi):
            terminal = terminal_by_doi.get(doi_id)
            _require(terminal is not None, "canonical case has no terminal DOI record")
            if _TERMINAL_STATUS_ADAPTER[terminal["terminal_status"]] != "DOWNLOADED":
                continue
            doi_cases = sorted(
                cases_by_doi[doi_id],
                key=lambda item: str(item["case_id"]),
            )
            counts = [int(item["qualified_panel_count"]) for item in doi_cases]
            strata = {
                "P1"
                if count == 1
                else "P2"
                if count == 2
                else "P3_4"
                if count in {3, 4}
                else "P5PLUS"
                for count in counts
            }
            _require(len(strata) == 1, "MULTI_STRATUM_CANONICAL_DOI")
            stratum = next(iter(strata))
            case_set = case_sets_by_doi.get(doi_id)
            _require(case_set is not None, "canonical case-set manifest lacks DOI")
            panel_descriptors = _panel_descriptors_for_case_set(case_set)
            case_stratum[doi_id] = stratum
            classification: dict[str, Any] = {
                "schema_version": "c2_v2_source_classification_v1",
                "doi_id": doi_id,
                "parent_doi_id": doi_id,
                "cluster_id": doi_id,
                "canonical_case_set_hash": case_set["canonical_case_set_hash"],
                "canonical_case_ids": [
                    item["case_id"] for item in doi_cases
                ],
                "canonical_case_record_hashes": [
                    item["record_hash"] for item in doi_cases
                ],
                "verified_panel_descriptors_sha256": _sha256_json(panel_descriptors),
                "qualified_panel_counts": counts,
                "derived_public_stratum": stratum,
                "derived_code_label": _code_label_for_stratum(stratum),
                "source_inventory_collection_hash": source_inventory_hash,
                "candidate_collection_hash": candidate_hash,
                "proposal_collection_hash": proposal_hash,
                "review_collection_hash": review_hash,
                "canonical_collection_hash": canonical_hash,
                "consumption_bijection_hash": consumption_bijection_hash,
                "panel_rule_hash": _rule_hash(P_RULE_ID, "1"),
            }
            _seal(classification, "record_hash")
            classifications.append(classification)
        dispositions: list[dict[str, Any]] = []
        classification_by_doi = {
            item["doi_id"]: item for item in classifications
        }
        terminal_outcomes_relative_path = "control/terminal_outcomes.jsonl"
        terminal_outcomes_sha256 = self.root.sha256(terminal_outcomes_relative_path)
        for terminal in sorted(
            self.terminal_rows, key=lambda item: int(item["input_index_1based"])
        ):
            doi_id = _require_doi(terminal.get("doi"), "terminal DOI is invalid")
            article_id = str(terminal["article_id"])
            terminal_status_raw = terminal.get("terminal_status")
            _require(
                isinstance(terminal_status_raw, str)
                and terminal_status_raw in _TERMINAL_STATUS_ADAPTER,
                "terminal status is not in the closed adapter",
            )
            terminal_status = _TERMINAL_STATUS_ADAPTER[terminal_status_raw]
            source_present = article_id in self.source_by_article
            doi_cases = sorted(
                cases_by_doi.get(doi_id, ()), key=lambda item: str(item["case_id"])
            )
            if terminal_status == "DOWNLOADED" and doi_id in case_stratum:
                disposition = "STRATIFIED_SOURCE_CANONICAL"
                reason = "VERIFIED_SOURCE_CANONICAL_ALL_CASES"
            elif terminal_status == "DOWNLOADED" and source_present:
                disposition = "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
                reason = "DOWNLOADED_SOURCE_NO_QUALIFYING_CASE"
            elif terminal_status == "DOWNLOADED":
                disposition = "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
                reason = "DOWNLOADED_WITHOUT_VERIFIED_SOURCE"
            else:
                disposition = _FINAL_DISPOSITION_BY_STATUS[terminal_status]
                reason = "TERMINAL_STATUS_PRECLUDES_P_CLASSIFICATION"
            evidence_binding = {
                "terminal_outcomes_relative_path": terminal_outcomes_relative_path,
                "terminal_outcomes_sha256": terminal_outcomes_sha256,
                "terminal_row_sha256": _sha256_json(terminal),
                "source_chunk_sha256": self.source_chunk_sha256,
                "attempt_count": len(terminal.get("rounds", {})),
                "terminal": True,
            }
            _require(
                evidence_binding["attempt_count"] == 3,
                "terminal attempt evidence is incomplete",
            )
            source_inventory_binding: dict[str, Any] | None = None
            if source_present:
                evidence = self.source_by_article[article_id]["source_evidence"]
                source_inventory_binding = {
                    "source_inventory_collection_hash": source_inventory_hash,
                    "source_descriptor_relative_path": getattr(
                        evidence, "descriptor_path", None
                    ),
                    "source_descriptor_sha256": getattr(
                        evidence, "descriptor_sha256", None
                    ),
                    "raw_asset_record_hashes": [
                        raw.asset_record["asset_record_hash"]
                        for raw in sorted(
                            (
                                raw
                                for raw in self.raw_assets
                                if raw.article_id == article_id
                            ),
                            key=lambda raw: str(raw.asset["asset_id"]),
                        )
                    ],
                }
            canonical_builder_binding: dict[str, Any] | None = None
            if terminal_status == "DOWNLOADED" and source_present:
                canonical_builder_binding = {
                    "canonical_builder_rule_id": CANONICAL_RULE_ID,
                    "canonical_builder_rule_version": "1",
                    "canonical_builder_rule_hash": _rule_hash(CANONICAL_RULE_ID, "1"),
                    "candidate_collection_hash": candidate_hash,
                    "proposal_collection_hash": proposal_hash,
                    "review_collection_hash": review_hash,
                    "canonical_collection_hash": canonical_hash,
                    "canonical_case_set_manifest_sha256": (
                        canonical_case_set_manifest_sha256
                    ),
                    "consumption_bijection_hash": consumption_bijection_hash,
                }
            value: dict[str, Any] = {
                "schema_version": "c2_v2_acquisition_disposition_v1",
                "doi_id": doi_id,
                "parent_doi_id": doi_id,
                "input_ordinal": terminal["input_index_1based"],
                "terminal_status_raw": terminal_status_raw,
                "terminal_status": terminal_status,
                "terminal_evidence_binding": evidence_binding,
                "final_disposition": disposition,
                "source_inventory_binding_or_null": source_inventory_binding,
                "canonical_builder_binding_or_null": canonical_builder_binding,
                "all_eligible_case_ids": (
                    [item["case_id"] for item in doi_cases]
                    if terminal_status == "DOWNLOADED"
                    else []
                ),
                "classification_reason": reason,
                "source_classification_record_hash_or_null": (
                    classification_by_doi[doi_id]["record_hash"]
                    if doi_id in classification_by_doi
                    else None
                ),
            }
            _seal(value, "record_hash")
            dispositions.append(value)
        _require(
            len(dispositions) == self.partition_records
            and [item["input_ordinal"] for item in dispositions]
            == list(range(1, self.partition_records + 1)),
            "acquisition disposition coverage is not exact",
        )
        summary: dict[str, Any] = {
            "schema_version": "c2_v2_p_evidence_summary_v1",
            "input_total": self.partition_records,
            "source_classification_count": len(classifications),
            "source_only_disposition_count": sum(
                item["final_disposition"] != "STRATIFIED_SOURCE_CANONICAL"
                for item in dispositions
            ),
            "classification_collection_hash": _sha256_json(
                [item["record_hash"] for item in classifications]
            ),
            "disposition_collection_hash": _sha256_json(
                [item["record_hash"] for item in dispositions]
            ),
            "canonical_case_set_manifest_sha256": canonical_case_set_manifest_sha256,
            "acquisition_disposition_doi_ids_sha256": _sha256_json(
                [item["doi_id"] for item in dispositions]
            ),
            "stratified_source_doi_ids_sha256": _sha256_json(
                [item["doi_id"] for item in classifications]
            ),
            "non_stratified_doi_ids_sha256": _sha256_json(
                [
                    item["doi_id"]
                    for item in dispositions
                    if item["final_disposition"] != "STRATIFIED_SOURCE_CANONICAL"
                ]
            ),
            "per_doi_case_set_hashes_sha256": _sha256_json(
                [
                    {
                        "doi_id": doi_id,
                        "canonical_case_set_hash": case_set[
                            "canonical_case_set_hash"
                        ],
                    }
                    for doi_id, case_set in case_sets_by_doi.items()
                ]
            ),
            "terminal_status_counts": {
                status: sum(item["terminal_status"] == status for item in dispositions)
                for status in sorted(set(_TERMINAL_STATUS_ADAPTER.values()))
            },
            "disposition_counts": _counts_by_value(
                item["final_disposition"] for item in dispositions
            ),
            "stratum_source_doi_counts": {
                stratum: sum(
                    item["derived_public_stratum"] == stratum
                    for item in classifications
                )
                for stratum in ("P1", "P2", "P3_4", "P5PLUS")
            },
            "stratum_independent_cluster_counts": {
                stratum: sum(
                    item["derived_public_stratum"] == stratum
                    and item["cluster_id"] == item["doi_id"]
                    for item in classifications
                )
                for stratum in ("P1", "P2", "P3_4", "P5PLUS")
            },
            "all_case_equal_weighting_rule_hash": _rule_hash(
                "DOI_CASE_AGGREGATION_V1", "1"
            ),
        }
        _seal(summary, "summary_hash")
        return classifications, dispositions, summary

    def build(self) -> SourceBearingExtensionResult:
        config_path = _write_json(
            self.root, "source_inventory_v2/fd_format_classifier_config.json", self.config
        )
        _require(
            self.root.sha256(config_path) == _sha256(_canonical_bytes(self.config) + b"\n"),
            "format configuration write hash mismatch",
        )
        self.classify_raw_assets()
        raw_inventory = [
            asset.asset_record for asset in sorted(
                self.raw_assets,
                key=lambda item: (item.input_ordinal, str(item.asset["asset_id"])),
            )
        ]
        raw_inventory_path = _write_jsonl(
            self.root, "source_inventory_v2/source_inventory.jsonl", raw_inventory
        )
        detected_path = _write_jsonl(
            self.root,
            "source_inventory_v2/detected_formats.jsonl",
            self.detected,
        )
        index_entries: list[dict[str, Any]] = []
        for node in sorted(self.nodes, key=lambda item: item["container_node_id"]):
            node_id = str(node["container_node_id"])
            account = self.accounts.get(node_id)
            _require(account is not None, "missing container account")
            index_entry = {
                "schema_version": "c2_v2_container_index_entry_v1",
                **node,
                "account_relative_path": self.account_paths[node_id],
                "account_sha256": self.root.sha256(self.account_paths[node_id]),
                "archive_accounting_hash": account["archive_accounting_hash"],
            }
            _seal(index_entry, "entry_hash")
            index_entries.append(index_entry)
        container_index: dict[str, Any] = {
            "schema_version": "c2_v2_container_accounting_index_v1",
            "format_classifier_config_hash": self.config["config_hash"],
            "container_nodes": index_entries,
        }
        _seal(container_index, "index_hash")
        container_index_path = _write_json(
            self.root, "source_inventory_v2/container_accounting_index.json", container_index
        )
        derived_path = _write_jsonl(
            self.root,
            "source_inventory_v2/derived_archive_members.jsonl",
            self.derived,
        )
        archive_exclusions_path = _write_jsonl(
            self.root,
            "source_inventory_v2/archive_source_only_dispositions.jsonl",
            sorted(
                self.archive_exclusions,
                key=lambda item: (
                    item["container_node_id"],
                    item["central_index"],
                ),
            ),
        )
        units_path = _write_jsonl(
            self.root,
            "cases_v2/consumable_source_units.jsonl",
            sorted(self.units, key=lambda item: item["unit_id"]),
        )
        source_inventory_hash = _sha256_json(
            [item["asset_record_hash"] for item in raw_inventory]
        )
        self._candidate_inputs: list[dict[str, Any]]
        consumption, self._candidate_inputs, candidates, exclusions = self._candidate_pipeline()
        for candidate_input in self._candidate_inputs:
            candidate_input["source_inventory_collection_hash"] = source_inventory_hash
            candidate_input["container_accounting_index_hash"] = container_index["index_hash"]
            expected = candidate_input.pop("record_hash")
            _seal(candidate_input, "record_hash")
            _require(
                expected != candidate_input["record_hash"],
                "candidate input hash must bind source/container indexes",
            )
        # Rebind the matching consumption rows after candidate-input hashes changed.
        by_input_id = {
            item["candidate_input_id"]: item for item in self._candidate_inputs
        }
        for record in consumption:
            if record["candidate_input_id_or_null"] is not None:
                record["candidate_input_record_hash_or_null"] = by_input_id[
                    record["candidate_input_id_or_null"]
                ]["record_hash"]
                record.pop("record_hash")
                _seal(record, "record_hash")
        consumption_path = _write_jsonl(
            self.root, "cases_v2/downstream_consumption.jsonl", consumption
        )
        exclusions_path = _write_jsonl(
            self.root, "cases_v2/source_only_exclusions.jsonl", exclusions
        )
        candidate_inputs_path = _write_jsonl(
            self.root, "cases_v2/candidate_set_inputs.jsonl", self._candidate_inputs
        )
        candidate_path = _write_jsonl(self.root, "cases_v2/candidates.jsonl", candidates)
        relevant_derived_ids = {item["derived_member_id"] for item in self.derived}
        derived_unit_ids = {
            item["unit_id"]
            for item in self.units
            if item["origin_kind"] in {"DERIVED_ARCHIVE_MEMBER", "DERIVED_XLSX_CONTAINER_SELF"}
        }
        _require(
            len(derived_unit_ids) == len(relevant_derived_ids),
            "DERIVED_MEMBER_CONSUMPTION_BIJECTION",
        )
        bijection: dict[str, Any] = {
            "schema_version": "c2_v2_consumption_bijection_validation_v1",
            "derived_member_ids": sorted(relevant_derived_ids),
            "derived_unit_ids": sorted(derived_unit_ids),
            "derived_member_unit_pairs": [
                {
                    "derived_member_id": derived["derived_member_id"],
                    "derived_member_record_hash": derived["record_hash"],
                    "unit_id": next(
                        unit["unit_id"]
                        for unit in self.units
                        if unit["origin_record_hash"] == derived["record_hash"]
                    ),
                    "unit_hash": next(
                        unit["unit_hash"]
                        for unit in self.units
                        if unit["origin_record_hash"] == derived["record_hash"]
                    ),
                }
                for derived in sorted(self.derived, key=lambda item: item["derived_member_id"])
            ],
            "unit_collection_hash": _sha256_json([item["unit_hash"] for item in self.units]),
            "consumption_collection_hash": _sha256_json(
                [item["record_hash"] for item in consumption]
            ),
            "candidate_input_collection_hash": _sha256_json(
                [item["record_hash"] for item in self._candidate_inputs]
            ),
            "candidate_collection_hash": _sha256_json(
                [item["record_hash"] for item in candidates]
            ),
            "source_only_exclusion_collection_hash": _sha256_json(
                [item["record_hash"] for item in exclusions]
            ),
        }
        _seal(bijection, "consumption_bijection_hash")
        bijection_path = _write_json(
            self.root, "cases_v2/consumption_bijection_validation.json", bijection
        )
        proposed, proposal_rejected, proposal_manifest = self._proposal_pipeline(
            candidates,
            self._candidate_inputs,
            consumption,
            source_inventory_hash,
            container_index["index_hash"],
            bijection["consumption_bijection_hash"],
        )
        proposed_path = _write_jsonl(self.root, "proposals_v2/proposed.jsonl", proposed)
        rejected_path = _write_jsonl(
            self.root, "proposals_v2/proposal_rejected.jsonl", proposal_rejected
        )
        proposal_manifest_path = _write_json(
            self.root, "proposals_v2/input_manifest.json", proposal_manifest
        )
        reviews, cases, protocol = self._review_and_canonical(
            candidates=candidates, proposed=proposed
        )
        protocol_path = _write_json(
            self.root, "review_v2/structural_protocol.json", protocol
        )
        review_path = _write_jsonl(
            self.root, "review_v2/structural_outcomes.jsonl", reviews
        )
        canonical_path = _write_jsonl(self.root, "canonical_v2/cases.jsonl", cases)
        canonical_summary: dict[str, Any] = {
            "schema_version": "c2_v2_canonical_summary_v1",
            "case_count": len(cases),
            "canonical_case_collection_hash": _sha256_json(
                [item["record_hash"] for item in cases]
            ),
            "proposal_collection_hash": _sha256_json(
                [item["record_hash"] for item in proposed]
            ),
            "review_collection_hash": _sha256_json(
                [item["record_hash"] for item in reviews]
            ),
            "canonical_builder_rule_hash": _rule_hash(CANONICAL_RULE_ID, "1"),
            "all_cases_retained": True,
        }
        _seal(canonical_summary, "summary_hash")
        canonical_summary_path = _write_json(
            self.root, "canonical_v2/canonical_summary.json", canonical_summary
        )
        case_set_manifest, case_sets_by_doi = _build_canonical_case_set_manifest(
            cases,
            applicable_doi_ids=_applicable_case_set_doi_ids(
                self.terminal_rows,
                self.source_by_article,
            ),
        )
        case_set_manifest_path = _write_json(
            self.root,
            "canonical_v2/canonical_case_set_manifest.json",
            case_set_manifest,
        )
        classifications, dispositions, p_summary = self._p_evidence(
            cases=cases,
            case_sets_by_doi=case_sets_by_doi,
            canonical_case_set_manifest_sha256=self.root.sha256(case_set_manifest_path),
            source_inventory_hash=source_inventory_hash,
            candidate_hash=_sha256_json([item["record_hash"] for item in candidates]),
            proposal_hash=_sha256_json([item["record_hash"] for item in proposed]),
            review_hash=_sha256_json([item["record_hash"] for item in reviews]),
            canonical_hash=canonical_summary["canonical_case_collection_hash"],
            consumption_bijection_hash=bijection["consumption_bijection_hash"],
        )
        classifications_path = _write_jsonl(
            self.root, "p_evidence_v2/source_classifications.jsonl", classifications
        )
        dispositions_path = _write_jsonl(
            self.root, "p_evidence_v2/acquisition_dispositions.jsonl", dispositions
        )
        p_summary_path = _write_json(self.root, "p_evidence_v2/p_summary.json", p_summary)
        validation = validate_source_bearing_extension(
            self.root,
            partition_records=self.partition_records,
            source_chunk_sha256=self.source_chunk_sha256,
            code_attestation=self.code_attestation,
        )
        validation_path = _write_json(
            self.root, "control/v2/source_bearing_extension_validation.json", validation
        )
        return SourceBearingExtensionResult(
            status="SOURCE_CLASSIFICATION_V2_COMPLETE",
            cases={
                "summary_relative_path": canonical_summary_path,
                "summary_sha256": self.root.sha256(canonical_summary_path),
                "summary_hash": canonical_summary["summary_hash"],
                "candidate_relative_path": candidate_path,
                "candidate_sha256": self.root.sha256(candidate_path),
                "candidate_count": len(candidates),
                "source_only_exclusion_relative_path": exclusions_path,
                "source_only_exclusion_sha256": self.root.sha256(exclusions_path),
            },
            proposals={
                "input_manifest_relative_path": proposal_manifest_path,
                "input_manifest_sha256": self.root.sha256(proposal_manifest_path),
                "proposed_relative_path": proposed_path,
                "proposed_sha256": self.root.sha256(proposed_path),
                "rejected_relative_path": rejected_path,
                "rejected_sha256": self.root.sha256(rejected_path),
                "proposal_count": len(proposed),
                "manifest_hash": proposal_manifest["manifest_hash"],
            },
            review={
                "protocol_relative_path": protocol_path,
                "protocol_sha256": self.root.sha256(protocol_path),
                "outcome_relative_path": review_path,
                "outcome_sha256": self.root.sha256(review_path),
                "review_count": len(reviews),
                "review_mode": REVIEW_MODE,
                "protocol_hash": protocol["protocol_hash"],
                "status": "STRUCTURAL_REVIEW_COMPLETE",
            },
            canonical={
                "cases_relative_path": canonical_path,
                "cases_sha256": self.root.sha256(canonical_path),
                "summary_relative_path": canonical_summary_path,
                "summary_sha256": self.root.sha256(canonical_summary_path),
                "case_set_manifest_relative_path": case_set_manifest_path,
                "case_set_manifest_sha256": self.root.sha256(case_set_manifest_path),
                "case_set_manifest_hash": case_set_manifest["case_set_manifest_hash"],
                "case_count": len(cases),
                "summary_hash": canonical_summary["summary_hash"],
                "status": "CANONICAL_SOURCE_DERIVED",
            },
            p_evidence={
                "classification_relative_path": classifications_path,
                "classification_sha256": self.root.sha256(classifications_path),
                "disposition_relative_path": dispositions_path,
                "disposition_sha256": self.root.sha256(dispositions_path),
                "summary_relative_path": p_summary_path,
                "summary_sha256": self.root.sha256(p_summary_path),
                "summary_hash": p_summary["summary_hash"],
                "status": "SOURCE_ONLY_AND_STRATIFIED_DERIVED",
            },
            extension_validation={
                "relative_path": validation_path,
                "sha256": self.root.sha256(validation_path),
                "validation_hash": validation["validation_hash"],
                "raw_inventory_sha256": self.root.sha256(raw_inventory_path),
                "detected_formats_sha256": self.root.sha256(detected_path),
                "container_index_sha256": self.root.sha256(container_index_path),
                "derived_members_sha256": self.root.sha256(derived_path),
                "archive_source_only_dispositions_sha256": self.root.sha256(
                    archive_exclusions_path
                ),
                "units_sha256": self.root.sha256(units_path),
                "consumption_sha256": self.root.sha256(consumption_path),
                "candidate_inputs_sha256": self.root.sha256(candidate_inputs_path),
                "bijection_sha256": self.root.sha256(bijection_path),
            },
        )


def _validate_raw_asset_records(
    root: _TargetRoot,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, bytes]]:
    by_hash: dict[str, Mapping[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for record in records:
        _verify_seal(record, "asset_record_hash", "source inventory record")
        _require(
            record.get("schema_version") == "c2_v2_raw_source_asset_v1",
            "source inventory schema mismatch",
        )
        relative = _require_relative(
            record.get("verified_relative_path"), "source inventory relative path invalid"
        )
        payload = root.read_bytes(relative)
        _require(
            _sha256(payload) == record.get("verified_file_sha256")
            and len(payload) == record.get("verified_bytes"),
            "raw source asset bytes changed",
        )
        descriptor_relative = _require_relative(
            record.get("source_descriptor_relative_path"),
            "source descriptor path invalid",
        )
        descriptor_payload = root.read_bytes(descriptor_relative)
        _require(
            _sha256(descriptor_payload) == record.get("source_descriptor_sha256"),
            "raw source descriptor bytes changed",
        )
        descriptor = _json_object(descriptor_payload, "raw source descriptor")
        assets = validate_source_evidence_descriptor_v2(
            descriptor,
            article_id=str(record.get("raw_article_id")),
            doi_id=str(record.get("parent_doi_id")),
            provenance_relative_path=str(record.get("provenance_relative_path")),
        )
        provenance = _json_object(
            root.read_bytes(
                _require_relative(
                    record.get("provenance_relative_path"),
                    "raw provenance path invalid",
                )
            ),
            "raw source provenance",
        )
        evidence_binding = provenance.get("source_evidence")
        _require(
            provenance.get("doi") == record.get("parent_doi_id")
            and isinstance(evidence_binding, Mapping)
            and evidence_binding.get("descriptor_path") == descriptor_relative
            and evidence_binding.get("descriptor_sha256")
            == record.get("source_descriptor_sha256"),
            "raw provenance descriptor linkage mismatch",
        )
        descriptor_asset = next(
            (
                asset
                for asset in assets
                if asset["asset_id"] == record.get("asset_id")
            ),
            None,
        )
        _require(
            descriptor_asset is not None
            and descriptor_asset["relative_path"] == relative
            and descriptor_asset["sha256"] == record.get("verified_file_sha256")
            and descriptor_asset["bytes"] == record.get("verified_bytes"),
            "raw source descriptor asset linkage mismatch",
        )
        _require(record["asset_record_hash"] not in by_hash, "duplicate raw source asset hash")
        by_hash[str(record["asset_record_hash"])] = record
        payloads[str(record["asset_record_hash"])] = payload
    return by_hash, payloads


def _validate_archive_accounts(
    root: _TargetRoot,
    index: Mapping[str, Any],
    config: Mapping[str, Any],
    archive_budget: _ArchiveRunBudget,
) -> dict[str, Mapping[str, Any]]:
    _verify_seal(index, "index_hash", "container accounting index")
    _require(
        index.get("schema_version") == "c2_v2_container_accounting_index_v1",
        "container index schema mismatch",
    )
    nodes = index.get("container_nodes")
    _require(
        isinstance(nodes, list)
        and index.get("format_classifier_config_hash") == config.get("config_hash")
        and [item.get("container_node_id") for item in nodes]
        == sorted(item.get("container_node_id") for item in nodes),
        "container index registry/order mismatch",
    )
    nodes_by_id: dict[str, Mapping[str, Any]] = {}
    origins: set[str] = set()
    for node in nodes:
        _verify_seal(node, "entry_hash", "container index entry")
        node_id = _require_identifier(
            node.get("container_node_id"), "container node ID invalid"
        )
        origin = _require_sha256(
            node.get("origin_record_hash"), "container node origin hash is invalid"
        )
        _require(
            node_id not in nodes_by_id and origin not in origins,
            "duplicate container node/origin",
        )
        nodes_by_id[node_id] = node
        origins.add(origin)
    for node_id, node in nodes_by_id.items():
        parent_id = node.get("parent_container_node_id_or_null")
        depth = node.get("depth")
        _require(
            isinstance(depth, int)
            and not isinstance(depth, bool)
            and depth >= 0
            and node.get("content_profile") in {"XLSX_V1", "GENERIC_ZIP_V1"},
            "container node metadata is invalid",
        )
        if parent_id is None:
            _require(depth == 0, "root container depth is invalid")
        else:
            _require(
                isinstance(parent_id, str)
                and parent_id in nodes_by_id
                and nodes_by_id[parent_id]["depth"] + 1 == depth
                and parent_id != node_id,
                "container parent graph is invalid",
            )
    result: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        node_id = _require_identifier(node.get("container_node_id"), "container node ID invalid")
        _require(node_id not in result, "duplicate container node")
        relative = _require_relative(
            node.get("account_relative_path"), "container account relative path invalid"
        )
        account = _json_object(root.read_bytes(relative), f"container account {node_id}")
        _verify_seal(account, "archive_accounting_hash", f"container account {node_id}")
        _require(
            account.get("container_node_id") == node_id
            and root.sha256(relative) == node.get("account_sha256")
            and account.get("archive_accounting_hash") == node.get("archive_accounting_hash")
            and all(
                account.get(field) == node.get(field)
                for field in (
                    "origin_record_hash",
                    "parent_doi_id",
                    "frozen_input_ordinal",
                    "verified_relative_path",
                    "verified_file_sha256",
                    "verified_bytes",
                    "content_profile",
                )
            )
            and account.get("container_format") == "ZIP_V1"
            and account.get("format_classifier_id") == FORMAT_CLASSIFIER_ID
            and account.get("format_classifier_version") == "1"
            and account.get("format_classifier_code_sha256")
            == config.get("format_classifier_code_sha256")
            and account.get("zip_v1_parser_id") == ZIP_PARSER_ID
            and account.get("zip_v1_parser_version") == "1"
            and account.get("zip_v1_parser_code_sha256")
            == config.get("zip_v1_parser_code_sha256"),
            "container account index binding mismatch",
        )
        payload = root.read_bytes(
            _require_relative(
                account.get("verified_relative_path"), "container source path invalid"
            )
        )
        _require(
            _sha256(payload) == account.get("verified_file_sha256")
            and len(payload) == account.get("verified_bytes"),
            "container source bytes changed",
        )
        archive = _parse_zip_v1(payload)
        archive_budget.reserve(payload, archive)
        _require(
            archive.central_directory_sha256 == account.get("central_directory_sha256")
            and archive.eocd_sha256 == account.get("eocd_sha256")
            and len(archive.entries) == account.get("physical_entry_count")
            and len(account.get("entries", [])) == len(archive.entries),
            "container central directory changed",
        )
        _require(
            account.get("central_entry_uncompressed_bytes")
            == sum(int(entry["uncompressed_bytes"]) for entry in account["entries"])
            and account.get("regular_member_uncompressed_bytes")
            == sum(
                int(entry["uncompressed_bytes"])
                for entry in account["entries"]
                if entry["disposition"] != "DIRECTORY_ENTRY"
            )
            and account.get("derived_member_uncompressed_bytes")
            == sum(
                int(entry["uncompressed_bytes"])
                for entry in account["entries"]
                if entry["derived_member_record_hash_or_null"] is not None
            )
            and account.get("source_only_excluded_uncompressed_bytes")
            == sum(
                int(entry["uncompressed_bytes"])
                for entry in account["entries"]
                if entry["archive_source_only_disposition_record_hash_or_null"]
                is not None
                and entry["disposition"] != "DIRECTORY_ENTRY"
            )
            and account["regular_member_uncompressed_bytes"]
            == account["derived_member_uncompressed_bytes"]
            + account["source_only_excluded_uncompressed_bytes"],
            "container byte accounting equation mismatch",
        )
        for observed, stored in zip(archive.entries, account["entries"], strict=True):
            _verify_seal(stored, "entry_hash", f"container {node_id} entry")
            _require(
                observed.index == stored.get("central_index")
                and _sha256(observed.raw_name) == stored.get("raw_name_sha256")
                and observed.selector == stored.get("selector")
                and observed.selector_casefold == stored.get("selector_casefold")
                and observed.header_sha256 == stored.get("central_header_sha256")
                and observed.compression_method == stored.get("compression_method")
                and observed.flag_bits == stored.get("flag_bits")
                and observed.crc32 == stored.get("crc32")
                and observed.compressed_bytes == stored.get("compressed_bytes")
                and observed.uncompressed_bytes == stored.get("uncompressed_bytes")
                and observed.external_attributes == stored.get("external_attributes")
                and observed.local_header_offset == stored.get("local_header_offset"),
                "container entry accounting mismatch",
            )
            derived_hash = stored["derived_member_record_hash_or_null"]
            exclusion_hash = stored[
                "archive_source_only_disposition_record_hash_or_null"
            ]
            if observed.is_directory:
                _require(
                    stored["disposition"] == "DIRECTORY_ENTRY"
                    and derived_hash is None
                    and exclusion_hash is not None
                    and stored["source_only_exclusion_reason_or_null"]
                    == "DIRECTORY_ENTRY",
                    "directory archive accounting disposition mismatch",
                )
            else:
                _require(
                    (derived_hash is None) != (exclusion_hash is None)
                    and stored["disposition"]
                    in {"DERIVED_MEMBER", "SOURCE_ONLY_EXCLUSION"},
                    "regular archive entry accounting disposition mismatch",
                )
        result[node_id] = account
    return result


def validate_source_bearing_extension(
    root: _TargetRoot,
    *,
    partition_records: int,
    source_chunk_sha256: str,
    code_attestation: SourceExtensionCodeAttestation | None = None,
) -> dict[str, Any]:
    """Independently replay V2 bytes -> account -> consumption -> canonical/P."""

    attestation = (
        verify_source_extension_code_attestation()
        if code_attestation is None
        else code_attestation
    )
    attestation.verify_runtime()
    config = _json_object(
        root.read_bytes("source_inventory_v2/fd_format_classifier_config.json"),
        "format classifier config",
    )
    _verify_seal(config, "config_hash", "format classifier config")
    expected_config = _format_config(attestation)
    _require(config == expected_config, "format classifier registry/code binding changed")
    inventory = _jsonl_objects(
        root.read_bytes("source_inventory_v2/source_inventory.jsonl"), "source inventory"
    )
    _require(
        [
            (item.get("frozen_input_ordinal"), item.get("asset_id"))
            for item in inventory
        ]
        == sorted(
            (item.get("frozen_input_ordinal"), item.get("asset_id"))
            for item in inventory
        ),
        "source inventory order is not deterministic",
    )
    raw_by_hash, _raw_payloads = _validate_raw_asset_records(root, inventory)
    detected = _jsonl_objects(
        root.read_bytes("source_inventory_v2/detected_formats.jsonl"), "detected formats"
    )
    detected_by_origin: dict[str, Mapping[str, Any]] = {}
    format_replay_budget = _ArchiveRunBudget()
    for item in detected:
        _verify_seal(item, "format_hash", "detected format")
        origin = str(item.get("origin_record_hash"))
        _require(origin not in detected_by_origin, "duplicate detected format origin")
        relative = _require_relative(
            item.get("verified_relative_path"), "detected format source path invalid"
        )
        payload = root.read_bytes(relative)
        _require(
            _sha256(payload) == item.get("verified_file_sha256")
            and len(payload) == item.get("verified_bytes"),
            "detected format source bytes changed",
        )
        replay = _detect_format(
            payload,
            archive_budget=format_replay_budget,
        )
        _require(
            [replay.container_format, replay.content_profile]
            == [item.get("container_format"), item.get("content_profile")],
            "FD format replay mismatch",
        )
        if origin in raw_by_hash:
            _validate_declared_format(raw_by_hash[origin], replay)
        detected_by_origin[origin] = item
    index = _json_object(
        root.read_bytes("source_inventory_v2/container_accounting_index.json"),
        "container accounting index",
    )
    accounts = _validate_archive_accounts(
        root,
        index,
        config,
        _ArchiveRunBudget(),
    )
    archive_exclusions = _jsonl_objects(
        root.read_bytes("source_inventory_v2/archive_source_only_dispositions.jsonl"),
        "archive source-only dispositions",
    )
    archive_exclusion_by_hash: dict[str, Mapping[str, Any]] = {}
    for exclusion in archive_exclusions:
        _verify_seal(
            exclusion,
            "record_hash",
            "archive source-only disposition",
        )
        _require(
            exclusion.get("container_node_id") in accounts
            and exclusion.get("reason_code") in _SOURCE_ONLY_REASONS,
            "archive source-only disposition is invalid",
        )
        digest = str(exclusion["record_hash"])
        _require(digest not in archive_exclusion_by_hash, "duplicate archive exclusion")
        archive_exclusion_by_hash[digest] = exclusion
    derived = _jsonl_objects(
        root.read_bytes("source_inventory_v2/derived_archive_members.jsonl"),
        "derived archive members",
    )
    derived_by_hash: dict[str, Mapping[str, Any]] = {}
    derived_by_id: dict[str, Mapping[str, Any]] = {}
    derived_by_container_entry: set[tuple[str, int, str]] = set()
    raw_by_article_asset = {
        (str(item["raw_article_id"]), str(item["asset_id"])): item
        for item in inventory
    }
    for item in derived:
        _verify_seal(item, "record_hash", "derived archive member")
        member_id = _require_identifier(
            item.get("derived_member_id"), "derived archive member ID is invalid"
        )
        digest = _require_sha256(
            item.get("record_hash"), "derived archive member hash is invalid"
        )
        node_id = item.get("container_node_id")
        _require(
            isinstance(node_id, str) and node_id in accounts,
            "derived archive member has unknown container",
        )
        central_index = item.get("central_index")
        central_header_sha256 = item.get("central_header_sha256")
        _require(
            isinstance(central_index, int)
            and not isinstance(central_index, bool)
            and isinstance(central_header_sha256, str)
            and member_id not in derived_by_id
            and digest not in derived_by_hash
            and (node_id, central_index, central_header_sha256)
            not in derived_by_container_entry,
            "duplicate derived archive member",
        )
        payload = root.read_bytes(
            _require_relative(item.get("verified_relative_path"), "derived member path invalid")
        )
        _require(
            _sha256(payload) == item.get("verified_file_sha256")
            and len(payload) == item.get("verified_bytes"),
            "derived archive member bytes changed",
        )
        detected_record = detected_by_origin.get(digest)
        _require(
            detected_record is not None
            and [detected_record["container_format"], detected_record["content_profile"]]
            == item.get("detected_format_tuple"),
            "derived member format replay mismatch",
        )
        account = accounts[node_id]
        container_payload = root.read_bytes(account["verified_relative_path"])
        archive = _parse_zip_v1(container_payload)
        matching_central = [
            entry
            for entry in archive.entries
            if entry.index == central_index
            and entry.header_sha256 == central_header_sha256
        ]
        _require(
            len(matching_central) == 1
            and payload == _read_zip_member(container_payload, matching_central[0])
            and item.get("parent_doi_id") == account.get("parent_doi_id")
            and item.get("frozen_input_ordinal")
            == account.get("frozen_input_ordinal"),
            "derived archive member extraction/provenance mismatch",
        )
        member_selector = item.get("member_selector")
        _require(
            isinstance(member_selector, str)
            and member_selector.rsplit("!", 1)[-1] == matching_central[0].selector,
            "derived archive member selector mismatch",
        )
        raw_record = raw_by_article_asset.get(
            (str(item.get("raw_article_id")), str(item.get("raw_asset_id")))
        )
        _require(
            raw_record is not None
            and raw_record.get("parent_doi_id") == item.get("parent_doi_id")
            and raw_record.get("frozen_input_ordinal")
            == item.get("frozen_input_ordinal"),
            "derived archive member raw provenance mismatch",
        )
        derived_by_hash[digest] = item
        derived_by_id[member_id] = item
        derived_by_container_entry.add((node_id, central_index, central_header_sha256))
    _require(
        [item.get("derived_member_id") for item in derived]
        == sorted(item.get("derived_member_id") for item in derived),
        "derived archive member order is not deterministic",
    )
    node_ids_by_origin: dict[str, list[str]] = defaultdict(list)
    for node in index["container_nodes"]:
        node_ids_by_origin[str(node["origin_record_hash"])].append(
            str(node["container_node_id"])
        )
    expected_detected_origins = set(raw_by_hash) | set(derived_by_hash)
    _require(
        set(detected_by_origin) == expected_detected_origins,
        "FD format detection coverage mismatch",
    )
    for origin, record in {**raw_by_hash, **derived_by_hash}.items():
        detected_record = detected_by_origin[origin]
        detected_node_ids = node_ids_by_origin.get(origin, [])
        _require(
            len(detected_node_ids) <= 1
            and detected_record.get("parent_doi_id") == record["parent_doi_id"]
            and detected_record.get("frozen_input_ordinal")
            == record["frozen_input_ordinal"]
            and detected_record.get("verified_relative_path")
            == record["verified_relative_path"]
            and detected_record.get("verified_file_sha256")
            == record["verified_file_sha256"]
            and detected_record.get("verified_bytes") == record["verified_bytes"]
            and detected_record.get("format_classifier_id") == FORMAT_CLASSIFIER_ID
            and detected_record.get("format_classifier_version") == "1"
            and detected_record.get("format_classifier_code_sha256")
            == config.get("format_classifier_code_sha256")
            and detected_record.get("format_detection_rule_hash")
            == config.get("format_detection_rule_hash")
            and detected_record.get("container_node_id_or_null")
            == (detected_node_ids[0] if detected_node_ids else None),
            "FD detected format lineage mismatch",
        )
        is_zip = detected_record.get("container_format") == "ZIP_V1"
        _require(
            is_zip == bool(detected_node_ids)
            and (
                not is_zip
                or detected_record.get("content_profile")
                == index["container_nodes"][
                    next(
                        position
                        for position, node in enumerate(index["container_nodes"])
                        if node["container_node_id"] == detected_node_ids[0]
                    )
                ]["content_profile"]
            ),
            "container/detected-format graph mismatch",
        )
    accounted_derived_hashes: set[str] = set()
    accounted_exclusion_hashes: set[str] = set()
    for account in accounts.values():
        accounted = {
            item["derived_member_record_hash_or_null"]
            for item in account["entries"]
            if item["derived_member_record_hash_or_null"] is not None
        }
        accounted_derived_hashes.update(str(value) for value in accounted)
        _require(
            accounted.issubset(derived_by_hash),
            "archive accounting refers to a missing derived member",
        )
        for entry in account["entries"]:
            exclusion_hash = entry[
                "archive_source_only_disposition_record_hash_or_null"
            ]
            if exclusion_hash is None:
                _require(
                    entry["derived_member_record_hash_or_null"] is not None,
                    "archive entry lacks a terminal accounting disposition",
                )
                continue
            accounted_exclusion_hashes.add(str(exclusion_hash))
            exclusion = archive_exclusion_by_hash.get(str(exclusion_hash))
            _require(
                exclusion is not None
                and exclusion["container_node_id"] == account["container_node_id"]
                and exclusion["central_index"] == entry["central_index"]
                and exclusion["central_header_sha256"] == entry["central_header_sha256"]
                and exclusion["reason_code"]
                == entry["source_only_exclusion_reason_or_null"],
                "archive exclusion/accounting binding mismatch",
            )
    _require(
        accounted_derived_hashes == set(derived_by_hash),
        "archive accounting/derived-member closure mismatch",
    )
    _require(
        accounted_exclusion_hashes == set(archive_exclusion_by_hash),
        "archive accounting/source-only exclusion closure mismatch",
    )
    for item in derived:
        account = accounts[str(item["container_node_id"])]
        matching = [
            entry
            for entry in account["entries"]
            if entry["central_index"] == item["central_index"]
            and entry["central_header_sha256"] == item["central_header_sha256"]
        ]
        _require(
            len(matching) == 1
            and matching[0]["derived_member_record_hash_or_null"] == item["record_hash"],
            "derived member central-directory binding mismatch",
        )
    units = _jsonl_objects(
        root.read_bytes("cases_v2/consumable_source_units.jsonl"), "source units"
    )
    units_by_id: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        _verify_seal(unit, "unit_hash", "source unit")
        unit_id = _require_identifier(unit.get("unit_id"), "source unit ID invalid")
        _require(unit_id not in units_by_id, "duplicate source unit")
        payload = root.read_bytes(
            _require_relative(unit.get("verified_relative_path"), "source unit path invalid")
        )
        _require(
            _sha256(payload) == unit.get("verified_file_sha256")
            and len(payload) == unit.get("verified_bytes"),
            "source unit bytes changed",
        )
        detected_record = detected_by_origin.get(str(unit.get("origin_record_hash")))
        _require(
            detected_record is not None
            and [
                detected_record["container_format"],
                detected_record["content_profile"],
            ]
            == unit.get("detected_format_tuple"),
            "source unit format replay mismatch",
        )
        units_by_id[unit_id] = unit
    _require(
        [item.get("unit_id") for item in units]
        == sorted(item.get("unit_id") for item in units),
        "source unit order is not deterministic",
    )
    expected_units_by_origin: dict[str, dict[str, Any]] = {}
    for origin, raw in raw_by_hash.items():
        detected_record = detected_by_origin[origin]
        profile = detected_record["content_profile"]
        if profile not in {"CSV_V1", "XLSX_V1"}:
            continue
        expected_units_by_origin[origin] = {
            "origin_kind": (
                "RAW_XLSX_CONTAINER_SELF"
                if profile == "XLSX_V1"
                else "RAW_NONCONTAINER_SELF"
            ),
            "parent_doi_id": raw["parent_doi_id"],
            "frozen_input_ordinal": raw["frozen_input_ordinal"],
            "verified_relative_path": raw["verified_relative_path"],
            "verified_file_sha256": raw["verified_file_sha256"],
            "verified_bytes": raw["verified_bytes"],
            "container_node_id_or_null": detected_record["container_node_id_or_null"],
            "raw_article_id": raw["raw_article_id"],
            "raw_asset_id": raw["asset_id"],
            "member_selector_or_null": None,
            "candidate_capable": True,
            "detected_format_tuple": [
                detected_record["container_format"],
                detected_record["content_profile"],
            ],
        }
    for origin, member in derived_by_hash.items():
        detected_record = detected_by_origin[origin]
        profile = detected_record["content_profile"]
        expected_units_by_origin[origin] = {
            "origin_kind": (
                "DERIVED_XLSX_CONTAINER_SELF"
                if profile == "XLSX_V1"
                else "DERIVED_ARCHIVE_MEMBER"
            ),
            "parent_doi_id": member["parent_doi_id"],
            "frozen_input_ordinal": member["frozen_input_ordinal"],
            "verified_relative_path": member["verified_relative_path"],
            "verified_file_sha256": member["verified_file_sha256"],
            "verified_bytes": member["verified_bytes"],
            "container_node_id_or_null": detected_record["container_node_id_or_null"],
            "raw_article_id": member["raw_article_id"],
            "raw_asset_id": member["raw_asset_id"],
            "member_selector_or_null": member["member_selector"],
            "candidate_capable": profile in {"CSV_V1", "XLSX_V1"},
            "detected_format_tuple": [
                detected_record["container_format"],
                detected_record["content_profile"],
            ],
        }
    units_by_origin: dict[str, Mapping[str, Any]] = {}
    for unit_id, unit in units_by_id.items():
        origin = _require_sha256(
            unit.get("origin_record_hash"), "source unit origin hash is invalid"
        )
        expected_unit = expected_units_by_origin.get(origin)
        _require(
            expected_unit is not None
            and origin not in units_by_origin
            and all(unit.get(field) == expected for field, expected in expected_unit.items())
            and unit.get("format_classifier_id") == FORMAT_CLASSIFIER_ID
            and unit.get("format_classifier_version") == "1"
            and unit.get("format_classifier_code_sha256")
            == config.get("format_classifier_code_sha256")
            and unit.get("format_detection_rule_hash")
            == config.get("format_detection_rule_hash"),
            "source unit origin/provenance binding mismatch",
        )
        units_by_origin[origin] = unit
    _require(
        set(units_by_origin) == set(expected_units_by_origin),
        "source unit origin coverage mismatch",
    )
    source_exclusions = _jsonl_objects(
        root.read_bytes("cases_v2/source_only_exclusions.jsonl"),
        "source-only exclusions",
    )
    source_exclusion_by_hash: dict[str, Mapping[str, Any]] = {}
    source_exclusion_by_unit: dict[str, Mapping[str, Any]] = {}
    for exclusion in source_exclusions:
        _verify_seal(exclusion, "record_hash", "source-only exclusion")
        unit_id = _require_identifier(
            exclusion.get("unit_id"), "source-only exclusion unit ID is invalid"
        )
        digest = _require_sha256(
            exclusion.get("record_hash"), "source-only exclusion hash is invalid"
        )
        _require(
            unit_id in units_by_id
            and unit_id not in source_exclusion_by_unit
            and digest not in source_exclusion_by_hash
            and exclusion.get("unit_hash") == units_by_id[unit_id]["unit_hash"]
            and exclusion.get("parent_doi_id")
            == units_by_id[unit_id]["parent_doi_id"]
            and exclusion.get("reason_code") in _SOURCE_ONLY_REASONS,
            "source-only exclusion binding is invalid",
        )
        source_exclusion_by_hash[digest] = exclusion
        source_exclusion_by_unit[unit_id] = exclusion
    consumption = _jsonl_objects(
        root.read_bytes("cases_v2/downstream_consumption.jsonl"), "downstream consumption"
    )
    consumption_by_unit: dict[str, Mapping[str, Any]] = {}
    consumption_by_link: dict[str, Mapping[str, Any]] = {}
    for record in consumption:
        _verify_seal(record, "record_hash", "downstream consumption")
        unit_id = _require_identifier(
            record.get("unit_id"), "downstream consumption unit ID is invalid"
        )
        link_hash = _require_sha256(
            record.get("consumption_link_hash"),
            "downstream consumption link hash is invalid",
        )
        _require(
            unit_id in units_by_id
            and unit_id not in consumption_by_unit
            and link_hash not in consumption_by_link,
            "DERIVED_MEMBER_CONSUMPTION_BIJECTION",
        )
        unit = units_by_id[unit_id]
        _require(
            record.get("unit_hash") == unit.get("unit_hash")
            and record.get("parent_doi_id") == unit.get("parent_doi_id")
            and record.get("origin_record_hash") == unit.get("origin_record_hash")
            and record.get("container_node_id_or_null")
            == unit.get("container_node_id_or_null")
            and link_hash
            == _sha256_json(
                {
                    "unit_id": unit_id,
                    "unit_hash": unit["unit_hash"],
                    "parent_doi_id": unit["parent_doi_id"],
                }
            ),
            "consumption unit binding mismatch",
        )
        disposition = record.get("consumption_disposition")
        _require(
            disposition in _CONSUMPTION_DISPOSITIONS and disposition != "REJECTION",
            "downstream consumption disposition is invalid",
        )
        if disposition == "SOURCE_ONLY_EXCLUSION":
            exclusion_hash = record.get("source_only_exclusion_record_hash_or_null")
            exclusion = source_exclusion_by_hash.get(str(exclusion_hash))
            _require(
                record.get("candidate_input_id_or_null") is None
                and record.get("candidate_input_record_hash_or_null") is None
                and exclusion is not None
                and exclusion["unit_id"] == unit_id
                and exclusion["reason_code"] == record.get("reason_code"),
                "source-only consumption exclusion binding mismatch",
            )
        else:
            _require(
                unit.get("candidate_capable") is True
                and record.get("source_only_exclusion_record_hash_or_null") is None
                and record.get("reason_code") == "BYTE_VALID_TABLE_UNIT",
                "candidate consumption disposition is invalid",
            )
        consumption_by_unit[unit_id] = record
        consumption_by_link[link_hash] = record
    _require(
        [item.get("unit_id") for item in consumption]
        == sorted(item.get("unit_id") for item in consumption),
        "consumption order is not deterministic",
    )
    _require(
        set(consumption_by_unit) == set(units_by_id),
        "DERIVED_MEMBER_CONSUMPTION_BIJECTION",
    )
    inputs = _jsonl_objects(
        root.read_bytes("cases_v2/candidate_set_inputs.jsonl"), "candidate inputs"
    )
    inputs_by_id: dict[str, Mapping[str, Any]] = {}
    inputs_by_unit: dict[str, Mapping[str, Any]] = {}
    for record in inputs:
        _verify_seal(record, "record_hash", "candidate input")
        input_id = _require_identifier(record.get("candidate_input_id"), "candidate input ID")
        unit_id = _require_identifier(
            record.get("unit_id"), "candidate input unit ID is invalid"
        )
        _require(
            input_id not in inputs_by_id and unit_id not in inputs_by_unit,
            "duplicate candidate input",
        )
        consumption_record = consumption_by_unit.get(unit_id)
        _require(
            unit_id in units_by_id
            and units_by_id[unit_id]["candidate_capable"] is True
            and record.get("unit_hash") == units_by_id[unit_id]["unit_hash"]
            and record.get("parent_doi_id") == units_by_id[unit_id]["parent_doi_id"]
            and consumption_record is not None
            and consumption_record["consumption_disposition"] == "CANDIDATE_SET_INPUT"
            and record.get("consumption_link_hash")
            == consumption_record["consumption_link_hash"]
            and record.get("record_hash")
            == consumption_record["candidate_input_record_hash_or_null"]
            and input_id == consumption_record["candidate_input_id_or_null"],
            "candidate input unit binding mismatch",
        )
        inputs_by_id[input_id] = record
        inputs_by_unit[unit_id] = record
    _require(
        [item.get("candidate_input_id") for item in inputs]
        == sorted(item.get("candidate_input_id") for item in inputs),
        "candidate input order is not deterministic",
    )
    source_inventory_hash = _sha256_json(
        [item["asset_record_hash"] for item in inventory]
    )
    for candidate_input in inputs:
        _require(
            candidate_input.get("source_inventory_collection_hash")
            == source_inventory_hash
            and candidate_input.get("container_accounting_index_hash")
            == index.get("index_hash")
            and candidate_input.get("candidate_builder_rule_id") == CANDIDATE_RULE_ID
            and candidate_input.get("candidate_builder_rule_version") == "1"
            and candidate_input.get("candidate_builder_rule_hash")
            == _rule_hash(CANDIDATE_RULE_ID, "1"),
            "candidate input registry binding mismatch",
        )
    expected_candidate_input_units = {
        unit_id
        for unit_id, unit in units_by_id.items()
        if unit["candidate_capable"]
    }
    _require(
        set(inputs_by_unit) == expected_candidate_input_units
        and {
            unit_id
            for unit_id, record in consumption_by_unit.items()
            if record["consumption_disposition"] == "CANDIDATE_SET_INPUT"
        }
        == expected_candidate_input_units,
        "candidate-capable source unit lacks exact candidate input",
    )
    for candidate_input in inputs:
        outcome = candidate_input.get("candidate_outcome")
        exclusion_hash = candidate_input.get(
            "source_only_exclusion_record_hash_or_null"
        )
        if outcome == "SOURCE_ONLY_EXCLUSION":
            exclusion = source_exclusion_by_hash.get(str(exclusion_hash))
            _require(
                candidate_input.get("candidate_ids") == []
                and candidate_input.get("candidate_record_hashes") == []
                and exclusion is not None
                and exclusion["unit_id"] == candidate_input["unit_id"],
                "candidate input source-only exclusion binding mismatch",
            )
        else:
            _require(
                outcome == "CANDIDATES_EMITTED"
                and exclusion_hash is None,
                "candidate input outcome is invalid",
            )
    _require(
        {
            exclusion["unit_id"]
            for exclusion in source_exclusions
        }
        == {
            unit_id
            for unit_id, record in consumption_by_unit.items()
            if record["consumption_disposition"] == "SOURCE_ONLY_EXCLUSION"
        }
        | {
            str(record["unit_id"])
            for record in inputs
            if record["candidate_outcome"] == "SOURCE_ONLY_EXCLUSION"
        },
        "source-only exclusion consumption coverage mismatch",
    )
    candidates = _jsonl_objects(root.read_bytes("cases_v2/candidates.jsonl"), "candidates")
    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    candidate_ids_by_input: dict[str, list[str]] = defaultdict(list)
    candidate_hashes_by_input: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        _verify_seal(candidate, "record_hash", "candidate")
        candidate_id = _require_identifier(candidate.get("candidate_id"), "candidate ID")
        _require(candidate_id not in candidates_by_id, "duplicate candidate")
        input_id = _require_identifier(
            candidate.get("candidate_input_id"), "candidate input ID is invalid"
        )
        candidate_input = inputs_by_id.get(input_id)
        _require(
            candidate_input is not None
            and candidate_input.get("candidate_outcome") == "CANDIDATES_EMITTED",
            "candidate has unknown/nonemitting input",
        )
        unit = units_by_id[str(candidate_input["unit_id"])]
        figure = raw_by_hash.get(str(candidate.get("figure_asset_record_hash")))
        caption = raw_by_hash.get(str(candidate.get("caption_asset_record_hash")))
        _require(
            candidate.get("parent_doi_id") == unit["parent_doi_id"]
            and candidate.get("unit_id") == unit["unit_id"]
            and candidate.get("unit_hash") == unit["unit_hash"]
            and candidate.get("consumption_link_hash")
            == candidate_input["consumption_link_hash"]
            and candidate.get("source_table_sha256") == unit["verified_file_sha256"]
            and candidate.get("source_table_bytes") == unit["verified_bytes"]
            and figure is not None
            and caption is not None
            and figure.get("parent_doi_id") == unit["parent_doi_id"]
            and caption.get("parent_doi_id") == unit["parent_doi_id"]
            and figure.get("raw_article_id") == unit["raw_article_id"]
            and caption.get("raw_article_id") == unit["raw_article_id"]
            and figure.get("declared_asset_kind") == "figure"
            and caption.get("declared_asset_kind") == "caption"
            and (
                candidate.get("case_group_or_null") is None
                or (
                    isinstance(candidate.get("case_group_or_null"), str)
                    and _IDENTIFIER_RE.fullmatch(candidate["case_group_or_null"])
                    is not None
                )
            )
            and isinstance(candidate.get("panel_id"), str)
            and _IDENTIFIER_RE.fullmatch(candidate["panel_id"]) is not None,
            "candidate source/provenance binding mismatch",
        )
        candidate_ids_by_input[input_id].append(candidate_id)
        candidate_hashes_by_input[input_id].append(str(candidate["record_hash"]))
        candidates_by_id[candidate_id] = candidate
    _require(
        [item.get("candidate_id") for item in candidates]
        == sorted(item.get("candidate_id") for item in candidates),
        "candidate order is not deterministic",
    )
    for input_id, candidate_input in inputs_by_id.items():
        emitted_ids = candidate_ids_by_input[input_id]
        _require(
            candidate_input.get("candidate_ids") == emitted_ids
            and candidate_input.get("candidate_record_hashes")
            == candidate_hashes_by_input[input_id]
            and (
                bool(emitted_ids)
                if candidate_input["candidate_outcome"] == "CANDIDATES_EMITTED"
                else not emitted_ids
            ),
            "candidate input candidate union mismatch",
        )
    bijection = _json_object(
        root.read_bytes("cases_v2/consumption_bijection_validation.json"),
        "consumption bijection",
    )
    _verify_seal(bijection, "consumption_bijection_hash", "consumption bijection")
    ordered_derived = sorted(derived, key=lambda item: str(item["derived_member_id"]))
    derived_units = [
        units_by_origin[str(item["record_hash"])] for item in ordered_derived
    ]
    expected_pairs = [
        {
            "derived_member_id": item["derived_member_id"],
            "derived_member_record_hash": item["record_hash"],
            "unit_id": units_by_origin[str(item["record_hash"])]["unit_id"],
            "unit_hash": units_by_origin[str(item["record_hash"])]["unit_hash"],
        }
        for item in ordered_derived
    ]
    _require(
        bijection.get("derived_member_ids")
        == [item["derived_member_id"] for item in ordered_derived]
        and bijection.get("derived_unit_ids")
        == sorted(item["unit_id"] for item in derived_units)
        and bijection.get("derived_member_unit_pairs") == expected_pairs
        and bijection.get("unit_collection_hash")
        == _sha256_json([item["unit_hash"] for item in units])
        and bijection.get("consumption_collection_hash")
        == _sha256_json([item["record_hash"] for item in consumption])
        and bijection.get("candidate_input_collection_hash")
        == _sha256_json([item["record_hash"] for item in inputs])
        and bijection.get("candidate_collection_hash")
        == _sha256_json([item["record_hash"] for item in candidates])
        and bijection.get("source_only_exclusion_collection_hash")
        == _sha256_json([item["record_hash"] for item in source_exclusions]),
        "DERIVED_MEMBER_CONSUMPTION_BIJECTION",
    )
    proposed = _jsonl_objects(root.read_bytes("proposals_v2/proposed.jsonl"), "proposals")
    rejected = _jsonl_objects(
        root.read_bytes("proposals_v2/proposal_rejected.jsonl"), "proposal rejected"
    )
    proposal_candidates: set[str] = set()
    proposals_by_id: dict[str, Mapping[str, Any]] = {}
    for proposals in (proposed, rejected):
        _require(
            [item.get("proposal_id") for item in proposals]
            == sorted(item.get("proposal_id") for item in proposals),
            "proposal order is not deterministic",
        )
    for proposal in [*proposed, *rejected]:
        _verify_seal(proposal, "record_hash", "proposal")
        proposal_id = _require_identifier(proposal.get("proposal_id"), "proposal ID")
        _require(proposal_id not in proposals_by_id, "duplicate proposal")
        candidate_id = _require_identifier(
            proposal.get("candidate_id"), "proposal candidate ID is invalid"
        )
        candidate = candidates_by_id.get(candidate_id)
        _require(
            candidate is not None and candidate_id not in proposal_candidates,
            "proposal candidate union mismatch",
        )
        candidate_input = inputs_by_id[str(candidate["candidate_input_id"])]
        consumption_record = consumption_by_link.get(
            str(candidate["consumption_link_hash"])
        )
        _require(
            consumption_record is not None
            and proposal.get("candidate_record_hash") == candidate["record_hash"]
            and proposal.get("candidate_input_record_hash")
            == candidate_input["record_hash"]
            and proposal.get("consumption_record_hash") == consumption_record["record_hash"]
            and proposal.get("unit_hash") == candidate["unit_hash"]
            and proposal.get("parent_doi_id") == candidate["parent_doi_id"],
            "proposal provenance binding mismatch",
        )
        proposal_candidates.add(candidate_id)
        proposals_by_id[proposal_id] = proposal
    _require(proposal_candidates == set(candidates_by_id), "proposal candidate union mismatch")
    proposal_manifest = _json_object(
        root.read_bytes("proposals_v2/input_manifest.json"),
        "proposal input manifest",
    )
    _verify_seal(proposal_manifest, "manifest_hash", "proposal input manifest")
    _require(
        proposal_manifest.get("candidate_collection_hash")
        == _sha256_json([item["record_hash"] for item in candidates])
        and proposal_manifest.get("candidate_set_input_collection_hash")
        == _sha256_json([item["record_hash"] for item in inputs])
        and proposal_manifest.get("consumption_collection_hash")
        == _sha256_json([item["record_hash"] for item in consumption])
        and proposal_manifest.get("consumption_bijection_hash")
        == bijection.get("consumption_bijection_hash")
        and proposal_manifest.get("source_inventory_collection_hash")
        == source_inventory_hash
        and proposal_manifest.get("container_accounting_index_hash")
        == index.get("index_hash")
        and proposal_manifest.get("format_classifier_config_hash")
        == config.get("config_hash")
        and proposal_manifest.get("proposal_rule_hash")
        == _rule_hash("c2_v2_proposal_builder_v1", "1")
        and proposal_manifest.get("review_protocol_hash")
        == config.get("review_protocol_hash"),
        "proposal input manifest binding mismatch",
    )
    protocol = _json_object(
        root.read_bytes("review_v2/structural_protocol.json"), "structural review protocol"
    )
    _verify_seal(protocol, "protocol_hash", "structural review protocol")
    _require(
        protocol.get("review_mode") == REVIEW_MODE
        and protocol.get("review_protocol_id") == REVIEW_MODE
        and protocol.get("review_protocol_version") == "1"
        and protocol.get("review_protocol_hash") == config.get("review_protocol_hash"),
        "structural review mode is invalid",
    )
    reviews = _jsonl_objects(
        root.read_bytes("review_v2/structural_outcomes.jsonl"), "structural outcomes"
    )
    _require(
        [item.get("proposal_id") for item in reviews]
        == sorted(item.get("proposal_id") for item in reviews),
        "structural review order is not deterministic",
    )
    review_by_proposal: dict[str, Mapping[str, Any]] = {}
    for review in reviews:
        _verify_seal(review, "record_hash", "structural review")
        proposal_id = _require_identifier(
            review.get("proposal_id"), "structural review proposal ID is invalid"
        )
        proposal = proposals_by_id.get(proposal_id)
        _require(
            proposal is not None
            and proposal_id not in review_by_proposal
            and review.get("proposal_record_hash") == proposal["record_hash"]
            and review.get("candidate_id") == proposal["candidate_id"]
            and review.get("candidate_record_hash")
            == candidates_by_id[str(proposal["candidate_id"])]["record_hash"]
            and review.get("review_mode") == REVIEW_MODE
            and review.get("review_protocol_hash") == protocol.get("protocol_hash")
            and review.get("outcome") in _STRUCTURAL_OUTCOMES,
            "review coverage/binding mismatch",
        )
        review_by_proposal[proposal_id] = review
    _require(
        set(review_by_proposal) == set(proposals_by_id),
        "review coverage/binding mismatch",
    )
    seen_group_panels: set[tuple[str, str, str]] = set()
    expected_review_outcomes: dict[str, str] = {}
    accepted_candidates: list[Mapping[str, Any]] = []
    for proposal_id in sorted(proposals_by_id):
        proposal = proposals_by_id[proposal_id]
        candidate = candidates_by_id[str(proposal["candidate_id"])]
        group = candidate["case_group_or_null"] or candidate["candidate_id"]
        panel_key = (
            str(candidate["parent_doi_id"]),
            str(group),
            str(candidate["panel_id"]),
        )
        _require(
            panel_key not in seen_group_panels,
            "DUPLICATE_PANEL_MEMBERSHIP",
        )
        outcome = "ACCEPTED_STRUCTURAL"
        seen_group_panels.add(panel_key)
        accepted_candidates.append(candidate)
        expected_review_outcomes[proposal_id] = outcome
    _require(
        {
            proposal_id: review["outcome"]
            for proposal_id, review in review_by_proposal.items()
        }
        == expected_review_outcomes,
        "structural review outcome recomputation mismatch",
    )
    cases = _jsonl_objects(root.read_bytes("canonical_v2/cases.jsonl"), "canonical cases")
    grouped_candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in accepted_candidates:
        group = candidate["case_group_or_null"] or candidate["candidate_id"]
        grouped_candidates[(str(candidate["parent_doi_id"]), str(group))].append(candidate)
    expected_cases_without_hash: list[dict[str, Any]] = []
    for (doi_id, group), group_candidates in sorted(grouped_candidates.items()):
        ordered_candidates = sorted(
            group_candidates, key=lambda item: str(item["panel_id"])
        )
        expected_cases_without_hash.append(
            {
                "schema_version": "c2_v2_canonical_case_v1",
                "case_id": f"case-{len(expected_cases_without_hash) + 1:06d}",
                "parent_doi_id": doi_id,
                "case_group": group,
                "case_kind": (
                    "single" if len(ordered_candidates) == 1 else "multi"
                ),
                "source_candidate_ids": [
                    item["candidate_id"] for item in ordered_candidates
                ],
                "source_candidate_record_hashes": [
                    item["record_hash"] for item in ordered_candidates
                ],
                "candidate_input_record_hashes": [
                    inputs_by_id[str(item["candidate_input_id"])]["record_hash"]
                    for item in ordered_candidates
                ],
                "consumption_link_hashes": [
                    item["consumption_link_hash"] for item in ordered_candidates
                ],
                "unit_hashes": [item["unit_hash"] for item in ordered_candidates],
                "verified_panel_ids": [
                    item["panel_id"] for item in ordered_candidates
                ],
                "qualified_panel_count": len(ordered_candidates),
                "canonical_builder_rule_hash": _rule_hash(CANONICAL_RULE_ID, "1"),
            }
        )
    _require(
        len(cases) == len(expected_cases_without_hash),
        "canonical case coverage mismatch",
    )
    cases_by_doi: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case, expected_case in zip(cases, expected_cases_without_hash, strict=True):
        _verify_seal(case, "record_hash", "canonical case")
        _require(
            _without(case, "record_hash") == expected_case,
            "canonical case provenance/binding mismatch",
        )
        cases_by_doi[str(case["parent_doi_id"])].append(case)
    canonical_summary = _json_object(
        root.read_bytes("canonical_v2/canonical_summary.json"),
        "canonical summary",
    )
    _verify_seal(canonical_summary, "summary_hash", "canonical summary")
    _require(
        canonical_summary.get("case_count") == len(cases)
        and canonical_summary.get("canonical_case_collection_hash")
        == _sha256_json([item["record_hash"] for item in cases])
        and canonical_summary.get("proposal_collection_hash")
        == _sha256_json([item["record_hash"] for item in proposed])
        and canonical_summary.get("review_collection_hash")
        == _sha256_json([item["record_hash"] for item in reviews])
        and canonical_summary.get("all_cases_retained") is True,
        "canonical summary binding mismatch",
    )
    case_set_manifest_relative_path = "canonical_v2/canonical_case_set_manifest.json"
    case_set_manifest = _json_object(
        root.read_bytes(case_set_manifest_relative_path),
        "canonical case-set manifest",
    )
    _verify_seal(
        case_set_manifest,
        "case_set_manifest_hash",
        "canonical case-set manifest",
    )
    terminal = _jsonl_objects(
        root.read_bytes("control/terminal_outcomes.jsonl"), "terminal outcomes"
    )
    _require(
        len(terminal) == partition_records
        and [item.get("input_index_1based") for item in terminal]
        == list(range(1, partition_records + 1)),
        "terminal outcome count/order mismatch",
    )
    terminal_by_doi: dict[str, Mapping[str, Any]] = {}
    for row in terminal:
        doi_id = _require_doi(row.get("doi"), "terminal outcome DOI is invalid")
        terminal_status_raw = row.get("terminal_status")
        _require(
            isinstance(terminal_status_raw, str)
            and terminal_status_raw in _TERMINAL_STATUS_ADAPTER
            and doi_id not in terminal_by_doi,
            "terminal outcome status/DOI is invalid",
        )
        terminal_by_doi[doi_id] = row
    inventory_by_article: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in inventory:
        inventory_by_article[str(record["raw_article_id"])].append(record)
    source_inventory_bindings: dict[str, dict[str, Any]] = {}
    for article_id, assets in inventory_by_article.items():
        ordered_assets = sorted(assets, key=lambda item: str(item["asset_id"]))
        descriptor_paths = {item["source_descriptor_relative_path"] for item in ordered_assets}
        descriptor_hashes = {item["source_descriptor_sha256"] for item in ordered_assets}
        _require(
            len(descriptor_paths) == len(descriptor_hashes) == 1,
            "source inventory descriptor binding mismatch",
        )
        source_inventory_bindings[article_id] = {
            "source_inventory_collection_hash": source_inventory_hash,
            "source_descriptor_relative_path": next(iter(descriptor_paths)),
            "source_descriptor_sha256": next(iter(descriptor_hashes)),
            "raw_asset_record_hashes": [
                item["asset_record_hash"] for item in ordered_assets
            ],
        }
    applicable_case_set_doi_ids = _applicable_case_set_doi_ids(
        terminal,
        source_inventory_bindings,
    )
    expected_case_set_manifest, _ = _build_canonical_case_set_manifest(
        cases,
        applicable_doi_ids=applicable_case_set_doi_ids,
    )
    _require(
        case_set_manifest == expected_case_set_manifest,
        "canonical case-set manifest binding mismatch",
    )
    case_sets_by_doi = {
        str(item["doi_id"]): item for item in case_set_manifest["case_sets"]
    }
    dispositions = _jsonl_objects(
        root.read_bytes("p_evidence_v2/acquisition_dispositions.jsonl"),
        "acquisition dispositions",
    )
    _require(
        len(dispositions) == partition_records
        and [item.get("input_ordinal") for item in dispositions]
        == list(range(1, partition_records + 1)),
        "acquisition disposition coverage mismatch",
    )
    classifications = _jsonl_objects(
        root.read_bytes("p_evidence_v2/source_classifications.jsonl"),
        "source classifications",
    )
    classifications_by_doi: dict[str, Mapping[str, Any]] = {}
    for item in classifications:
        _verify_seal(item, "record_hash", "source classification")
        doi_id = _require_doi(
            item.get("doi_id"), "source classification DOI is invalid"
        )
        _require(
            item.get("parent_doi_id") == doi_id
            and item.get("cluster_id") == doi_id
            and doi_id not in classifications_by_doi,
            "source classification DOI/cluster binding mismatch",
        )
        counts = [int(value) for value in item.get("qualified_panel_counts", [])]
        expected = {_stratum_for_panel_count(count) for count in counts}
        _require(
            len(expected) == 1
            and item.get("derived_public_stratum") in expected
            and item.get("derived_code_label")
            == _code_label_for_stratum(item["derived_public_stratum"]),
            "P stratum recomputation mismatch",
        )
        classifications_by_doi[doi_id] = item
    expected_classification_dois = {
        doi_id
        for doi_id in cases_by_doi
        if _TERMINAL_STATUS_ADAPTER[terminal_by_doi[doi_id]["terminal_status"]]
        == "DOWNLOADED"
    }
    _require(
        set(classifications_by_doi) == expected_classification_dois,
        "source classification canonical coverage mismatch",
    )
    for doi_id in sorted(expected_classification_dois):
        doi_cases = cases_by_doi[doi_id]
        expected_cases = sorted(doi_cases, key=lambda item: str(item["case_id"]))
        classification = classifications_by_doi[doi_id]
        case_set = case_sets_by_doi.get(doi_id)
        _require(case_set is not None, "source classification case-set is missing")
        expected_panel_descriptors = _panel_descriptors_for_case_set(case_set)
        _require(
            classification.get("canonical_case_ids")
            == [item["case_id"] for item in expected_cases]
            and classification.get("canonical_case_record_hashes")
            == [item["record_hash"] for item in expected_cases]
            and classification.get("qualified_panel_counts")
            == [item["qualified_panel_count"] for item in expected_cases]
            and classification.get("source_inventory_collection_hash")
            == source_inventory_hash
            and classification.get("candidate_collection_hash")
            == _sha256_json([item["record_hash"] for item in candidates])
            and classification.get("proposal_collection_hash")
            == _sha256_json([item["record_hash"] for item in proposed])
            and classification.get("review_collection_hash")
            == _sha256_json([item["record_hash"] for item in reviews])
            and classification.get("canonical_collection_hash")
            == canonical_summary["canonical_case_collection_hash"]
            and classification.get("consumption_bijection_hash")
            == bijection["consumption_bijection_hash"]
            and classification.get("canonical_case_set_hash")
            == case_set["canonical_case_set_hash"]
            and classification.get("verified_panel_descriptors_sha256")
            == _sha256_json(expected_panel_descriptors)
            and classification.get("panel_rule_hash") == _rule_hash(P_RULE_ID, "1"),
            "source classification lineage mismatch",
        )
    for row, disposition in zip(terminal, dispositions, strict=True):
        _verify_seal(disposition, "record_hash", "acquisition disposition")
        doi_id = _require_doi(row.get("doi"), "terminal outcome DOI is invalid")
        terminal_status_raw = row["terminal_status"]
        terminal_status = _TERMINAL_STATUS_ADAPTER[terminal_status_raw]
        article_id = str(row.get("article_id"))
        source_inventory_binding = source_inventory_bindings.get(article_id)
        doi_cases = sorted(
            cases_by_doi.get(doi_id, ()), key=lambda item: str(item["case_id"])
        )
        if terminal_status == "DOWNLOADED" and doi_id in classifications_by_doi:
            final_disposition = "STRATIFIED_SOURCE_CANONICAL"
            reason = "VERIFIED_SOURCE_CANONICAL_ALL_CASES"
        elif terminal_status == "DOWNLOADED" and source_inventory_binding is not None:
            final_disposition = "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
            reason = "DOWNLOADED_SOURCE_NO_QUALIFYING_CASE"
        elif terminal_status == "DOWNLOADED":
            final_disposition = "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
            reason = "DOWNLOADED_WITHOUT_VERIFIED_SOURCE"
        else:
            final_disposition = _FINAL_DISPOSITION_BY_STATUS[terminal_status]
            reason = "TERMINAL_STATUS_PRECLUDES_P_CLASSIFICATION"
        terminal_evidence_binding = {
            "terminal_outcomes_relative_path": "control/terminal_outcomes.jsonl",
            "terminal_outcomes_sha256": root.sha256("control/terminal_outcomes.jsonl"),
            "terminal_row_sha256": _sha256_json(row),
            "source_chunk_sha256": source_chunk_sha256,
            "attempt_count": len(row.get("rounds", {})),
            "terminal": True,
        }
        canonical_builder_binding = (
            {
                "canonical_builder_rule_id": CANONICAL_RULE_ID,
                "canonical_builder_rule_version": "1",
                "canonical_builder_rule_hash": _rule_hash(CANONICAL_RULE_ID, "1"),
                "candidate_collection_hash": _sha256_json(
                    [item["record_hash"] for item in candidates]
                ),
                "proposal_collection_hash": _sha256_json(
                    [item["record_hash"] for item in proposed]
                ),
                "review_collection_hash": _sha256_json(
                    [item["record_hash"] for item in reviews]
                ),
                "canonical_collection_hash": canonical_summary[
                    "canonical_case_collection_hash"
                ],
                "canonical_case_set_manifest_sha256": root.sha256(
                    case_set_manifest_relative_path
                ),
                "consumption_bijection_hash": bijection["consumption_bijection_hash"],
            }
            if terminal_status == "DOWNLOADED"
            and source_inventory_binding is not None
            else None
        )
        _require(
            disposition.get("doi_id") == doi_id
            and disposition.get("parent_doi_id") == doi_id
            and disposition.get("input_ordinal") == row.get("input_index_1based")
            and disposition.get("terminal_status_raw") == terminal_status_raw
            and disposition.get("terminal_status") == terminal_status
            and disposition.get("terminal_evidence_binding") == terminal_evidence_binding
            and disposition.get("final_disposition") == final_disposition
            and disposition.get("source_inventory_binding_or_null")
            == source_inventory_binding
            and disposition.get("canonical_builder_binding_or_null")
            == canonical_builder_binding
            and disposition.get("all_eligible_case_ids")
            == (
                [item["case_id"] for item in doi_cases]
                if terminal_status == "DOWNLOADED"
                else []
            )
            and disposition.get("classification_reason") == reason,
            "acquisition disposition terminal binding mismatch",
        )
        expected_classification_hash = (
            classifications_by_doi[doi_id]["record_hash"]
            if doi_id in classifications_by_doi
            else None
        )
        _require(
            disposition.get("source_classification_record_hash_or_null")
            == expected_classification_hash,
            "stratified disposition classification binding mismatch",
        )
    summary = _json_object(root.read_bytes("p_evidence_v2/p_summary.json"), "P summary")
    _verify_seal(summary, "summary_hash", "P summary")
    _require(
        summary.get("input_total") == partition_records
        and summary.get("source_classification_count") == len(classifications)
        and summary.get("source_only_disposition_count")
        == sum(
            item["final_disposition"] != "STRATIFIED_SOURCE_CANONICAL"
            for item in dispositions
        )
        and summary.get("classification_collection_hash")
        == _sha256_json([item["record_hash"] for item in classifications])
        and summary.get("disposition_collection_hash")
        == _sha256_json([item["record_hash"] for item in dispositions])
        and summary.get("canonical_case_set_manifest_sha256")
        == root.sha256(case_set_manifest_relative_path)
        and summary.get("acquisition_disposition_doi_ids_sha256")
        == _sha256_json([item["doi_id"] for item in dispositions])
        and summary.get("stratified_source_doi_ids_sha256")
        == _sha256_json([item["doi_id"] for item in classifications])
        and summary.get("non_stratified_doi_ids_sha256")
        == _sha256_json(
            [
                item["doi_id"]
                for item in dispositions
                if item["final_disposition"] != "STRATIFIED_SOURCE_CANONICAL"
            ]
        )
        and summary.get("per_doi_case_set_hashes_sha256")
        == _sha256_json(
            [
                {
                    "doi_id": item["doi_id"],
                    "canonical_case_set_hash": item["canonical_case_set_hash"],
                }
                for item in case_set_manifest["case_sets"]
            ]
        )
        and summary.get("terminal_status_counts")
        == {
            status: sum(item["terminal_status"] == status for item in dispositions)
            for status in sorted(set(_TERMINAL_STATUS_ADAPTER.values()))
        }
        and summary.get("disposition_counts")
        == _counts_by_value(item["final_disposition"] for item in dispositions)
        and summary.get("stratum_source_doi_counts")
        == {
            stratum: sum(
                item["derived_public_stratum"] == stratum
                for item in classifications
            )
            for stratum in ("P1", "P2", "P3_4", "P5PLUS")
        }
        and summary.get("stratum_independent_cluster_counts")
        == {
            stratum: sum(
                item["derived_public_stratum"] == stratum
                and item["cluster_id"] == item["doi_id"]
                for item in classifications
            )
            for stratum in ("P1", "P2", "P3_4", "P5PLUS")
        }
        and summary.get("all_case_equal_weighting_rule_hash")
        == _rule_hash("DOI_CASE_AGGREGATION_V1", "1"),
        "P summary input/lineage mismatch",
    )
    value: dict[str, Any] = {
        "schema_version": "c2_v2_source_bearing_extension_validation_v1",
        "status": "PASS",
        "source_chunk_sha256": source_chunk_sha256,
        "input_total": partition_records,
        "raw_asset_count": len(inventory),
        "detected_format_count": len(detected),
        "container_count": len(accounts),
        "derived_member_count": len(derived),
        "source_unit_count": len(units),
        "consumption_count": len(consumption),
        "candidate_input_count": len(inputs),
        "candidate_count": len(candidates),
        "proposal_count": len(proposed) + len(rejected),
        "review_count": len(reviews),
        "canonical_case_count": len(cases),
        "source_classification_count": len(classifications),
        "acquisition_disposition_count": len(dispositions),
        "format_config_hash": config["config_hash"],
        "consumption_bijection_hash": bijection["consumption_bijection_hash"],
        "canonical_case_set_manifest_hash": case_set_manifest["case_set_manifest_hash"],
        "approved_implementation_commit_full": (
            attestation.approved_implementation_commit_full
        ),
        "attestation_commit_full": attestation.attestation_commit_full,
        "code_attestation_manifest_sha256": attestation.manifest_sha256,
        "attested_code_blobs_sha256": attestation.code_blob_set_sha256,
    }
    return _seal(value, "validation_hash")


def build_source_bearing_extension(
    *,
    root: _TargetRoot,
    raw_reader: Any,
    records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
    partition_records: int,
    source_chunk_sha256: str,
    code_attestation: SourceExtensionCodeAttestation | None = None,
) -> SourceBearingExtensionResult:
    """Build every source-bearing artifact in an already-open private staging root.

    ``raw_reader`` is intentionally duck typed: only the single-open snapshot
    ``reads`` made by the generic finalizer is accepted.  Reopening raw paths
    would violate the raw-attempt single-read contract.
    """

    attestation = (
        verify_source_extension_code_attestation()
        if code_attestation is None
        else code_attestation
    )
    attestation.verify_runtime()
    _require(len(records) == partition_records, "source extension partition count mismatch")
    assets: list[_RawAsset] = []
    source_by_article: dict[str, Mapping[str, Any]] = {}
    by_article = {
        str(record["article_url"]).rstrip("/").rsplit("/", 1)[-1]: (ordinal, record)
        for ordinal, record in enumerate(records, start=1)
    }
    for article_id, entry in provenance.items():
        evidence = entry.get("source_evidence")
        if evidence is None:
            continue
        descriptor = getattr(evidence, "descriptor", None)
        _require(
            isinstance(descriptor, Mapping)
            and descriptor.get("schema_version") == "c2-source-evidence-v2",
            "NOT_SEALABLE_SOURCE_CLASSIFICATION_BUILDER_REQUIRED: "
            "source-bearing records require typed c2-source-evidence-v2 descriptors",
        )
        ordinal, record = by_article[article_id]
        doi_id = str(record["doi"])
        validated = validate_source_evidence_descriptor_v2(
            descriptor,
            article_id=article_id,
            doi_id=doi_id,
            provenance_relative_path=str(entry["relative_path"]),
        )
        source_by_article[article_id] = entry
        for asset in validated:
            relative = str(asset["relative_path"])
            snapshot = raw_reader.reads.get(relative)
            _require(snapshot is not None, "source asset was not FD-opened by raw validator")
            _require(
                snapshot.sha256 == asset["sha256"]
                and len(snapshot.payload) == asset["bytes"],
                "source asset snapshot does not match descriptor",
            )
            asset_record: dict[str, Any] = {
                "schema_version": "c2_v2_raw_source_asset_v1",
                "parent_doi_id": doi_id,
                "frozen_input_ordinal": ordinal,
                "raw_article_id": article_id,
                "asset_id": asset["asset_id"],
                "declared_asset_kind": asset["declared_asset_kind"],
                "declared_format_tuple": asset["declared_format_tuple"],
                "verified_relative_path": relative,
                "verified_file_sha256": snapshot.sha256,
                "verified_bytes": len(snapshot.payload),
                "source_descriptor_relative_path": evidence.descriptor_path,
                "source_descriptor_sha256": evidence.descriptor_sha256,
                "provenance_relative_path": entry["relative_path"],
            }
            _seal(asset_record, "asset_record_hash")
            assets.append(
                _RawAsset(
                    article_id=article_id,
                    doi_id=doi_id,
                    input_ordinal=ordinal,
                    asset=asset,
                    asset_record=asset_record,
                    payload=snapshot.payload,
                )
            )
    _require(assets, "source-bearing extension has no typed raw assets")
    return _Builder(
        root=root,
        raw_assets=assets,
        terminal_rows=terminal_rows,
        source_by_article=source_by_article,
        partition_records=partition_records,
        source_chunk_sha256=source_chunk_sha256,
        code_attestation=attestation,
    ).build()

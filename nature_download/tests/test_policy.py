from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path

import pytest

from nature_download.corpus.policy import (
    evaluate_crossref_item,
    evaluate_record,
    is_allowed_journal,
    normalize_cc_by_url,
    normalize_doi,
)


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_items() -> dict[str, dict]:
    payload = json.loads(
        (FIXTURES / "crossref_licenses.json").read_text(encoding="utf-8")
    )
    return {item["fixture_id"]: item for item in payload["items"]}


@pytest.mark.parametrize(
    "value",
    [
        "10.1038/example",
        "doi:10.1038/EXAMPLE",
        "https://doi.org/10.1038/example?utm_source=test#fragment",
        "https%3A%2F%2Fdoi.org%2F10.1038%2Fexample%3Futm%3D1",
    ],
)
def test_doi_aliases_have_one_canonical_identity(value: str) -> None:
    assert normalize_doi(value) == "10.1038/example"


@pytest.mark.parametrize(
    ("fixture_id", "accepted"),
    [
        ("nature_comm_cc_by_4", True),
        ("scientific_reports_cc_by_3", True),
        ("npj_cc_by_4", True),
        ("cc_by_nc", False),
        ("cc_by_sa", False),
        ("cc_by_nd", False),
        ("oa_only", False),
        ("missing_license", False),
        ("nature_main", False),
        ("unknown_journal", False),
    ],
)
def test_crossref_license_and_journal_gate(fixture_id: str, accepted: bool) -> None:
    decision = evaluate_crossref_item(
        fixture_items()[fixture_id],
        today=date(2026, 7, 12),
    )
    assert decision["download_eligible"] is accepted


def test_oa_signal_is_not_a_redistribution_license() -> None:
    decision = evaluate_crossref_item(fixture_items()["oa_only"])
    assert decision["open_access_signal"] is True
    assert decision["download_eligible"] is False
    assert "license-missing" in decision["reject_reasons"]


def test_crossref_license_records_version_source_evidence_and_effective_date() -> None:
    decision = evaluate_crossref_item(
        fixture_items()["nature_comm_cc_by_4"],
        today=date(2026, 7, 12),
    )
    assert decision["license"]["normalized_url"] == (
        "https://creativecommons.org/licenses/by/4.0/"
    )
    assert decision["license"]["version"] == "4.0"
    assert decision["license"]["effective_date"] == "2024-01-02"
    assert decision["license"]["content_version"] == "vor"
    assert decision["license"]["source"] == "crossref"
    assert decision["license"]["evidence"]["content-version"] == "vor"


def test_tdm_cc_by_does_not_override_restrictive_vor_license() -> None:
    item = {
        "DOI": "10.1038/s41467-024-00012-2",
        "container-title": ["Nature Communications"],
        "issued": {"date-parts": [[2024]]},
        "license": [
            {
                "URL": "https://creativecommons.org/licenses/by/4.0/",
                "content-version": "tdm",
            },
            {
                "URL": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                "content-version": "vor",
            },
        ],
    }
    decision = evaluate_crossref_item(item)
    assert decision["download_eligible"] is False
    assert "license-tdm-only" in decision["reject_reasons"]
    assert "license-disallowed-variant:by-nc-nd" in decision["reject_reasons"]


@pytest.mark.parametrize(
    "content_version",
    ["am", None, "VOR", "vor ", " vor", 1],
)
def test_crossref_cc_by_requires_exact_vor_content_version(
    content_version: object,
) -> None:
    license_record = {
        "URL": "https://creativecommons.org/licenses/by/4.0/",
    }
    if content_version is not None:
        license_record["content-version"] = content_version
    item = {
        "DOI": "10.1038/s41467-024-00013-3",
        "container-title": ["Nature Communications"],
        "issued": {"date-parts": [[2024]]},
        "license": [license_record],
    }

    decision = evaluate_crossref_item(item)

    assert decision["download_eligible"] is False
    assert any(
        reason.startswith("license-non-vor-content-version:")
        for reason in decision["reject_reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "normalized_url",
            "https://creativecommons.org/licenses/by/3.0/",
        ),
        ("license_id", "CC-BY-3.0"),
        ("version", "3.0"),
        ("content_version", "VOR"),
        ("source", "article_metadata"),
    ],
)
def test_revalidation_rejects_normalized_fields_that_contradict_raw_crossref(
    field: str,
    value: str,
) -> None:
    original = evaluate_crossref_item(
        fixture_items()["nature_comm_cc_by_4"],
        today=date(2026, 7, 12),
    )
    record = deepcopy(original)
    record["license"][field] = value

    decision = evaluate_record(record)

    assert decision["download_eligible"] is False
    assert "license-provenance-contradiction" in decision["reject_reasons"]
    assert (
        f"license-provenance-contradiction:{field}"
        in decision["reject_reasons"]
    )


def test_license_list_rejects_contradictory_top_level_source() -> None:
    decision = evaluate_record(
        {
            "doi": "10.1038/s41467-024-88888-8",
            "journal": "Nature Communications",
            "year": 2024,
            "license": [
                {
                    "URL": "https://creativecommons.org/licenses/by/4.0/",
                    "content-version": "vor",
                }
            ],
            "license_source": "article_metadata",
        }
    )

    assert decision["download_eligible"] is False
    assert (
        "license-provenance-contradiction:source"
        in decision["reject_reasons"]
    )


def test_revalidation_rejects_raw_evidence_that_disagrees_with_candidates() -> None:
    original = evaluate_crossref_item(
        fixture_items()["nature_comm_cc_by_4"],
        today=date(2026, 7, 12),
    )
    record = deepcopy(original)
    record["license"]["evidence"] = {
        **record["license"]["evidence"],
        "content-version": "tdm",
    }

    decision = evaluate_record(record)

    assert decision["download_eligible"] is False
    assert (
        "license-provenance-contradiction:evidence"
        in decision["reject_reasons"]
    )


def test_revalidation_uses_crossref_evidence_not_normalized_claims() -> None:
    decision = evaluate_record(
        {
            "doi": "10.1038/s41467-024-99998-8",
            "journal": "Nature Communications",
            "year": 2024,
            "license": {
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "normalized_url": (
                    "https://creativecommons.org/licenses/by/4.0/"
                ),
                "license_id": "CC-BY-4.0",
                "version": "4.0",
                "content_version": "vor",
                "source": "crossref",
                "evidence": {
                    "URL": "https://creativecommons.org/licenses/by/3.0/",
                    "content-version": "vor",
                },
            },
        }
    )

    assert decision["download_eligible"] is False
    assert "license-provenance-contradiction" in decision["reject_reasons"]


def test_nature_main_is_explicitly_rejected() -> None:
    decision = evaluate_crossref_item(fixture_items()["nature_main"])
    assert decision["download_eligible"] is False
    assert "journal-nature-main-excluded" in decision["reject_reasons"]


def test_npj_series_is_allowed_but_bare_npj_is_not() -> None:
    assert is_allowed_journal("npj Quantum Information")
    assert is_allowed_journal("NPJ   Biofilms and Microbiomes")
    assert not is_allowed_journal("npj")
    assert not is_allowed_journal("Nature")


def test_article_metadata_can_supply_exact_cc_by_evidence() -> None:
    html = (FIXTURES / "article_cc_by.html").read_text(encoding="utf-8")
    item = {
        "DOI": "10.1038/s41467-024-00010-0",
        "container-title": ["Nature Communications"],
        "issued": {"date-parts": [[2024]]},
        "license": [],
    }
    decision = evaluate_crossref_item(item, article_html=html)
    assert decision["download_eligible"] is True
    assert decision["license"]["source"] == "article_metadata"
    assert "citation_license" in decision["license"]["evidence"]
    revalidated = evaluate_record(decision)
    assert revalidated["download_eligible"] is True
    assert revalidated["license"]["source"] == "article_metadata"
    assert revalidated["license"]["evidence"] == decision["license"]["evidence"]


def test_article_metadata_evidence_must_bind_to_normalized_license() -> None:
    html = (FIXTURES / "article_cc_by.html").read_text(encoding="utf-8")
    record = evaluate_crossref_item(
        {
            "DOI": "10.1038/s41467-024-00010-0",
            "container-title": ["Nature Communications"],
            "issued": {"date-parts": [[2024]]},
            "license": [],
        },
        article_html=html,
    )
    record["license"]["evidence"] = (
        'meta[citation_license] content="'
        'https://creativecommons.org/licenses/by/3.0/"'
    )

    decision = evaluate_record(record)

    assert decision["download_eligible"] is False
    assert "license-provenance-contradiction" in decision["reject_reasons"]


def test_article_oa_text_without_license_is_rejected() -> None:
    html = (FIXTURES / "article_oa_only.html").read_text(encoding="utf-8")
    item = {
        "DOI": "10.1038/s41467-024-00011-1",
        "container-title": ["Nature Communications"],
        "issued": {"date-parts": [[2024]]},
        "is_open_access": True,
    }
    decision = evaluate_crossref_item(item, article_html=html)
    assert decision["download_eligible"] is False
    assert decision["reject_reasons"] == ["license-missing"]


def test_normalized_url_without_source_evidence_is_not_trusted() -> None:
    decision = evaluate_record(
        {
            "doi": "10.1038/s41467-024-99999-9",
            "journal": "Nature Communications",
            "year": 2024,
            "license": {
                "normalized_url": "https://creativecommons.org/licenses/by/4.0/"
            },
        }
    )
    assert decision["download_eligible"] is False
    assert "license-provenance-missing" in decision["reject_reasons"]


@pytest.mark.parametrize(
    "url",
    [
        "http://creativecommons.org/licenses/by/4.0/",
        "https://www.creativecommons.org/licenses/by/4.0/legalcode",
        "https://creativecommons.org/licenses/by/4.0/deed.en",
    ],
)
def test_cc_by_urls_are_canonicalized(url: str) -> None:
    assert normalize_cc_by_url(url) == (
        "https://creativecommons.org/licenses/by/4.0/",
        "4.0",
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/licenses/by-nd/4.0/",
        "https://example.org/licenses/by/4.0/",
    ],
)
def test_non_cc_by_urls_are_not_normalized(url: str) -> None:
    assert normalize_cc_by_url(url) is None

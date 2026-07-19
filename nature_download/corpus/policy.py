"""Machine-enforced journal and redistribution-license policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any, Iterable
from urllib.parse import unquote, urlparse, urlsplit


SCHEMA_VERSION = "1.0"
# CC-BY per-figure source-data publishers. Nature Communications, eLife, Life
# Science Alliance, and the admitted EMBO Press journals expose downloadable
# figure-local source data at meaningful rates.
# The Nature-portfolio "Communications X" siblings were re-verified with the
# harvester's OWN detector (find_source_data_links: <a> label matches
# "Source Data") and expose ~0% per-figure Source Data (their xlsx are labelled
# "Supplementary Data N", not per-figure Source Data). Scientific Reports and
# the npj titles likewise expose ~0%. Measured (find_source_data_links + xlsx,
# 25 CC-BY articles/journal, 2020-2023):
#   Nature Communications      per-fig SD 28%, joint(>=5-panel) 4.0%  [ADMIT]
#   Communications Biology     per-fig SD  0%   (xlsx = "Supplementary Data")
#   Communications Medicine    per-fig SD  0%
#   Communications Chemistry   per-fig SD  0%
#   Communications Earth&Env   per-fig SD  0%
#   Communications Physics     per-fig SD  0%
# (An earlier broad `MOESM*.xlsx` regex over-counted supplementary xlsx and
# falsely suggested the siblings qualified; corrected here.) eLife is the
# non-Springer per-figure source-data publisher. Life Science Alliance's
# Silverchair HTML exposes source XLSX/CSV inside the owning figure container
# (rather than merely as whole-article supplements). EMBO Press journals hosted
# on SpringerLink are also admitted because they expose CC-BY articles with
# per-figure Source Data. All gates (CC-BY + per-figure source data + >=5
# panels) still validate every item.
ALLOWED_JOURNALS = frozenset(
    {
        "nature communications",
        "scientific reports",
        "elife",
        "life science alliance",
        "embo journal",
        "the embo journal",
        "molecular systems biology",
        "embo reports",
        "embo molecular medicine",
    }
)
ALLOWED_CC_BY_VERSIONS = frozenset({"3.0", "4.0"})
LICENSE_META_NAMES = frozenset(
    {
        "citation_license",
        "dc.rights",
        "dc.rights.license",
        "dcterms.license",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = unquote(str(value)).strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"} or (
            parsed.hostname or ""
        ).casefold() not in {"doi.org", "dx.doi.org"}:
            return None
        text = parsed.path.lstrip("/")
    else:
        text = re.split(r"[?#]", text, maxsplit=1)[0]
    doi = text.strip().casefold()
    if (
        not doi.startswith("10.")
        or "/" not in doi
        or any(character.isspace() for character in doi)
    ):
        return None
    return doi


def normalize_journal(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_allowed_journal(value: Any) -> bool:
    journal = normalize_journal(value).casefold()
    if journal in ALLOWED_JOURNALS:
        return True
    return bool(re.fullmatch(r"npj\s+\S(?:.*\S)?", journal))


def journal_rejection_reason(value: Any) -> str | None:
    journal = normalize_journal(value)
    if not journal:
        return "journal-missing"
    if is_allowed_journal(journal):
        return None
    if journal.casefold() == "nature":
        return "journal-nature-main-excluded"
    return "journal-not-allowed"


def normalize_cc_by_url(value: Any) -> tuple[str, str] | None:
    """Return the canonical URL and version for exact CC BY 3.0/4.0 URLs."""
    if not value:
        return None
    raw = str(value).strip()
    if not re.match(r"^https?://", raw, flags=re.I):
        return None
    parsed = urlparse(raw)
    host = (parsed.hostname or "").casefold()
    if host.startswith("www."):
        host = host[4:]
    if host != "creativecommons.org":
        return None
    path = re.sub(r"/+", "/", parsed.path.casefold())
    match = re.fullmatch(
        r"/licenses/by/(3\.0|4\.0)(?:/(?:legalcode|deed(?:\.[a-z-]+)?))?/?",
        path,
    )
    if not match:
        return None
    version = match.group(1)
    if version not in ALLOWED_CC_BY_VERSIONS:
        return None
    return f"https://creativecommons.org/licenses/by/{version}/", version


def _license_variant(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = urlparse(str(value).strip())
    except Exception:
        return None
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if host != "creativecommons.org":
        return None
    match = re.search(r"/licenses/([^/]+)/([^/]+)", parsed.path.casefold())
    return match.group(1) if match else None


def _date_from_parts(value: Any) -> str | None:
    if not value:
        return None
    parts = value.get("date-parts") if isinstance(value, dict) else value
    if not isinstance(parts, list) or not parts:
        return None
    if parts and isinstance(parts[0], list):
        parts = parts[0]
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day).isoformat()
    except (TypeError, ValueError, IndexError):
        return None


def _first_date(item: dict[str, Any]) -> str | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = item.get(key)
        parsed = _date_from_parts(value)
        if parsed:
            return parsed
    return None


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


class _ArticleMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.license_candidates: list[dict[str, str]] = []
        self.journal: str | None = None
        self.doi: str | None = None
        self.article_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {str(key).casefold(): value for key, value in attrs if value}
        if tag.casefold() == "meta":
            name = (values.get("name") or values.get("property") or "").casefold()
            content = values.get("content")
            if name in LICENSE_META_NAMES and content:
                self.license_candidates.append(
                    {
                        "url": content,
                        "evidence": f'meta[{name}] content="{content}"',
                    }
                )
            elif name in {"citation_journal_title", "prism.publicationname"} and content:
                self.journal = content
            elif name in {"citation_doi", "dc.identifier"} and content:
                self.doi = content
            elif name in {"citation_public_url", "og:url"} and content:
                self.article_url = content
        elif tag.casefold() == "link":
            rel = (values.get("rel") or "").casefold().split()
            href = values.get("href")
            if "license" in rel and href:
                self.license_candidates.append(
                    {
                        "url": href,
                        "evidence": f'link[rel=license] href="{href}"',
                    }
                )


def parse_article_metadata(html: str | None) -> dict[str, Any]:
    parser = _ArticleMetadataParser()
    if html:
        parser.feed(html)
    return {
        "license_candidates": parser.license_candidates,
        "journal": parser.journal,
        "doi": normalize_doi(parser.doi),
        "article_url": parser.article_url,
    }


@dataclass(frozen=True)
class LicenseEvidence:
    url: str
    normalized_url: str
    license_id: str
    version: str
    effective_date: str | None
    content_version: str | None
    source: str
    evidence: Any


def _crossref_candidates(
    item: dict[str, Any],
) -> Iterable[tuple[Any, str | None, str, Any, Any]]:
    licenses = item.get("license") or []
    if isinstance(licenses, dict):
        licenses = [licenses]
    for entry in licenses:
        if not isinstance(entry, dict):
            continue
        url = entry.get("URL") or entry.get("url")
        content_version = entry.get("content-version")
        yield (
            url,
            _date_from_parts(entry.get("start")),
            "crossref",
            entry,
            content_version,
        )


def _metadata_candidates(
    metadata: dict[str, Any],
) -> Iterable[tuple[Any, str | None, str, Any, str | None]]:
    for entry in metadata.get("license_candidates") or []:
        if not isinstance(entry, dict):
            continue
        yield (
            entry.get("url"),
            None,
            "article_metadata",
            entry.get("evidence") or entry,
            "vor",
        )


def _select_license(
    item: dict[str, Any],
    article_metadata: dict[str, Any],
    *,
    today: date,
) -> tuple[LicenseEvidence | None, list[str]]:
    candidates = list(_crossref_candidates(item)) + list(_metadata_candidates(article_metadata))
    if not candidates:
        return None, ["license-missing"]

    variants: set[str] = set()
    future_cc_by = False
    tdm_cc_by = False
    non_vor_versions: set[str] = set()
    for raw_url, effective_date, source, evidence, content_version in candidates:
        if source == "crossref":
            if content_version == "vor":
                pass
            elif str(content_version or "").strip().casefold() == "tdm":
                if normalize_cc_by_url(raw_url):
                    tdm_cc_by = True
                continue
            else:
                if normalize_cc_by_url(raw_url):
                    non_vor_versions.add(
                        str(content_version).strip()
                        if content_version is not None
                        and str(content_version).strip()
                        else "missing"
                    )
                else:
                    variant = _license_variant(raw_url)
                    if variant:
                        variants.add(variant)
                continue
        normalized = normalize_cc_by_url(raw_url)
        if not normalized:
            variant = _license_variant(raw_url)
            if variant:
                variants.add(variant)
            continue
        canonical_url, version = normalized
        if effective_date:
            try:
                if date.fromisoformat(effective_date) > today:
                    future_cc_by = True
                    continue
            except ValueError:
                effective_date = None
        return (
            LicenseEvidence(
                url=str(raw_url),
                normalized_url=canonical_url,
                license_id=f"CC-BY-{version}",
                version=version,
                effective_date=effective_date,
                content_version=content_version,
                source=source,
                evidence=evidence,
            ),
            [],
        )

    reasons: list[str] = []
    if future_cc_by:
        reasons.append("license-not-yet-effective")
    if tdm_cc_by:
        reasons.append("license-tdm-only")
    reasons.extend(
        f"license-non-vor-content-version:{version}"
        for version in sorted(non_vor_versions)
    )
    if variants:
        reasons.extend(f"license-disallowed-variant:{variant}" for variant in sorted(variants))
    if not reasons:
        reasons.append("license-not-verifiable-cc-by")
    return None, reasons


def evaluate_crossref_item(
    item: dict[str, Any],
    *,
    article_html: str | None = None,
    retrieved_at: str | None = None,
    require_cc_by: bool = True,
    today: date | None = None,
) -> dict[str, Any]:
    """Evaluate a Crossref work without treating OA signals as a license."""
    retrieved_at = retrieved_at or utc_now()
    today = today or datetime.now(timezone.utc).date()
    metadata = parse_article_metadata(article_html)
    journal = normalize_journal(
        _first_text(item.get("container-title"))
        or _first_text(item.get("short-container-title"))
        or metadata.get("journal")
    )
    journal_reason = journal_rejection_reason(journal)
    license_evidence, license_reasons = _select_license(item, metadata, today=today)
    reasons = ([journal_reason] if journal_reason else []) + license_reasons

    doi = normalize_doi(item.get("DOI") or item.get("doi") or metadata.get("doi"))
    if not doi:
        reasons.append("doi-missing")
    policy_accepted = not reasons
    download_eligible = policy_accepted and require_cc_by
    queue_reasons = list(reasons)
    if policy_accepted and not require_cc_by:
        queue_reasons.append("cc-by-gate-not-enabled")

    published_date = _first_date(item)
    year = int(published_date[:4]) if published_date else item.get("year")
    raw_article_url = (
        metadata.get("article_url")
        or item.get("URL")
        or item.get("url")
    )
    if doi and doi.startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        article_url = f"https://www.nature.com/articles/{suffix}"
    else:
        article_url = raw_article_url or (
            f"https://doi.org/{doi}" if doi else None
        )
    evidence_dict = asdict(license_evidence) if license_evidence else {
        "url": None,
        "normalized_url": None,
        "license_id": None,
        "version": None,
        "effective_date": None,
        "content_version": None,
        "source": None,
        "evidence": None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "doi": doi,
        "title": _first_text(item.get("title")),
        "journal": journal,
        "year": year,
        "published_date": published_date,
        "article_url": article_url,
        "license": evidence_dict,
        "license_candidates": item.get("license") or [],
        "retrieved_at": retrieved_at,
        "journal_allowed": journal_reason is None,
        "policy_accepted": policy_accepted,
        "require_cc_by": require_cc_by,
        "download_eligible": download_eligible,
        "reject_reasons": queue_reasons,
        "open_access_signal": bool(
            item.get("is_open_access")
            or item.get("isOpenAccess")
            or item.get("is-oa")
        ),
    }


def _article_metadata_evidence_url(evidence: Any) -> str | None:
    if not isinstance(evidence, str):
        return None
    match = re.fullmatch(
        r'(?:meta\[[^\]\r\n]+\] content|link\[rel=license\] href)="([^"\r\n]+)"',
        evidence.strip(),
        flags=re.I,
    )
    return match.group(1) if match else None


def _reject_license_decision(
    decision: dict[str, Any],
    reasons: Iterable[str],
) -> None:
    added = [reason for reason in reasons if reason]
    if not added:
        return
    decision["policy_accepted"] = False
    decision["download_eligible"] = False
    decision["reject_reasons"] = sorted(
        set(list(decision.get("reject_reasons") or []) + added)
    )


def _license_contradictions(
    original: dict[str, Any],
    selected: dict[str, Any],
    *,
    authoritative_source: str | None,
    stated_source: Any,
    stated_evidence: Any,
) -> list[str]:
    fields: set[str] = set()
    if (
        authoritative_source
        and stated_source is not None
        and stated_source != authoritative_source
    ):
        fields.add("source")

    for key in (
        "url",
        "normalized_url",
        "license_id",
        "version",
        "effective_date",
        "content_version",
    ):
        if (
            key in original
            and original.get(key) is not None
            and original.get(key) != selected.get(key)
        ):
            fields.add(key)

    if stated_evidence is not None:
        if authoritative_source == "crossref":
            if not isinstance(stated_evidence, dict):
                fields.add("evidence")
            else:
                if stated_evidence != selected.get("evidence"):
                    fields.add("evidence")
                evidence_url = stated_evidence.get("URL") or stated_evidence.get(
                    "url"
                )
                if (
                    stated_evidence.get("content-version") != "vor"
                    or normalize_cc_by_url(evidence_url)
                    != normalize_cc_by_url(selected.get("normalized_url"))
                ):
                    fields.add("evidence")
        elif authoritative_source == "article_metadata":
            evidence_url = _article_metadata_evidence_url(stated_evidence)
            if (
                not evidence_url
                or normalize_cc_by_url(evidence_url)
                != normalize_cc_by_url(selected.get("normalized_url"))
            ):
                fields.add("evidence")

    if not fields:
        return []
    return ["license-provenance-contradiction"] + [
        f"license-provenance-contradiction:{field}"
        for field in sorted(fields)
    ]


def evaluate_record(
    record: dict[str, Any],
    *,
    require_cc_by: bool = True,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Revalidate a discovery/provenance record without trusting OA flags."""
    original_license_value = record.get("license")
    original_license = (
        original_license_value
        if isinstance(original_license_value, dict)
        else None
    )
    nested_source = (
        original_license.get("source") if original_license is not None else None
    )
    nested_evidence = (
        original_license.get("evidence") if original_license is not None else None
    )
    top_level_source = record.get("license_source")
    top_level_evidence = record.get("license_evidence")
    stated_source = nested_source or top_level_source
    stated_evidence = (
        nested_evidence if nested_evidence is not None else top_level_evidence
    )

    input_reasons: list[str] = []
    if (
        nested_source is not None
        and top_level_source is not None
        and nested_source != top_level_source
    ):
        input_reasons.extend(
            [
                "license-provenance-contradiction",
                "license-provenance-contradiction:source",
            ]
        )
    if (
        nested_evidence is not None
        and top_level_evidence is not None
        and nested_evidence != top_level_evidence
    ):
        input_reasons.extend(
            [
                "license-provenance-contradiction",
                "license-provenance-contradiction:evidence",
            ]
        )

    raw_candidates = record.get("license_candidates")
    raw_candidates_supplied = bool(raw_candidates)
    if (
        raw_candidates is not None
        and not isinstance(raw_candidates, (dict, list))
    ):
        input_reasons.append("license-candidates-invalid")
        raw_license: Any = []
        raw_candidates_supplied = False
    else:
        raw_license = raw_candidates

    synthesized_from_normalized = False
    if not raw_license:
        normalized = original_license
        if (
            isinstance(normalized, dict)
            and stated_source == "crossref"
            and isinstance(stated_evidence, dict)
        ):
            raw_license = [dict(stated_evidence)]
        elif isinstance(normalized, dict) and (
            normalized.get("normalized_url") or normalized.get("url")
        ):
            synthesized_from_normalized = True
            metadata_evidence_url = (
                _article_metadata_evidence_url(stated_evidence)
                if stated_source == "article_metadata"
                else None
            )
            entry: dict[str, Any] = {
                "URL": metadata_evidence_url
                or normalized.get("url")
                or normalized.get("normalized_url")
            }
            content_version = normalized.get("content_version")
            if content_version:
                entry["content-version"] = content_version
            elif normalized.get("source") == "article_metadata":
                entry["content-version"] = "vor"
            effective = normalized.get("effective_date")
            if effective:
                try:
                    parsed = date.fromisoformat(str(effective))
                    entry["start"] = {"date-parts": [[parsed.year, parsed.month, parsed.day]]}
                except ValueError:
                    pass
            raw_license = [entry]
        elif isinstance(original_license_value, list):
            raw_license = original_license_value
            raw_candidates_supplied = bool(raw_license)
        else:
            raw_license = []

    authoritative_source: str | None
    if raw_candidates_supplied or isinstance(original_license_value, list):
        authoritative_source = "crossref"
    elif stated_source == "crossref" and isinstance(stated_evidence, dict):
        authoritative_source = "crossref"
    elif stated_source == "article_metadata":
        authoritative_source = "article_metadata"
    else:
        authoritative_source = None

    item = {
        "DOI": record.get("doi") or record.get("DOI"),
        "title": record.get("title") or [],
        "container-title": record.get("journal") or record.get("container-title") or [],
        "URL": record.get("article_url") or record.get("url") or record.get("URL"),
        "license": raw_license,
        "is_open_access": record.get("is_open_access")
        or record.get("open_access_signal"),
    }
    published_date = record.get("published_date")
    if published_date:
        try:
            parsed = date.fromisoformat(str(published_date))
            item["published"] = {
                "date-parts": [[parsed.year, parsed.month, parsed.day]]
            }
        except ValueError:
            pass
    elif record.get("year"):
        item["issued"] = {"date-parts": [[record["year"]]]}
    decision = evaluate_crossref_item(
        item,
        retrieved_at=retrieved_at or record.get("retrieved_at"),
        require_cc_by=require_cc_by,
    )
    selected_license = (
        decision.get("license")
        if isinstance(decision.get("license"), dict)
        else {}
    )
    contradiction_reasons = list(input_reasons)
    if original_license is not None:
        contradiction_reasons.extend(
            _license_contradictions(
                original_license,
                selected_license,
                authoritative_source=authoritative_source,
                stated_source=stated_source,
                stated_evidence=stated_evidence,
            )
        )
    elif isinstance(original_license_value, list) or raw_candidates_supplied:
        contradiction_reasons.extend(
            _license_contradictions(
                {},
                selected_license,
                authoritative_source="crossref",
                stated_source=stated_source,
                stated_evidence=stated_evidence,
            )
        )

    provenance_missing = synthesized_from_normalized and (
        stated_source not in {"crossref", "article_metadata"}
        or not stated_evidence
        or (
            stated_source == "crossref"
            and not isinstance(stated_evidence, dict)
        )
        or (
            stated_source == "article_metadata"
            and not _article_metadata_evidence_url(stated_evidence)
        )
    )
    if provenance_missing:
        contradiction_reasons.append("license-provenance-missing")
    _reject_license_decision(decision, contradiction_reasons)

    if (
        decision.get("policy_accepted")
        and original_license is not None
        and normalize_cc_by_url(
            original_license.get("normalized_url") or original_license.get("url")
        )
    ):
        preserved = dict(decision["license"])
        for key in (
            "url",
            "normalized_url",
            "license_id",
            "version",
            "effective_date",
            "content_version",
            "source",
            "evidence",
        ):
            if original_license.get(key) is not None:
                preserved[key] = original_license[key]
        decision["license"] = preserved
    if synthesized_from_normalized and authoritative_source == "article_metadata":
        decision["license_candidates"] = record.get("license_candidates") or []
    for key in ("authors", "abstract", "pmcid", "pmc_url", "pmid"):
        if key in record:
            decision[key] = record[key]
    return decision

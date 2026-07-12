"""Per-article provenance manifests and checksum validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

from .policy import evaluate_record, normalize_cc_by_url, normalize_doi


SCHEMA_VERSION = "1.0"
SOURCE_DATA_ORIGINS = frozenset(
    {"supplementary_information", "reconstructed", "not_present"}
)
RECONSTRUCTED_STATUSES = frozenset({"unverified", "verified"})


class ProvenanceError(ValueError):
    """Raised when provenance would otherwise be guessed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid-metadata-json:{path}:{exc}") from exc


def _article_id(record: dict[str, Any]) -> str:
    article_url = str(record.get("article_url") or record.get("url") or "")
    match = re.search(r"/articles/([^/?#]+)", article_url)
    if match:
        return match.group(1)
    doi = normalize_doi(record.get("doi") or record.get("DOI"))
    if doi and "/" in doi:
        return doi.split("/", 1)[1]
    return "unknown"


def _resolve_recorded_file(article_dir: Path, raw_path: Any, fallback_dir: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                article_dir / path,
                article_dir / fallback_dir / path.name,
            ]
        )
    else:
        candidates.append(article_dir / fallback_dir / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _relative_path(path: Path, article_dir: Path) -> str:
    try:
        return path.resolve().relative_to(article_dir.resolve()).as_posix()
    except ValueError:
        raise ProvenanceError(f"file-outside-article-directory:{path}") from None


def _file_entry(
    path: Path | None,
    article_dir: Path,
    *,
    kind: str,
    source_url: str | None,
    rejection_reason: str | None = None,
    source_data_origin: str | None = None,
    verification_status: str | None = None,
    verification_evidence: str | None = None,
) -> dict[str, Any]:
    exists = bool(path and path.is_file())
    entry: dict[str, Any] = {
        "kind": kind,
        "path": _relative_path(path, article_dir) if exists and path else None,
        "sha256": sha256_file(path) if exists and path else None,
        "size_bytes": path.stat().st_size if exists and path else None,
        "source_url": source_url,
        "download_status": "downloaded" if exists else "missing",
        "rejection_reason": rejection_reason if not exists else None,
    }
    if kind == "source_data":
        entry.update(
            {
                "source_data_origin": source_data_origin,
                "verification_status": verification_status,
                "verification_evidence": verification_evidence,
            }
        )
    return entry


def _tracked_figure_entries(article_dir: Path) -> tuple[list[dict[str, Any]], set[Path]]:
    metadata = _load_json(article_dir / "meta" / "figures.json", [])
    if not isinstance(metadata, list):
        metadata = []
    entries: list[dict[str, Any]] = []
    tracked: set[Path] = set()
    for raw in metadata:
        if not isinstance(raw, dict):
            continue
        image = _resolve_recorded_file(article_dir, raw.get("image_file"), "figures")
        caption = _resolve_recorded_file(article_dir, raw.get("caption_file"), "figures")
        source_url = raw.get("source_url") or raw.get("image_url")
        entries.append(
            _file_entry(
                image,
                article_dir,
                kind="figure",
                source_url=raw.get("image_url") or source_url,
                rejection_reason="recorded-figure-file-missing",
            )
        )
        if image:
            tracked.add(image)
        if raw.get("caption_file") or caption:
            entries.append(
                _file_entry(
                    caption,
                    article_dir,
                    kind="caption",
                    source_url=source_url,
                    rejection_reason="recorded-caption-file-missing",
                )
            )
            if caption:
                tracked.add(caption)
    return entries, tracked


def _tracked_source_entries(
    article_dir: Path,
    *,
    explicit_origin: str | None,
    reconstructed_verification: str,
    reconstructed_evidence: str | None,
    reconstructed_source_url: str | None,
) -> tuple[list[dict[str, Any]], set[Path], str]:
    metadata = _load_json(article_dir / "meta" / "source_data.json", [])
    if not isinstance(metadata, list):
        metadata = []
    entries: list[dict[str, Any]] = []
    tracked: set[Path] = set()
    inferred_origin = "supplementary_information" if metadata else explicit_origin
    source_dir = article_dir / "source_data"
    present_files = sorted(path.resolve() for path in source_dir.glob("**/*") if path.is_file())

    if present_files and inferred_origin not in {
        "supplementary_information",
        "reconstructed",
    }:
        raise ProvenanceError(
            f"source-data-origin-required:{article_dir.name}; "
            "choose supplementary_information or reconstructed"
        )
    if not present_files:
        return [], set(), "not_present"
    if inferred_origin == "reconstructed" and reconstructed_verification == "verified":
        if not reconstructed_evidence:
            raise ProvenanceError(
                "reconstructed-verification-evidence-required"
            )

    by_name: dict[str, dict[str, Any]] = {}
    for raw in metadata:
        if not isinstance(raw, dict):
            continue
        name = raw.get("saved_name")
        if name:
            by_name[str(name)] = raw

    for path in present_files:
        raw = by_name.get(path.name, {})
        source_url = raw.get("url") or (
            reconstructed_source_url
            if inferred_origin == "reconstructed"
            else None
        )
        origin = inferred_origin
        verification = (
            "source-provided"
            if origin == "supplementary_information"
            else reconstructed_verification
        )
        entries.append(
            _file_entry(
                path,
                article_dir,
                kind="source_data",
                source_url=source_url,
                source_data_origin=origin,
                verification_status=verification,
                verification_evidence=(
                    raw.get("label")
                    if origin == "supplementary_information"
                    else reconstructed_evidence
                ),
            )
        )
        tracked.add(path)
    return entries, tracked, str(inferred_origin)


def _untracked_figure_entries(
    article_dir: Path,
    tracked: set[Path],
    article_url: str | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    figures_dir = article_dir / "figures"
    for path in sorted(path.resolve() for path in figures_dir.glob("**/*") if path.is_file()):
        if path in tracked:
            continue
        kind = "caption" if path.suffix.casefold() in {".txt", ".md"} else "figure"
        entries.append(
            _file_entry(
                path,
                article_dir,
                kind=kind,
                source_url=article_url,
                rejection_reason=None,
            )
        )
    return entries


def build_article_manifest(
    record: dict[str, Any],
    content_root: str | Path,
    *,
    require_cc_by: bool = True,
    download_status: str | None = None,
    rejection_reason: str | None = None,
    source_data_origin: str | None = None,
    reconstructed_verification: str = "unverified",
    reconstructed_evidence: str | None = None,
) -> dict[str, Any]:
    if source_data_origin and source_data_origin not in SOURCE_DATA_ORIGINS:
        raise ProvenanceError(f"invalid-source-data-origin:{source_data_origin}")
    if reconstructed_verification not in RECONSTRUCTED_STATUSES:
        raise ProvenanceError(
            f"invalid-reconstructed-verification:{reconstructed_verification}"
        )

    decision = evaluate_record(record, require_cc_by=require_cc_by)
    article_id = _article_id(decision)
    article_dir = Path(content_root) / article_id
    files: list[dict[str, Any]] = []
    article_source_origin = "not_present"
    if article_dir.is_dir():
        figure_entries, figure_tracked = _tracked_figure_entries(article_dir)
        files.extend(figure_entries)
        files.extend(
            _untracked_figure_entries(
                article_dir,
                figure_tracked,
                decision.get("article_url"),
            )
        )
        source_entries, _, article_source_origin = _tracked_source_entries(
            article_dir,
            explicit_origin=source_data_origin
            or record.get("source_data_origin"),
            reconstructed_verification=record.get(
                "source_data_verification", reconstructed_verification
            ),
            reconstructed_evidence=record.get(
                "source_data_verification_evidence", reconstructed_evidence
            ),
            reconstructed_source_url=record.get(
                "reconstructed_source_url"
            )
            or decision.get("article_url"),
        )
        files.extend(source_entries)

    reject_reasons = list(decision.get("reject_reasons") or [])
    if rejection_reason:
        reject_reasons.append(rejection_reason)
    if not decision.get("download_eligible"):
        resolved_status = "rejected"
    elif download_status:
        resolved_status = download_status
    elif not article_dir.exists():
        resolved_status = "not_started"
    elif any(entry["download_status"] == "missing" for entry in files):
        resolved_status = "partial"
    elif files:
        resolved_status = "downloaded"
    else:
        resolved_status = "empty"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "doi": decision.get("doi"),
        "journal": decision.get("journal"),
        "year": decision.get("year"),
        "published_date": decision.get("published_date"),
        "article_url": decision.get("article_url"),
        "license": decision.get("license"),
        "license_source": (decision.get("license") or {}).get("source"),
        "license_evidence": (decision.get("license") or {}).get("evidence"),
        "retrieved_at": decision.get("retrieved_at"),
        "manifest_generated_at": utc_now(),
        "journal_allowed": decision.get("journal_allowed"),
        "policy_accepted": decision.get("policy_accepted"),
        "require_cc_by": require_cc_by,
        "download_eligible": decision.get("download_eligible"),
        "download_status": resolved_status,
        "rejection_reasons": sorted(set(reject_reasons)),
        "source_data_origin": article_source_origin,
        "files": sorted(
            files,
            key=lambda entry: (str(entry.get("kind")), str(entry.get("path"))),
        ),
    }
    return manifest


def build_corpus_manifest(
    records: Iterable[dict[str, Any]],
    content_root: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        doi = normalize_doi(record.get("doi") or record.get("DOI"))
        dedup_key = doi or f"missing:{len(manifests)}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        manifests.append(build_article_manifest(record, content_root, **kwargs))
    return manifests


def _safe_manifest_file_path(value: Any) -> PurePosixPath | None:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        return None
    return path


def _path_has_symlink(path: Path, article_dir: Path) -> bool:
    current = article_dir
    try:
        parts = path.relative_to(article_dir).parts
    except ValueError:
        return True
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate_article_manifest(
    manifest: dict[str, Any],
    *,
    content_root: str | Path | None = None,
) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest-invalid"]

    errors: list[str] = []
    if not normalize_doi(manifest.get("doi")):
        errors.append("doi-missing")
    if manifest.get("download_eligible"):
        if not manifest.get("journal_allowed"):
            errors.append("journal-not-allowed")
        license_value = manifest.get("license")
        if not isinstance(license_value, dict):
            errors.append("license-record-invalid")
            license_record: dict[str, Any] = {}
        else:
            license_record = license_value
        normalized = normalize_cc_by_url(
            license_record.get("normalized_url") or license_record.get("url")
        )
        if not normalized:
            errors.append("license-not-verifiable-cc-by")
        else:
            canonical_url, version = normalized
            if (
                license_record.get("normalized_url") is not None
                and license_record.get("normalized_url") != canonical_url
            ):
                errors.append("license-normalized-url-mismatch")
            if (
                license_record.get("version") is not None
                and license_record.get("version") != version
            ):
                errors.append("license-version-mismatch")
            if (
                license_record.get("license_id") is not None
                and license_record.get("license_id") != f"CC-BY-{version}"
            ):
                errors.append("license-id-mismatch")

        license_source = manifest.get("license_source")
        nested_source = license_record.get("source")
        if not license_source:
            errors.append("license-source-missing")
        if not nested_source:
            errors.append("license-record-source-missing")
        if (
            license_source
            and nested_source
            and license_source != nested_source
        ):
            errors.append("license-source-mismatch")

        raw_evidence = manifest.get("license_evidence")
        nested_evidence = license_record.get("evidence")
        if not raw_evidence:
            errors.append("license-evidence-missing")
        if (
            nested_evidence is not None
            and raw_evidence is not None
            and nested_evidence != raw_evidence
        ):
            errors.append("license-evidence-mismatch")
        if (
            (nested_source or license_source) == "crossref"
        ):
            if license_record.get("content_version") != "vor":
                errors.append("license-crossref-content-version-not-vor")
            if not isinstance(raw_evidence, dict):
                errors.append("license-crossref-evidence-invalid")
            else:
                if raw_evidence.get("content-version") != "vor":
                    errors.append(
                        "license-crossref-evidence-content-version-not-vor"
                    )
                evidence_url = raw_evidence.get("URL") or raw_evidence.get("url")
                if (
                    not normalized
                    or normalize_cc_by_url(evidence_url) != normalized
                ):
                    errors.append("license-crossref-evidence-url-mismatch")
                if (
                    license_record.get("url") is not None
                    and license_record.get("url") != evidence_url
                ):
                    errors.append("license-crossref-raw-url-mismatch")
    elif not manifest.get("rejection_reasons"):
        errors.append("rejected-without-reason")
    if manifest.get("download_eligible") and not manifest.get("require_cc_by"):
        errors.append("download-eligible-without-cc-by-gate")

    origin = manifest.get("source_data_origin")
    if not isinstance(origin, str) or origin not in SOURCE_DATA_ORIGINS:
        errors.append("source-data-origin-invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        errors.append("files-invalid")
        raw_files = []

    article_dir = (
        Path(content_root) / _article_id(manifest)
        if content_root is not None
        else None
    )
    seen_paths: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict):
            errors.append("file-entry-invalid")
            continue
        kind = entry.get("kind")
        if not isinstance(kind, str) or kind not in {
            "figure",
            "caption",
            "source_data",
        }:
            errors.append("file-kind-invalid")
        if kind == "source_data":
            file_origin = entry.get("source_data_origin")
            if not isinstance(file_origin, str) or file_origin not in {
                "supplementary_information",
                "reconstructed",
            }:
                errors.append("source-data-file-origin-invalid")
            if (
                file_origin == "reconstructed"
                and entry.get("verification_status") == "verified"
                and not entry.get("verification_evidence")
            ):
                errors.append("reconstructed-verification-evidence-missing")
            if (
                file_origin == "reconstructed"
                and not entry.get("verification_status")
            ):
                errors.append("reconstructed-verification-status-missing")
            if (
                origin in {
                    "supplementary_information",
                    "reconstructed",
                }
                and file_origin != origin
            ):
                errors.append("source-data-file-origin-mismatch")

        status = entry.get("download_status")
        if not isinstance(status, str) or status not in {
            "downloaded",
            "missing",
        }:
            errors.append("file-download-status-invalid")
        raw_path = entry.get("path")
        safe_path = (
            _safe_manifest_file_path(raw_path)
            if raw_path is not None
            else None
        )
        if raw_path is not None and safe_path is None:
            errors.append(f"file-path-invalid:{raw_path}")
        if safe_path is not None:
            path_key = safe_path.as_posix().casefold()
            if path_key in seen_paths:
                errors.append(f"file-path-duplicate:{safe_path.as_posix()}")
            seen_paths.add(path_key)

        if status == "downloaded":
            if safe_path is None:
                errors.append("downloaded-file-path-missing")
            checksum = entry.get("sha256")
            if not checksum:
                errors.append("downloaded-file-checksum-missing")
            elif (
                not isinstance(checksum, str)
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            ):
                errors.append(f"downloaded-file-checksum-invalid:{raw_path}")
            size_bytes = entry.get("size_bytes")
            if (
                not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
            ):
                errors.append(f"downloaded-file-size-invalid:{raw_path}")
            source_url = entry.get("source_url")
            if not isinstance(source_url, str) or not source_url:
                errors.append(f"downloaded-file-source-url-missing:{raw_path}")
            if article_dir is not None and safe_path is not None:
                path = article_dir.joinpath(*safe_path.parts)
                resolved = path.resolve()
                try:
                    resolved.relative_to(article_dir.resolve())
                except ValueError:
                    errors.append(f"file-outside-article-directory:{raw_path}")
                    continue
                if _path_has_symlink(path, article_dir):
                    errors.append(f"file-symlink:{raw_path}")
                elif not path.is_file():
                    errors.append(f"file-missing:{raw_path}")
                else:
                    if sha256_file(path) != checksum:
                        errors.append(f"checksum-mismatch:{raw_path}")
                    if path.stat().st_size != size_bytes:
                        errors.append(f"size-mismatch:{raw_path}")
    return sorted(set(errors))

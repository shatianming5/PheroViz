from __future__ import annotations

import json
from pathlib import Path

import pytest

from nature_download.corpus.provenance import (
    ProvenanceError,
    build_article_manifest,
    sha256_file,
    validate_article_manifest,
)


ARTICLE_ID = "s41467-024-12345-6"


def accepted_record() -> dict:
    return {
        "doi": f"10.1038/{ARTICLE_ID}",
        "journal": "Nature Communications",
        "year": 2024,
        "published_date": "2024-05-06",
        "article_url": f"https://www.nature.com/articles/{ARTICLE_ID}",
        "license": {
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "normalized_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_id": "CC-BY-4.0",
            "version": "4.0",
            "effective_date": "2024-05-06",
            "source": "crossref",
            "evidence": {"URL": "http://creativecommons.org/licenses/by/4.0/"},
        },
    }


def make_downloaded_article(content_root: Path) -> Path:
    article = content_root / ARTICLE_ID
    figures = article / "figures"
    source_data = article / "source_data"
    meta = article / "meta"
    figures.mkdir(parents=True)
    source_data.mkdir()
    meta.mkdir()
    image = figures / "fig_001.png"
    caption = figures / "fig_001.txt"
    source = source_data / "Source_Data_Fig_1.csv"
    image.write_bytes(b"png fixture")
    caption.write_text("caption fixture", encoding="utf-8")
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    (meta / "figures.json").write_text(
        json.dumps(
            [
                {
                    "figure_tag": "fig_001",
                    "image_file": str(image),
                    "caption_file": str(caption),
                    "image_url": "https://media.springernature.com/fig1.png",
                    "source_url": f"https://www.nature.com/articles/{ARTICLE_ID}/figures/1",
                }
            ]
        ),
        encoding="utf-8",
    )
    (meta / "source_data.json").write_text(
        json.dumps(
            [
                {
                    "label": "Source Data Fig. 1",
                    "url": "https://static-content.springer.com/source-data.csv",
                    "saved_name": source.name,
                }
            ]
        ),
        encoding="utf-8",
    )
    return article


def test_manifest_records_checksums_sources_and_si_origin(workdir: Path) -> None:
    content = workdir / "content"
    article = make_downloaded_article(content)
    manifest = build_article_manifest(accepted_record(), content)
    assert manifest["download_eligible"] is True
    assert manifest["source_data_origin"] == "supplementary_information"
    by_kind = {
        entry["kind"]: entry
        for entry in manifest["files"]
        if entry["kind"] in {"figure", "caption", "source_data"}
    }
    assert by_kind["figure"]["sha256"] == sha256_file(
        article / "figures" / "fig_001.png"
    )
    assert by_kind["caption"]["source_url"].endswith("/figures/1")
    assert by_kind["source_data"]["source_data_origin"] == "supplementary_information"
    assert by_kind["source_data"]["verification_status"] == "source-provided"
    assert by_kind["source_data"]["source_url"].startswith("https://")
    assert validate_article_manifest(manifest, content_root=content) == []


def test_checksum_validation_detects_mutation(workdir: Path) -> None:
    content = workdir / "content"
    article = make_downloaded_article(content)
    manifest = build_article_manifest(accepted_record(), content)
    (article / "figures" / "fig_001.png").write_bytes(b"mutated")
    errors = validate_article_manifest(manifest, content_root=content)
    assert "checksum-mismatch:figures/fig_001.png" in errors


def test_untracked_source_data_requires_explicit_origin(workdir: Path) -> None:
    content = workdir / "content"
    source_dir = content / ARTICLE_ID / "source_data"
    source_dir.mkdir(parents=True)
    (source_dir / "reconstructed.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="source-data-origin-required"):
        build_article_manifest(accepted_record(), content)


def test_reconstructed_data_defaults_to_unverified(workdir: Path) -> None:
    content = workdir / "content"
    source_dir = content / ARTICLE_ID / "source_data"
    source_dir.mkdir(parents=True)
    (source_dir / "reconstructed.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    manifest = build_article_manifest(
        accepted_record(),
        content,
        source_data_origin="reconstructed",
    )
    entry = next(item for item in manifest["files"] if item["kind"] == "source_data")
    assert entry["source_data_origin"] == "reconstructed"
    assert entry["verification_status"] == "unverified"
    assert entry["source_url"] == accepted_record()["article_url"]


def test_reconstructed_verified_requires_evidence(workdir: Path) -> None:
    content = workdir / "content"
    source_dir = content / ARTICLE_ID / "source_data"
    source_dir.mkdir(parents=True)
    (source_dir / "reconstructed.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="evidence-required"):
        build_article_manifest(
            accepted_record(),
            content,
            source_data_origin="reconstructed",
            reconstructed_verification="verified",
        )


def test_rejected_article_is_a_valid_fail_closed_manifest(workdir: Path) -> None:
    record = accepted_record()
    record["journal"] = "Nature"
    manifest = build_article_manifest(record, workdir / "content")
    assert manifest["download_status"] == "rejected"
    assert "journal-nature-main-excluded" in manifest["rejection_reasons"]
    assert validate_article_manifest(manifest) == []


def test_corrupt_existing_metadata_fails_explicitly(workdir: Path) -> None:
    content = workdir / "content"
    metadata = content / ARTICLE_ID / "meta"
    metadata.mkdir(parents=True)
    (metadata / "figures.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="invalid-metadata-json"):
        build_article_manifest(accepted_record(), content)

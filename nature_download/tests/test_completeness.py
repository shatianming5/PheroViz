from __future__ import annotations

import json
from pathlib import Path

from corpus.build_combined_records import iter_provenance_records
from corpus.completeness import (
    inspect_article_payload,
    quarantine_incomplete_article_dirs,
)


def _article(
    root: Path,
    name: str,
    *,
    figure: bool = True,
    source: bool = True,
    metadata: bool = True,
) -> Path:
    article = root / name
    if figure:
        figures = article / "figures"
        figures.mkdir(parents=True)
        (figures / "fig_001.png").write_bytes(b"png")
    if source:
        source_data = article / "source_data"
        source_data.mkdir(parents=True, exist_ok=True)
        (source_data / "source.xlsx").write_bytes(b"source")
    if metadata:
        meta = article / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "figures.json").write_text("[]", encoding="utf-8")
    return article


def test_payload_gate_requires_figure_source_and_metadata(workdir: Path) -> None:
    content = workdir / "content"
    complete = _article(content, "complete")
    figure_only = _article(content, "figure-only", source=False)
    source_only = _article(content, "source-only", figure=False)

    assert inspect_article_payload(complete).complete
    assert inspect_article_payload(figure_only).reasons == ("source-data-missing",)
    assert inspect_article_payload(source_only).reasons == ("figure-image-missing",)

    quarantined = quarantine_incomplete_article_dirs(content)

    assert [path.name for path in quarantined] == ["figure-only", "source-only"]
    assert complete.exists()
    assert not figure_only.exists()
    assert not source_only.exists()
    assert (content / "_rejected_no_source" / "figure-only").is_dir()
    assert (content / "_rejected_no_source" / "source-only").is_dir()


def test_combined_records_excludes_and_quarantines_incomplete_articles(
    workdir: Path,
) -> None:
    content = workdir / "content"
    complete = _article(content, "lsa.complete")
    incomplete = _article(content, "lsa.incomplete", source=False)
    (complete / "meta" / "provenance.json").write_text(
        json.dumps({"doi": "10.26508/lsa.complete"}),
        encoding="utf-8",
    )
    (incomplete / "meta" / "provenance.json").write_text(
        json.dumps({"doi": "10.26508/lsa.incomplete"}),
        encoding="utf-8",
    )

    records = iter_provenance_records(content)

    assert records == [{"doi": "10.26508/lsa.complete"}]
    assert not incomplete.exists()
    assert (content / "_rejected_no_source" / "lsa.incomplete").is_dir()

from __future__ import annotations

from pathlib import Path

import embo_harvest


FIXTURES = Path(__file__).parent / "fixtures"


def sample_html() -> str:
    return (FIXTURES / "embo_article.html").read_text(encoding="utf-8")


def test_panel_count_accepts_spaced_capital_parentheses() -> None:
    assert embo_harvest.panel_count("( A )( B )( C )( D )( E )") == 5


def test_is_archive_junk_filters_os_generated_members() -> None:
    # macOS AppleDouble resource-fork junk and Office lock temp files are junk...
    assert embo_harvest.is_archive_junk("__MACOSX/Figure 6/6B/._Chemotaxis.xlsx")
    assert embo_harvest.is_archive_junk("Figure 1/._Figure 1A-H.xlsx")
    assert embo_harvest.is_archive_junk("Figure 6/6B/~$Chemotaxis.xlsx")
    # ...but genuine per-figure source-data members are NOT junk.
    assert not embo_harvest.is_archive_junk("Figure 6/6B/Chemotaxis.xlsx")
    assert not embo_harvest.is_archive_junk("SourceData_Fig1.csv")


def test_build_moesm_source_map_filters_to_downloadable_source_types() -> None:
    source_map = embo_harvest.build_moesm_source_map(
        sample_html(), "10.1038/s44320-026-00206-9"
    )
    assert sorted(source_map) == ["MOESM3"]
    assert source_map["MOESM3"]["ext"] == "xlsx"
    assert source_map["MOESM3"]["url"].endswith("MOESM3_ESM.xlsx")


def test_license_gate_detects_cc_by_4_html() -> None:
    assert embo_harvest.has_cc_by_4_license(sample_html()) is True
    assert embo_harvest.has_cc_by_4_license("https://creativecommons.org/licenses/by-nc/4.0/") is False


def test_article_id_sanitizes_doi_suffix() -> None:
    assert (
        embo_harvest.article_id_from_doi("10.1038/s44320-026-00206-9")
        == "s44320-026-00206-9"
    )
    assert embo_harvest.article_id_from_doi("10.7554/elife.12345") is None


def test_parse_article_assets_associates_figure_source_and_largest_png() -> None:
    figures, per_figure_keys = embo_harvest.parse_article_assets(
        sample_html(), "10.1038/s44320-026-00206-9"
    )
    assert per_figure_keys == ["MOESM3"]
    assert len(figures) == 1
    figure = figures[0]
    assert figure["number"] == 2
    assert figure["panels"] == 5
    assert figure["source_keys"] == ["MOESM3"]
    assert figure["source_records"][0]["url"].endswith("MOESM3_ESM.xlsx")
    assert "/lw1200/" in figure["image_url"]

from __future__ import annotations

import argparse

from corpus.policy import is_allowed_journal
import lsa_harvest


ARTICLE_HTML = """
<html>
  <head>
    <link rel="license" href="https://creativecommons.org/licenses/by/4.0/">
  </head>
  <body>
    <div class="fig pos-float type-featured" id="F2">
      <a class="highwire-figure-link highwire-figure-link-download"
         href="https://www.life-science-alliance.org/content/lsa/6/1/e202201499/F2.large.jpg?download=true">
        Download figure
      </a>
      <div class="fig-caption">
        <span class="fig-label">Figure 1.</span>
        A representative result. (A, B, C, D, E) Five measured panels.
      </div>
      <div class="supplementary-material source-data" id="DC2">
        <a href="https://www.life-science-alliance.org/content/lsa/6/1/e202201499/F2/DC2/embed/inline-supplementary-material-2.xlsx?download=true">
          [LSA-2022-01499_SdataF1.xlsx]
        </a>
      </div>
    </div>
    <div class="fig pos-float type-figure" id="F3">
      <a class="highwire-figure-link highwire-figure-link-download"
         href="https://www.life-science-alliance.org/content/lsa/6/1/e202201499/F3.large.jpg?download=true">
        Download figure
      </a>
      <div class="fig-caption">
        <span class="fig-label">Figure 2.</span>
        (A, B, C, D, E) This source link belongs to another figure and is rejected.
      </div>
      <div class="supplementary-material source-data" id="DC3">
        <a href="https://www.life-science-alliance.org/content/lsa/6/1/e202201499/F99/DC3/embed/inline-supplementary-material-3.xlsx?download=true">
          [wrong-figure.xlsx]
        </a>
      </div>
    </div>
    <div class="fig type-supplementary-material" id="F4">
      <div class="fig-caption">
        <span class="fig-label">Figure S1.</span>
        (A, B, C, D, E) Supplementary figure.
      </div>
      <div class="supplementary-material source-data">
        <a href="https://www.life-science-alliance.org/content/lsa/6/1/e202201499/F4/DC4/embed/inline-supplementary-material-4.xlsx?download=true">
          [supplement.xlsx]
        </a>
      </div>
    </div>
  </body>
</html>
"""


def test_article_identity_accepts_only_lsa_dois() -> None:
    assert lsa_harvest.article_identity("10.26508/lsa.202201499") == (
        "lsa.202201499",
        "e202201499",
    )
    assert lsa_harvest.article_identity("10.7554/elife.12345") is None


def test_license_gate_detects_only_cc_by_4_html() -> None:
    assert lsa_harvest.has_cc_by_4_license(ARTICLE_HTML) is True
    assert (
        lsa_harvest.has_cc_by_4_license(
            "https://creativecommons.org/licenses/by-nc/4.0/"
        )
        is False
    )


def test_parse_assets_requires_figure_local_source_data() -> None:
    figures = lsa_harvest.parse_article_assets(
        ARTICLE_HTML,
        "10.26508/lsa.202201499",
        article_url_value="https://www.life-science-alliance.org/content/6/1/e202201499",
    )
    assert [(figure["number"], figure["panels"]) for figure in figures] == [
        (1, 5),
        (2, 5),
    ]
    assert figures[0]["image_url"].endswith("/F2.large.jpg?download=true")
    assert figures[0]["source_records"] == [
        {
            "url": (
                "https://www.life-science-alliance.org/content/lsa/6/1/"
                "e202201499/F2/DC2/embed/inline-supplementary-material-2.xlsx"
                "?download=true"
            ),
            "name": "inline-supplementary-material-2.xlsx",
            "ext": "xlsx",
            "label": "[LSA-2022-01499_SdataF1.xlsx]",
            "asset_id": "2",
        }
    ]
    assert figures[1]["source_records"] == []


def test_policy_allows_verified_lsa_journal() -> None:
    assert is_allowed_journal("Life Science Alliance")


def test_run_treats_existing_article_as_a_successful_resume(
    monkeypatch, tmp_path
) -> None:
    item = {
        "DOI": "10.26508/lsa.202201499",
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
    }
    article = tmp_path / "lsa.202201499"
    (article / "figures").mkdir(parents=True)
    (article / "source_data").mkdir()
    (article / "meta").mkdir()
    (article / "figures" / "fig_001.png").write_bytes(b"png")
    (article / "source_data" / "source.xlsx").write_bytes(b"data")
    (article / "meta" / "figures.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(lsa_harvest, "maybe_force_ipv4", lambda: False)
    monkeypatch.setattr(
        lsa_harvest,
        "crossref_items",
        lambda *args, **kwargs: iter([item]),
    )
    args = argparse.Namespace(
        require_cc_by=True,
        out=str(tmp_path),
        min_panels=5,
        from_date="2018-01-01",
        until_date="2025-12-31",
        max_articles=1,
        sleep=0,
        timeout=5,
        max_retries=1,
    )
    assert lsa_harvest.run(args) == 0

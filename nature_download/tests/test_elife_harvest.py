from __future__ import annotations

import elife_harvest


# A realistic (trimmed) slice of the official eLife JSON API payload shape.
# Figure blocks live nested under `body`; each has `assets[]`. Main figures use
# ids like `fig1`/`fig2`; supplements use `fig1s1`. `sourceData[]` carries the
# per-figure XLSX on cdn.elifesciences.org. Captions are structured nodes.
API_PAYLOAD = {
    "id": "16349",
    "body": [
        {
            "type": "section",
            "content": [
                {
                    "type": "figure",
                    "assets": [
                        {
                            "id": "fig1",
                            "label": "Figure 1",
                            "title": "Neuronal AMPK regulates behaviour.",
                            "caption": [
                                {
                                    "text": "(A) local to distal transition. "
                                    "(B) forward locomotion. (C) reversals. "
                                    "(D) speed. (E) dwelling."
                                }
                            ],
                            "image": {
                                "uri": "https://iiif.elifesciences.org/lax/"
                                "16349%2Felife-16349-fig1-v1.tif",
                                "source": {
                                    "uri": "https://iiif.elifesciences.org/lax/"
                                    "16349%2Felife-16349-fig1-v1.tif/full/"
                                    "full/0/default.jpg"
                                },
                            },
                            "sourceData": [
                                {
                                    "id": "SD1",
                                    "uri": "https://cdn.elifesciences.org/"
                                    "articles/16349/"
                                    "elife-16349-fig1-data1-v1.xlsx",
                                },
                                {
                                    "id": "SD2",
                                    "uri": "https://cdn.elifesciences.org/"
                                    "articles/16349/"
                                    "elife-16349-fig1-data2-v1.xlsx",
                                },
                            ],
                        },
                        {
                            "id": "fig1s1",
                            "label": "Figure 1—figure supplement 1",
                            "title": "Supplement.",
                            "caption": [{"text": "(A) control."}],
                            "image": {
                                "uri": "https://iiif.elifesciences.org/lax/"
                                "16349%2Felife-16349-fig1-figsupp1-v1.tif"
                            },
                            "sourceData": [
                                {
                                    "id": "SD3",
                                    "uri": "https://cdn.elifesciences.org/"
                                    "articles/16349/"
                                    "elife-16349-fig1-figsupp1-data1-v1.xlsx",
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "type": "figure",
            "assets": [
                {
                    "id": "fig2",
                    "label": "Figure 2",
                    "title": "",
                    "caption": [{"text": "(A) one. (B) two."}],
                    "image": {
                        "uri": "https://iiif.elifesciences.org/lax/"
                        "16349%2Felife-16349-fig2-v1.tif"
                    },
                    "sourceData": [
                        {
                            "id": "SD4",
                            # Wrong host -> must be rejected.
                            "uri": "https://evil.example.org/articles/16349/"
                            "elife-16349-fig2-data1-v1.xlsx",
                        },
                        {
                            "id": "SD5",
                            "uri": "https://cdn.elifesciences.org/articles/"
                            "16349/elife-16349-fig2-data1-v1.xlsx",
                        },
                    ],
                }
            ],
        },
        {
            # Tables are not figures -> ignored by the figure-asset walker.
            "type": "table",
            "assets": [{"id": "table1", "label": "Table 1"}],
        },
    ],
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_caption_to_text_flattens_structured_nodes():
    node = [
        {"text": "Panel A."},
        {"content": [{"text": "nested"}, {"text": "deeper"}]},
    ]
    assert elife_harvest._caption_to_text(node) == "Panel A. nested deeper"


def test_iter_figure_assets_walks_nested_figures_only():
    ids = [a.get("id") for a in elife_harvest._iter_figure_assets(API_PAYLOAD)]
    # Both figure blocks (nested + top-level) yield their assets; the table
    # block does not.
    assert ids == ["fig1", "fig1s1", "fig2"]


def test_asset_source_xlsx_filters_host_and_article():
    asset = API_PAYLOAD["body"][1]["assets"][0]  # fig2
    urls = elife_harvest._asset_source_xlsx(asset, "16349")
    assert urls == [
        "https://cdn.elifesciences.org/articles/16349/"
        "elife-16349-fig2-data1-v1.xlsx"
    ]
    # Wrong article id must not match.
    assert elife_harvest._asset_source_xlsx(asset, "99999") == []


def test_asset_png_url_builds_iiif_default_png():
    asset = API_PAYLOAD["body"][0]["content"][0]["assets"][0]  # fig1
    assert elife_harvest._asset_png_url(asset) == (
        "https://iiif.elifesciences.org/lax/"
        "16349%2Felife-16349-fig1-v1.tif/full/full/0/default.png"
    )


def test_fetch_article_assets_collects_sources_and_main_figures(monkeypatch):
    def fake_request(session, url, **kwargs):
        assert url.endswith("/16349")
        assert kwargs.get("accept") == elife_harvest.ELIFE_API_ACCEPT
        return _FakeResponse(API_PAYLOAD)

    monkeypatch.setattr(elife_harvest, "request_with_retries", fake_request)

    sources, figures = elife_harvest.fetch_article_assets(
        session=None,
        numeric_id="16349",
        timeout=10,
        sleep=0,
        max_retries=3,
    )

    # Source data from main figures AND supplements, deduped + sorted, with the
    # wrong-host URL rejected.
    assert sources == [
        "https://cdn.elifesciences.org/articles/16349/"
        "elife-16349-fig1-data1-v1.xlsx",
        "https://cdn.elifesciences.org/articles/16349/"
        "elife-16349-fig1-data2-v1.xlsx",
        "https://cdn.elifesciences.org/articles/16349/"
        "elife-16349-fig1-figsupp1-data1-v1.xlsx",
        "https://cdn.elifesciences.org/articles/16349/"
        "elife-16349-fig2-data1-v1.xlsx",
    ]

    # Only main figures (fig1, fig2); the fig1s1 supplement is excluded.
    assert [f["number"] for f in figures] == [1, 2]
    fig1 = figures[0]
    assert fig1["panels"] == 5  # (A)-(E)
    assert fig1["caption"].startswith("Neuronal AMPK regulates behaviour.")
    assert fig1["image_url"].endswith("/full/full/0/default.png")


def test_fetch_article_assets_empty_when_no_figures(monkeypatch):
    monkeypatch.setattr(
        elife_harvest,
        "request_with_retries",
        lambda session, url, **kwargs: _FakeResponse({"id": "1", "body": []}),
    )
    sources, figures = elife_harvest.fetch_article_assets(
        session=None, numeric_id="1", timeout=10, sleep=0, max_retries=3
    )
    assert sources == []
    assert figures == []

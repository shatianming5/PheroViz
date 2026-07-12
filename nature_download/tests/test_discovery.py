from __future__ import annotations

from nature_download.corpus import discovery


def test_discovery_uses_cursor_pagination_and_filters(monkeypatch):
    calls = []
    pages = [
        {
            "message": {
                "items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}],
                "next-cursor": "cursor-2",
            }
        },
        {
            "message": {
                "items": [{"DOI": "10.1/c"}],
                "next-cursor": "cursor-3",
            }
        },
    ]

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return pages.pop(0)

    monkeypatch.setattr(discovery, "_request_json", fake_request)
    items = discovery.crossref_discover(
        "machine learning",
        rows=3,
        journal="Nature Communications",
        from_date="2024-01-01",
        until_date="2026-07-12",
        sleep=0,
        page_size=2,
    )

    assert [item["DOI"] for item in items] == ["10.1/a", "10.1/b", "10.1/c"]
    first = calls[0][1]["params"]
    second = calls[1][1]["params"]
    assert first["cursor"] == "*"
    assert second["cursor"] == "cursor-2"
    assert first["query.container-title"] == "Nature Communications"
    assert first["filter"] == (
        "type:journal-article,from-pub-date:2024-01-01,"
        "until-pub-date:2026-07-12"
    )


def test_discovery_rejects_invalid_bounds():
    for kwargs in (
        {"rows": 0},
        {"rows": 1, "page_size": 0},
        {"rows": 1, "from_date": "not-a-date"},
    ):
        try:
            discovery.crossref_discover("x", sleep=0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")

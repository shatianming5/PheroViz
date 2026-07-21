"""Regression tests for PV_ISSN_FILTER exact-journal enumeration.

nature_all_in_one reads PV_ISSN_FILTER at import time to (a) restrict the
Crossref filter to specific ISSNs so cursor pagination walks an entire journal
exhaustively, and (b) drop the query.container-title bias, which otherwise
CONFLICTS with the ISSN filter and returns zero results.
"""
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]


def _probe(env_extra: dict[str, str]) -> dict[str, str]:
    code = (
        "import json, nature_all_in_one as m;"
        "print(json.dumps({"
        "'filter': m._PV_DATE_FILTER,"
        "'issn': m._PV_ISSN_FILTER,"
        "}))"
    )
    import os

    env = dict(os.environ)
    env.update(env_extra)
    out = subprocess.check_output(
        [sys.executable, "-c", code], cwd=str(MODULE_DIR), env=env, text=True
    )
    import json

    return json.loads(out.strip().splitlines()[-1])


def test_issn_filter_absent_by_default() -> None:
    info = _probe({"PV_FROM_DATE": "2020-01-01", "PV_UNTIL_DATE": "2020-12-31", "PV_ISSN_FILTER": ""})
    assert "issn:" not in info["filter"]
    assert info["filter"] == "type:journal-article,from-pub-date:2020-01-01,until-pub-date:2020-12-31"


def test_issn_filter_injected_into_crossref_filter() -> None:
    info = _probe(
        {
            "PV_FROM_DATE": "2020-01-01",
            "PV_UNTIL_DATE": "2023-12-31",
            "PV_ISSN_FILTER": "2041-1723",
        }
    )
    assert info["filter"].endswith(",issn:2041-1723")
    assert info["issn"] == "2041-1723"


def test_issn_filter_supports_multiple_issns() -> None:
    info = _probe(
        {
            "PV_FROM_DATE": "2020-01-01",
            "PV_UNTIL_DATE": "2023-12-31",
            "PV_ISSN_FILTER": "2041-1723, 2050-084X",
        }
    )
    assert "issn:2041-1723" in info["filter"]
    assert "issn:2050-084X" in info["filter"]

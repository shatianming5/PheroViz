from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from nature_download.corpus.cli import LicenseGateError, authorize_direct_download
from nature_download.nature_all_in_one import build_parser, cmd_auto, cmd_postfetch


def test_new_corpus_commands_enable_gate_by_default() -> None:
    parser = build_parser()
    discover = parser.parse_args(["discover", "--query", "test"])
    validate = parser.parse_args(["validate", "--jsonl", "records.jsonl"])
    manifest = parser.parse_args(
        [
            "build-manifest",
            "--jsonl",
            "records.jsonl",
            "--content-root",
            "content",
        ]
    )
    assert discover.require_cc_by is True
    assert validate.require_cc_by is True
    assert manifest.require_cc_by is True


def test_legacy_download_commands_require_explicit_gate() -> None:
    parser = build_parser()
    assert parser.parse_args(["search", "--query", "test"]).require_cc_by is False
    assert parser.parse_args(["postfetch", "--jsonl", "x"]).require_cc_by is False
    assert parser.parse_args(["auto"]).require_cc_by is False


def test_direct_download_fails_before_network_without_gate() -> None:
    with pytest.raises(LicenseGateError, match="--require-cc-by"):
        authorize_direct_download(
            url="https://www.nature.com/articles/nature00001/figures/1",
            doi=None,
            require_cc_by=False,
            mailto=None,
            timeout=1,
            max_retries=1,
            sleep=0,
        )


def test_legacy_batch_paths_fail_before_io_without_gate() -> None:
    with pytest.raises(LicenseGateError, match="postfetch refused"):
        cmd_postfetch(type("Args", (), {"require_cc_by": False})())
    with pytest.raises(LicenseGateError, match="auto download refused"):
        cmd_auto(type("Args", (), {"require_cc_by": False})())


def test_postfetch_rejects_non_cc_by_before_download(workdir: Path) -> None:
    records = workdir / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "doi": "10.1038/s41467-024-00003-3",
                "journal": "Nature Communications",
                "year": 2024,
                "article_url": (
                    "https://www.nature.com/articles/s41467-024-00003-3"
                ),
                "license_candidates": [
                    {
                        "URL": (
                            "https://creativecommons.org/licenses/by-nc/4.0/"
                        )
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = workdir / "content"
    cmd_postfetch(
        argparse.Namespace(
            require_cc_by=True,
            jsonl=str(records),
            out=str(output),
            sort="input",
            processed_file=None,
            max_articles=0,
        )
    )
    rejection = json.loads(
        (output / "_license_rejections.jsonl").read_text(encoding="utf-8")
    )
    assert rejection["download_status"] == "rejected"
    assert "license-disallowed-variant:by-nc" in rejection["rejection_reason"]

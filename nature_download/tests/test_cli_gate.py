from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import nature_download.nature_all_in_one as module
from nature_download.corpus.cli import (
    LicenseGateError,
    authorize_direct_download,
    cmd_propose_cases,
)
from nature_download.corpus.proposals import PROPOSAL_RULE_V4
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
    cases = parser.parse_args(
        [
            "build-cases",
            "--corpus-manifest",
            "manifest.jsonl",
            "--content-root",
            "content",
            "--out",
            "cases",
        ]
    )
    proposals = parser.parse_args(
        [
            "propose-cases",
            "--candidates",
            "candidates.jsonl",
            "--out",
            "proposals",
        ]
    )
    reviews = parser.parse_args(
        [
            "review-proposals",
            "--proposed",
            "proposed.jsonl",
            "--out",
            "reviews",
            "--judge-model",
            "judge-a",
            "--judge-model",
            "judge-b",
        ]
    )
    benchmark = parser.parse_args(
        [
            "assemble-benchmark",
            "--candidates",
            "batch-a.jsonl",
            "--candidates",
            "batch-b.jsonl",
            "--evidence",
            "evidence.json",
            "--proposed",
            "proposed.jsonl",
            "--reviews",
            "reviews.jsonl",
            "--seed",
            "17",
            "--out",
            "benchmark",
        ]
    )
    derived = parser.parse_args(
        [
            "derive-multi-proposals",
            "--proposed",
            "proposed.jsonl",
            "--reviews",
            "reviews.jsonl",
            "--evidence",
            "evidence.json",
            "--out",
            "derived",
        ]
    )
    assert discover.require_cc_by is True
    assert validate.require_cc_by is True
    assert manifest.require_cc_by is True
    assert cases.max_xlsx_sheets == 256
    assert proposals.max_rows == 100_000
    assert proposals.max_columns == 64
    assert reviews.judge_model == ["judge-a", "judge-b"]
    assert reviews.resume is False
    assert reviews.allow_dirty is False
    assert benchmark.candidates == ["batch-a.jsonl", "batch-b.jsonl"]
    assert benchmark.proposed == ["proposed.jsonl"]
    assert benchmark.reviews == ["reviews.jsonl"]
    assert benchmark.evidence == ["evidence.json"]
    assert benchmark.train_ratio == 0.8
    assert benchmark.seed == 17
    assert derived.proposed == ["proposed.jsonl"]
    assert derived.reviews == ["reviews.jsonl"]
    assert derived.evidence == ["evidence.json"]


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


def test_unreachable_article_returns_zero_not_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(module, "polite_get", unavailable)
    assert (
        module.postfetch_one(
            "https://www.nature.com/articles/s41467-025-00000-0",
            "unused",
            3,
            0,
            1,
            1,
        )
        == 0
    )


def test_source_metadata_keeps_distinct_urls_with_the_same_label(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
) -> None:
    article_url = "https://www.nature.com/articles/s41467-019-00001-1"
    html = """
        <a href="https://static.example/source-a.xlsx">Source Data</a>
        <a href="https://static.example/source-b.xlsx">Source Data</a>
    """

    class Response:
        text = html
        url = article_url

    monkeypatch.setattr(module, "polite_get", lambda *args, **kwargs: Response())

    def fake_download(url, out_path, **kwargs):
        out_path.write_bytes(url.encode("utf-8"))
        return str(out_path), None, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    monkeypatch.setattr(module, "download_binary", fake_download)

    assert module.cmd_source(
        argparse.Namespace(
            url=article_url,
            out=str(workdir),
            section_id=None,
            filter=None,
            sleep=0,
            timeout=1,
            max_retries=1,
            _license_prevalidated=True,
            _defer_payload_gate=True,
        )
    )
    metadata = json.loads(
        (
            workdir
            / "s41467-019-00001-1"
            / "meta"
            / "source_data.json"
        ).read_text(encoding="utf-8")
    )
    assert [entry["url"] for entry in metadata] == [
        "https://static.example/source-a.xlsx",
        "https://static.example/source-b.xlsx",
    ]


def test_nature_payload_gate_quarantines_figure_only_article(workdir: Path) -> None:
    article = workdir / "s41467-019-00002-2"
    figures = article / "figures"
    metadata = article / "meta"
    figures.mkdir(parents=True)
    metadata.mkdir()
    (figures / "fig_001.png").write_bytes(b"png")
    (metadata / "figures.json").write_text("[]", encoding="utf-8")

    assert module.complete_or_quarantine_article(article) is False
    assert not article.exists()
    assert (
        workdir
        / "_rejected_no_source"
        / "s41467-019-00002-2"
        / "figures"
        / "fig_001.png"
    ).is_file()


def test_source_command_preserves_already_complete_article(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
) -> None:
    article = workdir / "s41467-019-00003-3"
    figures = article / "figures"
    source_data = article / "source_data"
    metadata = article / "meta"
    figures.mkdir(parents=True)
    source_data.mkdir()
    metadata.mkdir()
    (figures / "fig_001.png").write_bytes(b"png")
    source = source_data / "source.xlsx"
    source.write_bytes(b"source")
    (metadata / "figures.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "polite_get",
        lambda *args, **kwargs: pytest.fail("complete article must not be re-fetched"),
    )

    assert module.cmd_source(
        argparse.Namespace(
            url="https://www.nature.com/articles/s41467-019-00003-3",
            out=str(workdir),
            section_id=None,
            filter=None,
            sleep=0,
            timeout=1,
            max_retries=1,
            _license_prevalidated=True,
        )
    )
    assert source.is_file()
    assert article.is_dir()


def test_proposal_output_cannot_overwrite_candidate_directory(
    workdir: Path,
) -> None:
    candidates = workdir / "candidates.jsonl"
    candidates.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        cmd_propose_cases(
            argparse.Namespace(
                candidates=str(candidates),
                out=str(workdir),
                max_file_bytes=1,
                max_rows=1,
                max_columns=2,
            )
        )


def test_propose_cli_uses_v4_rule(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_propose_cases(**kwargs):
        captured.update(kwargs)
        return [], [], {
            "single_proposals": 0,
            "multi_panel_proposals": 0,
            "rejected": 0,
        }

    monkeypatch.setattr(
        "nature_download.corpus.cli.propose_cases",
        fake_propose_cases,
    )
    monkeypatch.setattr(
        "nature_download.corpus.cli.write_proposal_outputs",
        lambda *args: None,
    )
    cmd_propose_cases(
        argparse.Namespace(
            candidates=str(workdir / "candidates.jsonl"),
            out=str(workdir / "proposals"),
            max_file_bytes=1,
            max_rows=1,
            max_columns=2,
        )
    )
    assert captured["rule_version"] == PROPOSAL_RULE_V4

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest

import experiments.c2_full_replacement_finalizer as full_replacement
import experiments.c2_remediation_root_finalizer as remediation
import experiments.c2_terminal_finalizer as terminal
import experiments.cli as cli
from experiments.c2_m1_trust_boundary import (
    M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE,
    M1ExternalTrustLockUnavailable,
)


class _ExplodingCallerPath:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"M1 gate accessed caller-controlled value via {name}")

    def __fspath__(self) -> str:
        raise AssertionError("M1 gate accessed caller-controlled filesystem path")


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".c2_m1_trust_boundary_test_work"
    path = parent / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, mode=0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _assert_unavailable(call: Callable[[], object]) -> None:
    with pytest.raises(M1ExternalTrustLockUnavailable) as raised:
        call()
    assert raised.value.code == M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE
    assert str(raised.value) == M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE


def test_terminal_public_apis_deny_before_caller_path_access_or_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("terminal-library") as workspace:
        caller_path = _ExplodingCallerPath()
        output_path = workspace / "must-not-exist" / "final-report.json"

        monkeypatch.setattr(
            terminal,
            "_prepare_finalization_for_testing",
            lambda _manifest: pytest.fail("terminal core was reached"),
        )
        monkeypatch.setattr(
            terminal,
            "_write_final_report_for_testing",
            lambda _finalized, _output: pytest.fail("terminal writer was reached"),
        )

        _assert_unavailable(lambda: terminal.prepare_finalization(caller_path))  # type: ignore[arg-type]
        _assert_unavailable(lambda: terminal.finalize_manifest(caller_path))  # type: ignore[arg-type]
        _assert_unavailable(
            lambda: terminal.write_final_report(caller_path, output_path)  # type: ignore[arg-type]
        )
        _assert_unavailable(
            lambda: terminal.finalize_to_path(caller_path, output_path)  # type: ignore[arg-type]
        )

        assert not output_path.parent.exists()


def test_remediation_public_api_denies_before_paths_or_source_bearing_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("remediation-library") as workspace:
        caller_path = _ExplodingCallerPath()
        target_root = workspace / "must-not-exist" / "remediation-root"

        monkeypatch.setattr(
            remediation,
            "_finalize_remediation_root_for_testing",
            lambda **_kwargs: pytest.fail("remediation core was reached"),
        )

        _assert_unavailable(
            lambda: remediation.finalize_remediation_root(
                chunk_id="001",
                raw_root=caller_path,  # type: ignore[arg-type]
                target_root=target_root,
                source_chunk=caller_path,  # type: ignore[arg-type]
                frozen_universe=caller_path,  # type: ignore[arg-type]
                freeze_summary=caller_path,  # type: ignore[arg-type]
                worktree=caller_path,  # type: ignore[arg-type]
                source_bearing_v2=False,
            )
        )

        assert not target_root.parent.exists()


def test_full_replacement_public_apis_deny_before_paths_or_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("full-replacement-library") as workspace:
        caller_path = _ExplodingCallerPath()
        output_path = workspace / "must-not-exist" / "final-report.json"

        monkeypatch.setattr(
            full_replacement,
            "load_production_policy",
            lambda: pytest.fail("production policy resolver was reached"),
        )
        assert not hasattr(
            full_replacement,
            "finalize_synthetic_to_path_for_testing",
        )
        with pytest.raises(ImportError):
            from experiments.c2_full_replacement_finalizer import (
                finalize_synthetic_to_path_for_testing,
            )

        _assert_unavailable(
            lambda: full_replacement.prepare_full_replacement_finalization(
                caller_path  # type: ignore[arg-type]
            )
        )
        _assert_unavailable(
            lambda: full_replacement.write_full_replacement_report(
                caller_path,  # type: ignore[arg-type]
                output_path,
            )
        )
        _assert_unavailable(
            lambda: full_replacement.finalize_to_path(
                caller_path,  # type: ignore[arg-type]
                output_path,
            )
        )

        assert not output_path.parent.exists()


def test_terminal_cli_denies_before_parsing_or_dispatching_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("terminal-cli") as workspace:
        output_path = workspace / "must-not-exist" / "final-report.json"

        monkeypatch.setattr(
            cli,
            "finalize_c2_to_path",
            lambda *_args: pytest.fail("terminal CLI dispatched finalization"),
        )
        assert (
            cli.main(
                [
                    "c2-terminal-finalize",
                    "does-not-exist-manifest.json",
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE in captured.err
        assert not output_path.parent.exists()

        assert (
            terminal.main(
                [
                    "does-not-exist-manifest.json",
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE in captured.err
        assert not output_path.parent.exists()


@pytest.mark.parametrize(
    "handler",
    (
        cli._c2_terminal_finalize_command,
        cli._c2_remediation_root_finalize_command,
        cli._c2_full_replacement_finalize_command,
    ),
)
def test_c2_cli_handlers_deny_before_accessing_the_parsed_namespace(
    handler: Callable[[object], int],
) -> None:
    _assert_unavailable(lambda: handler(_ExplodingCallerPath()))


def test_remediation_cli_denies_before_dispatching_default_source_bearing_false(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("remediation-cli") as workspace:
        target_root = workspace / "must-not-exist" / "remediation-root"

        monkeypatch.setattr(
            cli,
            "finalize_remediation_root",
            lambda **_kwargs: pytest.fail("remediation CLI dispatched finalization"),
        )
        assert (
            cli.main(
                [
                    "c2-remediation-root-finalize",
                    "001",
                    "--raw-root",
                    "does-not-exist-raw",
                    "--target-root",
                    str(target_root),
                    "--source-chunk",
                    "does-not-exist-source",
                    "--frozen-universe",
                    "does-not-exist-universe",
                    "--freeze-summary",
                    "does-not-exist-summary",
                    "--worktree",
                    "does-not-exist-worktree",
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE in captured.err
        assert not target_root.parent.exists()


def test_full_replacement_cli_denies_before_dispatching_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("full-replacement-cli") as workspace:
        output_path = workspace / "must-not-exist" / "final-report.json"

        monkeypatch.setattr(
            cli,
            "finalize_c2_full_replacement_v2_to_path",
            lambda *_args: pytest.fail("full-replacement CLI dispatched finalization"),
        )
        assert (
            cli.main(
                [
                    "c2-full-replacement-finalize",
                    "does-not-exist-manifest.json",
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE in captured.err
        assert not output_path.parent.exists()

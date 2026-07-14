from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest

import experiments.c2_full_replacement_evidence as full_replacement_evidence
import experiments.c2_full_replacement_finalizer as full_replacement
import experiments.c2_full_replacement_policy as full_replacement_policy
import experiments.c2_remediation_root_finalizer as remediation
import experiments.c2_remediation_preflight as preflight
import experiments.c2_source_bearing_extension as source_extension
import experiments.c2_terminal_finalizer as terminal
import experiments.cli as cli
from experiments.c2_m1_trust_boundary import (
    M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE,
    M1ExternalTrustLockUnavailable,
)


class _ExplodingCallerPath:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"M1 gate accessed caller-controlled value via {name}")

    def __hash__(self) -> int:
        raise AssertionError("M1 gate used a caller-controlled cache key")

    def __fspath__(self) -> str:
        raise AssertionError("M1 gate accessed caller-controlled filesystem path")


class _ExplodingCallerReport(dict[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"M1 gate inspected caller-controlled report key {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("M1 gate iterated a caller-controlled report")

    def items(self) -> object:
        raise AssertionError("M1 gate inspected caller-controlled report items")

    def get(self, key: str, default: object = None) -> object:
        raise AssertionError(f"M1 gate inspected caller-controlled report key {key}")


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


def test_terminal_public_apis_deny_before_caller_path_access_or_output_creation() -> None:
    with _workspace("terminal-library") as workspace:
        caller_path = _ExplodingCallerPath()
        output_path = workspace / "must-not-exist" / "final-report.json"

        assert not any(
            hasattr(terminal, name)
            for name in (
                "_prepare_finalization_for_testing",
                "_finalize_manifest_for_testing",
                "_write_final_report_for_testing",
                "_finalize_to_path_for_testing",
            )
        )

        _assert_unavailable(lambda: terminal.prepare_finalization(caller_path))  # type: ignore[arg-type]
        _assert_unavailable(lambda: terminal.finalize_manifest(caller_path))  # type: ignore[arg-type]
        _assert_unavailable(
            lambda: terminal._read_json_object(caller_path, "sentinel")  # type: ignore[arg-type]
        )
        _assert_unavailable(
            lambda: terminal.write_final_report(caller_path, output_path)  # type: ignore[arg-type]
        )
        _assert_unavailable(
            lambda: terminal.finalize_to_path(caller_path, output_path)  # type: ignore[arg-type]
        )

        assert not output_path.parent.exists()


def test_remediation_public_api_denies_before_paths_or_source_bearing_selector() -> None:
    with _workspace("remediation-library") as workspace:
        caller_path = _ExplodingCallerPath()
        target_root = workspace / "must-not-exist" / "remediation-root"

        assert not hasattr(
            remediation,
            "_finalize_remediation_root_for_testing",
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
        _assert_unavailable(
            lambda: remediation._RawRootReader(caller_path)  # type: ignore[arg-type]
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
        assert not any(
            hasattr(full_replacement, name)
            for name in (
                "prepare_full_replacement_finalization_for_testing",
                "write_full_replacement_report_for_testing",
            )
        )
        with pytest.raises(ImportError):
            from experiments.c2_full_replacement_finalizer import (
                finalize_synthetic_to_path_for_testing,
            )
        with pytest.raises(ModuleNotFoundError):
            import tests._c2_full_replacement_test_support  # type: ignore[import-not-found]

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


def test_full_replacement_evidence_and_source_extension_paths_deny_before_inputs() -> None:
    caller_path = _ExplodingCallerPath()

    _assert_unavailable(
        lambda: full_replacement_evidence.load_and_validate_raw_evidence(
            caller_path,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )
    )
    _assert_unavailable(
        lambda: source_extension.verify_source_extension_code_attestation_for_testing(
            caller_path  # type: ignore[arg-type]
        )
    )
    _assert_unavailable(
        lambda: source_extension.validate_source_bearing_extension(
            caller_path,  # type: ignore[arg-type]
            partition_records=1,
            source_chunk_sha256="0" * 64,
        )
    )
    _assert_unavailable(
        lambda: source_extension.build_source_bearing_extension(
            root=caller_path,  # type: ignore[arg-type]
            raw_reader=caller_path,
            records=caller_path,
            provenance=caller_path,
            terminal_rows=caller_path,
            partition_records=1,
            source_chunk_sha256="0" * 64,
        )
    )
    _assert_unavailable(
        lambda: source_extension.validate_source_bearing_extension_for_testing(
            caller_path,  # type: ignore[arg-type]
            partition_records=1,
            source_chunk_sha256="0" * 64,
        )
    )
    _assert_unavailable(
        lambda: source_extension.build_source_bearing_extension_for_testing(
            root=caller_path,  # type: ignore[arg-type]
            raw_reader=caller_path,
            records=caller_path,
            provenance=caller_path,
            terminal_rows=caller_path,
            partition_records=1,
            source_chunk_sha256="0" * 64,
        )
    )


def test_validation_entrypoints_deny_before_caller_controlled_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_path = _ExplodingCallerPath()
    caller_report = _ExplodingCallerReport()

    _assert_unavailable(lambda: terminal.validate_final_report(caller_report))
    _assert_unavailable(lambda: terminal._validate_final_report(caller_report))
    _assert_unavailable(
        lambda: terminal._validate_schema(
            caller_report,
            "c2_terminal_final_report.schema.json",
            "sentinel",
        )
    )
    _assert_unavailable(
        lambda: full_replacement.validate_synthetic_final_report_for_testing(
            caller_report,
            caller_path,  # type: ignore[arg-type]
        )
    )
    _assert_unavailable(
        lambda: full_replacement_evidence._validate_schema(
            caller_report,
            "c2_full_replacement_final_report_v2.schema.json",
            "sentinel",
        )
    )
    _assert_unavailable(
        lambda: full_replacement_evidence._validate_trusted_evidence_metadata(
            caller_path,  # type: ignore[arg-type]
            caller_path,  # type: ignore[arg-type]
            caller_path,  # type: ignore[arg-type]
            "sentinel",
        )
    )
    _assert_unavailable(
        lambda: full_replacement._build_test_report(  # type: ignore[arg-type]
            caller_path,
            caller_path,
        )
    )
    _assert_unavailable(
        lambda: full_replacement._build_validated_test_report(  # type: ignore[arg-type]
            caller_path,
            caller_path,
        )
    )
    _assert_unavailable(
        lambda: full_replacement._validate_ledger_structure(  # type: ignore[arg-type]
            caller_path,
            caller_path,
        )
    )
    admission = full_replacement.ValidatedFullReplacementAdmission(
        policy=caller_path,  # type: ignore[arg-type]
        evidence=caller_path,  # type: ignore[arg-type]
    )
    _assert_unavailable(lambda: admission.report)
    _assert_unavailable(
        lambda: full_replacement_policy.compile_synthetic_policy_for_testing(
            caller_report
        )
    )
    _assert_unavailable(
        lambda: source_extension.validate_source_evidence_descriptor_v2(
            caller_report,
            article_id=caller_path,  # type: ignore[arg-type]
            doi_id=caller_path,  # type: ignore[arg-type]
            provenance_relative_path=caller_path,  # type: ignore[arg-type]
        )
    )

    def _unexpected_fstat(*_args: object, **_kwargs: object) -> object:
        pytest.fail("M1 gate accessed a caller-controlled evidence descriptor")

    with monkeypatch.context() as patched:
        patched.setattr(full_replacement_evidence.os, "fstat", _unexpected_fstat)
        _assert_unavailable(
            lambda: full_replacement_evidence.validate_trusted_directory_descriptor(
                caller_path,  # type: ignore[arg-type]
                caller_path,  # type: ignore[arg-type]
            )
        )
        _assert_unavailable(
            lambda: full_replacement_evidence.validate_trusted_regular_file_descriptor(
                caller_path,  # type: ignore[arg-type]
                caller_path,  # type: ignore[arg-type]
            )
        )


def test_cached_schema_validation_denies_after_m1_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patched:
        patched.setattr(
            terminal,
            "require_external_m1_trust_lock",
            lambda: None,
        )
        terminal._schema_validator("c2_terminal_final_report.schema.json")
    with monkeypatch.context() as patched:
        patched.setattr(
            full_replacement_evidence,
            "require_external_m1_trust_lock",
            lambda: None,
        )
        full_replacement_evidence._schema_validator(
            "c2_full_replacement_final_report_v2.schema.json"
        )

    caller_path = _ExplodingCallerPath()
    caller_report = _ExplodingCallerReport()
    _assert_unavailable(
        lambda: terminal._schema_validator(caller_path)  # type: ignore[arg-type]
    )
    _assert_unavailable(lambda: terminal.validate_final_report(caller_report))
    _assert_unavailable(lambda: terminal._validate_final_report(caller_report))
    _assert_unavailable(
        lambda: terminal._validate_schema(
            caller_report,
            "c2_terminal_final_report.schema.json",
            "sentinel",
        )
    )
    _assert_unavailable(
        lambda: full_replacement_evidence._schema_validator(
            caller_path  # type: ignore[arg-type]
        )
    )
    _assert_unavailable(
        lambda: full_replacement_evidence._validate_schema(
            caller_report,
            "c2_full_replacement_final_report_v2.schema.json",
            "sentinel",
        )
    )


def test_preflight_library_paths_deny_before_plan_or_evidence_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_value = _ExplodingCallerPath()

    def _unexpected_access(*_args: object, **_kwargs: object) -> object:
        pytest.fail("preflight accessed a caller-controlled plan or filesystem path")

    with monkeypatch.context() as patched:
        patched.setattr(preflight, "_parse_plan", _unexpected_access)
        patched.setattr(preflight, "open_trusted_directory", _unexpected_access)
        patched.setattr(preflight.os, "open", _unexpected_access)
        patched.setattr(preflight.Path, "read_text", _unexpected_access)

        _assert_unavailable(
            lambda: preflight.run_preflight(caller_value)  # type: ignore[arg-type]
        )
        _assert_unavailable(
            lambda: preflight.run_preflight_for_testing(
                caller_value,  # type: ignore[arg-type]
                test_bindings=caller_value,  # type: ignore[arg-type]
            )
        )
        _assert_unavailable(
            lambda: preflight._load_plan_from_path(caller_value)  # type: ignore[arg-type]
        )


def test_preflight_module_main_and_cli_deny_before_parsing_or_opening_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("preflight-cli") as workspace:
        plan_path = workspace / "must-not-exist" / "plan.json"

        def _unexpected_access(*_args: object, **_kwargs: object) -> object:
            pytest.fail(
                "preflight CLI accessed a caller-controlled plan or filesystem path"
            )

        with monkeypatch.context() as patched:
            patched.setattr(preflight, "_build_parser", _unexpected_access)
            patched.setattr(preflight, "_parse_plan", _unexpected_access)
            patched.setattr(
                preflight,
                "open_trusted_directory",
                _unexpected_access,
            )
            patched.setattr(preflight.os, "open", _unexpected_access)
            patched.setattr(preflight.Path, "read_text", _unexpected_access)

            _assert_unavailable(
                lambda: preflight.main(_ExplodingCallerPath())  # type: ignore[arg-type]
            )
            assert not plan_path.parent.exists()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.c2_remediation_preflight",
                str(plan_path),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE in result.stderr
        assert not plan_path.parent.exists()


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

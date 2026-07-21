from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from openpyxl import Workbook
import pytest
import yaml

from experiments.baseline_specs.build_matplotagent_v2 import (
    MaterializationError,
    materialize_matplotagent_v2,
    normalization_code_hash,
    normalize_source_to_csv,
)
from experiments.external_baselines import MatPlotAgentProvider
from experiments.manifest import load_dataset_manifest, select_case
from experiments.matrix import load_and_expand_matrix
from experiments.models import sha256_file
from experiments.providers import GenerationRequest
from tests.test_experiment_support import experiment_workspace


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
PARENT = (
    REPO_ROOT
    / "nature_download"
    / "outputs"
    / "combined_verified_400"
    / "final_benchmark_renderable_v1_seed0"
    / "benchmark_manifest.json"
)
PARENT_SHA256 = (
    "6c6c5cb50603d899a9be0f41c9e562517c92ae15477610e9111f6504cd1757b3"
)
SPEC = (
    REPO_ROOT
    / "agent"
    / "experiments"
    / "baseline_specs"
    / "matplotagent-single-test-normalized-v2.json"
)
MATRIX_TEMPLATE = (
    REPO_ROOT
    / "agent"
    / "experiments"
    / "matrices"
    / "baseline_matplotagent_single_normalized_v2.yaml"
)
V1_SPEC = (
    REPO_ROOT
    / "agent"
    / "experiments"
    / "baseline_specs"
    / "matplotagent-single-test-renderable-v1.json"
)
V1_MATRIX = (
    REPO_ROOT
    / "agent"
    / "experiments"
    / "matrices"
    / "baseline_matplotagent_single_renderable_v1.yaml"
)
V1_SPEC_SHA256 = (
    "6acde4f8f6e99d9eebf06dc009bff9bb58d39124a1e7e4c661118c8f3f21b394"
)
V1_MATRIX_SHA256 = (
    "6e2542f402b49229851a91958ea1ae137316d8c0549b4a025651172de611ce2b"
)
OUTCOME_FIELDS = {
    "failed_case_ids",
    "method_results",
    "method_scores",
    "production_artifacts",
    "review_outcomes",
    "run_records",
}


@pytest.fixture
def workdir() -> Path:
    with experiment_workspace("matplotagent-v2-normalizer") as workspace:
        yield workspace


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _compatible(case: dict) -> bool:
    expectation = case.get("evaluation_expectation")
    return (
        case.get("split") == "test"
        and case.get("panel_count") == 1
        and isinstance(case.get("data_path"), str)
        and Path(case["data_path"]).suffix.casefold()
        in {".csv", ".xlsx", ".xlsm"}
        and isinstance(case.get("user_goal"), str)
        and bool(case["user_goal"].strip())
        and isinstance(expectation, dict)
        and expectation.get("panel_groups") == []
    )


def _write_workbook(
    path: Path,
    *,
    sheet: str = "Data",
    mixed: bool = False,
    formula: bool = False,
) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    worksheet.append(["Category", "Value"])
    worksheet.append(["A", 1])
    worksheet.append(["B", "mixed" if mixed else 2])
    if formula:
        worksheet["B3"] = "=1+1"
    workbook.save(path)
    workbook.close()


def test_v2_selection_is_metadata_only_and_covers_six_dois() -> None:
    parent_bytes = PARENT.read_bytes()
    parent = json.loads(parent_bytes)
    spec = _load(SPEC)
    selected = [case for case in parent["cases"] if _compatible(case)]

    assert hashlib.sha256(parent_bytes).hexdigest() == PARENT_SHA256
    assert len(selected) == 16
    assert len({case["doi"] for case in selected}) == 6
    assert spec["selection"]["selected_case_ids"] == [
        case["case_id"] for case in selected
    ]
    assert set(spec["selection"]["forbidden_inputs"]) == OUTCOME_FIELDS

    mutated = deepcopy(parent["cases"])
    for index, case in enumerate(mutated):
        case["run_records"] = [{"status": "failed"}]
        case["method_scores"] = {"score": index}
        case["production_artifacts"] = [f"artifact-{index}"]
    assert [case["case_id"] for case in mutated if _compatible(case)] == [
        case["case_id"] for case in selected
    ]
    assert PARENT.read_bytes() == parent_bytes


def test_normalizer_is_deterministic_and_fails_closed(workdir: Path) -> None:
    workbook = workdir / "table.xlsx"
    _write_workbook(workbook)
    first = workdir / "first.csv"
    second = workdir / "second.csv"
    expected_hash = sha256_file(workbook)

    first_audit = normalize_source_to_csv(
        source_path=workbook,
        source_sha256=expected_hash,
        sheet="Data",
        destination=first,
    )
    second_audit = normalize_source_to_csv(
        source_path=workbook,
        source_sha256=expected_hash,
        sheet="Data",
        destination=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_audit["normalized_csv_sha256"] == sha256_file(first)
    assert first_audit["normalized_csv_sha256"] == second_audit[
        "normalized_csv_sha256"
    ]
    assert first_audit["column_types"] == ["string", "numeric"]

    with pytest.raises(MaterializationError, match="explicit source sheet"):
        normalize_source_to_csv(
            source_path=workbook,
            source_sha256=expected_hash,
            sheet=None,
            destination=workdir / "missing-sheet.csv",
        )
    with pytest.raises(MaterializationError, match="missing or ambiguous"):
        normalize_source_to_csv(
            source_path=workbook,
            source_sha256=expected_hash,
            sheet="Wrong",
            destination=workdir / "wrong-sheet.csv",
        )
    with pytest.raises(MaterializationError, match="SHA-256 mismatch"):
        normalize_source_to_csv(
            source_path=workbook,
            source_sha256="0" * 64,
            sheet="Data",
            destination=workdir / "wrong-hash.csv",
        )
    with pytest.raises(MaterializationError, match="missing or a symlink"):
        normalize_source_to_csv(
            source_path=workdir / "absent.xlsx",
            source_sha256="0" * 64,
            sheet="Data",
            destination=workdir / "absent.csv",
        )


def test_normalizer_rejects_mixed_dtypes_and_uncached_formulas(
    workdir: Path,
) -> None:
    mixed = workdir / "mixed.xlsx"
    formula = workdir / "formula.xlsx"
    _write_workbook(mixed, mixed=True)
    _write_workbook(formula, formula=True)

    with pytest.raises(MaterializationError, match="ambiguous dtypes"):
        normalize_source_to_csv(
            source_path=mixed,
            source_sha256=sha256_file(mixed),
            sheet="Data",
            destination=workdir / "mixed.csv",
        )
    with pytest.raises(MaterializationError, match="uncached formula"):
        normalize_source_to_csv(
            source_path=formula,
            source_sha256=sha256_file(formula),
            sheet="Data",
            destination=workdir / "formula.csv",
        )


def test_materialized_v2_binds_normalized_inputs_for_both_methods() -> None:
    parent = _load(PARENT)
    parent_by_id = {case["case_id"]: case for case in parent["cases"]}
    with experiment_workspace("matplotagent-v2-materialized") as workspace:
        summary = materialize_matplotagent_v2(
            repo_root=REPO_ROOT,
            workspace_root=WORKSPACE_ROOT,
            output_root=workspace / "runtime",
        )
        manifest_path = Path(summary["manifest"])
        manifest = _load(manifest_path)
        matrix_path = Path(summary["matrix"])
        specs = load_and_expand_matrix(matrix_path)

        assert summary["model_calls"] == 0
        assert summary["cases"] == 16
        assert summary["unique_dois"] == 6
        assert summary["normalization_code_sha256"] == normalization_code_hash()
        assert len(manifest["cases"]) == 16
        assert len(specs) == 16 * 2 * 3
        assert {spec.panel_count for spec in specs} == {1}
        assert {spec.seed for spec in specs} == {0, 1, 2}
        assert {spec.budget_value for spec in specs} == {1.0}
        assert {
            spec.method for spec in specs
        } == {
            "matplotagent_native_normalized_v2",
            "pheroviz_model_spec_normalized_v2",
        }

        binding_by_id = {
            item["case_id"]: item
            for item in manifest["provenance"]["compatibility_derivation"][
                "case_bindings"
            ]
        }
        for case in manifest["cases"]:
            parent_case = parent_by_id[case["case_id"]]
            binding = binding_by_id[case["case_id"]]
            assert case["input_track"] == "table_instruction"
            assert case["sheet"] is None
            assert Path(case["data_path"]).suffix == ".csv"
            assert sha256_file(Path(case["data_path"])) == case["data_sha256"]
            assert binding["parent_source_sha256"] == parent_case["data_sha256"]
            assert binding["parent_source_sheet"] == parent_case.get("sheet")
            assert binding["normalized_csv_sha256"] == case["data_sha256"]
            assert binding["instruction_sha256"] == hashlib.sha256(
                json.dumps(
                    parent_case["user_goal"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            assert binding["expectation_sha256"] == hashlib.sha256(
                json.dumps(
                    parent_case["evaluation_expectation"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            assert case["user_goal"] == parent_case["user_goal"]
            assert case["evaluation_expectation"] == parent_case[
                "evaluation_expectation"
            ]

        grouped: dict[tuple[str, int], list] = {}
        for item in specs:
            grouped.setdefault((item.case_id, item.seed), []).append(item)
        assert len(grouped) == 16 * 3
        for pair in grouped.values():
            assert len(pair) == 2
            assert len({item.dataset_manifest_hash for item in pair}) == 1
            assert len({item.case_id for item in pair}) == 1
            assert len({item.backbone for item in pair}) == 1
            assert len({item.budget_value for item in pair}) == 1

        loaded = load_dataset_manifest(
            manifest_path,
            dataset_mode="sealed_benchmark",
        )
        assert len(loaded) == 16
        assert all(case.panel_count == 1 and case.split == "test" for case in loaded)

        external_spec = next(
            item
            for item in specs
            if item.method == "matplotagent_native_normalized_v2"
        )
        selected_case = select_case(loaded, external_spec.case_id)
        adapter_output = workspace / "adapter"
        adapter_output.mkdir()
        request = GenerationRequest(
            spec=external_spec,
            dataset_manifest_path=manifest_path,
            output_dir=adapter_output,
            call_index=1,
            remaining_renders=1,
            remaining_seconds=None,
            deadline_monotonic=None,
            history=(),
            previous_candidate=None,
        )
        provider = MatPlotAgentProvider(
            repo_path=WORKSPACE_ROOT / "repo" / "baseline_repos" / "MatPlotAgent",
            python_executable=str(
                WORKSPACE_ROOT / "venvs" / "matplotagent-linux" / "bin" / "python"
            ),
            check_dependencies=False,
            environ={
                "MATPLOTAGENT_API_KEY": "test-only",
                "MATPLOTAGENT_BASE_URL": "https://example.invalid/v1",
            },
        )
        invocation = provider._prepare_invocation(request, selected_case)
        adapter_config = json.loads(
            (adapter_output / "baseline_input.json").read_text(encoding="utf-8")
        )
        assert invocation.result_manifest == adapter_output / "driver_result.json"
        assert sha256_file(adapter_output / "workspace" / "data.csv") == (
            selected_case.payload["data_sha256"]
        )
        assert sha256_file(Path(adapter_config["table_files"][0])) == (
            selected_case.payload["data_sha256"]
        )


def test_v2_template_is_portable_disjoint_and_secret_free() -> None:
    spec = _load(SPEC)
    matrix = yaml.safe_load(MATRIX_TEMPLATE.read_text(encoding="utf-8"))
    paths = [SPEC, MATRIX_TEMPLATE, Path(spec["builder_contract"]["builder_path"])]
    for path in paths:
        path = path if path.is_absolute() else REPO_ROOT / path
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "tommy" not in text.casefold()
        assert "sk-" not in text

    assert spec["license_status"] == "not_declared"
    assert spec["public_interface"]["cohesion_metric"] == "NA"
    assert spec["chartcoder"]["status"] == "blocked"
    assert not spec["chartcoder"]["result_row"]
    assert spec["v1_isolation"]["overwrite_or_merge"] is False
    assert matrix["compatibility_subtrack"]["selected_cases"] == 16
    assert matrix["compatibility_subtrack"]["selected_dois"] == 6
    assert matrix["compatibility_subtrack"]["cohesion"] == "NA"
    assert matrix["compatibility_subtrack"]["v1_artifact_root_untouched"]
    assert matrix["portable_template"]["artifact_root_name"] != (
        spec["v1_isolation"]["completed_v1_artifact_root_name"]
    )
    assert sha256_file(REPO_ROOT / spec["builder_contract"]["builder_path"]) == (
        spec["builder_contract"]["builder_sha256"]
    )
    assert normalization_code_hash() == spec["normalization"]["code_sha256"]
    configured_root = (
        REPO_ROOT
        / "agent"
        / "experiments"
        / "runs"
        / "production"
        / matrix["portable_template"]["artifact_root_name"]
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(configured_root)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert ignored.returncode == 0


def test_v1_specs_are_unchanged() -> None:
    assert sha256_file(V1_SPEC) == V1_SPEC_SHA256
    assert sha256_file(V1_MATRIX) == V1_MATRIX_SHA256

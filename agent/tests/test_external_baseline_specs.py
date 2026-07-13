from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from experiments.baseline_specs.build_portable_subtracks import (
    materialize_subtracks,
)
from experiments.manifest import load_dataset_manifest
from experiments.matrix import load_and_expand_matrix
from experiments.models import sha256_file
from tests.test_experiment_support import experiment_workspace


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "agent" / "experiments"
SPECS_ROOT = EXPERIMENTS_ROOT / "baseline_specs"
MATRICES_ROOT = EXPERIMENTS_ROOT / "matrices"
PARENT_PATH = (
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
CONFIGS = {
    "matplotagent": {
        "track": "table_instruction",
        "spec": "matplotagent-single-test-renderable-v1.json",
        "selected_count": 2,
        "allowed_suffixes": {".csv"},
        "selected_case_set_sha256": (
            "c0322341004517463f291e0151ea5e02b92296bdab6c5b7e2026654cceb7aac2"
        ),
        "selected_parent_cases_sha256": (
            "79e287eeb455a88c171c4d8f02fc53291cc74a2e88fb3dad926fc618b6b62019"
        ),
        "matrix": "baseline_matplotagent_single_renderable_v1.yaml",
        "external_method": "matplotagent_native",
        "external_provider": (
            "experiments.external_baselines:MatPlotAgentProvider"
        ),
        "repo_relative": "baseline_repos/MatPlotAgent",
        "commit": "9cafa262aae7bdf85fccf6d02b2153fb772bc376",
        "python_relative": ".baseline_envs/matplotagent/bin/python",
        "audit": "agent/experiments/baseline_audits/matplotagent.json",
        "audit_sha256": (
            "9528e845e67a26867c57d85f30e8982f715d8c23248765ba68f9e26b8038d4af"
        ),
        "dependency_modules": ["openai", "tenacity", "matplotlib"],
    },
    "nvagent": {
        "track": "table_nl_instruction",
        "spec": "nvagent-single-test-renderable-v1.json",
        "selected_count": 16,
        "allowed_suffixes": {".csv", ".tsv", ".xls", ".xlsx", ".xlsm"},
        "selected_case_set_sha256": (
            "e8726db8bdbb5c9485a5d4685f22cbf9a057bc11b686d71ccbacc54de6c16b74"
        ),
        "selected_parent_cases_sha256": (
            "80ad7b4742223a0b35c5bf8eee2eba47d5f2e58d3cfab9a7892349ff60d56b48"
        ),
        "matrix": "baseline_nvagent_single_renderable_v1.yaml",
        "external_method": "nvagent_native",
        "external_provider": "experiments.external_baselines:NvAgentProvider",
        "repo_relative": "baseline_repos/nvAgent",
        "commit": "a37209e675813a25241e83f2fe56a87657a49ef6",
        "python_relative": ".baseline_envs/nvagent/bin/python",
        "audit": "agent/experiments/baseline_audits/nvagent.json",
        "audit_sha256": (
            "6cd142519ab3a0ad1c5b33a9ca9924852e9617f57d26b23bbfe7427015961a99"
        ),
        "dependency_modules": [
            "openai",
            "duckdb",
            "pandas",
            "matplotlib",
            "seaborn",
            "sqlglot",
            "func_timeout",
            "attr",
            "tqdm",
        ],
    },
}
OUTCOME_FIELDS = {
    "failed_case_ids",
    "method_results",
    "method_scores",
    "production_artifacts",
    "review_outcomes",
    "run_records",
}


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _compatible(case: dict, allowed_suffixes: set[str]) -> bool:
    instruction = next(
        (
            case.get(name)
            for name in ("instruction", "user_goal", "nl_instruction", "query")
            if isinstance(case.get(name), str) and case[name].strip()
        ),
        None,
    )
    expectation = case.get("evaluation_expectation")
    return (
        case.get("split") == "test"
        and case.get("panel_count") == 1
        and isinstance(case.get("data_path"), str)
        and Path(case["data_path"]).suffix.lower() in allowed_suffixes
        and instruction is not None
        and isinstance(expectation, dict)
        and expectation.get("panel_groups") == []
    )


def _selected_ids(cases: list[dict], allowed_suffixes: set[str]) -> list[str]:
    return [
        case["case_id"]
        for case in cases
        if _compatible(case, allowed_suffixes)
    ]


@pytest.mark.parametrize("baseline", sorted(CONFIGS))
def test_compatibility_derivation_is_parent_bound_and_outcome_independent(
    baseline: str,
) -> None:
    config = CONFIGS[baseline]
    parent_bytes = PARENT_PATH.read_bytes()
    parent = json.loads(parent_bytes)
    spec_path = SPECS_ROOT / config["spec"]
    spec = _load_json(spec_path)
    selected = _selected_ids(parent["cases"], config["allowed_suffixes"])

    assert hashlib.sha256(parent_bytes).hexdigest() == PARENT_SHA256
    assert spec["parent_manifest"]["sha256"] == PARENT_SHA256
    assert spec["selection"]["selected_case_ids"] == selected
    assert len(selected) == config["selected_count"]
    assert _canonical_hash(sorted(selected)) == config[
        "selected_case_set_sha256"
    ]
    assert spec["selection"]["selected_case_set_sha256"] == (
        config["selected_case_set_sha256"]
    )
    assert spec["selection"]["selected_parent_cases_sha256"] == (
        config["selected_parent_cases_sha256"]
    )
    assert set(spec["selection"]["forbidden_inputs"]) == OUTCOME_FIELDS
    assert spec["selection"]["allowed_case_mutations"] == [
        "input_track",
        "runtime_root_remap",
    ]
    assert spec["selection"]["declared_field_addition"] == {
        "input_track": config["track"]
    }

    mutated = deepcopy(parent["cases"])
    for index, case in enumerate(mutated):
        case["run_records"] = [{"status": "failed" if index % 2 else "completed"}]
        case["method_scores"] = {"data_fidelity": index / 100}
        case["production_artifacts"] = [f"artifact-{index}"]
    assert _selected_ids(mutated, config["allowed_suffixes"]) == selected
    assert PARENT_PATH.read_bytes() == parent_bytes


def test_portable_builder_materializes_sealed_manifests_and_exact_inputs() -> None:
    parent = _load_json(PARENT_PATH)
    parent_by_id = {case["case_id"]: case for case in parent["cases"]}
    with experiment_workspace("portable-baseline-materialization") as workspace:
        output = workspace / "runtime"
        summary = materialize_subtracks(
            repo_root=REPO_ROOT,
            workspace_root=REPO_ROOT.parent,
            output_root=output,
        )

        assert summary["model_calls"] == 0
        for baseline, config in CONFIGS.items():
            spec_path = SPECS_ROOT / config["spec"]
            manifest_path = Path(summary["subtracks"][baseline]["manifest"])
            manifest = _load_json(manifest_path)
            absolute_values: list[Path] = []

            def collect_absolute(value: object) -> None:
                if isinstance(value, dict):
                    for child in value.values():
                        collect_absolute(child)
                elif isinstance(value, list):
                    for child in value:
                        collect_absolute(child)
                elif isinstance(value, str) and Path(value).is_absolute():
                    absolute_values.append(Path(value))

            collect_absolute(manifest)
            assert absolute_values
            assert all(path.is_relative_to(REPO_ROOT) for path in absolute_values)
            assert len(manifest["cases"]) == config["selected_count"]
            assert manifest["manifest_hash"] == _canonical_hash(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_hash"
                }
            )
            derivation = manifest["provenance"]["compatibility_derivation"]
            assert derivation["spec_sha256"] == sha256_file(spec_path)
            assert derivation["parent_manifest_sha256"] == PARENT_SHA256
            assert derivation["selected_case_set_sha256"] == config[
                "selected_case_set_sha256"
            ]
            assert derivation["runtime_repo_root"] == str(REPO_ROOT)
            assert derivation["derivation_hash"] == _canonical_hash(
                {
                    key: value
                    for key, value in derivation.items()
                    if key != "derivation_hash"
                }
            )
            bindings = {
                item["case_id"]: item
                for item in derivation["case_bindings"]
            }
            for derived in manifest["cases"]:
                original = parent_by_id[derived["case_id"]]
                assert derived["input_track"] == config["track"]
                for key in set(original) - {"data_path", "panels"}:
                    assert derived[key] == original[key]
                assert Path(derived["data_path"]).is_relative_to(REPO_ROOT)
                original_parts = Path(original["data_path"]).parts
                source_relative = Path(
                    *original_parts[original_parts.index("nature_download") :]
                )
                assert Path(derived["data_path"]).relative_to(REPO_ROOT) == (
                    source_relative
                )
                for original_panel, derived_panel in zip(
                    original["panels"],
                    derived["panels"],
                    strict=True,
                ):
                    for key in set(original_panel) - {"data_path"}:
                        assert derived_panel[key] == original_panel[key]
                    panel_parts = Path(original_panel["data_path"]).parts
                    panel_relative = Path(
                        *panel_parts[panel_parts.index("nature_download") :]
                    )
                    assert Path(derived_panel["data_path"]).relative_to(
                        REPO_ROOT
                    ) == panel_relative
                binding = bindings[derived["case_id"]]
                assert binding["parent_case_sha256"] == _canonical_hash(original)
                assert binding["runtime_case_sha256"] == _canonical_hash(derived)
                assert binding["source_sha256"] == original["data_sha256"]
                assert binding["instruction_sha256"] == _canonical_hash(
                    original["user_goal"]
                )
                assert binding["expectation_sha256"] == _canonical_hash(
                    original["evaluation_expectation"]
                )
                assert sha256_file(Path(derived["data_path"])) == (
                    original["data_sha256"]
                )

            loaded = load_dataset_manifest(
                manifest_path,
                dataset_mode="sealed_benchmark",
            )
            assert [case.case_id for case in loaded] == [
                case["case_id"] for case in manifest["cases"]
            ]
            assert all(
                case.panel_count == 1 and case.split == "test"
                for case in loaded
            )


@pytest.mark.parametrize("baseline", sorted(CONFIGS))
def test_fixed_external_runtime_and_license_limitations(baseline: str) -> None:
    config = CONFIGS[baseline]
    spec = _load_json(SPECS_ROOT / config["spec"])
    runtime = spec["external_runtime"]
    checkout = (REPO_ROOT.parent / config["repo_relative"]).resolve()
    python = (REPO_ROOT.parent / config["python_relative"]).absolute()

    assert spec["license_status"] == "not_declared"
    assert runtime["repo"] == {
        "root": "workspace_root",
        "relative_path": config["repo_relative"],
        "commit": config["commit"],
    }
    assert runtime["python"] == {
        "root": "workspace_root",
        "relative_path": config["python_relative"],
    }
    assert runtime["audit"] == {
        "path": config["audit"],
        "sha256": config["audit_sha256"],
    }
    assert sha256_file(REPO_ROOT / config["audit"]) == config["audit_sha256"]
    assert python.is_file()
    dependency_probe = subprocess.run(
        [
            str(python),
            "-c",
            ";".join(
                f"import {module}" for module in config["dependency_modules"]
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert dependency_probe.returncode == 0, dependency_probe.stderr
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual_commit == config["commit"]
    assert spec["chartcoder"]["status"] == "blocked"
    assert spec["chartcoder"]["license_status"] == "not_declared"
    builder = spec["builder_contract"]
    assert sha256_file(REPO_ROOT / builder["builder_path"]) == (
        builder["builder_sha256"]
    )
    for binding in runtime["evaluator_files"]:
        assert sha256_file(REPO_ROOT / binding["path"]) == binding["sha256"]


@pytest.mark.parametrize("baseline", sorted(CONFIGS))
def test_matrix_is_exactly_paired_single_panel_and_cohesion_na(
    baseline: str,
) -> None:
    config = CONFIGS[baseline]
    template_path = MATRICES_ROOT / config["matrix"]
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert template["portable_template"]["direct_dry_run"] is False
    assert template["dataset_manifest"] == "__MATERIALIZED_BY_BUILDER__"
    with experiment_workspace(f"portable-matrix-{baseline}") as workspace:
        summary = materialize_subtracks(
            repo_root=REPO_ROOT,
            workspace_root=REPO_ROOT.parent,
            output_root=workspace / "runtime",
        )
        matrix_path = Path(summary["subtracks"][baseline]["matrix"])
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        specs = load_and_expand_matrix(matrix_path)

        assert all(
            Path(value).is_absolute()
            for value in (
                matrix["dataset_manifest"],
                matrix["artifact_root"],
                matrix["repo_root"],
            )
        )
        assert len(specs) == config["selected_count"] * 2 * 3
        assert matrix["dataset_manifest_sha256"] == sha256_file(
            Path(matrix["dataset_manifest"])
        )
        assert matrix["backbones"] == ["gpt-4o-mini"]
        assert matrix["seeds"] == [0, 1, 2]
        assert matrix["budgets"] == [{"type": "renders", "value": 1}]
        assert matrix["splits"] == ["test"]
        assert matrix["compatibility_subtrack"]["cohesion"] == "NA"
        assert matrix["compatibility_subtrack"]["multi_panel_cases"] == 0
        assert matrix["compatibility_subtrack"]["license_status"] == "not_declared"
        assert not matrix["compatibility_subtrack"]["chartcoder_result_row"]
        assert matrix["compatibility_subtrack"]["reported_metrics"] == [
            "data_fidelity",
            "execution_success",
        ]

        methods = {spec.method for spec in specs}
        assert methods == {config["external_method"], "pheroviz_model_spec"}
        grouped: dict[tuple[str, int], list] = {}
        for item in specs:
            assert item.panel_count == 1
            assert item.split == "test"
            assert item.backbone == "gpt-4o-mini"
            assert item.budget_type == "renders" and item.budget_value == 1
            assert item.schedule == "best_of_n"
            grouped.setdefault((item.case_id, item.seed), []).append(item)
        assert len(grouped) == config["selected_count"] * 3
        assert all(
            {item.method for item in pair}
            == {config["external_method"], "pheroviz_model_spec"}
            for pair in grouped.values()
        )
        phero = [item for item in specs if item.method == "pheroviz_model_spec"]
        assert all(
            item.provider == "experiments.providers:UnifiedBenchmarkProvider"
            and item.method_config["initial_generation"] == "model_spec"
            and item.method_config["memory_mode"] == "none"
            and item.provider_options["manifest_data_root"] == str(REPO_ROOT)
            for item in phero
        )
        external = [
            item for item in specs if item.method == config["external_method"]
        ]
        assert all(
            item.provider == config["external_provider"]
            and item.provider_options["repo_path"]
            == str((REPO_ROOT.parent / config["repo_relative"]).resolve())
            and item.provider_options["python_executable"]
            == str((REPO_ROOT.parent / config["python_relative"]).absolute())
            for item in external
        )
        if baseline == "matplotagent":
            assert all(
                item.method_config["mode"] == "direct"
                and item.method_config["visual_refine"] is False
                for item in external
            )
        else:
            assert all(
                item.method_config["openai_compatible"] is True
                and item.provider_options["openai_compatible"] is True
                for item in external
            )

        configured_root = Path(matrix["artifact_root"])
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", str(configured_root)],
            cwd=REPO_ROOT,
            check=False,
        )
        assert ignored.returncode == 0
        assert not configured_root.exists()


def test_subtracks_are_disjoint_and_chartcoder_has_no_result_row() -> None:
    with experiment_workspace("portable-disjoint-matrices") as workspace:
        summary = materialize_subtracks(
            repo_root=REPO_ROOT,
            workspace_root=REPO_ROOT.parent,
            output_root=workspace / "runtime",
        )
        all_specs = {
            baseline: load_and_expand_matrix(
                Path(summary["subtracks"][baseline]["matrix"])
            )
            for baseline in CONFIGS
        }
        run_names = {
            baseline: {spec.run_name for spec in specs}
            for baseline, specs in all_specs.items()
        }
        roots = {
            baseline: {spec.artifact_root for spec in specs}
            for baseline, specs in all_specs.items()
        }

        assert run_names["matplotagent"].isdisjoint(run_names["nvagent"])
        assert roots["matplotagent"].isdisjoint(roots["nvagent"])
        for specs in all_specs.values():
            assert all(
                "chartcoder" not in spec.method.casefold() for spec in specs
            )
            assert all(
                "chartcoder" not in spec.run_name.casefold() for spec in specs
            )


def test_specs_and_matrices_contain_no_secret_values() -> None:
    paths = [
        *SPECS_ROOT.rglob("*.json"),
        *(MATRICES_ROOT / config["matrix"] for config in CONFIGS.values()),
    ]
    forbidden_key_tokens = ("api_key", "apikey", "auth_token", "password", "secret")
    forbidden_value_prefixes = ("sk-", "Bearer ")

    def inspect(value: object, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                qualified = f"{path}.{key}" if path else str(key)
                assert not any(
                    token in str(key).casefold() for token in forbidden_key_tokens
                ), qualified
                inspect(child, qualified)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")
        elif isinstance(value, str):
            assert not value.startswith(forbidden_value_prefixes), path

    for path in paths:
        value = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".json"
            else yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        inspect(value)


def test_tracked_templates_have_no_machine_local_absolute_roots() -> None:
    paths = [
        SPECS_ROOT / "build_portable_subtracks.py",
        *(SPECS_ROOT / config["spec"] for config in CONFIGS.values()),
        *(MATRICES_ROOT / config["matrix"] for config in CONFIGS.values()),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "tommy" not in text.casefold()

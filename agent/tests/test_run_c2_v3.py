from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import run_c2_v3
from experiments.manifest import DatasetCase, ManifestError, verify_case_data_files
from run_c2_v3 import C2PipelineError, _manifest_case_info, _write_matrices


def test_c2_matrix_shards_incompatible_panel_counts(tmp_path: Path) -> None:
    manifest = tmp_path / "proposed.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()

    matrices = _write_matrices(
        output_dir=output,
        dataset_manifest=manifest,
        dataset_mode="legacy",
        cases=[("single", 1), ("p5", 5), ("p6", 6)],
        profile="smoke",
        rounds_per_case=1,
        model="test-model",
        offline_defaults=False,
        manifest_data_root=tmp_path,
    )

    payloads = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in matrices
    }

    assert payloads["panel_count_1.json"]["budgets"] == [
        {"type": "renders", "value": 1}
    ]
    assert payloads["panel_count_5.json"]["budgets"] == [
        {"type": "renders", "value": 5}
    ]
    assert payloads["panel_count_6.json"]["budgets"] == [
        {"type": "renders", "value": 6}
    ]
    assert payloads["panel_count_5.json"]["case_ids"] == ["p5"]


def test_sealed_manifest_min_dois_counts_unique_filtered_clusters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [
        DatasetCase("p5-a", 5, "test", {"doi": "10.1000/a"}),
        DatasetCase("p5-b", 5, "test", {"doi": "10.1000/a"}),
        DatasetCase("p5-c", 5, "test", {"doi": "10.1000/c"}),
    ]
    monkeypatch.setattr(run_c2_v3, "load_dataset_manifest", lambda *_, **__: cases)

    selected = _manifest_case_info(
        tmp_path / "manifest.json",
        manifest_data_root=tmp_path,
        dataset_mode="sealed_benchmark",
        case_kind="multi",
        max_cases=None,
        min_panels=5,
        min_dois=2,
    )

    assert selected == [("p5-a", 5), ("p5-b", 5), ("p5-c", 5)]
    with pytest.raises(C2PipelineError, match="only 2 independent DOI clusters"):
        _manifest_case_info(
            tmp_path / "manifest.json",
            manifest_data_root=tmp_path,
            dataset_mode="sealed_benchmark",
            case_kind="multi",
            max_cases=None,
            min_panels=5,
            min_dois=3,
        )


def test_runtime_rejects_exploratory_normalizer_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "exploratory_normalizer"
    source_dir.mkdir()
    source = source_dir / "table.csv"
    source.write_text("Category,Value\nA,1\n", encoding="utf-8")
    case = DatasetCase(
        "case-a",
        1,
        "test",
        {
            "eligible_for_experiment": True,
            "data_path": str(source),
            "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )

    with pytest.raises(ManifestError, match="uses an exploratory-normalizer source"):
        verify_case_data_files(case, manifest_path=tmp_path / "manifest.json")

from __future__ import annotations

import json
from pathlib import Path

from run_c2_v3 import _write_matrices


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

from __future__ import annotations

import json
from pathlib import Path

import yaml


AGENT_ROOT = Path(__file__).resolve().parents[1]


def test_open_weight_registry_matches_remote_model_audit() -> None:
    registry = yaml.safe_load(
        (AGENT_ROOT / "configs" / "model_registry.yml").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (
            AGENT_ROOT
            / "experiments"
            / "model_audits"
            / "qwen2p5_coder_7b_remote.json"
        ).read_text(encoding="utf-8")
    )
    configured = registry["generators"]["open_weight"]

    assert audit["schema_version"] == "1.0"
    assert audit["model_id"] == configured["request_model"]
    assert audit["revision"] == configured["revision"]
    assert audit["license"].casefold() == configured["license"].casefold()
    assert audit["parameter_count"] == configured["parameter_count"]
    assert audit["official_revision_api"].endswith(
        f"/revision/{configured['revision']}?blobs=true"
    )
    assert audit["remote_cache"]["allowed_gpu_indices"] == [1, 2, 3, 4]
    assert audit["remote_cache"]["excluded_gpu_indices"] == [0]

    files = audit["files"]
    assert [item["name"] for item in files] == [
        f"model-{index:05d}-of-00004.safetensors"
        for index in range(1, 5)
    ]
    assert sum(item["bytes"] for item in files) == audit["weight_bytes"]
    assert all(
        len(item["sha256"]) == 64
        and set(item["sha256"]) <= set("0123456789abcdef")
        for item in files
    )
    assert audit["verification"] == {
        "official_sha256_matches_remote_cache": True,
        "official_bytes_match_remote_cache": True,
        "all_required_weight_shards_present": True,
        "model_service_started": False,
        "gpu_workload_started": False,
    }

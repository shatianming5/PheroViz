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


def test_visual_judges_are_exactly_frozen_for_c5() -> None:
    registry = yaml.safe_load(
        (AGENT_ROOT / "configs" / "model_registry.yml").read_text(
            encoding="utf-8"
        )
    )
    judges = registry["visual_judges"]

    assert set(judges) == {"primary", "secondary"}
    assert {
        judges["primary"]["request_model"],
        judges["secondary"]["request_model"],
    } == {"claude-sonnet-4.6", "gemini-3.5-flash"}
    assert {
        judges["primary"]["judge_id"],
        judges["secondary"]["judge_id"],
    } == {"visual-form-primary-v1", "visual-form-secondary-v1"}
    for judge in judges.values():
        assert judge["served_model"] == judge["request_model"]
        assert judge["protocol"] == "anthropic_messages"
        assert judge["endpoint_class"] == "anthropic_compatibility_gateway"
        assert judge["base_url_env"] == "ANTHROPIC_BASE_URL"
        assert judge["api_key_env"] == "ANTHROPIC_AUTH_TOKEN"
        assert judge["max_tokens"] == 1024
        assert judge["timeout_seconds"] == 180
        assert judge["connect_timeout_seconds"] == 10
        assert judge["retries"] == 2
        assert judge["rubric_version"] == "visual-form-v1"
        assert judge["rubric_hash"] == (
            "259261257f56eacd8c700747e127ff71373c5d9f973b0ffb25e356be0c0ccc16"
        )
        assert judge["prompt_hash"] == (
            "fa2c1f6199df16fe6d2e47e413f43f1637e464608e212b2a01f66305ab078de1"
        )
        assert judge["model_cutoff"] is None


def test_c5_provenance_schemas_are_versioned() -> None:
    schema_root = AGENT_ROOT / "experiments" / "schemas"
    for name in (
        "c5_rejudge_batch.schema.json",
        "c5_rejudge_sidecar.schema.json",
        "c5_summary_provenance.schema.json",
    ):
        schema = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["properties"]["schema_version"]["const"] == "2.0"

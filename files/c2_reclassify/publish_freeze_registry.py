#!/usr/bin/env python3
"""Publish auditable metadata for every C2 reclassification freeze."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nature_download.corpus.reviews import (  # noqa: E402
    REVIEW_RUBRIC_V3_HASH,
    REVIEW_RUBRIC_V4_HASH,
)


FREEZES = (
    (
        "v3_strict",
        ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800",
        "diagnostic_only_stale_v3_strict_scope",
    ),
    (
        "v4_strict",
        ROOT / "files/c2_reclassify/frozen_v4_20260721T190322Z",
        "diagnostic_only_historic_reject_selected_or_overlapping_strict_scope",
    ),
    (
        "v4_full_normalizer",
        ROOT
        / "files/c2_reclassify_pool/frozen_v4_full_nonoverlap_20260721T191806Z",
        "sealed_blocked_exploratory_normalizer",
    ),
    (
        "v4_raw_seed",
        ROOT / "files/c2_reclassify_raw_p5/frozen_raw_p5_v4_20260721T192239Z",
        "diagnostic_only_superseded_by_direct_raw_p5",
    ),
    (
        "v4_direct_raw_p5",
        ROOT
        / "files/c2_reclassify_raw_p5/frozen_direct_raw_p5_v4_20260721T193342Z",
        "diagnostic_only_k62_insufficient",
    ),
)
OUTPUT = ROOT / "files/c2_reclassify/freeze_registry.json"
REGISTRY_GENERATIONS = OUTPUT.parent / "freeze_registry_generations"
REGISTRY_CURRENT = OUTPUT.parent / "freeze_registry_current"
V3_PREDECESSOR_PATH_CLARIFICATION = (
    ROOT
    / "files/c2_reclassify"
    / "v3_mutable_predecessor_path_clarification_20260722T035220_0800.json"
)
V3_NAMED_IMMUTABLE_REVIEW_BINDING_ADDENDUM = (
    ROOT
    / "files/c2_reclassify"
    / "v3_named_immutable_review_binding_addendum_20260722T040852_0800.json"
)
V3_DUAL_DIAGNOSTIC_TRANSPORT_CLARIFICATION = (
    ROOT
    / "files/c2_reclassify"
    / "v3_dual_diagnostic_transport_clarification_20260722T041121_0800.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_durable(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def replace_symlink(path: Path, target: str) -> None:
    temporary = path.with_name(path.name + ".next")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"refusing to replace existing registry link temp: {temporary}")
    os.symlink(target, temporary)
    os.replace(temporary, path)


def ensure_symlink(path: Path, target: str) -> None:
    if path.is_symlink() and os.readlink(path) == target:
        return
    replace_symlink(path, target)


def publish_registry_atomically(registry: dict[str, Any]) -> tuple[str, str]:
    canonical = registry["registry_sha256"]
    serialized = json.dumps(
        registry, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    generation_name = f"registry-{canonical}"
    generation = REGISTRY_GENERATIONS / generation_name
    REGISTRY_GENERATIONS.mkdir(parents=True, exist_ok=True)

    if generation.exists():
        registry_path = generation / OUTPUT.name
        sidecar_path = generation / OUTPUT.with_suffix(".sha256").name
        if (
            not registry_path.is_file()
            or not sidecar_path.is_file()
            or registry_path.read_text(encoding="utf-8") != serialized
            or sidecar_path.read_text(encoding="utf-8").strip() != canonical
        ):
            raise ValueError(f"existing registry generation differs: {generation}")
    else:
        temporary_generation = REGISTRY_GENERATIONS / f".{generation_name}.next"
        if temporary_generation.exists() or temporary_generation.is_symlink():
            raise ValueError(
                f"refusing to replace registry generation temp: {temporary_generation}"
            )
        temporary_generation.mkdir()
        registry_path = temporary_generation / OUTPUT.name
        sidecar_path = temporary_generation / OUTPUT.with_suffix(".sha256").name
        write_durable(registry_path, serialized)
        write_durable(sidecar_path, canonical + "\n")
        registry_path.chmod(0o444)
        sidecar_path.chmod(0o444)
        temporary_generation.chmod(0o555)
        os.replace(temporary_generation, generation)

    # Both compatibility paths traverse this one symlink, so subsequent generation
    # changes switch the registry and its sidecar together in one rename.
    replace_symlink(
        REGISTRY_CURRENT,
        str(Path("freeze_registry_generations") / generation_name),
    )
    ensure_symlink(
        OUTPUT,
        str(Path(REGISTRY_CURRENT.name) / OUTPUT.name),
    )
    ensure_symlink(
        OUTPUT.with_suffix(".sha256"),
        str(Path(REGISTRY_CURRENT.name) / OUTPUT.with_suffix(".sha256").name),
    )
    if (
        sha256_file(OUTPUT)
        != hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        or OUTPUT.with_suffix(".sha256").read_text(encoding="utf-8").strip()
        != canonical
    ):
        raise ValueError("atomic registry publication verification failed")
    return canonical, sha256_file(OUTPUT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def input_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("inputs")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    raw = manifest.get("input")
    if isinstance(raw, dict):
        return [dict(raw)]
    raise ValueError("freeze manifest has no input entry")


def classifier_metadata(rule_version: str) -> dict[str, Any]:
    if rule_version == "simple-2d-v4":
        proposals_path = ROOT / "nature_download/corpus/proposals.py"
        reviews_path = ROOT / "nature_download/corpus/reviews.py"
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if subprocess.check_output(
            ["git", "show", f"HEAD:{proposals_path.relative_to(ROOT)}"],
            cwd=ROOT,
        ) != proposals_path.read_bytes():
            raise ValueError("working classifier differs from HEAD")
        return {
            "proposal_rule_version": rule_version,
            "classifier_git_head": head,
            "classifier_file": str(proposals_path.relative_to(ROOT)),
            "classifier_file_sha256": sha256_file(proposals_path),
            "reviewer_file": str(reviews_path.relative_to(ROOT)),
            "reviewer_file_sha256": sha256_file(reviews_path),
            "review_rubric_hash": REVIEW_RUBRIC_V4_HASH,
        }
    if rule_version == "simple-2d-v3":
        return {
            "proposal_rule_version": rule_version,
            "classifier_file_sha256": None,
            "classifier_hash_status": (
                "not recorded by the historic V3 freeze; current code hash is not "
                "substituted retroactively"
            ),
            "review_rubric_hash": REVIEW_RUBRIC_V3_HASH,
        }
    raise ValueError(f"unsupported frozen rule version: {rule_version}")


def summarize_input(entry: dict[str, Any], rule_version: str) -> dict[str, Any]:
    frozen = Path(str(entry["frozen_path"]))
    source = Path(str(entry["source_path"]))
    rows = read_jsonl(frozen)
    frozen_hash = sha256_file(frozen)
    source_hash = sha256_file(source) if source.is_file() else None
    rules = sorted({str(row.get("proposal_rule_version") or "") for row in rows})
    if rules != [rule_version]:
        raise ValueError(f"{frozen}: rule mismatch {rules}")
    return {
        "label": entry.get("label"),
        "scope": entry.get("scope"),
        "source_path": str(source),
        "source_sha256_declared": entry["source_sha256"],
        "source_sha256_verified": source_hash,
        "source_hash_matches_declared": source_hash == entry["source_sha256"],
        "frozen_path": str(frozen),
        "proposal_sha256_declared": entry["frozen_sha256"],
        "proposal_sha256_verified": frozen_hash,
        "proposal_hash_matches_declared": frozen_hash == entry["frozen_sha256"],
        "records": len(rows),
        "single_panels": sum(
            row.get("proposal_type") == "single_panel" for row in rows
        ),
        "multi_panel_parents": sum(
            row.get("proposal_type") == "multi_panel" for row in rows
        ),
        "distinct_doi_clusters": len(
            {str(row.get("doi") or "") for row in rows if row.get("doi")}
        ),
        "eligible_for_experiment_true": sum(
            row.get("eligible_for_experiment") is True for row in rows
        ),
        "frozen_file_readonly": not bool(frozen.stat().st_mode & 0o222),
        "proposal_rule_versions_verified": rules,
    }


def hash_bound_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    internal_hash = manifest.pop("manifest_sha256")
    if internal_hash != canonical_hash(manifest):
        raise ValueError(f"{path}: internal manifest hash mismatch")
    sidecars = (path.with_suffix(path.suffix + ".sha256"), path.with_suffix(".sha256"))
    if not any(
        sidecar.is_file()
        and sidecar.read_text(encoding="utf-8").strip() == internal_hash
        for sidecar in sidecars
    ):
        raise ValueError(f"{path}: internal manifest hash sidecar mismatch")
    manifest["manifest_sha256"] = internal_hash
    return manifest, internal_hash, sha256_file(path)


def v3_review_input_binding_status() -> dict[str, Any]:
    clarification, clarification_internal_hash, clarification_file_hash = (
        hash_bound_manifest(V3_PREDECESSOR_PATH_CLARIFICATION)
    )
    if clarification.get("requires_fresh_review") is not True:
        raise ValueError("V3 predecessor clarification must require fresh review")
    if clarification.get("stale_review_or_evidence_reuse_forbidden") is not True:
        raise ValueError("V3 predecessor clarification must forbid stale reuse")

    observations = clarification.get("historical_mutable_path_observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("V3 predecessor clarification lacks path observations")

    addendum, addendum_internal_hash, addendum_file_hash = hash_bound_manifest(
        V3_NAMED_IMMUTABLE_REVIEW_BINDING_ADDENDUM
    )
    if (
        addendum.get("requires_fresh_review") is not True
        or addendum.get("stale_review_or_evidence_reuse_forbidden") is not True
        or addendum.get("no_in_place_overwrite") is not True
    ):
        raise ValueError("named immutable V3 addendum lacks required safeguards")
    source_freeze = addendum.get("source_v3_freeze")
    if not isinstance(source_freeze, dict):
        raise ValueError("named immutable V3 addendum lacks source freeze")
    frozen_manifest_path = (
        ROOT
        / "files/c2_reclassify/frozen_v3_20260722T024015_0800/freeze_manifest.json"
    )
    if source_freeze.get("file_sha256") != sha256_file(frozen_manifest_path):
        raise ValueError("named immutable V3 addendum source freeze mismatch")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_by_label = {
        str(entry["label"]): entry for entry in frozen_manifest.get("inputs", [])
    }
    expected_frozen_labels = {
        "casecount_strict": "priority_casecount_strict",
        "full_strict": "priority_full_strict",
    }
    historical_context = addendum.get("historical_predecessor_observation_context")
    if not isinstance(historical_context, dict) or (
        historical_context.get("clarification_file_sha256") != clarification_file_hash
        or historical_context.get("clarification_internal_sha256")
        != clarification_internal_hash
    ):
        raise ValueError("named immutable V3 addendum historical context mismatch")

    named_binding = addendum.get("canonical_named_immutable_review_binding")
    if not isinstance(named_binding, dict):
        raise ValueError("named immutable V3 addendum lacks named binding")
    binding_path = Path(str(named_binding["binding_manifest_path"]))
    binding, binding_internal_hash, binding_file_hash = hash_bound_manifest(binding_path)
    if (
        named_binding.get("binding_manifest_file_sha256") != binding_file_hash
        or named_binding.get("binding_manifest_internal_sha256")
        != binding_internal_hash
        or binding.get("source_v3_freeze_manifest_sha256")
        != sha256_file(frozen_manifest_path)
    ):
        raise ValueError("named immutable V3 binding manifest mismatch")

    binding_by_label = {
        str(entry["label"]): entry
        for entry in binding.get("authoritative_proposal_inputs", [])
    }
    named_inputs = []
    for entry in named_binding.get("inputs", []):
        label = str(entry["label"])
        binding_entry = binding_by_label.get(label)
        frozen_entry = frozen_by_label.get(expected_frozen_labels.get(label, ""))
        if not isinstance(binding_entry, dict) or not isinstance(frozen_entry, dict):
            raise ValueError(f"named immutable V3 binding lacks {label}")
        named_path = Path(str(entry["named_readonly_input"]))
        frozen_path = Path(str(entry["frozen_v3_source"]))
        named_hash = sha256_file(named_path)
        frozen_hash = sha256_file(frozen_path)
        if (
            named_hash != entry.get("sha256")
            or named_hash != binding_entry.get("review_input_sha256")
            or frozen_hash != binding_entry.get("source_v3_frozen_sha256")
            or frozen_path != Path(str(frozen_entry["frozen_path"]))
            or frozen_hash != frozen_entry.get("frozen_sha256")
            or named_hash != frozen_hash
            or bool(named_path.stat().st_mode & 0o222)
            or bool(frozen_path.stat().st_mode & 0o222)
        ):
            raise ValueError(f"{named_path}: named immutable V3 input mismatch")
        named_inputs.append(
            {
                "label": label,
                "named_readonly_path": str(named_path),
                "frozen_v3_path": str(frozen_path),
                "sha256": named_hash,
                "byte_identical_to_frozen_v3_source": True,
                "readonly": True,
            }
        )
    if {entry["label"] for entry in named_inputs} != {
        "casecount_strict",
        "full_strict",
    }:
        raise ValueError("unexpected named immutable V3 input labels")

    return {
        "historical_predecessor_path_clarification": {
            "path": str(V3_PREDECESSOR_PATH_CLARIFICATION),
            "file_sha256": clarification_file_hash,
            "internal_sha256": clarification_internal_hash,
            "original_current_sha256_fields_are_historical_observations_only": True,
            "historical_observation_only": True,
        },
        "named_immutable_review_input_binding": {
            "addendum_path": str(V3_NAMED_IMMUTABLE_REVIEW_BINDING_ADDENDUM),
            "addendum_file_sha256": addendum_file_hash,
            "addendum_internal_sha256": addendum_internal_hash,
            "binding_manifest_path": str(binding_path),
            "binding_manifest_file_sha256": binding_file_hash,
            "binding_manifest_internal_sha256": binding_internal_hash,
            "inputs": named_inputs,
            "requires_fresh_review": True,
            "stale_review_or_evidence_reuse_forbidden": True,
            "no_in_place_overwrite": True,
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": addendum[
            "prior_review_or_evidence_status"
        ],
    }


def v3_dual_transport_binding_status() -> dict[str, Any]:
    clarification, clarification_internal_hash, clarification_file_hash = (
        hash_bound_manifest(V3_DUAL_DIAGNOSTIC_TRANSPORT_CLARIFICATION)
    )
    if (
        clarification.get("requires_fresh_review") is not True
        or clarification.get("stale_review_or_evidence_reuse_forbidden") is not True
        or clarification.get("no_in_place_overwrite") is not True
    ):
        raise ValueError("V3 dual transport clarification lacks required safeguards")

    frozen_manifest_path = (
        ROOT
        / "files/c2_reclassify/frozen_v3_20260722T024015_0800/freeze_manifest.json"
    )
    source_freeze = clarification.get("source_v3_freeze")
    if not isinstance(source_freeze, dict) or (
        source_freeze.get("file_sha256") != sha256_file(frozen_manifest_path)
    ):
        raise ValueError("V3 dual transport source freeze mismatch")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_by_label = {
        str(entry["label"]): entry for entry in frozen_manifest.get("inputs", [])
    }
    expected_frozen_labels = {
        "casecount_strict": "priority_casecount_strict",
        "full_strict": "priority_full_strict",
    }

    historical = clarification.get("historical_predecessor_observation_context")
    if not isinstance(historical, dict):
        raise ValueError("V3 dual transport clarification lacks historical context")
    historical_path = Path(str(historical["path"]))
    _, historical_internal_hash, historical_file_hash = hash_bound_manifest(
        historical_path
    )
    if (
        historical.get("file_sha256") != historical_file_hash
        or historical.get("internal_sha256") != historical_internal_hash
    ):
        raise ValueError("V3 dual transport historical context mismatch")

    transports = clarification.get("valid_diagnostic_transports")
    if not isinstance(transports, dict):
        raise ValueError("V3 dual transport clarification lacks transports")
    controlled = transports.get("controlled_alias_binding")
    named = transports.get("named_readonly_mirror_binding")
    if not isinstance(controlled, dict) or not isinstance(named, dict):
        raise ValueError("V3 dual transport clarification lacks a transport")

    controlled_path = Path(str(controlled["path"]))
    controlled_manifest, controlled_internal_hash, controlled_file_hash = (
        hash_bound_manifest(controlled_path)
    )
    if (
        controlled.get("file_sha256") != controlled_file_hash
        or controlled.get("internal_sha256") != controlled_internal_hash
    ):
        raise ValueError("V3 controlled transport clarification mismatch")
    live_binding = controlled_manifest.get("controlled_live_input_manifest")
    if not isinstance(live_binding, dict):
        raise ValueError("V3 controlled transport lacks input manifest")
    input_manifest_path = Path(str(live_binding["path"]))
    input_manifest_hash = sha256_file(input_manifest_path)
    input_sidecar = Path(str(live_binding["file_sha256_sidecar"]))
    if (
        input_manifest_hash != controlled.get("input_manifest_file_sha256")
        or input_manifest_hash != live_binding.get("file_sha256")
        or not input_sidecar.is_file()
        or input_sidecar.read_text(encoding="utf-8").strip() != input_manifest_hash
    ):
        raise ValueError("V3 controlled transport input manifest mismatch")
    controlled_by_label = {
        str(entry["batch"]): entry for entry in live_binding.get("inputs", [])
    }

    named_path = Path(str(named["path"]))
    named_manifest, named_internal_hash, named_file_hash = hash_bound_manifest(
        named_path
    )
    if (
        named.get("file_sha256") != named_file_hash
        or named.get("internal_sha256") != named_internal_hash
        or named_manifest.get("source_v3_freeze_manifest_sha256")
        != sha256_file(frozen_manifest_path)
    ):
        raise ValueError("V3 named mirror transport mismatch")
    named_by_label = {
        str(entry["label"]): entry
        for entry in named_manifest.get("authoritative_proposal_inputs", [])
    }

    triple_inputs = []
    for entry in transports.get("verified_equivalent_inputs", []):
        label = str(entry["label"])
        frozen_entry = frozen_by_label.get(expected_frozen_labels.get(label, ""))
        controlled_entry = controlled_by_label.get(label)
        named_entry = named_by_label.get(label)
        if not all(
            isinstance(item, dict)
            for item in (frozen_entry, controlled_entry, named_entry)
        ):
            raise ValueError(f"V3 dual transport lacks {label}")
        frozen_path = Path(str(entry["frozen_v3_source"]))
        live_path = Path(str(entry["controlled_alias_input"]))
        mirror_path = Path(str(entry["named_readonly_mirror_input"]))
        frozen_hash = sha256_file(frozen_path)
        live_hash = sha256_file(live_path)
        mirror_hash = sha256_file(mirror_path)
        if (
            frozen_path != Path(str(frozen_entry["frozen_path"]))
            or frozen_hash != frozen_entry.get("frozen_sha256")
            or frozen_hash != controlled_entry.get("frozen_v3_input_sha256")
            or frozen_hash != named_entry.get("source_v3_frozen_sha256")
            or live_hash != controlled_entry.get("controlled_live_input_sha256")
            or mirror_hash != named_entry.get("review_input_sha256")
            or frozen_hash != entry.get("sha256")
            or frozen_hash != live_hash
            or frozen_hash != mirror_hash
            or bool(frozen_path.stat().st_mode & 0o222)
            or bool(live_path.stat().st_mode & 0o222)
            or bool(mirror_path.stat().st_mode & 0o222)
        ):
            raise ValueError(f"V3 dual transport input mismatch: {label}")
        triple_inputs.append(
            {
                "label": label,
                "sha256": frozen_hash,
                "frozen_v3_path": str(frozen_path),
                "controlled_alias_path": str(live_path),
                "named_readonly_mirror_path": str(mirror_path),
                "all_three_byte_identical": True,
                "all_three_readonly": True,
            }
        )
    if {entry["label"] for entry in triple_inputs} != set(expected_frozen_labels):
        raise ValueError("unexpected V3 dual transport input labels")

    return {
        "dual_diagnostic_transport_clarification": {
            "path": str(V3_DUAL_DIAGNOSTIC_TRANSPORT_CLARIFICATION),
            "file_sha256": clarification_file_hash,
            "internal_sha256": clarification_internal_hash,
            "controlled_alias_binding": {
                "path": str(controlled_path),
                "file_sha256": controlled_file_hash,
                "internal_sha256": controlled_internal_hash,
                "input_manifest_path": str(input_manifest_path),
                "input_manifest_sha256": input_manifest_hash,
            },
            "named_readonly_mirror_binding": {
                "path": str(named_path),
                "file_sha256": named_file_hash,
                "internal_sha256": named_internal_hash,
            },
            "verified_equivalent_inputs": triple_inputs,
            "both_transports_valid_for_diagnostic_only": True,
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": clarification[
            "prior_review_or_evidence_status"
        ],
    }


def main() -> int:
    freezes = []
    for label, directory, status in FREEZES:
        manifest_path = directory / "freeze_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_path)
        if manifest.get("manifest_sha256") != canonical_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ):
            raise ValueError(f"{manifest_path}: internal manifest hash mismatch")
        rule_version = str(manifest["proposal_rule_version"])
        freeze = {
            "label": label,
            "sealed_status": status,
            "freeze_manifest_path": str(manifest_path),
            "freeze_manifest_sha256": manifest_hash,
            "freeze_manifest_internal_sha256": manifest["manifest_sha256"],
            "classifier": classifier_metadata(rule_version),
            "requires_fresh_review": manifest.get("requires_fresh_review") is True,
            "stale_review_or_evidence_reuse_forbidden": (
                manifest.get("stale_review_or_evidence_reuse_forbidden") is True
            ),
            "inputs": [
                summarize_input(entry, rule_version)
                for entry in input_entries(manifest)
            ],
        }
        if label == "v3_strict":
            freeze["review_input_binding_status"] = (
                v3_dual_transport_binding_status()
            )
        freezes.append(freeze)
    registry = {
        "schema_version": "c2-reclassification-freeze-registry-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "priority_or_reject_scopes_diagnostic_only": True,
        "sealed_eligible_freezes": [],
        "publication_contract": {
            "mode": "atomic-generation-symlink",
            "current_generation_pointer": str(REGISTRY_CURRENT),
            "compatibility_paths": [
                str(OUTPUT),
                str(OUTPUT.with_suffix(".sha256")),
            ],
        },
        "integrity_rule": (
            "Only a freeze with sealed_status=eligible_for_sealed_review could be "
            "used for a final C2 review/benchmark. This registry currently contains "
            "no such freeze."
        ),
        "freezes": freezes,
    }
    registry["registry_sha256"] = canonical_hash(registry)
    published_canonical, published_file_hash = publish_registry_atomically(registry)
    if published_canonical != registry["registry_sha256"]:
        raise ValueError("published registry canonical hash mismatch")
    print(
        json.dumps(
            {
                **registry,
                "published_registry_file_sha256": published_file_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

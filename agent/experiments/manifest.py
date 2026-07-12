from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from .models import ProvenanceError


class ManifestError(ProvenanceError):
    """Raised when a dataset manifest violates the experiment protocol."""


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    panel_count: Optional[int]
    split: Optional[str]
    payload: Dict[str, Any]


def _load_manifest_object(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            raise ManifestError(
                f"Dataset manifests must use .json, .yaml, or .yml: {path}"
            )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"Cannot load dataset manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError("Dataset manifest must contain an object")
    return data


def validate_manifest(data: Mapping[str, Any]) -> list[DatasetCase]:
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("Dataset manifest requires a non-empty cases list")

    cases: list[DatasetCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ManifestError(f"cases[{index}] must be an object")
        raw_case_id = raw_case.get("case_id")
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise ManifestError(
                f"cases[{index}].case_id must be a non-empty string"
            )
        case_id = raw_case_id.strip()
        if case_id in seen:
            raise ManifestError(f"Duplicate case_id in dataset manifest: {case_id}")
        seen.add(case_id)

        panel_count = raw_case.get("panel_count")
        if panel_count is not None and (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
        ):
            raise ManifestError(
                f"cases[{index}].panel_count must be a positive integer"
            )

        split = raw_case.get("split")
        if split is not None:
            if not isinstance(split, str) or not split.strip():
                raise ManifestError(
                    f"cases[{index}].split must be a non-empty string"
                )
            split = split.strip()

        cases.append(
            DatasetCase(
                case_id=case_id,
                panel_count=panel_count,
                split=split,
                payload=dict(raw_case),
            )
        )
    return cases


def load_dataset_manifest(path: Path) -> list[DatasetCase]:
    return validate_manifest(_load_manifest_object(path))


def select_case(
    cases: Sequence[DatasetCase],
    case_id: str,
) -> DatasetCase:
    matches = [case for case in cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ManifestError(
            f"Dataset manifest has no unique case_id {case_id!r}"
        )
    return matches[0]


def verify_case_metadata(
    case: DatasetCase,
    *,
    panel_count: Optional[int],
    split: Optional[str],
) -> None:
    if case.panel_count != panel_count:
        raise ManifestError(
            f"case_id {case.case_id!r} panel_count changed: "
            f"{panel_count!r} != {case.panel_count!r}"
        )
    if case.split != split:
        raise ManifestError(
            f"case_id {case.case_id!r} split changed: "
            f"{split!r} != {case.split!r}"
        )

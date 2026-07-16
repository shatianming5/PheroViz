"""Deterministic paper-level splits with manifest and cutoff provenance."""

from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .policy import normalize_doi
from .provenance import sha256_bytes


SCHEMA_VERSION = "1.0"
SPLIT_NAMES = ("train", "val", "test")


def _canonical_manifest_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def _model_cutoff_stratum(
    record: dict[str, Any],
    cutoff: date | None,
) -> str:
    if cutoff is None:
        return "unconfigured"
    published_date = record.get("published_date")
    if published_date:
        try:
            value = date.fromisoformat(str(published_date))
            return "pre_cutoff" if value < cutoff else "post_cutoff"
        except ValueError:
            pass
    try:
        year = int(record.get("year"))
    except (TypeError, ValueError):
        return "unknown_publication_date"
    if year < cutoff.year:
        return "pre_cutoff"
    if year > cutoff.year:
        return "post_cutoff"
    return "unknown_within_cutoff_year"


def _allocation_counts(n: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = [n * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = n - sum(counts)
    priority = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in priority[:remainder]:
        counts[index] += 1
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if n >= len(positive):
        for empty_index in (index for index in positive if counts[index] == 0):
            donor = max(
                (index for index in positive if counts[index] > 1),
                key=lambda index: (counts[index], ratios[index], -index),
            )
            counts[donor] -= 1
            counts[empty_index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def generate_split_bundle(
    manifests: Iterable[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    source_manifest_sha256: str | None = None,
    model_cutoff_date: str | None = None,
) -> dict[str, Any]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(
        sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("split ratios must be non-negative and sum to 1")
    cutoff = date.fromisoformat(model_cutoff_date) if model_cutoff_date else None

    deduped: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        if not manifest.get("download_eligible"):
            continue
        doi = normalize_doi(manifest.get("doi"))
        if not doi:
            continue
        candidate = dict(manifest)
        candidate["doi"] = doi
        current = deduped.get(doi)
        if current is None:
            deduped[doi] = candidate
            continue
        current_json = json.dumps(current, sort_keys=True, ensure_ascii=False)
        candidate_json = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if candidate_json < current_json:
            deduped[doi] = candidate

    canonical_records = [deduped[doi] for doi in sorted(deduped)]
    manifest_hash = source_manifest_sha256 or _canonical_manifest_hash(
        canonical_records
    )
    dois = sorted(deduped)
    random.Random(seed).shuffle(dois)
    counts = _allocation_counts(len(dois), ratios)

    assignments: dict[str, str] = {}
    offset = 0
    for split_name in SPLIT_NAMES:
        end = offset + counts[split_name]
        assignments.update({doi: split_name for doi in dois[offset:end]})
        offset = end

    papers = [
        {
            "doi": doi,
            "split": assignments[doi],
            "year": deduped[doi].get("year"),
            "published_date": deduped[doi].get("published_date"),
            "model_cutoff_stratum": _model_cutoff_stratum(
                deduped[doi], cutoff
            ),
        }
        for doi in sorted(assignments)
    ]
    split_sets = {
        name: {paper["doi"] for paper in papers if paper["split"] == name}
        for name in SPLIT_NAMES
    }
    if any(
        split_sets[left] & split_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise AssertionError("paper-level splits overlap")

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "counts": {name: len(split_sets[name]) for name in SPLIT_NAMES},
        "source_manifest_sha256": manifest_hash,
        "model_cutoff_date": model_cutoff_date,
        "papers": papers,
    }


def write_split_bundle(bundle: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    papers = bundle.get("papers") or []
    for split_name in SPLIT_NAMES:
        with (output / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for paper in papers:
                if paper.get("split") == split_name:
                    handle.write(
                        json.dumps(paper, ensure_ascii=False, sort_keys=True) + "\n"
                    )

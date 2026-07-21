from __future__ import annotations

from nature_download.corpus.splits import generate_split_bundle


def manifests() -> list[dict]:
    records = []
    for index in range(12):
        records.append(
            {
                "doi": f"10.1038/paper-{index:02d}",
                "year": 2018 + index,
                "published_date": f"{2018 + index}-01-01",
                "download_eligible": True,
            }
        )
    records.append(dict(records[0]))
    return records


def test_splits_are_deterministic_deduplicated_and_disjoint() -> None:
    first = generate_split_bundle(
        manifests(),
        seed=42,
        source_manifest_sha256="abc123",
        model_cutoff_date="2024-06-01",
    )
    second = generate_split_bundle(
        list(reversed(manifests())),
        seed=42,
        source_manifest_sha256="abc123",
        model_cutoff_date="2024-06-01",
    )
    assert first == second
    assert len(first["papers"]) == 12
    sets = {
        split: {
            paper["doi"]
            for paper in first["papers"]
            if paper["split"] == split
        }
        for split in ("train", "val", "test")
    }
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert first["source_manifest_sha256"] == "abc123"
    assert all(first["counts"][name] > 0 for name in ("train", "val", "test"))


def test_cutoff_is_never_guessed() -> None:
    without_cutoff = generate_split_bundle(manifests(), seed=7)
    assert {
        paper["model_cutoff_stratum"] for paper in without_cutoff["papers"]
    } == {"unconfigured"}
    with_cutoff = generate_split_bundle(
        manifests(),
        seed=7,
        model_cutoff_date="2024-06-01",
    )
    assert {"pre_cutoff", "post_cutoff"}.issubset(
        {
            paper["model_cutoff_stratum"]
            for paper in with_cutoff["papers"]
        }
    )

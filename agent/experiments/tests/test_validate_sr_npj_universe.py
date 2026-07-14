from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.experiments.validate_sr_npj_universe import (
    ValidationError,
    _canonical_sha256,
    _read_statuses,
)


def test_canonical_sha256_is_key_order_independent() -> None:
    assert _canonical_sha256({"b": 2, "a": 1}) == _canonical_sha256(
        {"a": 1, "b": 2}
    )


def test_read_statuses_rejects_duplicate_article(tmp_path: Path) -> None:
    path = tmp_path / "skipped.txt"
    path.write_text("article\tno-figures\narticle\tno-source-data\n")
    with pytest.raises(ValidationError, match="duplicate article"):
        _read_statuses(path)


def test_empty_sha_reference() -> None:
    assert hashlib.sha256(b"").hexdigest() == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_canonical_encoding_is_compact() -> None:
    value = {"rounds": ["initial", "retry1", "retry2"]}
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _canonical_sha256(value) == expected

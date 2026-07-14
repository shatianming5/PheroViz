"""Production boundary for C2 full-replacement finalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .c2_m1_trust_boundary import require_external_m1_trust_lock
from .c2_full_replacement_policy import (
    C2FullReplacementPolicyError,
    load_production_policy,
)
from .models import ProvenanceError


class C2FullReplacementError(ProvenanceError):
    """Raised when V2.1 full-replacement evidence cannot finalize safely."""


def prepare_full_replacement_finalization(manifest_path: Path) -> Any:
    """Deny production preparation before inspecting a caller-selected manifest."""

    require_external_m1_trust_lock()
    del manifest_path
    try:
        load_production_policy()
    except C2FullReplacementPolicyError as exc:
        raise C2FullReplacementError(str(exc)) from exc
    raise AssertionError("Stage-B production policy resolver must not return in Stage A")


def write_full_replacement_report(finalized: Any, output_path: Path) -> Path:
    """Deny production publication before inspecting an output path or evidence."""

    require_external_m1_trust_lock()
    del finalized, output_path
    raise AssertionError("Stage-A production finalization cannot reach publication")


def finalize_to_path(
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Deny production finalization before inspecting either caller path."""

    require_external_m1_trust_lock()
    del output_path
    prepare_full_replacement_finalization(manifest_path)
    raise AssertionError("Stage-A production finalization cannot reach publication")

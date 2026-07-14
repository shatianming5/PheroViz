"""Deny-only boundary for the unavailable external M1 trust lock.

This repository has no independently signed, deployment-pinned M1 artifact or
adapter.  Production C2 routes must call this no-input helper before examining
any caller-controlled value.  It deliberately has no resource path, selector,
environment fallback, test fixture, or allowing branch.
"""

from __future__ import annotations

from typing import NoReturn

from .models import ProvenanceError


M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE = "M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE"


class M1ExternalTrustLockUnavailable(ProvenanceError):
    """Raised while no independently verified external M1 lock is available."""

    code = M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(self.code)


def require_external_m1_trust_lock() -> NoReturn:
    """Deny production C2 finalization without an external M1 verifier."""

    raise M1ExternalTrustLockUnavailable()

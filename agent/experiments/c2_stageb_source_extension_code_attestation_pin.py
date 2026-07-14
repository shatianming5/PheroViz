"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256: str | None = None
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = False

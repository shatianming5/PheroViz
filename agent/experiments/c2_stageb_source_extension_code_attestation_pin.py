"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256 = (
    "5b6c52e37b268fba2e443b0ff4dd97e5006c2cd536323a778274b1231da2d6bc"
)
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = True

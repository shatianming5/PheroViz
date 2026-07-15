"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256 = (
    "1407743c874493b57babc93da968a20e3cc592bd008693ee3b2f3ae68929fb87"
)
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = True

"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256 = (
    "206524eb7d1c71c459a6e54fb29c5a3c143b1c38de70ff2ad3a9ff71926ae68f"
)
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = True

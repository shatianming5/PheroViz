"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256 = (
    "1b46a969150ce9fd0dde6bd71a0bf225403907022ddee893674557ea9153a67c"
)
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = True

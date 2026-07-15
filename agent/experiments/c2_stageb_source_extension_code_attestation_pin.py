"""Compile-time pin for the fixed source-extension registry resource.

This small module breaks the registry/manifest digest cycle: the attested loader
is immutable before the manifest is generated, while this pin is populated only
after that manifest and registry exist.
"""

from __future__ import annotations


SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256 = (
    "0c04ba41242ab09ba686a274fe3cd7dccb3f72a044770897b8a856dccd7b4d06"
)
SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = True

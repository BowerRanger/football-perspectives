"""Awaiting later phase — see Phase 0 of broadcast-mono pipeline plan."""

import pytest

pytest.skip(
    "awaiting later phase: imports a module deleted in Phase 0 of the "
    "broadcast-mono pipeline rewrite",
    allow_module_level=True,
)

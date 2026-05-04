"""Minimal GVHMR config registration for inference-only usage.

GVHMR ships a ``register_store_gvhmr()`` function at
``third_party/gvhmr/hmr4d/configs/__init__.py`` which imports the entire
dataset hierarchy (BEDLAM, H36M, 3DPW, EMDB, RICH, AMASS).  Those imports
pull in training-only dependencies (cython_bbox, lapx, and heavy dataset
utilities) that we don't need for demo inference.

This module provides a reduced registration that imports only the three
modules required to resolve the demo config's ``defaults:`` list:

  - ``model/gvhmr/gvhmr_pl_demo``       → hmr4d.model.gvhmr.gvhmr_pl_demo
  - ``network/gvhmr/relative_transformer`` → hmr4d.network.gvhmr.relative_transformer
  - ``endecoder/gvhmr/v1_amass_local_bedlam_cam`` → hmr4d.model.gvhmr.utils.endecoder

Each of those modules registers its config with the hydra ConfigStore at
import time via ``MainStore.store(...)`` decorators with
``builds(Cls, populate_full_signature=True)``.  Because the configs are
auto-generated from the class signatures, they always match the
implementation — unlike the dead ``siga24_release.yaml`` which references a
non-existent ``NetworkEncoderRoPEV2`` class.
"""

from __future__ import annotations


def register_minimal_gvhmr() -> None:
    """Import only the GVHMR modules needed for demo inference config resolution.

    Must be called inside an ``initialize_config_module`` context so the
    registrations are visible to ``hydra.compose``.
    """
    # Import order matters only to surface errors early — the ConfigStore is a
    # global singleton, so registration side effects accumulate regardless.
    import hmr4d.model.gvhmr.utils.endecoder  # noqa: F401
    import hmr4d.network.gvhmr.relative_transformer  # noqa: F401
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401

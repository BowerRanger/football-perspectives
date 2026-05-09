"""Extract the SMPL neutral body model into a Blender-friendly npz.

The bundled ``SMPL_NEUTRAL.pkl`` (under
``third_party/gvhmr/inputs/checkpoints/body_models/smpl/``) is in the
chumpy/Python-2 pickle format, which Blender's bundled Python cannot
load. This script runs in the project's main venv (where chumpy is
available via the ``hmr`` extras) and writes a plain NumPy npz the
Blender FBX exporter can read.

Output: ``data/models/smpl_neutral.npz`` (gitignored — SMPL data is
not redistributable).

Contains:
    v_template  (6890, 3)  float32   mean-shape vertex positions, y-up
    faces       (~13776, 3) int32    mesh triangulation
    weights     (6890, 24) float32   per-vertex skinning weights

Run::

    python scripts/extract_smpl_neutral.py

Idempotent: skips re-extraction if the npz already exists and is newer
than the pkl.
"""

from __future__ import annotations

import inspect as _inspect
import pickle
import sys
from pathlib import Path

import numpy as np

# chumpy (SMPL's legacy array library) uses ``inspect.getargspec`` (gone
# in Python 3.11) and numpy aliases (gone in numpy >= 1.20). Apply the
# same shim pattern as ``src/utils/gvhmr_estimator.py`` before importing
# anything that triggers a chumpy import.
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]
for _alias, _target in (
    ("bool", bool), ("int", int), ("float", float),
    ("complex", complex), ("object", object),
    ("str", str), ("unicode", str),
):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)


REPO_ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = (
    REPO_ROOT
    / "third_party" / "gvhmr" / "inputs" / "checkpoints" / "body_models"
    / "smpl" / "SMPL_NEUTRAL.pkl"
)
OUT_PATH = REPO_ROOT / "data" / "models" / "smpl_neutral.npz"


def main() -> int:
    if not PKL_PATH.exists():
        sys.stderr.write(f"SMPL pkl not found at {PKL_PATH}\n")
        return 1

    if OUT_PATH.exists() and OUT_PATH.stat().st_mtime >= PKL_PATH.stat().st_mtime:
        print(f"[smpl-extract] {OUT_PATH} is up to date, skipping")
        return 0

    with PKL_PATH.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    # SMPL fields may be chumpy objects; np.asarray coerces both.
    v_template = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    weights = np.asarray(data["weights"], dtype=np.float32)

    expected_v = 6890
    expected_j = 24
    if v_template.shape != (expected_v, 3):
        sys.stderr.write(
            f"unexpected v_template shape: {v_template.shape}\n"
        )
        return 2
    if weights.shape != (expected_v, expected_j):
        sys.stderr.write(f"unexpected weights shape: {weights.shape}\n")
        return 2
    if faces.ndim != 2 or faces.shape[1] != 3:
        sys.stderr.write(f"unexpected faces shape: {faces.shape}\n")
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        v_template=v_template,
        faces=faces,
        weights=weights,
    )
    print(
        f"[smpl-extract] wrote {OUT_PATH} "
        f"(v={v_template.shape[0]}, f={faces.shape[0]}, w={weights.shape})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

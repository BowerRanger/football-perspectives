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
    v_template       (6890, 3)        float32   mean-shape vertices, y-up
    faces            (~13776, 3)      int32     mesh triangulation
    weights          (6890, 24)       float32   per-vertex skinning weights
    joint_positions  (24, 3)          float32   neutral T-pose joints (J_regressor @ v_template)
    shapedirs        (6890, 3, 10)    float32   first-10 SMPL shape blendshapes
    joint_shapedirs  (24, 3, 10)      float32   J_regressor @ shapedirs (joint betas response)

``shapedirs`` and ``joint_shapedirs`` carry the SMPL beta-space basis
that the web viewer's SkinnedMesh uses to apply per-player body shape
from HMR-fitted ``betas``. Only the first 10 components are kept (the
SMPL ``betas`` vector); ``shapedirs[:, :, 10:]`` in the source pkl are
DMPL/extension coefficients that HMR doesn't emit.

Run::

    python scripts/extract_smpl_neutral.py

Idempotent: skips re-extraction iff the npz already exists, is newer
than the pkl, AND contains every expected key. Adding a new key to the
output here invalidates the cached npz on the next run.
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

# Number of SMPL shape blendshapes carried over from the source pkl. The
# pkl ships 300 columns (10 SMPL betas + DMPL extensions); HMR's
# ``betas`` is the first 10, so we slice and discard the rest to keep
# the on-disk size and the JSON payload sent to the browser bounded.
_N_BETAS = 10

# Keys the consuming code (web server endpoint, blender FBX exporter)
# expects in the npz. Adding a new key here forces re-extraction on
# the next run even when the source pkl hasn't changed.
_EXPECTED_KEYS = {
    "v_template", "faces", "weights", "joint_positions",
    "shapedirs", "joint_shapedirs",
}


def main() -> int:
    if not PKL_PATH.exists():
        sys.stderr.write(f"SMPL pkl not found at {PKL_PATH}\n")
        return 1

    if OUT_PATH.exists() and OUT_PATH.stat().st_mtime >= PKL_PATH.stat().st_mtime:
        try:
            existing_keys = set(np.load(OUT_PATH, allow_pickle=False).files)
        except Exception:
            existing_keys = set()
        if _EXPECTED_KEYS.issubset(existing_keys):
            print(f"[smpl-extract] {OUT_PATH} is up to date, skipping")
            return 0
        missing = _EXPECTED_KEYS - existing_keys
        print(
            f"[smpl-extract] {OUT_PATH} missing keys {sorted(missing)}; "
            "re-extracting"
        )

    with PKL_PATH.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    # SMPL fields may be chumpy objects; np.asarray coerces both.
    v_template = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    weights = np.asarray(data["weights"], dtype=np.float32)

    # The joint positions are computed from v_template via the
    # J_regressor matrix. We store these so the Blender exporter places
    # bones at the actual SMPL joint locations (matching the mesh)
    # rather than relying on the hand-typed SMPL_REST_JOINTS_YUP table,
    # which has the pelvis re-centred to (0,0,0) and so sits ~22cm
    # above the real joint positions.
    j_reg = data["J_regressor"]
    if hasattr(j_reg, "todense"):
        j_reg = j_reg.todense()
    j_regressor = np.asarray(j_reg, dtype=np.float32)
    joint_positions = (j_regressor @ v_template).astype(np.float32)

    # Shape blendshapes: SMPL's ``shapedirs`` is (V, 3, 300). The first
    # ``_N_BETAS`` (= 10) columns are the SMPL body-shape basis that
    # HMR fits. The remaining columns are DMPL/extension coefficients
    # we don't use and don't ship to the browser.
    shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
    shapedirs = shapedirs_full[:, :, :_N_BETAS].copy()

    # Pre-multiply the joint regressor through the shape basis. With
    # this the browser can recover beta-adjusted joint positions
    # without shipping the full (24, 6890) J_regressor: simply
    # ``joint_positions_shaped = joint_positions + joint_shapedirs @ betas``.
    joint_shapedirs = np.einsum(
        "jv,vbk->jbk", j_regressor, shapedirs
    ).astype(np.float32)

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
    if joint_positions.shape != (expected_j, 3):
        sys.stderr.write(
            f"unexpected joint_positions shape: {joint_positions.shape}\n"
        )
        return 2
    if shapedirs.shape != (expected_v, 3, _N_BETAS):
        sys.stderr.write(f"unexpected shapedirs shape: {shapedirs.shape}\n")
        return 2
    if joint_shapedirs.shape != (expected_j, 3, _N_BETAS):
        sys.stderr.write(
            f"unexpected joint_shapedirs shape: {joint_shapedirs.shape}\n"
        )
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        v_template=v_template,
        faces=faces,
        weights=weights,
        joint_positions=joint_positions,
        shapedirs=shapedirs,
        joint_shapedirs=joint_shapedirs,
    )
    print(
        f"[smpl-extract] wrote {OUT_PATH} "
        f"(v={v_template.shape[0]}, f={faces.shape[0]}, "
        f"w={weights.shape}, j={joint_positions.shape}, "
        f"shapedirs={shapedirs.shape})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

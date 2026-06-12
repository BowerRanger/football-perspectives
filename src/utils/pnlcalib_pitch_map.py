"""Map PnLCalib keypoint IDs to world coordinates in THIS project's frame.

Recovers the proven keypoint-table loader from the (shelved) calibration
work: it imports PnLCalib's own ``keypoint_world_coords_2D`` /
``keypoint_aux_world_coords_2D`` tables via a temporary sys.path swap, so the
table is never hand-transcribed.

IMPORTANT: ``utils/utils_calib.py`` RE-CENTRES the table at import time
(``[x - 52.5, y - 34]``), so the imported values are in PnLCalib's centred
pitch frame (x in [-52.5, 52.5], y in [-34, 34]); kp1 imports as
``[-52.5, -34]``, NOT ``[0, 0]``. Goalpost tops {12,15,16,19} are z=-2.44
(z-down). This project is near-left-corner (x in [0,105], y toward the far
touchline in [0,68], z-up). The point transform — identical to the position
half of ``neural_calibrator.convert_pnlcalib_to_ours`` — is:
``x_ours = x + 52.5, y_ours = 34 - y, z_ours = -z``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PNLCALIB_ROOT = _REPO_ROOT / "third_party" / "PnLCalib"
_HALF_LENGTH = 52.5
_HALF_WIDTH = 34.0

# PnLCalib assigns z = -2.44 to the four goalpost-top keypoints (crossbars).
_GOALPOST_TOP_KEYS = {12, 15, 16, 19}
_GOALPOST_Z = -2.44


def _load_pnlcalib_keypoint_table() -> dict[int, tuple[float, float, float]]:
    """Import PnLCalib's keypoint world tables; return {kp_id: (x, y, z)} in
    PnLCalib's own centred frame. kp_id is 1-based: 1..57 main, 58..73 aux."""
    if not _PNLCALIB_ROOT.exists():
        raise RuntimeError(
            f"PnLCalib submodule missing at {_PNLCALIB_ROOT}. "
            "Run `git submodule update --init --recursive`."
        )
    src_path = str(_REPO_ROOT / "src")
    removed = [p for p in sys.path if p == src_path or p.rstrip("/") == src_path]
    for p in removed:
        sys.path.remove(p)
    sys.path.insert(0, str(_PNLCALIB_ROOT))
    try:
        from utils.utils_calib import (  # type: ignore
            keypoint_aux_world_coords_2D,
            keypoint_world_coords_2D,
        )
    finally:
        try:
            sys.path.remove(str(_PNLCALIB_ROOT))
        except ValueError:
            pass
        for p in removed:
            sys.path.append(p)

    table: dict[int, tuple[float, float, float]] = {}
    for idx, (xw, yw) in enumerate(keypoint_world_coords_2D):
        kp_id = idx + 1
        zw = _GOALPOST_Z if kp_id in _GOALPOST_TOP_KEYS else 0.0
        table[kp_id] = (float(xw), float(yw), float(zw))
    for idx, (xw, yw) in enumerate(keypoint_aux_world_coords_2D):
        table[idx + 1 + 57] = (float(xw), float(yw), 0.0)
    return table


_TABLE: dict[int, tuple[float, float, float]] | None = None


def _get_table() -> dict[int, tuple[float, float, float]]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _load_pnlcalib_keypoint_table()
    return _TABLE


def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    if name == "NUM_KEYPOINTS":
        return len(_get_table())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def keypoint_world_xyz_ours(kp_id: int) -> tuple[float, float, float] | None:
    """World (x, y, z) in THIS project's pitch frame for 1-based PnLCalib
    keypoint ``kp_id``, or None if unknown. The imported table is in
    PnLCalib's CENTRED frame; convert to ours (matches
    ``convert_pnlcalib_to_ours`` for a position)."""
    entry = _get_table().get(kp_id)
    if entry is None:
        return None
    x, y, z = entry
    return (x + _HALF_LENGTH, _HALF_WIDTH - y, -z)

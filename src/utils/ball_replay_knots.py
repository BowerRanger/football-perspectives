"""Cross-replay fixes -> hybrid-trajectory ``Knot``s (operator-wins +
physical-volume gates).

Cross-replay triangulated fixes (``src/utils/ball_cross_replay.py``,
``src/schemas/ball_fixes.py::BallFix``) are the pipeline's only ABSOLUTE
3-D ground truth (sub-20cm campaign W5u-z2: 0.04-0.61m vs operator rays,
first-absolute-GT finding). But they come from a SEPARATE geometric
pipeline (a partner shot's camera + a sync offset) that can be locally
plausible yet globally wrong: W6 found s013's replay-group partner
cameras all reprojected at 3-6px per-frame residual while their fixes
landed 6-8m underground — invisible to any lateral/pixel gate, visible
only once you ask "is this point physically possible" or "does it agree
with an independent observation".

Two gates run here, at Knot-construction time (downstream of, and in
addition to, ``ball_cross_replay``'s pair-level gates
``pair_geometry_trusted``/``filter_physical_fixes``/``filter_moving_fixes``,
which reject whole untrustworthy PAIRS before a ``BallFixSet`` is even
written):

1. **Physical volume** — a fix outside the court volume a real ball can
   occupy (``ball_cross_replay.physically_possible``) is dropped
   regardless of anything else. Catches an isolated bad fix inside an
   otherwise-trusted pair (the fraction-based pair gate only rejects the
   whole pair once BAD/total exceeds ``max_impossible_frac``).
2. **Operator wins** — a fix whose 3-D position, reprojected into a
   manual anchor's own camera frame (the closest one within
   ``adjacent_frames``), lands more than ``tol_px`` from that anchor's
   clicked pixel disagrees with the operator's ground truth and is
   dropped. Manual anchors always override automatic data (CLAUDE.md
   invariant); this is that invariant's cross-replay instance.

Pure module: no file/video access.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_cross_replay import Cams, physically_possible
from src.utils.ball_hybrid_types import Knot
from src.utils.camera_projection import project_world_to_image


def fixes_to_knots(
    fixes: Iterable,
    manual_anchors: Iterable[BallAnchor],
    cams: Cams,
    *,
    tol_px: float,
    adjacent_frames: int = 1,
    weight: float = 1.0,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[Knot], list[dict]]:
    """Turn cross-replay ``fixes`` into depth-hard trajectory ``Knot``s.

    Args:
        fixes: iterable of objects with ``.frame`` (int) and ``.xyz``
            (3-tuple) — :class:`~src.schemas.ball_fixes.BallFix` is the
            expected input, but anything duck-typed the same way works
            (including a previously-gated ``Knot``, so this composes).
        manual_anchors: this shot's own operator clicks
            (``BallAnchorSet.anchors``). Anchors with ``image_xy is None``
            (``off_screen_flight``) never constrain a fix.
        cams: ``{frame: (K, R, t)}`` for this shot, matching
            ``ball_cross_replay.Cams``.
        tol_px: reprojection-disagreement tolerance in PIXELS, checked at
            the nearest manual anchor's own frame.
        adjacent_frames: how many frames away a manual anchor may sit from
            a fix's frame and still gate it (0 = exact frame only). The
            single CLOSEST anchor in range is checked (ties toward the
            exact frame) — a fix is judged against the one click most
            likely to be the same real ball position, not against every
            anchor in the window.
        weight: relative solver weight to stamp on every accepted Knot
            (the trajectory layer's fit weighting is its own concern;
            this is just a pass-through default).
        distortion: this shot's ``(k1, k2)`` radial distortion.

    Returns:
        ``(knots, dropped)``. ``knots`` are depth-hard
        (``kind="fix"``, ``source="fix"``). ``dropped`` has one dict per
        rejected fix: always ``{"frame", "reason", ...}`` with
        ``reason`` one of ``"physically_impossible"`` (+ ``"xyz"``) or
        ``"manual_anchor_conflict"`` (+ ``"anchor_frame"``, ``"px_dist"``,
        ``"tol_px"``).
    """
    anchors_by_frame: dict[int, BallAnchor] = {
        a.frame: a for a in manual_anchors if a.image_xy is not None
    }

    knots: list[Knot] = []
    dropped: list[dict] = []

    for fx in fixes:
        frame = int(fx.frame)
        xyz = (float(fx.xyz[0]), float(fx.xyz[1]), float(fx.xyz[2]))

        if not physically_possible(xyz):
            dropped.append({
                "frame": frame,
                "reason": "physically_impossible",
                "xyz": list(xyz),
            })
            continue

        conflict = _manual_anchor_conflict(
            frame, xyz, anchors_by_frame, cams,
            tol_px=tol_px, adjacent_frames=adjacent_frames,
            distortion=distortion,
        )
        if conflict is not None:
            dropped.append({
                "frame": frame,
                "reason": "manual_anchor_conflict",
                **conflict,
            })
            continue

        knots.append(Knot(
            frame=frame, xyz=xyz, kind="fix", depth_hard=True,
            source="fix", weight=weight,
        ))

    return knots, dropped


def _manual_anchor_conflict(
    frame: int,
    xyz: tuple[float, float, float],
    anchors_by_frame: dict[int, BallAnchor],
    cams: Cams,
    *,
    tol_px: float,
    adjacent_frames: int,
    distortion: tuple[float, float],
) -> dict | None:
    """``None`` when no nearby manual anchor disagrees; else a detail
    dict (``anchor_frame``, ``px_dist``, ``tol_px``) for the drop record.
    """
    candidates = sorted(
        (f for f in anchors_by_frame if abs(f - frame) <= adjacent_frames),
        key=lambda f: abs(f - frame),
    )
    if not candidates:
        return None
    anchor_frame = candidates[0]
    if anchor_frame not in cams:
        return None
    anchor = anchors_by_frame[anchor_frame]
    K, R, t = cams[anchor_frame]
    uv = project_world_to_image(
        K, R, t, distortion,
        np.asarray(xyz, dtype=np.float64).reshape(1, 3),
    )[0]
    dist = float(np.hypot(
        uv[0] - anchor.image_xy[0], uv[1] - anchor.image_xy[1],
    ))
    if dist <= tol_px:
        return None
    return {"anchor_frame": anchor_frame, "px_dist": dist, "tol_px": tol_px}

"""Single source of truth for sub-pixel-experiment measurements.

Loads an anchor JSON, runs the same solve the camera stage runs, and
reports per-anchor mean / max landmark pixel deviance, the locked
camera centre, and a flag for >2 m body motion (the user-supplied
hard-failure threshold for the experiment).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.utils.anchor_solver import (
    _estimate_lens_from_best_anchor,
    _estimate_lens_jointly,
    _is_rich,
    refine_with_bounded_motion,
    refine_with_shared_translation,
    reprojection_residual_for_anchor,
    solve_anchors_jointly,
)
from src.utils.camera_projection import project_world_to_image


class AnchorMetrics(NamedTuple):
    frame: int
    n_landmarks: int
    mean_px: float
    max_px: float
    is_rich: bool


def _landmark_residuals(
    anchor: Anchor,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> np.ndarray:
    if not anchor.landmarks:
        return np.empty(0)
    pts = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    obs = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
    proj = project_world_to_image(K, R, t, distortion, pts)
    return np.linalg.norm(proj - obs, axis=1)


def _per_anchor_metrics(
    anchors: tuple[Anchor, ...],
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    distortion: tuple[float, float],
) -> list[AnchorMetrics]:
    by_frame = {a.frame: a for a in anchors}
    out: list[AnchorMetrics] = []
    for frame, (K, R, t) in sorted(per_anchor_KRt.items()):
        anchor = by_frame[frame]
        residuals = _landmark_residuals(anchor, K, R, t, distortion)
        if residuals.size == 0:
            out.append(AnchorMetrics(frame, 0, 0.0, 0.0, _is_rich(anchor)))
            continue
        out.append(AnchorMetrics(
            frame=frame,
            n_landmarks=len(residuals),
            mean_px=float(residuals.mean()),
            max_px=float(residuals.max()),
            is_rich=_is_rich(anchor),
        ))
    return out


def _print_metrics(
    label: str,
    metrics: list[AnchorMetrics],
    camera_centre: np.ndarray | None,
    distortion: tuple[float, float],
    body_motion_m: float | None,
) -> None:
    print(f"\n=== {label} ===")
    print(f"{'frame':>5} {'n':>3} {'mean':>7} {'max':>7} {'flags':>10}")
    fails = []
    for m in metrics:
        gate_mean = m.mean_px < 1.0
        gate_max = m.max_px < 3.0
        gate = "PASS" if (gate_mean and gate_max) else "FAIL"
        if gate == "FAIL":
            fails.append(m.frame)
        rich = "*" if m.is_rich else " "
        print(
            f"{m.frame:5d} {m.n_landmarks:3d} {m.mean_px:7.3f} {m.max_px:7.3f}  {gate} {rich}"
        )
    if camera_centre is not None:
        print(
            f"C_locked = ({camera_centre[0]:.3f}, {camera_centre[1]:.3f}, "
            f"{camera_centre[2]:.3f})  distortion = ({distortion[0]:+.4f}, "
            f"{distortion[1]:+.4f})"
        )
    if body_motion_m is not None:
        flag = "  [FAIL >2m]" if body_motion_m > 2.0 else ""
        print(f"body motion across anchor frames: {body_motion_m:.3f} m{flag}")
    if fails:
        print(f"FAILED anchors: {fails}")
    else:
        print("ALL ANCHORS PASS (<1 px mean, <3 px max)")


def _body_motion(
    per_anchor_KRt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    anchors: tuple[Anchor, ...],
) -> float:
    """Max pairwise distance between any two per-anchor camera centres,
    restricted to rich anchors. Line-only anchors have under-determined C
    (the perpendicular-to-line component is unconstrained), so including
    them in motion accounting inflates the metric to meaningless values.
    """
    by_frame = {a.frame: a for a in anchors}
    Cs = []
    for af, (_K, R, t) in per_anchor_KRt.items():
        a = by_frame.get(af)
        if a is None or not _is_rich(a):
            continue
        C = -R.astype(np.float64).T @ t.astype(np.float64)
        Cs.append(C)
    if len(Cs) < 2:
        return 0.0
    Cs_arr = np.stack(Cs)
    diff = Cs_arr.max(axis=0) - Cs_arr.min(axis=0)
    return float(np.linalg.norm(diff))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("--no-relock", action="store_true",
                        help="Skip refine_with_shared_translation (solo-solve only).")
    parser.add_argument("--motion-budget-m", type=float, default=0.0,
                        help="When >0, use bounded-motion relock with this budget "
                             "(metres) instead of strict static-camera relock.")
    parser.add_argument("--lens-prior", action="store_true",
                        help="Estimate (cx, cy, k1, k2) from the single best anchor.")
    parser.add_argument("--lens-prior-joint", action="store_true",
                        help="Estimate (cx, cy, k1, k2) jointly across all rich anchors.")
    parser.add_argument("--drop-landmarks", action="append", default=[],
                        help="Substring(s) of landmark.name to filter out. Repeatable.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print solver debug logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    aset = AnchorSet.load(args.anchors_path)
    drop = tuple(args.drop_landmarks)
    if drop:
        cleaned: list[Anchor] = []
        for a in aset.anchors:
            filtered = tuple(
                lm for lm in a.landmarks
                if not any(d in lm.name for d in drop)
            )
            cleaned.append(Anchor(frame=a.frame, landmarks=filtered, lines=a.lines))
        anchors = tuple(cleaned)
        n_dropped = sum(len(a.landmarks) - len(c.landmarks)
                        for a, c in zip(aset.anchors, cleaned))
        print(f"dropped {n_dropped} landmarks matching {drop}")
    else:
        anchors = tuple(aset.anchors)

    lens_prior = None
    if args.lens_prior_joint:
        lens_prior = _estimate_lens_jointly(anchors, image_size=aset.image_size)
        print(f"joint lens prior: {lens_prior}")
    elif args.lens_prior:
        lens_prior = _estimate_lens_from_best_anchor(anchors, image_size=aset.image_size)
        print(f"single-anchor lens prior: {lens_prior}")

    sol = solve_anchors_jointly(
        anchors, image_size=aset.image_size, lens_prior=lens_prior,
    )
    if not args.no_relock:
        if args.motion_budget_m > 0.0:
            sol = refine_with_bounded_motion(anchors, sol, args.motion_budget_m)
        else:
            sol = refine_with_shared_translation(anchors, sol)

    body_motion = _body_motion(sol.per_anchor_KRt, anchors) if args.no_relock else 0.0
    label = (
        "solo-solve (no relock)" if args.no_relock else "joint + static-camera relock"
    )
    metrics = _per_anchor_metrics(anchors, sol.per_anchor_KRt, sol.distortion)
    centre = (
        np.asarray(sol.camera_centre) if sol.camera_centre is not None else None
    )
    _print_metrics(label, metrics, centre, sol.distortion, body_motion if args.no_relock else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

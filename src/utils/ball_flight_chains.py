"""Gravity-arc refit of airborne anchor chains (sub-20cm campaign W3).

In events mode, ``airborne_low/mid/high`` anchors historically resolved via
ray ∩ a fixed height plane (z = 1.0/6.0/15.0) — bucket placeholders with
metre-scale depth error. This module re-resolves every maximal run of
airborne anchors bracketed by two hard knots (touch/kick/bounce/goal/…,
whose worlds are already resolved) with a single gravity-arc fit:

- the two hard-knot worlds are (near-)hard constraints,
- every chain anchor's clicked pixel is a reprojection observation (a ray,
  not a plane),
- the airborne buckets demote to one-sided z-range hinges.

Two hard 3-D knots + gravity fully determine the 6-DOF arc, depth included
(the 2026-05-31 C2 result), so interior airborne positions — and every
in-between frame the interpolator later renders — inherit physical depth.

Chains missing a bracket, or whose fit residual exceeds the gate, keep
their bucket resolution and are flagged in the diagnostics instead.
Pure numpy/scipy via ``bundle_adjust`` — no torch, no video.
"""

from __future__ import annotations

import logging

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_anchor_heights import (
    AIRBORNE_BUCKETS,
    HARD_KNOT_STATES,
    airborne_bucket_range,
)
from src.utils.bundle_adjust import fit_parabola_to_image_observations
from src.utils.camera_projection import project_world_to_image

logger = logging.getLogger(__name__)

# Sanity caps on an accepted arc (anything beyond is a mis-fit, not football).
_MAX_LAUNCH_SPEED_M_S = 60.0
_MAX_ARC_Z_M = 40.0
# An extra (detection) observation joins the stage-2 refit only when the
# anchors-only arc already reprojects within this of it — junk detections
# (players, static lock-ons) never poison the fit.
_EXTRA_OBS_GATE_PX = 30.0


def _arc_residual_px(p0, v0, obs, Ks, Rs, ts, distortion, fps, f0) -> float:
    """Median reprojection error of the arc against ``obs`` pixels."""
    errs = []
    g = np.array([0.0, 0.0, -9.81])
    for (f, uv), K, R, t in zip(obs, Ks, Rs, ts):
        tt = (f - f0) / fps
        w = p0 + v0 * tt + 0.5 * g * tt * tt
        proj = project_world_to_image(K, R, t, distortion,
                                      np.asarray(w).reshape(1, 3))[0]
        errs.append(float(np.linalg.norm(proj - np.asarray(uv, dtype=float))))
    return float(np.median(errs)) if errs else float("inf")


def _chains(anchor_by_frame, world_for_anchor):
    """Yield (start_hard_frame, [airborne frames...], end_hard_frame)."""
    frames = sorted(anchor_by_frame)
    runs: list[tuple[int | None, list[int], int | None]] = []
    current_air: list[int] = []
    last_hard: int | None = None
    for f in frames:
        state = anchor_by_frame[f].state
        if state in AIRBORNE_BUCKETS:
            current_air.append(f)
            continue
        is_hard = (state in HARD_KNOT_STATES
                   and world_for_anchor.get(f) is not None)
        if current_air:
            runs.append((last_hard, current_air, f if is_hard else None))
            current_air = []
        if is_hard:
            last_hard = f
        elif state not in AIRBORNE_BUCKETS:
            # off_screen_flight or an unresolved hard state breaks the chain.
            last_hard = None
    if current_air:
        runs.append((last_hard, current_air, None))
    return runs


def refit_airborne_chains(
    *,
    anchor_by_frame: dict[int, BallAnchor],
    world_for_anchor: dict[int, tuple[float, float, float] | None],
    per_frame_K,
    per_frame_R,
    per_frame_t,
    distortion: tuple[float, float],
    fps: float,
    max_residual_px: float = 5.0,
    extra_observations: dict[int, tuple[float, float]] | None = None,
) -> tuple[dict[int, tuple[float, float, float]], list[dict]]:
    """Return ``({airborne_frame: refit_world}, diagnostics)``.

    Only interior airborne-bucket anchors of successfully fitted chains
    appear in the updates map; everything else is untouched.
    ``extra_observations`` (W4) are real in-span detection pixels — they
    densify the fit and keep chains determined when interior anchor clicks
    are absent (e.g. a hold-out run or a lightly-anchored clip).
    """
    updates: dict[int, tuple[float, float, float]] = {}
    diags: list[dict] = []
    extra_observations = extra_observations or {}
    for start, air_frames, end in _chains(anchor_by_frame, world_for_anchor):
        if start is None or end is None:
            diags.append({
                "kind": "underconstrained_chain",
                "air_frames": list(air_frames),
                "note": "airborne run lacks a bracketing hard knot; "
                        "bucket heights kept — add a bracketing anchor",
            })
            continue
        chain_frames = [start, *air_frames, end]
        anchor_obs = []
        for f in chain_frames:
            anc = anchor_by_frame[f]
            if (anc.image_xy is not None and f in per_frame_K
                    and f in per_frame_R and f in per_frame_t):
                anchor_obs.append((f, (float(anc.image_xy[0]),
                                       float(anc.image_xy[1]))))
        extras = []
        anchored = {f for f, _ in anchor_obs}
        for f in sorted(extra_observations):
            if (start < f < end and f not in anchored
                    and f in per_frame_K and f in per_frame_R
                    and f in per_frame_t):
                uv = extra_observations[f]
                extras.append((f, (float(uv[0]), float(uv[1]))))
        if (len(anchor_obs) < 2 or anchor_obs[0][0] != start
                or anchor_obs[-1][0] != end
                or len(anchor_obs) + len(extras) < 3):
            diags.append({
                "kind": "underconstrained_chain",
                "air_frames": list(air_frames),
                "note": "insufficient pixel observations for a chain fit",
            })
            continue

        def _cams_for(o):
            return ([np.asarray(per_frame_K[f]) for f, _ in o],
                    [np.asarray(per_frame_R[f]) for f, _ in o],
                    [np.asarray(per_frame_t[f]) for f, _ in o])

        f0 = start
        knots = {
            0: np.asarray(world_for_anchor[start], dtype=float),
            end - f0: np.asarray(world_for_anchor[end], dtype=float),
        }
        z_ranges = {
            f - f0: bucket
            for f in air_frames
            if (bucket := airborne_bucket_range(
                anchor_by_frame[f].state)) is not None
        }

        def _fit(o):
            Ks, Rs, ts = _cams_for(o)
            return fit_parabola_to_image_observations(
                o, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
                distortion=distortion, p0_fixed=None,
                knot_frames={k - (o[0][0] - f0): w for k, w in knots.items()},
                z_range_frames={k - (o[0][0] - f0): b
                                for k, b in z_ranges.items()},
            )

        # Two-stage robust fit: anchors first (operator clicks are trusted),
        # then only extras the anchors-only arc already agrees with — and
        # the refit must not degrade the anchor residual (W5b).
        n_extra_used = 0
        try:
            p0, v0, residual = _fit(anchor_obs)
            aK, aR, aT = _cams_for(anchor_obs)
            anchor_res = _arc_residual_px(
                p0, v0, anchor_obs, aK, aR, aT, distortion, fps, f0)
            inliers = []
            for f, uv in extras:
                K = np.asarray(per_frame_K[f])
                R = np.asarray(per_frame_R[f])
                t = np.asarray(per_frame_t[f])
                if _arc_residual_px(p0, v0, [(f, uv)], [K], [R], [t],
                                    distortion, fps, f0) <= _EXTRA_OBS_GATE_PX:
                    inliers.append((f, uv))
            if inliers:
                merged = sorted([*anchor_obs, *inliers], key=lambda o: o[0])
                p0_2, v0_2, residual_2 = _fit(merged)
                anchor_res_2 = _arc_residual_px(
                    p0_2, v0_2, anchor_obs, aK, aR, aT, distortion, fps, f0)
                if anchor_res_2 <= anchor_res + 1.0:
                    p0, v0 = p0_2, v0_2
                    anchor_res = anchor_res_2
                    n_extra_used = len(inliers)
            # Acceptance grades the arc against the trusted anchor clicks.
            residual = float(anchor_res)
        except Exception as exc:  # noqa: BLE001 — fit failure keeps buckets
            diags.append({
                "kind": "chain_fit", "accepted": False,
                "span": [start, end], "error": str(exc),
            })
            continue
        arc_z = [
            float(p0[2] + v0[2] * ((f - f0) / fps)
                  - 0.5 * 9.81 * ((f - f0) / fps) ** 2)
            for f in chain_frames
        ]
        ok = (
            np.isfinite(residual)
            and residual <= max_residual_px
            and float(np.linalg.norm(v0)) <= _MAX_LAUNCH_SPEED_M_S
            and all(-1.0 <= z <= _MAX_ARC_Z_M for z in arc_z)
        )
        diags.append({
            "kind": "chain_fit", "accepted": bool(ok),
            "span": [start, end], "n_air": len(air_frames),
            "n_extra_obs": len(extras), "n_extra_used": n_extra_used,
            "residual_px": float(residual),
        })
        if not ok:
            logger.info(
                "ball flight-chain [%d,%d]: fit rejected "
                "(residual %.1fpx > %.1f or implausible arc) — "
                "bucket heights kept", start, end, residual, max_residual_px,
            )
            continue
        g = np.array([0.0, 0.0, -9.81])
        for f in air_frames:
            t = (f - f0) / fps
            w = p0 + v0 * t + 0.5 * g * t * t
            updates[f] = (float(w[0]), float(w[1]), float(w[2]))
    return updates, diags


__all__ = ["refit_airborne_chains"]

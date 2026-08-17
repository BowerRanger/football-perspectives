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

logger = logging.getLogger(__name__)

# Sanity caps on an accepted arc (anything beyond is a mis-fit, not football).
_MAX_LAUNCH_SPEED_M_S = 60.0
_MAX_ARC_Z_M = 40.0


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
) -> tuple[dict[int, tuple[float, float, float]], list[dict]]:
    """Return ``({airborne_frame: refit_world}, diagnostics)``.

    Only interior airborne-bucket anchors of successfully fitted chains
    appear in the updates map; everything else is untouched.
    """
    updates: dict[int, tuple[float, float, float]] = {}
    diags: list[dict] = []
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
        obs = []
        for f in chain_frames:
            anc = anchor_by_frame[f]
            if (anc.image_xy is not None and f in per_frame_K
                    and f in per_frame_R and f in per_frame_t):
                obs.append((f, (float(anc.image_xy[0]),
                                float(anc.image_xy[1]))))
        if len(obs) < 3 or obs[0][0] != start or obs[-1][0] != end:
            diags.append({
                "kind": "underconstrained_chain",
                "air_frames": list(air_frames),
                "note": "insufficient pixel observations for a chain fit",
            })
            continue
        f0 = obs[0][0]
        Ks = [np.asarray(per_frame_K[f]) for f, _ in obs]
        Rs = [np.asarray(per_frame_R[f]) for f, _ in obs]
        ts = [np.asarray(per_frame_t[f]) for f, _ in obs]
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
        try:
            p0, v0, residual = fit_parabola_to_image_observations(
                obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
                distortion=distortion, p0_fixed=None,
                knot_frames=knots, z_range_frames=z_ranges,
            )
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

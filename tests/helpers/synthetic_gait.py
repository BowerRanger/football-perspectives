"""Analytic synthetic-walk generator — the ground-truth oracle for
foot-contact tests (see docs/superpowers/plans/2026-09-02-foot-contact-
locomotion.md, Task 1).

Builds an alternating-stance walk cycle "backwards": pick exact foot
world targets per frame (constant during stance, smoothly interpolated
during swing/flight), then solve a sagittal-plane 2-link IK (hip pitch +
knee flexion, both pure rotations about the local x axis) so the SMPL FK
foot joint lands EXACTLY on the target every frame. Stance feet are
therefore stationary to floating-point precision, not just approximately
so — this is what lets the contact-detection and foot-lock tests assert
exact tolerances instead of "close enough".

Independent of ``src/`` on purpose (per the plan): the 2-link IK law-of-
cosines math is duplicated here rather than imported, since Task 7's
``src/utils/foot_lock.py`` doesn't exist yet when this fixture is
written and — even once it does — this fixture must stay a trustworthy
oracle that isn't circularly validating itself against the code it's
supposed to be testing.

Geometry conventions (derived from ``SMPL_REST_JOINTS_YUP``, see that
table's docstring in ``src/utils/smpl_skeleton.py``):

  - Canonical +y is up, canonical x is lateral (left/right), and — because
    the SMPL leg chain's local rotations are pure x-axis rotations —
    canonical (y, z) is the plane a leg actually swings in. Canonical x
    per leg joint is therefore FIXED regardless of pose (only rotates
    the y/z components), so a leg's reachable target set is a 2-D disc
    in that joint's own (y, z) plane.
  - We pick ``root_R`` so canonical -z faces the direction of travel:
    ``root_R`` is built from an orthonormal frame
    ``(right, up, backward) = (col_x, col_y, col_z)`` with
    ``forward = (cos(direction_deg), sin(direction_deg), 0)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from src.utils.smpl_skeleton import SMPL_PARENTS, SMPL_REST_JOINTS_YUP

# SMPL joint indices used by the leg chain (see CLAUDE.md: hips 1/2, knees
# 4/5, ankles 7/8, feet(toes) 10/11).
_HIP = {0: 1, 1: 2}
_KNEE = {0: 4, 1: 5}
_FOOT = {0: 10, 1: 11}

# Swing peak lift height (metres) and the world-frame sole target.
_SWING_LIFT_M = 0.12
_STANCE_Z = 0.0

# Fraction of a full gait cycle that a single foot spends in stance.
# < 0.5 so the two feet's stance windows don't fully tile the cycle,
# leaving a "flight" gap where neither foot is down (see module
# docstring / test_walk_swing_foot_lifts and the foot_quality
# contact_ratio metric, which wants some non-trivial flight fraction).
_STANCE_FRAC = 0.4


class GaitTrack(NamedTuple):
    frames: np.ndarray          # (F,) int64
    thetas: np.ndarray          # (F, 24, 3) axis-angle
    root_R: np.ndarray          # (F, 3, 3)
    root_t: np.ndarray          # (F, 3) pitch metres
    betas: np.ndarray           # (10,)
    contacts_true: np.ndarray   # (F, 2) bool [L, R]
    fps: float


def _leg_rest_geometry(side: int) -> tuple[float, float, float, float]:
    """Return ``(L1, L2, phi1, phi2)`` for one leg's sagittal 2-link chain.

    ``L1``/``L2`` are the (y, z)-plane lengths of the hip->knee and
    knee->foot rest segments (the latter lumps the knee->ankle and
    ankle->foot segments together since the ankle's theta is never
    articulated by this fixture — see module docstring). ``phi1``/``phi2``
    are those segments' rest angles in the (y, z) plane, measured the
    same way :func:`_solve_sagittal_ik` measures target angles
    (``atan2(z, y)``).
    """
    hip, knee, foot = _HIP[side], _KNEE[side], _FOOT[side]
    d1 = SMPL_REST_JOINTS_YUP[knee, 1:3] - SMPL_REST_JOINTS_YUP[hip, 1:3]
    d2 = SMPL_REST_JOINTS_YUP[foot, 1:3] - SMPL_REST_JOINTS_YUP[knee, 1:3]
    L1 = float(np.linalg.norm(d1))
    L2 = float(np.linalg.norm(d2))
    phi1 = float(math.atan2(d1[1], d1[0]))
    phi2 = float(math.atan2(d2[1], d2[0]))
    return L1, L2, phi1, phi2


def _solve_sagittal_ik(
    target_yz: np.ndarray, L1: float, L2: float, phi1: float, phi2: float,
) -> tuple[float, float]:
    """Law-of-cosines 2-link IK in the canonical (y, z) plane.

    Returns ``(theta_hip, theta_knee)`` — axis-angle magnitudes for a
    pure local-x rotation at each joint — such that forward-kinematics
    (hip rotation then knee rotation, both about local x) places the
    foot exactly at ``target_yz`` relative to the hip (or as close as
    physically possible if the target is out of reach, in which case the
    distance is clamped).

    This is the same law-of-cosines construction Task 7's
    ``lock_feet_ik`` (src/utils/foot_lock.py) implements for the real
    pipeline — duplicated here per the plan so this fixture is not
    circularly dependent on the code it's used to test.
    """
    ty, tz = float(target_yz[0]), float(target_yz[1])
    d = math.hypot(ty, tz)
    d = min(max(d, abs(L1 - L2) + 1e-6), L1 + L2 - 1e-6)

    cos_a = (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)
    interior_a = math.acos(min(max(cos_a, -1.0), 1.0))

    beta = math.atan2(tz, ty)
    hip_global_angle = beta - interior_a
    theta_hip = hip_global_angle - phi1

    knee_pos = np.array(
        [L1 * math.cos(hip_global_angle), L1 * math.sin(hip_global_angle)]
    )
    to_target = np.array([ty, tz]) - knee_pos
    link2_global_angle = math.atan2(to_target[1], to_target[0])
    theta_knee = link2_global_angle - phi2 - theta_hip

    return theta_hip, theta_knee


def _root_R_for_direction(direction_deg: float) -> np.ndarray:
    """Constant root rotation mapping canonical (x, y, z) to a pitch-world
    frame where canonical +y is world +z (up) and canonical -z faces
    ``direction_deg`` (degrees, world XY plane, 0 = +x)."""
    d = math.radians(direction_deg)
    forward = np.array([math.cos(d), math.sin(d), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    backward = -forward
    right = np.cross(up, backward)  # matches canonical x = y_hat x z_hat
    return np.column_stack([right, up, backward])


def make_walk(
    n_frames: int = 120,
    fps: float = 25.0,
    speed: float = 2.0,
    stride_s: float = 0.6,
    direction_deg: float = 0.0,
) -> GaitTrack:
    """Build an analytic alternating-stance walk cycle in the pitch frame.

    See module docstring for the geometric construction. ``stride_s`` is
    the duration (seconds) of one full gait cycle (both feet through
    stance + swing once); each foot's own stance window is
    ``_STANCE_FRAC`` of that.
    """
    root_R_const = _root_R_for_direction(direction_deg)
    right_dir = root_R_const[:, 0]
    forward_dir = -root_R_const[:, 2]

    geom = {side: _leg_rest_geometry(side) for side in (0, 1)}

    cycle_t = float(stride_s)
    stance_dur = _STANCE_FRAC * cycle_t
    swing_dur = cycle_t - stance_dur
    stance_excursion = speed * stance_dur
    # Forward offset of the planted foot ahead of the hip at touchdown;
    # symmetric placement (A at touchdown, A - stance_excursion at
    # toe-off) minimises the peak horizontal reach required of the leg.
    forward_amp = 0.5 * stance_excursion

    # Pelvis height chosen so the leg never needs to reach outside
    # [|L1-L2|, L1+L2] anywhere in the cycle, with a safety margin.
    L1, L2, _, _ = geom[0]
    max_reach = 0.9 * (L1 + L2)
    min_reach = 1.5 * abs(L1 - L2) + 0.05
    h_sq = max_reach**2 - forward_amp**2
    pelvis_height = math.sqrt(max(h_sq, min_reach**2))

    def pelvis_xy(t: float) -> np.ndarray:
        return speed * t * forward_dir[:2]

    def stance_target_world(side: int, k: int, touchdown_offset: float) -> np.ndarray:
        touchdown_time = touchdown_offset + k * cycle_t
        centre_xy = pelvis_xy(touchdown_time) + forward_amp * forward_dir[:2]
        lateral = SMPL_REST_JOINTS_YUP[_FOOT[side], 0] * right_dir[:2]
        xy = centre_xy + lateral
        return np.array([xy[0], xy[1], _STANCE_Z])

    def _smoothstep(u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def foot_target_world(side: int, t: float) -> tuple[np.ndarray, bool]:
        # side 0 (L) stance window starts at phase 0; side 1 (R) is
        # offset by half a cycle so the two feet alternate.
        touchdown_offset = 0.0 if side == 0 else 0.5 * cycle_t
        u = (t - touchdown_offset) % cycle_t
        k = math.floor((t - touchdown_offset) / cycle_t)
        if u < stance_dur:
            return stance_target_world(side, k, touchdown_offset), True
        swing_start_t = touchdown_offset + k * cycle_t + stance_dur
        liftoff = stance_target_world(side, k, touchdown_offset)
        landing = stance_target_world(side, k + 1, touchdown_offset)
        frac = (t - swing_start_t) / swing_dur
        ease = _smoothstep(frac)
        pos = liftoff + (landing - liftoff) * ease
        pos[2] = _SWING_LIFT_M * math.sin(math.pi * min(max(frac, 0.0), 1.0))
        return pos, False

    frames = np.arange(n_frames, dtype=np.int64)
    thetas = np.zeros((n_frames, 24, 3), dtype=np.float64)
    root_t = np.zeros((n_frames, 3), dtype=np.float64)
    root_R = np.tile(root_R_const, (n_frames, 1, 1))
    contacts_true = np.zeros((n_frames, 2), dtype=bool)

    for f in range(n_frames):
        t = f / fps
        pelvis = np.array([*pelvis_xy(t), pelvis_height])
        root_t[f] = pelvis

        for side in (0, 1):
            hip, knee = _HIP[side], _KNEE[side]
            L1, L2, phi1, phi2 = geom[side]
            target_world, contact = foot_target_world(side, t)
            contacts_true[f, side] = contact

            hip_world = pelvis + root_R_const @ SMPL_REST_JOINTS_YUP[hip]
            delta = target_world - hip_world
            local = root_R_const.T @ delta
            theta_hip, theta_knee = _solve_sagittal_ik(
                np.array([local[1], local[2]]), L1, L2, phi1, phi2,
            )
            thetas[f, hip] = [theta_hip, 0.0, 0.0]
            thetas[f, knee] = [theta_knee, 0.0, 0.0]

    return GaitTrack(
        frames=frames,
        thetas=thetas,
        root_R=root_R,
        root_t=root_t,
        betas=np.zeros(10, dtype=np.float64),
        contacts_true=contacts_true,
        fps=float(fps),
    )

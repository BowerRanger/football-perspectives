"""Stance-pinned root solve + foot-lock two-bone IK + penetration guard —
components [C]/[D] of
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md
(plan Tasks 4 and 7).

Three independent numerics passes, all numpy/scipy only:

- :func:`solve_root_with_pins` — ``hmr_world``'s stance-pinned root solve
  (component [C]): per constrained frame, the pin implies a root
  translation; the implied-minus-carrier delta is interpolated smoothly
  (PCHIP) across unconstrained frames and decays back to zero beyond the
  first/last constrained frame.
- :func:`lock_feet_ik` — ``refined_poses``'s foot-lock finale (component
  [D], part 1): re-pins each stance span to its own median FK foot
  position on the FINAL smoothed track, nudges the root a little, then
  solves an analytic two-bone (hip/knee) IK so the ankle lands where the
  foot needs to be, preserving the foot's global orientation via an
  ankle counter-rotation.
- :func:`penetration_guard` — component [D], part 2: a raise-only pass
  that guarantees no sole-proxy penetration remains, applied very last.

FK convention throughout (matches ``src/utils/smpl_skeleton.py`` exactly):
``thetas[:, 0]`` (the per-frame global orient) is IGNORED — ``root_R``
already carries the root joint's world orientation. Only ``thetas[:, 1:]``
drive the articulated pose; applying both would double-count orientation.
Joint indices (SMPL-24, see CLAUDE.md): hips 1/2, knees 4/5, ankles 7/8,
feet(toes) 10/11, side order [L, R] throughout.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d
from scipy.spatial.transform import Rotation

from src.utils.foot_contact import ContactSpan, FootContacts
from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    axis_angle_to_matrix,
    beta_adjusted_rest_joints,
    compute_all_joint_worlds_batch,
    compute_canonical_joints_batch,
    load_smpl_neutral_model,
)

_EPS = 1e-9

_HIP_IDX = (SMPL_JOINT_NAMES.index("l_hip"), SMPL_JOINT_NAMES.index("r_hip"))
_KNEE_IDX = (SMPL_JOINT_NAMES.index("l_knee"), SMPL_JOINT_NAMES.index("r_knee"))
_ANKLE_IDX = (SMPL_JOINT_NAMES.index("l_ankle"), SMPL_JOINT_NAMES.index("r_ankle"))
_FOOT_IDX = (SMPL_JOINT_NAMES.index("l_foot"), SMPL_JOINT_NAMES.index("r_foot"))

# 5-frame triangular kernel used by lock_feet_ik's root micro-correction
# low-pass (radius 2, weights 1-2-3-2-1 normalized).
_ROOT_CORR_KERNEL = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
_ROOT_CORR_KERNEL = _ROOT_CORR_KERNEL / _ROOT_CORR_KERNEL.sum()

# Width-3 triangular kernel used by penetration_guard's smoothing pass
# (radius 1, weights 1-2-1 normalized).
_PEN_GUARD_KERNEL = np.array([0.25, 0.5, 0.25])

# Default foot-landing tolerance beyond which a clamped IK solve is
# rejected and the whole span is skipped rather than mangled (see
# lock_feet_ik). Overridable via lock_feet_ik's skip_pin_err_m kwarg,
# wired from config as refined_poses.foot_lock.skip_pin_err_m.
_IK_SKIP_TOLERANCE_M = 0.04


def _resolve_rest_joints(
    betas: np.ndarray | None, rest_joints: np.ndarray | None,
) -> np.ndarray:
    """``rest_joints`` wins when given; otherwise beta-adjust from
    ``betas`` (gracefully falling back to the mean-shape constant table
    when ``data/models/smpl_neutral.npz`` is absent — see
    ``beta_adjusted_rest_joints``'s own docstring)."""
    if rest_joints is not None:
        return np.asarray(rest_joints, dtype=np.float64)
    return beta_adjusted_rest_joints(betas, load_smpl_neutral_model())


# ---------------------------------------------------------------------
# [C] Task 4: stance-pinned root solve
# ---------------------------------------------------------------------


def solve_root_with_pins(
    *,
    root_carrier: np.ndarray,
    root_R: np.ndarray,
    thetas: np.ndarray,
    betas: np.ndarray | None,
    contacts: FootContacts,
    fps: float,
    max_correction_m: float = 0.5,
    decay_s: float = 0.6,
    rest_joints: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Stance-pin the root translation on top of a dense carrier path.

    Per constrained frame (one or both feet in a contact span), the pin
    implies a root translation: ``implied = pin - root_R @ canon[foot]``
    — ``foot`` is the SMPL foot/toe joint (10/11), NOT the ankle (7/8)
    (multi-foot double support uses the quality-weighted mean of the two
    implied roots). Pinning the FOOT/toe rather than the ankle is
    deliberate: during real stance the ankle is not physically
    stationary (the tibia rotates over the planted toe as the body
    advances — heel lift), so an implied-root formula built on the ankle
    forces the anatomically-stationary toe to sweep instead, which is
    exactly the Wave-4 stance-skate root cause this function fixes (see
    ``src.utils.foot_contact``'s ``_SMPL_FOOT_IDX`` docstring for the
    full diagnosis). ``delta = implied - root_carrier`` is clamped to
    ``max_correction_m``, then interpolated (monotone-cubic, exact at
    constrained nodes) across every unconstrained frame between the
    first and last constrained frame, and linearly decayed to zero over
    ``decay_s * fps`` frames beyond either end (pure carrier beyond
    that). Returns ``root_carrier + delta_dense``.

    Args:
        root_carrier: (F, 3) dense per-frame root translation to
            correct (today's ankle-mid anchor, or any other carrier).
        root_R: (F, 3, 3) per-frame root world rotation.
        thetas: (F, 24, 3) axis-angle; ``thetas[:, 0]`` ignored.
        betas: (10,) SMPL shape, used only when ``rest_joints`` is
            ``None`` (beta-adjusts the rest joint table).
        contacts: per-foot spans + pins + quality (see
            ``src.utils.foot_contact.FootContacts``).
        fps: clip frame rate, drives the decay-window frame count.
        max_correction_m: hard clamp on ``|delta|`` per constrained
            frame (guards against a bad pin overriding the carrier).
        decay_s: seconds over which ``delta`` decays to zero beyond the
            first/last constrained frame.
        rest_joints: optional (24, 3) beta-adjusted rest joint override.

    Returns:
        ``(root_t, stats)`` where ``stats`` has keys
        ``constrained_frames``, ``mean_delta_m``, ``max_delta_m``,
        ``clamped_frames``.
    """
    root_carrier = np.asarray(root_carrier, dtype=np.float64)
    root_R = np.asarray(root_R, dtype=np.float64)
    n = int(root_carrier.shape[0])
    rest = _resolve_rest_joints(betas, rest_joints)

    stats = {
        "constrained_frames": 0,
        "mean_delta_m": 0.0,
        "max_delta_m": 0.0,
        "clamped_frames": 0,
    }
    if n == 0:
        return root_carrier.copy(), stats

    canon = compute_canonical_joints_batch(thetas, rest)  # (F, 24, 3)
    # Per-side FOOT (toe) offset from root, rotated into world
    # orientation (not translated — this is the "if root_t were 0" foot
    # position). See this function's docstring for why the foot/toe
    # (10/11), not the ankle (7/8), is the joint pinned here.
    off = np.empty((n, 2, 3), dtype=np.float64)
    for side, foot_idx in enumerate(_FOOT_IDX):
        off[:, side, :] = np.einsum("fba,fa->fb", root_R, canon[:, foot_idx, :])

    active_pin = np.full((n, 2, 3), np.nan, dtype=np.float64)
    active_quality = np.zeros((n, 2), dtype=np.float64)
    for span in contacts.spans:
        s, e, side = int(span.start), int(span.end), int(span.side)
        active_pin[s:e, side, :] = np.asarray(span.pin, dtype=np.float64)
        active_quality[s:e, side] = np.asarray(contacts.quality, dtype=np.float64)[s:e, side]

    delta = np.zeros((n, 3), dtype=np.float64)
    constrained = np.zeros(n, dtype=bool)
    clamped_frames = 0

    for f in range(n):
        active_sides = [s for s in (0, 1) if not np.isnan(active_pin[f, s, 0])]
        if not active_sides:
            continue
        constrained[f] = True
        if len(active_sides) == 1:
            s = active_sides[0]
            implied = active_pin[f, s] - off[f, s]
        else:
            weights = np.array([active_quality[f, s] for s in active_sides])
            if weights.sum() <= 0.0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            implied = np.zeros(3, dtype=np.float64)
            for w, s in zip(weights, active_sides):
                implied += w * (active_pin[f, s] - off[f, s])

        d = implied - root_carrier[f]
        dn = float(np.linalg.norm(d))
        if dn > max_correction_m and dn > _EPS:
            d = d * (max_correction_m / dn)
            clamped_frames += 1
        delta[f] = d

    idx = np.where(constrained)[0]
    delta_dense = np.zeros((n, 3), dtype=np.float64)
    if idx.size > 0:
        first, last = int(idx[0]), int(idx[-1])
        rng = np.arange(first, last + 1)
        for axis in range(3):
            vals = delta[idx, axis]
            if idx.size >= 2:
                interp = PchipInterpolator(idx.astype(np.float64), vals, extrapolate=False)
                delta_dense[first:last + 1, axis] = interp(rng.astype(np.float64))
            else:
                delta_dense[first:last + 1, axis] = vals[0]

        decay_frames = max(float(decay_s) * float(fps), _EPS)
        edge_first = delta_dense[first].copy()
        edge_last = delta_dense[last].copy()
        for f in range(0, first):
            alpha = max(0.0, 1.0 - (first - f) / decay_frames)
            delta_dense[f] = alpha * edge_first
        for f in range(last + 1, n):
            alpha = max(0.0, 1.0 - (f - last) / decay_frames)
            delta_dense[f] = alpha * edge_last

    root_t = root_carrier + delta_dense
    norms = np.linalg.norm(delta_dense, axis=1)
    stats["constrained_frames"] = int(idx.size)
    stats["mean_delta_m"] = float(norms.mean())
    stats["max_delta_m"] = float(norms.max())
    stats["clamped_frames"] = int(clamped_frames)
    return root_t, stats


# ---------------------------------------------------------------------
# [D] Task 7: foot-lock two-bone IK
# ---------------------------------------------------------------------


class _LegChain(NamedTuple):
    Rl_hip: np.ndarray
    Rl_knee: np.ndarray
    Rl_ankle: np.ndarray
    R_hip: np.ndarray
    R_knee: np.ndarray
    R_ankle: np.ndarray
    pos_hip: np.ndarray
    pos_knee: np.ndarray
    pos_ankle: np.ndarray
    pos_foot: np.ndarray


class _LegSolve(NamedTuple):
    theta_hip: np.ndarray
    theta_knee: np.ndarray
    theta_ankle: np.ndarray
    foot_world: np.ndarray
    hip_delta_rad: float
    knee_delta_rad: float
    # Small, well-defined DELTA rotations (axis + signed angle) for the
    # hip/knee/ankle edits, each expressed in the frame its own theta
    # lives in (world/canonical for the hip, since its parent's
    # canonical rotation is always identity; the hip's/knee's own
    # local-to-parent frame otherwise). Blending (lock_feet_ik) scales
    # these SMALL angles by the ease weight and composes them onto the
    # OLD rotation MATRIX — never by linearly interpolating two
    # absolute axis-angle vectors. That matters because axis-angle is
    # not a globally continuous parametrization (scipy's
    # ``as_rotvec()`` always returns the canonical <=pi representation,
    # while an upstream track's thetas may be an unwrapped >pi
    # representation of the identical rotation) — subtracting two
    # absolute rotvecs in that situation manufactures a large spurious
    # "delta" for what is geometrically a tiny rotation. Composing a
    # small delta onto the old MATRIX sidesteps the ambiguity entirely.
    hip_axis: np.ndarray
    hip_angle: float
    knee_axis: np.ndarray
    knee_angle: float
    ankle_axis: np.ndarray
    ankle_angle: float


def _fk_leg_chain(
    theta_row: np.ndarray, rest: np.ndarray, hip: int, knee: int, ankle: int, foot: int,
) -> _LegChain:
    """Canonical (root-local) FK for one leg chain, one frame.

    Hip's parent is the pelvis, whose canonical global rotation is
    always identity (``thetas[:, 0]`` ignored) — so ``R_hip`` is just
    the hip's own local rotation, matching ``compute_canonical_joints_batch``.
    """
    Rl_hip = axis_angle_to_matrix(theta_row[hip])
    Rl_knee = axis_angle_to_matrix(theta_row[knee])
    Rl_ankle = axis_angle_to_matrix(theta_row[ankle])
    Rl_foot = axis_angle_to_matrix(theta_row[foot])

    R_hip = Rl_hip
    pos_hip = rest[hip]
    R_knee = R_hip @ Rl_knee
    pos_knee = pos_hip + R_hip @ (rest[knee] - rest[hip])
    R_ankle = R_knee @ Rl_ankle
    pos_ankle = pos_knee + R_knee @ (rest[ankle] - rest[knee])
    R_foot = R_ankle @ Rl_foot
    pos_foot = pos_ankle + R_ankle @ (rest[foot] - rest[ankle])
    return _LegChain(
        Rl_hip=Rl_hip, Rl_knee=Rl_knee, Rl_ankle=Rl_ankle,
        R_hip=R_hip, R_knee=R_knee, R_ankle=R_ankle,
        pos_hip=pos_hip, pos_knee=pos_knee, pos_ankle=pos_ankle, pos_foot=pos_foot,
    )


def _minimal_rotation_axis_angle(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """``(axis, angle)`` of the minimal rotation taking unit vector ``a``
    to unit vector ``b``. ``axis`` is an arbitrary unit vector when
    ``angle`` is ~0 (no rotation needed)."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if c < -1.0 + 1e-10:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return axis / np.linalg.norm(axis), math.pi
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    return axis, math.acos(c)


def _rotate_local(theta_row_joint: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Compose a delta rotation (``axis``/``angle``, small, well-defined)
    onto a joint's CURRENT local rotation (given as its axis-angle
    theta), returning the resulting rotation as a matrix. The delta is
    applied on the LEFT (pre-multiplied) — i.e. in the same frame the
    delta's axis was computed in, consistent throughout this module."""
    R_old = axis_angle_to_matrix(theta_row_joint)
    R_delta = Rotation.from_rotvec(axis * angle).as_matrix()
    return R_delta @ R_old


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector ``v`` by ``angle`` radians about unit ``axis``
    (Rodrigues' formula)."""
    axis = axis / np.linalg.norm(axis)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - math.cos(angle))
    )


def _solve_leg_frame(
    *,
    target_ankle_world: np.ndarray,
    root_R_f: np.ndarray,
    root_t_f: np.ndarray,
    theta_row: np.ndarray,
    rest: np.ndarray,
    hip: int,
    knee: int,
    ankle: int,
    foot: int,
    ik_max_joint_delta_deg: float,
) -> _LegSolve:
    """Analytic two-bone (hip/knee) IK landing the ankle at
    ``target_ankle_world``, then an ankle counter-rotation that keeps
    the foot's GLOBAL orientation unchanged despite the hip/knee edit.

    Works in the root-local canonical frame (``H`` = canonical hip
    position, constant per player — the hip's canonical position never
    depends on thetas since its parent, the pelvis, always has identity
    canonical rotation).

    Algorithm (see module docstring / plan Task 7 for the derivation).
    This is a provably-exact (when unclamped) two-bone placement: rather
    than "flex the knee about its current bend axis, then aim the hip"
    (which implicitly assumes the knee's bend axis is perpendicular to
    the hip->knee rest offset — NOT quite true for the real SMPL rest
    table, which has a small off-axis component, so that decomposition
    silently fails to converge on some frames), this directly places
    the knee at the geometrically required position:

      1. Law-of-cosines: the hip-side angle ``alpha`` between H->knee
         and H->target that makes the H-knee-target triangle have sides
         ``L1``, ``L2``, ``d`` (``d`` = the clipped, reachable distance
         from H to the target).
      2. Rotate ``u = normalize(target - H)`` by ``+-alpha`` about the
         normal of the (current knee direction, target direction) plane
         (falling back to the knee's current bend axis when that plane
         is degenerate) to get ``required_knee_dir`` — picking the sign
         that stays on the same side as the CURRENT knee direction, to
         avoid an elbow flip. By construction of the law of cosines,
         ``knee_new = H + L1*required_knee_dir`` is EXACTLY ``L2`` from
         the target — no perpendicularity assumption needed.
      3. Hip delta = minimal rotation from the current to the required
         knee direction (this is exact, in the WORLD/canonical frame,
         since the hip's canonical parent rotation is always identity).
      4. Knee delta = minimal rotation (computed in world frame, then
         conjugated into the knee's LOCAL frame by the — possibly
         clamped — new hip rotation) from "where the ankle offset would
         point if only the hip changed" to "where it needs to point to
         reach the target from the (possibly-clamped) new knee
         position".
      5. Clamp each delta's ANGLE (not an absolute rotvec difference —
         see ``_LegSolve``) to ``ik_max_joint_delta_deg`` independently.
      6. Counter-rotate the ankle's LOCAL rotation so its GLOBAL
         rotation — and therefore the foot's global orientation — is
         unchanged from before this function ran.
    """
    chain = _fk_leg_chain(theta_row, rest, hip, knee, ankle, foot)
    H = rest[hip]
    L1 = float(np.linalg.norm(rest[knee] - rest[hip]))
    L2 = float(np.linalg.norm(rest[ankle] - rest[knee]))
    max_rad = math.radians(float(ik_max_joint_delta_deg))

    T_local = root_R_f.T @ (np.asarray(target_ankle_world, dtype=np.float64) - root_t_f)

    # --- 1/2/3: place the knee -----------------------------------------
    dir_current = chain.pos_knee - H
    nc = float(np.linalg.norm(dir_current))
    dir_current = dir_current / nc if nc > _EPS else np.array([0.0, -1.0, 0.0])

    d = float(np.linalg.norm(T_local - H))
    d = float(np.clip(d, abs(L1 - L2) + _EPS, L1 + L2 - _EPS))
    u = (T_local - H) / d
    alpha = math.acos(np.clip((L1 ** 2 + d ** 2 - L2 ** 2) / (2.0 * L1 * d), -1.0, 1.0))

    plane_normal = np.cross(dir_current, u)
    if np.linalg.norm(plane_normal) < 1e-7:
        theta_knee_vec = np.asarray(theta_row[knee], dtype=np.float64)
        theta_knee_norm = float(np.linalg.norm(theta_knee_vec))
        bend_axis_local = (
            theta_knee_vec / theta_knee_norm if theta_knee_norm > _EPS
            else np.array([1.0, 0.0, 0.0])
        )
        plane_normal = chain.R_hip @ bend_axis_local
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    req_pos = _rotate_vector(u, plane_normal, alpha)
    req_neg = _rotate_vector(u, plane_normal, -alpha)
    required_knee_dir = (
        req_pos if np.dot(req_pos, dir_current) >= np.dot(req_neg, dir_current) else req_neg
    )

    hip_axis, hip_angle_raw = _minimal_rotation_axis_angle(dir_current, required_knee_dir)
    hip_angle_c = float(min(hip_angle_raw, max_rad))

    Rl_hip_c = _rotate_local(theta_row[hip], hip_axis, hip_angle_c)  # parent (pelvis) canonical rot is I
    R_hip_c = Rl_hip_c
    pos_knee_c = H + R_hip_c @ (rest[knee] - rest[hip])

    # --- 4: place the ankle (knee's own local rotation) -----------------
    ankle_offset_dir_rest = (rest[ankle] - rest[knee])
    ankle_offset_dir_rest = ankle_offset_dir_rest / np.linalg.norm(ankle_offset_dir_rest)
    ankle_dir_with_old_local_knee = R_hip_c @ chain.Rl_knee @ ankle_offset_dir_rest
    ankle_dir_with_old_local_knee = ankle_dir_with_old_local_knee / np.linalg.norm(
        ankle_dir_with_old_local_knee
    )

    dt = T_local - pos_knee_c
    ndt = float(np.linalg.norm(dt))
    target_dir_from_knee = dt / ndt if ndt > _EPS else ankle_dir_with_old_local_knee

    world_knee_axis, world_knee_angle = _minimal_rotation_axis_angle(
        ankle_dir_with_old_local_knee, target_dir_from_knee
    )
    # Conjugate the WORLD delta into the knee's LOCAL frame (local_rot[knee]
    # operates after R_hip_c is applied): a world rotation W corresponds to
    # local delta L = R_hip_c^T @ W @ R_hip_c, i.e. axis R_hip_c^T @ axis_W,
    # same angle (conjugation by an orthogonal matrix preserves angle).
    knee_axis = R_hip_c.T @ world_knee_axis
    knee_axis = knee_axis / np.linalg.norm(knee_axis)
    knee_angle_c = float(min(world_knee_angle, max_rad))

    Rl_knee_c = _rotate_local(theta_row[knee], knee_axis, knee_angle_c)
    R_knee_c = R_hip_c @ Rl_knee_c
    pos_ankle_c = pos_knee_c + R_knee_c @ (rest[ankle] - rest[knee])

    # --- 6: ankle counter-rotation ---------------------------------------
    # Choose the ankle's new LOCAL rotation so its GLOBAL rotation (and
    # hence the foot's global orientation) is exactly what it was before
    # this function touched the hip/knee. The needed LOCAL delta (in the
    # frame ankle's own theta lives in, i.e. after global_rot[knee] is
    # applied) is a small, well-defined rotation on its own — safe to
    # convert to axis/angle directly (see _matrix_axis_angle docstring).
    R_delta_ankle_local = R_knee_c.T @ chain.R_knee
    ankle_axis, ankle_angle = _matrix_axis_angle(R_delta_ankle_local)
    Rl_ankle_c = _rotate_local(theta_row[ankle], ankle_axis, ankle_angle)
    R_ankle_c = chain.R_knee @ chain.Rl_ankle  # == R_knee_c @ Rl_ankle_c, preserved
    pos_foot_c = pos_ankle_c + R_ankle_c @ (rest[foot] - rest[ankle])

    foot_world_new = root_R_f @ pos_foot_c + root_t_f

    return _LegSolve(
        theta_hip=Rotation.from_matrix(Rl_hip_c).as_rotvec(),
        theta_knee=Rotation.from_matrix(Rl_knee_c).as_rotvec(),
        theta_ankle=Rotation.from_matrix(Rl_ankle_c).as_rotvec(),
        foot_world=foot_world_new,
        hip_delta_rad=hip_angle_c,
        knee_delta_rad=knee_angle_c,
        hip_axis=hip_axis,
        hip_angle=hip_angle_c,
        knee_axis=knee_axis,
        knee_angle=knee_angle_c,
        ankle_axis=ankle_axis,
        ankle_angle=ankle_angle,
    )


def _matrix_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """``(axis, angle)`` for a rotation matrix, via scipy's canonical
    (angle in [0, pi]) rotvec. Safe to use here because the caller only
    ever RECONSTRUCTS a matrix from this axis/angle (Rodrigues' formula
    is exact for any valid axis/angle pair representing ``R``) — it is
    never compared against another absolute rotvec, which is the
    pattern that manufactures spurious deltas (see ``_LegSolve``)."""
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle < _EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return rotvec / angle, angle


def _ease_weights(length: int, edge_ease_frames: int) -> np.ndarray:
    """Per-frame ease weight for one span: 1 inside, linear 0->1 (and
    1->0) over ``edge_ease_frames`` frames at each edge. Overlapping
    ramps on a short span take the more conservative (smaller) weight."""
    w = np.ones(length, dtype=np.float64)
    e = max(int(edge_ease_frames), 0)
    if e <= 0 or length == 0:
        return w
    ramp_len = min(e, length)
    for k in range(ramp_len):
        w[k] = min(w[k], k / e)
    for k in range(ramp_len):
        idx = length - 1 - k
        w[idx] = min(w[idx], k / e)
    return w


def lock_feet_ik(
    *,
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    betas: np.ndarray | None,
    contacts: FootContacts,
    fps: float,
    target_foot_z: float = 0.02,
    ik_max_joint_delta_deg: float = 10.0,
    max_residual_correction_m: float = 0.15,
    edge_ease_frames: int = 3,
    rest_joints: np.ndarray | None = None,
    skip_pin_err_m: float = _IK_SKIP_TOLERANCE_M,
    resolved_pin_err_m: float | None = None,
    resolved_max_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Foot-lock finale: re-pin each stance span to its own FK median and
    land the foot there via root micro-correction + two-bone leg IK.

    Meant to run LAST, after all smoothing, on the final track. For each
    span in ``contacts.spans`` (using ``FootContacts`` only for
    ``start``/``end``/``side`` — the pin is RE-derived fresh from this
    track's own FK, not the possibly-stale ``span.pin`` the contacts
    were originally detected with):

      1. Pin = (median FK foot-joint XY over the span, ``target_foot_z``).
      2. Ease weight ``w(f)``: 1 inside the span, linear 0->1 (1->0) over
         ``edge_ease_frames`` at each edge.
      3. Root micro-correction: per frame, the mean over active spans of
         ``(pin - foot_fk) * w(f)``, low-passed with a 5-frame
         triangular kernel, clamped to ``max_residual_correction_m``,
         added to ``root_t`` (all three axes).
      4. Two-bone hip/knee IK (see ``_solve_leg_frame``) lands the ANKLE
         at ``pin + (ankle_fk - foot_fk)`` (preserving the current
         ankle->foot offset, so the foot lands on ``pin`` once the
         ankle counter-rotation preserves that offset's world direction
         too) using the root-corrected track. Per-joint deltas are
         clamped to ``ik_max_joint_delta_deg``; if the CLAMPED solve
         still leaves the foot more than ``skip_pin_err_m`` (default
         4 cm) from ``pin`` anywhere in the span, the whole span is
         skipped (thetas left untouched — ``new_thetas`` starts as a
         copy of the input, so "restoring" is simply not writing into
         it).
      5. Theta edits (hip/knee/ankle) are blended per frame by ``w(f)``.
      6. HONESTY CHECK (Wave 5 addition — see docs/superpowers/plans/
         2026-09-02-foot-contact-locomotion.md's Wave 5 report): a span
         that is clamp-FEASIBLE (survives step 4's ``skip_pin_err_m``
         check) can still be a span the evidence can't actually support
         as a stable stance — e.g. a noisy far/small player whose
         underlying per-frame FK positions disagree with their own
         span median by more than the IK's joint/root clamps can close.
         ``resolved_pin_err_m`` is a SECOND, normally-tighter tolerance
         on the same ``max_err`` step 4 already computes: a span whose
         clamped solve still leaves ``max_err > resolved_pin_err_m``
         (but <= ``skip_pin_err_m``, so step 4 didn't reject it) is
         treated exactly like a skip for POSE-OUTPUT purposes — its
         theta edits are discarded (``new_thetas`` keeps the input for
         those frames, same "restoring is simply not writing into it"
         mechanism as step 4) — but counted separately in
         ``stats["spans_unresolved"]`` rather than
         ``stats["spans_skipped"]``, so a caller can distinguish
         "IK genuinely infeasible under clamp" from "IK geometrically
         reached the pin but the result isn't trustworthy enough to
         apply/report as a verified stance" (false pinning is worse
         than honest free motion — the whole point of this check).
         ``resolved_pin_err_m=None`` (the default) makes this a no-op:
         internally treated as exactly ``skip_pin_err_m``, so every
         span that passes step 4 also passes this check and
         ``spans_unresolved`` stays 0 — existing callers that don't
         pass this kwarg see byte-identical behaviour to before this
         addition. When given, it is clamped to
         ``min(resolved_pin_err_m, skip_pin_err_m)`` so it can only ever
         tighten (never loosen) the effective bar, regardless of what a
         caller passes.

         ``max_err`` alone (worst absolute distance from the pin, over
         all frames in the span) misses one failure mode: a span can
         have EVERY frame within ``resolved_pin_err_m`` of the pin and
         still contain a single outlier frame — typically the first or
         last, still mid-transition into/out of stance — that sits far
         enough from its immediate NEIGHBOURS (not the pin) to register
         as a large instantaneous foot velocity once measured frame-to-
         frame, which is what ``scripts/eval_foot_quality.py``'s skate
         metric actually computes. ``resolved_max_step_m``, when given
         (``None`` disables it, matching ``resolved_pin_err_m``'s no-op
         convention), additionally requires the largest CONSECUTIVE-
         frame displacement within the span to be at or below this
         bound; a span failing either check is unresolved.

         ``stats["resolved_spans"]`` collects the ``ContactSpan``s
         (fresh pin, same side/start/end as the input span) that passed
         ALL checks — the "verified/effective contact set" downstream
         stages persist as the ``{pid}_resolved_contacts.json`` sidecar
         (see ``src.stages.refined_poses``) and prefer over the raw
         detection-time contacts for reporting/evaluation.

    ``root_R`` is never modified.

    Returns:
        ``(thetas', root_t', stats)`` — ``stats`` has keys
        ``spans_locked`` (spans that were BOTH clamp-feasible AND
        verified — the only ones whose theta edits were actually
        applied), ``spans_skipped`` (clamp-infeasible, step 4),
        ``spans_unresolved`` (clamp-feasible but not verified, step 6 —
        theta edits discarded same as a skip), ``mean_pin_err_m_before``,
        ``mean_pin_err_m_after`` (both computed across every
        non-clamp-skipped span, i.e. locked + unresolved, unchanged
        meaning from before this addition), ``max_root_corr_m``,
        ``max_joint_delta_deg``, and ``resolved_spans`` (a tuple of
        ``ContactSpan`` — the verified/effective contact set, see
        step 6 above).
    """
    thetas_in = np.asarray(thetas, dtype=np.float64)
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t_in = np.asarray(root_t, dtype=np.float64)
    n = int(thetas_in.shape[0])
    rest = _resolve_rest_joints(betas, rest_joints)
    resolved_pin_err_eff = (
        float(skip_pin_err_m) if resolved_pin_err_m is None
        else min(float(resolved_pin_err_m), float(skip_pin_err_m))
    )

    stats = {
        "spans_locked": 0,
        "spans_skipped": 0,
        "spans_unresolved": 0,
        "mean_pin_err_m_before": 0.0,
        "mean_pin_err_m_after": 0.0,
        "max_root_corr_m": 0.0,
        "max_joint_delta_deg": 0.0,
    }
    stats["resolved_spans"] = ()
    if n == 0 or not contacts.spans:
        return thetas_in.copy(), root_t_in.copy(), stats

    fw0 = compute_all_joint_worlds_batch(thetas_in, root_R, root_t_in, rest)

    # --- 1. pins (median FK foot-joint XY over each span) --------------
    pins: list[np.ndarray] = []
    for span in contacts.spans:
        foot_idx = _FOOT_IDX[span.side]
        seg = fw0[span.start:span.end, foot_idx, :2]
        xy = np.median(seg, axis=0) if seg.shape[0] else np.zeros(2)
        pins.append(np.array([xy[0], xy[1], float(target_foot_z)], dtype=np.float64))

    # --- 2/3. ease weights + root micro-correction ----------------------
    span_weights: list[np.ndarray] = []
    raw_corr = np.zeros((n, 3), dtype=np.float64)
    corr_count = np.zeros(n, dtype=np.float64)
    for pin, span in zip(pins, contacts.spans):
        length = int(span.end - span.start)
        w = _ease_weights(length, edge_ease_frames)
        span_weights.append(w)
        foot_idx = _FOOT_IDX[span.side]
        seg_foot = fw0[span.start:span.end, foot_idx, :]
        raw_corr[span.start:span.end] += (pin[None, :] - seg_foot) * w[:, None]
        corr_count[span.start:span.end] += 1.0

    active = corr_count > 0
    mean_corr = np.zeros((n, 3), dtype=np.float64)
    mean_corr[active] = raw_corr[active] / corr_count[active, None]

    padded = np.pad(mean_corr, ((2, 2), (0, 0)), mode="edge")
    smooth_corr = np.zeros((n, 3), dtype=np.float64)
    for k, wgt in enumerate(_ROOT_CORR_KERNEL):
        smooth_corr += wgt * padded[k:k + n]

    corr_norms = np.linalg.norm(smooth_corr, axis=1)
    over = corr_norms > max_residual_correction_m
    if np.any(over):
        scale = np.ones(n, dtype=np.float64)
        scale[over] = max_residual_correction_m / corr_norms[over]
        smooth_corr = smooth_corr * scale[:, None]

    root_t_corr = root_t_in + smooth_corr
    stats["max_root_corr_m"] = float(np.linalg.norm(smooth_corr, axis=1).max()) if n else 0.0

    fw_corr = compute_all_joint_worlds_batch(thetas_in, root_R, root_t_corr, rest)

    # --- 4/5/6. per-span two-bone IK + blend + honesty check ------------
    new_thetas = thetas_in.copy()
    max_delta_deg = 0.0
    resolved_spans: list[ContactSpan] = []

    for pin, span, w in zip(pins, contacts.spans, span_weights):
        side = span.side
        hip, knee, ankle, foot = (
            _HIP_IDX[side], _KNEE_IDX[side], _ANKLE_IDX[side], _FOOT_IDX[side],
        )
        frame_solves: list[_LegSolve] = []
        for f in range(span.start, span.end):
            ankle_fk = fw_corr[f, ankle]
            foot_fk = fw_corr[f, foot]
            target_ankle_world = pin + (ankle_fk - foot_fk)
            solve = _solve_leg_frame(
                target_ankle_world=target_ankle_world,
                root_R_f=root_R[f],
                root_t_f=root_t_corr[f],
                theta_row=thetas_in[f],
                rest=rest,
                hip=hip, knee=knee, ankle=ankle, foot=foot,
                ik_max_joint_delta_deg=ik_max_joint_delta_deg,
            )
            frame_solves.append(solve)

        if frame_solves:
            max_err = max(
                float(np.linalg.norm(s.foot_world - pin)) for s in frame_solves
            )
            max_step = max(
                (
                    float(np.linalg.norm(
                        frame_solves[k + 1].foot_world - frame_solves[k].foot_world
                    ))
                    for k in range(len(frame_solves) - 1)
                ),
                default=0.0,
            )
        else:
            max_err = 0.0
            max_step = 0.0

        if max_err > skip_pin_err_m:
            stats["spans_skipped"] += 1
            continue
        step_ok = resolved_max_step_m is None or max_step <= float(resolved_max_step_m)
        if max_err > resolved_pin_err_eff or not step_ok:
            # Clamp-feasible (would have locked under the old, single-
            # threshold logic) but not trustworthy enough to verify —
            # see step 6 in this function's docstring. The step check
            # catches a case max_err alone misses: a span can have every
            # frame within resolved_pin_err_eff of the PIN yet still
            # contain one outlier frame (typically the first/last, still
            # mid-transition into/out of stance) that sits far enough
            # from its NEIGHBOURS to read as a large instantaneous skate
            # spike once measured frame-to-frame — the actual quantity
            # scripts/eval_foot_quality.py's skate metric computes.
            # Treated exactly like a skip for pose output (thetas left
            # untouched), but counted separately so callers can tell the
            # two cases apart.
            stats["spans_unresolved"] += 1
            continue

        stats["spans_locked"] += 1
        resolved_spans.append(
            ContactSpan(side=side, start=int(span.start), end=int(span.end), pin=pin.copy())
        )
        for k, f in enumerate(range(span.start, span.end)):
            solve = frame_solves[k]
            wf = float(w[k])
            if wf <= 0.0:
                pass  # exact no-op: new_thetas[f] already == thetas_in[f]
            elif wf >= 1.0:
                new_thetas[f, hip] = solve.theta_hip
                new_thetas[f, knee] = solve.theta_knee
                new_thetas[f, ankle] = solve.theta_ankle
            else:
                # Ease the edit in/out by scaling each SMALL delta
                # rotation's angle by wf and composing it onto the
                # ORIGINAL matrix — not a linear blend of the two
                # absolute rotvecs (see _LegSolve docstring for why
                # that is unsafe for an unwrapped input convention).
                new_thetas[f, hip] = Rotation.from_matrix(
                    _rotate_local(thetas_in[f, hip], solve.hip_axis, solve.hip_angle * wf)
                ).as_rotvec()
                new_thetas[f, knee] = Rotation.from_matrix(
                    _rotate_local(thetas_in[f, knee], solve.knee_axis, solve.knee_angle * wf)
                ).as_rotvec()
                new_thetas[f, ankle] = Rotation.from_matrix(
                    _rotate_local(thetas_in[f, ankle], solve.ankle_axis, solve.ankle_angle * wf)
                ).as_rotvec()
            max_delta_deg = max(
                max_delta_deg,
                math.degrees(solve.hip_delta_rad),
                math.degrees(solve.knee_delta_rad),
            )

    # --- stats: pin error before/after, over every span/frame -----------
    fw_final = compute_all_joint_worlds_batch(new_thetas, root_R, root_t_corr, rest)
    errs_before: list[float] = []
    errs_after: list[float] = []
    for pin, span in zip(pins, contacts.spans):
        foot_idx = _FOOT_IDX[span.side]
        seg_before = fw_corr[span.start:span.end, foot_idx, :]
        seg_after = fw_final[span.start:span.end, foot_idx, :]
        errs_before.extend(np.linalg.norm(seg_before - pin, axis=1).tolist())
        errs_after.extend(np.linalg.norm(seg_after - pin, axis=1).tolist())

    stats["mean_pin_err_m_before"] = float(np.mean(errs_before)) if errs_before else 0.0
    stats["mean_pin_err_m_after"] = float(np.mean(errs_after)) if errs_after else 0.0
    stats["max_joint_delta_deg"] = float(max_delta_deg)
    stats["resolved_spans"] = tuple(resolved_spans)

    return new_thetas, root_t_corr, stats


# ---------------------------------------------------------------------
# [D] Task 7: penetration guard
# ---------------------------------------------------------------------


def penetration_guard(
    *,
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    betas: np.ndarray | None,
    sole_clearance_m: float = 0.025,
    rest_joints: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Raise-only pass: guarantees no sole-proxy penetration remains.

    ``deficit[f] = max(0, sole_clearance_m - min(lower foot joint z))``.
    The applied raise is a rolling max of ``deficit`` over a +/-2 frame
    window, then a width-3 triangular smooth — in that order, the
    rolling-max window (radius 2) is strictly wider than the smoothing
    kernel (radius 1), which guarantees ``raise[f] >= deficit[f]`` for
    every ``f`` (every smoothing tap at ``f`` is itself a rolling max
    that already covers ``f``), i.e. penetration is always fully
    cleared, never partially averaged away. Never lowers ``root_t.z``
    and is a no-op when the track is already clear.

    Returns:
        ``(root_t', stats)`` — ``stats`` has keys ``frames_raised``,
        ``max_raise_cm``.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t = np.asarray(root_t, dtype=np.float64)
    n = int(thetas.shape[0])
    rest = _resolve_rest_joints(betas, rest_joints)

    if n == 0:
        return root_t.copy(), {"frames_raised": 0, "max_raise_cm": 0.0}

    fw = compute_all_joint_worlds_batch(thetas, root_R, root_t, rest)
    feet_z = fw[:, list(_FOOT_IDX), 2]
    lower_z = feet_z.min(axis=1)
    deficit = np.clip(float(sole_clearance_m) - lower_z, 0.0, None)

    rmax = maximum_filter1d(deficit, size=5, mode="nearest")
    padded = np.pad(rmax, 1, mode="edge")
    raise_arr = (
        _PEN_GUARD_KERNEL[0] * padded[:-2]
        + _PEN_GUARD_KERNEL[1] * padded[1:-1]
        + _PEN_GUARD_KERNEL[2] * padded[2:]
    )

    root_t_new = root_t.copy()
    root_t_new[:, 2] += raise_arr

    stats = {
        "frames_raised": int(np.count_nonzero(raise_arr > 1e-9)),
        "max_raise_cm": float(raise_arr.max() * 100.0),
    }
    return root_t_new, stats

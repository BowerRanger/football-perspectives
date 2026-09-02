"""Stance-pinned root solve — component [C] of
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md
(plan Task 4). Task 7 (``lock_feet_ik`` / ``penetration_guard``,
component [D]) lands in a follow-up append to this module.

:func:`solve_root_with_pins` is ``hmr_world``'s stance-pinned root solve:
per constrained frame, the pin implies a root translation; the
implied-minus-carrier delta is interpolated smoothly (PCHIP) across
unconstrained frames and decays back to zero beyond the first/last
constrained frame.

FK convention throughout (matches ``src/utils/smpl_skeleton.py`` exactly):
``thetas[:, 0]`` (the per-frame global orient) is IGNORED — ``root_R``
already carries the root joint's world orientation. Only ``thetas[:, 1:]``
drive the articulated pose; applying both would double-count orientation.
Joint indices (SMPL-24, see CLAUDE.md): hips 1/2, knees 4/5, ankles 7/8,
feet(toes) 10/11, side order [L, R] throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.utils.foot_contact import FootContacts
from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    beta_adjusted_rest_joints,
    compute_canonical_joints_batch,
    load_smpl_neutral_model,
)

_EPS = 1e-9

_ANKLE_IDX = (SMPL_JOINT_NAMES.index("l_ankle"), SMPL_JOINT_NAMES.index("r_ankle"))


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
    implies a root translation: ``implied = pin - root_R @ canon[ankle]``
    (multi-foot double support uses the quality-weighted mean of the two
    implied roots). ``delta = implied - root_carrier`` is clamped to
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
    # Per-side ankle offset from root, rotated into world orientation
    # (not translated — this is the "if root_t were 0" ankle position).
    off = np.empty((n, 2, 3), dtype=np.float64)
    for side, ankle_idx in enumerate(_ANKLE_IDX):
        off[:, side, :] = np.einsum("fba,fa->fb", root_R, canon[:, ankle_idx, :])

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

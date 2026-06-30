"""Categorical spin presets exposed on ``kick`` and ``volley`` ball
anchors.

When the user tags a kick (or volley) with a preset like
``"instep_curl_right"``, the ball stage translates the tag to a 3-vector
angular-velocity seed and feeds it to ``fit_magnus_trajectory`` as the
LM starting point. With a hint in hand the stage also relaxes the
"is Magnus better than parabola?" acceptance threshold, so subtle curl
that doesn't quite beat the strict 20 % residual-improvement bar still
makes it into the final trajectory.

Pitch coordinates per CLAUDE.md: x along the nearside touchline,
y across the pitch, z up. The curl presets seed a vertical (±z) axis;
top/back spin presets compute a horizontal axis perpendicular to the
parabola's initial velocity so the resulting Magnus force points
correctly down (top-spin → dipping) or up (back-spin → floating).
"""

from __future__ import annotations

import math

import numpy as np


# Typical magnitudes for football spin, used as LM starting points only —
# the optimiser is free to refine. Sources: published video-tracking
# studies of curled free kicks (≈ 10–20 rad/s side-spin) and driven /
# chipped shots (≈ 40–60 rad/s).
_CURL_MAGNITUDE_RAD_S = 15.0
_SPIN_MAGNITUDE_RAD_S = 50.0

# Geometric thresholds for derive_spin_seed.
# Minimum horizontal direction change (degrees) to classify as a curling
# foot contact.
_MIN_CURL_ANGLE_DEG = 20.0
# Minimum exit elevation angle (degrees) to classify a foot kick as lofted
# (producing a backspin seed rather than a side-spin seed).
_MIN_LOFT_ELEVATION_DEG = 25.0

# Bones that map to foot contacts.
_FOOT_BONES: frozenset[str] = frozenset({"l_foot", "r_foot"})


VALID_SPIN_PRESETS: frozenset[str] = frozenset({
    "none",
    "instep_curl_right",
    "instep_curl_left",
    "outside_curl_right",
    "outside_curl_left",
    "topspin",
    "backspin",
    "knuckle",
})


# Anchor states on which a ``spin`` preset is accepted. ``player_touch``
# subsumes the deprecated ``kick`` / ``volley`` anchors: the user picks
# ``player_touch`` and an explicit ``touch_type`` ("shot" / "volley") to
# opt into the spin pathway.
SPIN_ENABLED_STATES: frozenset[str] = frozenset({"player_touch"})


# Values accepted on the new ``touch_type`` sub-tag of a player_touch
# anchor. "shot" = deliberate strike toward goal; "volley" = mid-flight
# strike. Both gate the spin sub-dropdown in the UI.
VALID_TOUCH_TYPES: frozenset[str] = frozenset({"shot", "volley"})


# Touch types that enable the spin hint. Both values currently do, but
# the indirection keeps the door open for future touch_type values
# (e.g. "header", "chest") that don't carry spin.
SPIN_ENABLED_TOUCH_TYPES: frozenset[str] = frozenset({"shot", "volley"})


def omega_seed_from_preset(
    preset: str | None,
    v0: np.ndarray | None,
) -> np.ndarray:
    """Return the angular-velocity seed (rad/s, world frame) for a spin
    preset.

    Args:
        preset: one of ``VALID_SPIN_PRESETS`` or ``None``. ``None``,
            ``"none"``, and ``"knuckle"`` all return a zero vector
            (the LM is then free to find or stay at zero spin).
        v0: parabola initial velocity, used to orient top/back-spin
            seeds perpendicular to the ball's direction of travel.
            Required for ``"topspin"`` and ``"backspin"`` — pass
            ``None`` to seed at zero when v0 is unavailable.

    Raises:
        ValueError: if ``preset`` is not in ``VALID_SPIN_PRESETS``.
    """
    if preset is None or preset in ("none", "knuckle"):
        return np.zeros(3, dtype=float)
    if preset not in VALID_SPIN_PRESETS:
        raise ValueError(
            f"unknown spin preset {preset!r}; valid: {sorted(VALID_SPIN_PRESETS)}"
        )
    # Curl presets: vertical axis. Sign convention follows the right-hand
    # rule applied to a ball moving roughly along +x:
    #   +z spin (ω = (0, 0, +1)) → Magnus force has +y component → ball
    #   curves toward +y (across the pitch from camera-near to far).
    # That matches a right-foot inside-of-foot strike for a player kicking
    # roughly toward +x with the ball curling right-to-left in the camera
    # view (toward the far touchline).
    if preset == "instep_curl_right":
        return np.array([0.0, 0.0, +1.0]) * _CURL_MAGNITUDE_RAD_S
    if preset == "instep_curl_left":
        return np.array([0.0, 0.0, -1.0]) * _CURL_MAGNITUDE_RAD_S
    if preset == "outside_curl_right":
        # Right-foot outside-of-foot curls opposite to instep.
        return np.array([0.0, 0.0, -1.0]) * _CURL_MAGNITUDE_RAD_S
    if preset == "outside_curl_left":
        return np.array([0.0, 0.0, +1.0]) * _CURL_MAGNITUDE_RAD_S
    # Top / back spin: horizontal axis perpendicular to the ball's
    # initial horizontal direction. Magnus force F ∝ ω × v.
    #   top-spin → F points down → ball dips → ω = (v.y, -v.x, 0) / |·|
    #     so ω × v has z-component = ω_x v_y - ω_y v_x = v_y² + v_x² > 0
    #     ⇒ +z. We want F downward (−z), so flip sign.
    # Easiest mental check: for v = (1, 0, 0), top-spin axis is (0, +1, 0)
    # and ω × v = (0, 0, -1) ⇒ down. ✓
    if preset in ("topspin", "backspin"):
        if v0 is None:
            return np.zeros(3, dtype=float)
        vh = np.array([float(v0[0]), float(v0[1]), 0.0])
        vh_norm = float(np.linalg.norm(vh))
        if vh_norm < 1e-6:
            return np.zeros(3, dtype=float)
        # Rotate horizontal v0 by +90° around +z: (vx, vy) → (-vy, vx).
        axis = np.array([-vh[1], vh[0], 0.0]) / vh_norm
        if preset == "backspin":
            axis = -axis
        return axis * _SPIN_MAGNITUDE_RAD_S
    # Unreachable — preset guarded above.
    return np.zeros(3, dtype=float)


def derive_spin_seed(
    contact_bone: str,
    v_in: np.ndarray,
    v_out: np.ndarray,
) -> np.ndarray | None:
    """Derive a geometric spin seed from the ball's approach and exit vectors.

    This function infers a likely angular-velocity seed from the geometry of
    a player contact event, using the contact bone, ball approach velocity
    (``v_in``), and ball exit velocity (``v_out``).  The seed is passed to
    ``fit_magnus_trajectory`` as the LM starting point when no manual spin
    preset is on the anchor.

    Args:
        contact_bone: the SMPL bone at the contact site, e.g. ``"l_foot"``,
            ``"r_foot"``, ``"head"``, ``"chest"``.  See
            ``ball_anchor_heights.BONE_TO_SMPL_INDEX`` for the full vocabulary.
        v_in: ball approach velocity (world frame, m/s), 3-vector.
        v_out: ball exit velocity (world frame, m/s), 3-vector.

    Returns:
        * An angular-velocity seed (rad/s, world frame) when a clear spin
          pattern is detected.
        * A **zero vector** (not ``None``) for headers — we assert
          approximately zero spin rather than leaving the fit unconstrained.
        * ``None`` when no pattern is recognised, so the caller falls back to
          the unconstrained (strict-gate) Magnus fit.

    Rules (in priority order):

    1. **Head** → ``np.zeros(3)``  (controlled spin is rare; seed zero).
    2. **Foot + large horizontal direction change** (angle between horizontal
       projections of ``v_in`` and ``v_out`` exceeds
       :data:`_MIN_CURL_ANGLE_DEG`) → side-spin about world-z; sign given by
       ``cross(v_in_xy, v_out_xy).z``.  Magnitude = :data:`_CURL_MAGNITUDE_RAD_S`.
    3. **Foot + lofted exit** (exit elevation angle >
       :data:`_MIN_LOFT_ELEVATION_DEG`) → backspin about the horizontal axis
       perpendicular to the exit direction (``up × v_out_hat``), with the sign
       that gives backspin (top of ball moving backward relative to travel,
       producing an upward Magnus force).  Magnitude =
       :data:`_SPIN_MAGNITUDE_RAD_S`.
    4. **Otherwise** → ``None``.
    """
    v_in = np.asarray(v_in, dtype=float)
    v_out = np.asarray(v_out, dtype=float)

    # Rule 1: head — zero spin assertion.
    if contact_bone == "head":
        return np.zeros(3, dtype=float)

    # Rules 2 & 3 apply to foot bones only.
    if contact_bone not in _FOOT_BONES:
        return None

    # --- Rule 2: side-spin from horizontal direction change ---------------
    v_in_xy = np.array([v_in[0], v_in[1], 0.0], dtype=float)
    v_out_xy = np.array([v_out[0], v_out[1], 0.0], dtype=float)
    norm_in = float(np.linalg.norm(v_in_xy))
    norm_out = float(np.linalg.norm(v_out_xy))

    if norm_in > 1e-6 and norm_out > 1e-6:
        cos_angle = float(np.dot(v_in_xy, v_out_xy) / (norm_in * norm_out))
        # Clamp for numerical safety.
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg >= _MIN_CURL_ANGLE_DEG:
            cross_z = float(np.cross(v_in_xy, v_out_xy)[2])
            sign = 1.0 if cross_z >= 0.0 else -1.0
            return np.array([0.0, 0.0, sign * _CURL_MAGNITUDE_RAD_S], dtype=float)

    # --- Rule 3: backspin from lofted exit --------------------------------
    out_speed = float(np.linalg.norm(v_out))
    if out_speed > 1e-6:
        out_h = float(np.linalg.norm([v_out[0], v_out[1]]))
        # Elevation angle of exit vector.
        elevation_deg = math.degrees(math.atan2(float(v_out[2]), out_h))
        if elevation_deg > _MIN_LOFT_ELEVATION_DEG:
            # Horizontal direction of travel.
            v_out_h = np.array([v_out[0], v_out[1], 0.0], dtype=float)
            h_norm = float(np.linalg.norm(v_out_h))
            if h_norm < 1e-6:
                return None
            v_out_h /= h_norm
            # Backspin axis: perpendicular to travel in the horizontal plane.
            # ``up × v_out_hat`` gives the axis to the left of travel; for
            # backspin we want the top of the ball moving backward relative to
            # travel direction, which produces an upward Magnus force.
            # F = k (ω × v); to get F_z > 0 with v roughly horizontal we need
            # (ω × v).z > 0.  Using ω = up × v_hat = (-v_y, v_x, 0) and
            # v = (v_x, v_y, 0):
            #   (ω × v).z = ω_x v_y - ω_y v_x = -v_y v_y - v_x v_x = -(v_x²+v_y²)
            # That is negative — so we flip the sign to get backspin (upward force).
            # ω = -(up × v_out_hat) = (v_y, -v_x, 0)
            axis = np.array([v_out_h[1], -v_out_h[0], 0.0], dtype=float)
            axis /= float(np.linalg.norm(axis))
            return axis * _SPIN_MAGNITUDE_RAD_S

    # Rule 4: no pattern detected.
    return None

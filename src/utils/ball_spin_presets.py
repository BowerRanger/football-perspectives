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

import numpy as np


# Typical magnitudes for football spin, used as LM starting points only —
# the optimiser is free to refine. Sources: published video-tracking
# studies of curled free kicks (≈ 10–20 rad/s side-spin) and driven /
# chipped shots (≈ 40–60 rad/s).
_CURL_MAGNITUDE_RAD_S = 15.0
_SPIN_MAGNITUDE_RAD_S = 50.0


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

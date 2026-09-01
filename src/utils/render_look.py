"""Pure look/color/camera math for the render stage (no bpy)."""
from __future__ import annotations

import numpy as np

# Rest-pose height fractions (0=sole, 1=crown). Arms inherit the shirt
# color in v1 (long-sleeve reading; acceptable under the toon look).
_ZONES = (
    (0.15, "socks"),
    (0.48, "skin"),      # legs
    (0.58, "shorts"),
    (0.86, "shirt"),     # torso + arms
    (1.01, "skin"),      # head/neck
)


def kit_zone_for_height_fraction(f: float) -> str:
    f = float(min(max(f, 0.0), 1.0))
    for upper, zone in _ZONES:
        if f < upper:
            return zone
    return "skin"


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_str: str) -> tuple[float, float, float, float]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected #RRGGBB, got {hex_str!r}")
    srgb = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [0.0 if v == 0.0 else 1.0 if v == 1.0 else _srgb_to_linear(v)
           for v in srgb]
    return (lin[0], lin[1], lin[2], 1.0)


def resolve_player_colors(
    teams_cfg: dict,
    team_class: dict[str, tuple[str, str]],
) -> dict[str, dict[str, tuple]]:
    defaults = teams_cfg.get("defaults", {})
    overrides = teams_cfg.get("by_player", {})
    fallback = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}
    out: dict[str, dict[str, tuple]] = {}
    for pid, (team, _cls) in team_class.items():
        key = overrides.get(pid, team)
        kit = defaults.get(key, fallback)
        out[pid] = {part: hex_to_linear_rgba(kit.get(part, fallback[part]))
                    for part in ("shirt", "shorts", "socks")}
    return out


def blender_camera_world_matrix(
    R: list[list[float]], t: list[float],
) -> list[list[float]]:
    """OpenCV world->camera (R, t) to a Blender camera world matrix.

    OpenCV camera axes: +X right, +Y down, +Z forward. Blender cameras
    look down -Z with +Y up, so the rotation columns flip on Y and Z.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    C = -R.T @ t
    R_bl = R.T @ np.diag([1.0, -1.0, -1.0])
    M = np.eye(4)
    M[:3, :3] = R_bl
    M[:3, 3] = C
    return [[float(v) for v in row] for row in M]


def lens_mm_from_K(
    K: list[list[float]], width_px: int, sensor_mm: float = 36.0,
) -> float:
    fx = float(K[0][0])
    return fx * sensor_mm / float(width_px)

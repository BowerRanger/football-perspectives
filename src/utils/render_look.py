"""Pure look/color/camera math for the render stage (no bpy)."""
from __future__ import annotations

import numpy as np

from src.utils.team_roles import derive_kit_role

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
    """Resolve each player's shirt/shorts/socks colour from team config.

    ``team_class`` maps ``player_id -> (team, class_name)`` in the real
    tracking vocabulary (``team`` in ``"A"|"B"|"referee"|"unknown"``,
    ``class_name`` in ``"player"|"goalkeeper"|"referee"|"ball"`` — see
    ``src/schemas/tracks.py``). That vocabulary does not match
    ``render.teams.defaults``'s keys directly (``home``/``away``/
    ``home_gk``/``away_gk``/``referee``/``unknown``), so every player is
    routed through :func:`src.utils.team_roles.derive_kit_role` — the same
    mapping the export/UE kit path uses — to get the canonical kit role
    first. A ``by_player`` override (keyed by role, e.g. ``{"P003":
    "away"}``) always wins over the derived role.

    Falls back gracefully when ``defaults`` doesn't have every role: a
    missing ``*_gk`` role falls back to that side's outfield kit, and
    anything still unresolved falls back to a neutral gray.
    """
    defaults = teams_cfg.get("defaults", {})
    overrides = teams_cfg.get("by_player", {})
    fallback = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}
    out: dict[str, dict[str, tuple]] = {}
    for pid, (team, cls) in team_class.items():
        role = overrides.get(pid) or derive_kit_role(team, cls)
        kit = defaults.get(role)
        if kit is None and role.endswith("_gk"):
            kit = defaults.get(role[: -len("_gk")])
        if kit is None:
            kit = fallback
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


def merge_partial(defaults: dict, overrides: dict | None) -> dict:
    """Shallow-merge a possibly-``None``/partial override dict over
    ``defaults``.

    Used by ``blender_render_scene.py``'s ``_resolve_style`` for the
    ``palette``/``post``/``post.duotone`` sub-blocks: a caller can
    override a single nested key (e.g. only ``post.grain``) without
    having to repeat every sibling key. Every key absent from
    ``overrides`` falls back to ``defaults`` — the byte-identical-
    defaults guarantee new style keys must preserve.
    """
    return {**defaults, **(overrides or {})}


def duotone_colors(
    duotone: dict | None,
) -> tuple[tuple, tuple] | None:
    """Resolve a ``style.post.duotone`` block to ``(shadow, highlight)``
    linear RGBA pairs, or ``None`` when the duotone effect is inactive.

    Both endpoints are required: a duotone gradient with only one color
    set is ambiguous (there's no principled fallback for the other
    end), so it's treated as fully off rather than guessing — same
    "absent key means off" posture as every other post-effect default.
    """
    if not duotone:
        return None
    shadow = duotone.get("shadow")
    highlight = duotone.get("highlight")
    if not shadow or not highlight:
        return None
    return hex_to_linear_rgba(shadow), hex_to_linear_rgba(highlight)


def post_style_is_active(post: dict | None) -> bool:
    """True when at least one ``style.post`` effect deviates from its
    neutral/off default.

    Gates whether the render script bothers attaching a post-processing
    compositor graph at all: an all-default post block (e.g. the
    ``_resolve_style`` fallback when the caller never asked for ``post``)
    must never attach a compositor node group, or it would trip the same
    cross-pass "compositor poisoning" gotcha documented for AOV — a later
    render call with no post effects of its own would still see
    ``scene.compositing_node_group`` from a previous call attached.
    """
    if not post:
        return False
    if float(post.get("glare", 0.0)) > 0.0:
        return True
    if float(post.get("grain", 0.0)) > 0.0:
        return True
    if float(post.get("vignette", 0.0)) > 0.0:
        return True
    if int(post.get("posterize", 0) or 0) >= 2:
        return True
    if float(post.get("saturation", 1.0)) != 1.0:
        return True
    if duotone_colors(post.get("duotone")) is not None:
        return True
    return False


def grain_noise_pixels(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Flat float32 RGBA noise buffer for the compositor film-grain
    overlay blend, sized exactly ``width`` x ``height`` (the Blender
    ``Image`` datablock the render script bakes this into must match
    the render's own resolution pixel-for-pixel).

    Grayscale (R==G==B per pixel), normally distributed and centered at
    0.5 — the neutral point for an ``OVERLAY`` blend, so mixing this in
    at ``Fac=0`` is a true no-op and increasing ``Fac`` smoothly ramps
    grain visibility. Deterministic per ``(width, height, seed)`` via a
    local ``Generator`` (never perturbs global numpy random state, so
    it's safe to call from a bpy-free unit test).
    """
    rng = np.random.default_rng(seed)
    mono = rng.normal(loc=0.5, scale=0.14, size=(height, width)).astype(np.float32)
    np.clip(mono, 0.0, 1.0, out=mono)
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., 0] = mono
    rgba[..., 1] = mono
    rgba[..., 2] = mono
    rgba[..., 3] = 1.0
    return rgba.reshape(-1)

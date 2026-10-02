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
    role_overrides: dict[str, str] | None = None,
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
    first. Role precedence: a config ``by_player`` override (keyed by
    role, e.g. ``{"P003": "away"}``) > ``role_overrides`` (the operator's
    ``players.json`` kit roles, via ``player_names.load_kit_roles`` — the
    same source the export stage honours) > the derived role.

    Falls back gracefully when ``defaults`` doesn't have every role: a
    missing ``*_gk`` role falls back to that side's outfield kit, and
    anything still unresolved falls back to a neutral gray.
    """
    fallback = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}
    return {
        pid: {part: hex_to_linear_rgba(kit.get(part, fallback[part]))
              for part in ("shirt", "shorts", "socks")}
        for pid, kit in _resolve_kits(teams_cfg, team_class, role_overrides).items()
    }


def _resolve_kits(
    teams_cfg: dict,
    team_class: dict[str, tuple[str, str]],
    role_overrides: dict[str, str] | None,
) -> dict[str, dict]:
    """``player_id -> raw kit dict`` (hex values + kit options) — the
    role resolution shared by :func:`resolve_player_colors` and
    :func:`resolve_player_looks`."""
    defaults = teams_cfg.get("defaults", {})
    overrides = teams_cfg.get("by_player", {})
    roles = role_overrides or {}
    fallback = {"shirt": "#888888", "shorts": "#666666", "socks": "#888888"}
    out: dict[str, dict] = {}
    for pid in sorted(set(team_class) | set(roles)):
        team, cls = team_class.get(pid, ("unknown", "player"))
        role = overrides.get(pid) or roles.get(pid) or derive_kit_role(team, cls)
        kit = defaults.get(role)
        if kit is None and role.endswith("_gk"):
            kit = defaults.get(role[: -len("_gk")])
        out[pid] = kit if kit is not None else fallback
    return out


# --- Per-player look: kit options + appearance -----------------------------

DEFAULT_SKIN_HEX = "#c68863"
DEFAULT_HAIR_HEX = "#2b1d14"
DEFAULT_BOOTS_HEX = "#1c1c1c"
_SLEEVES = ("short", "long")


def resolve_player_looks(
    teams_cfg: dict,
    team_class: dict[str, tuple[str, str]],
    role_overrides: dict[str, str] | None = None,
    appearance: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict]:
    """Full per-player look for the anatomical-zone body.

    Returns ``{pid: {"colors": {zone: linear_rgba}, "sleeves": "short"|
    "long", "gloves": bool}}`` where ``colors`` covers every zone
    :func:`anatomical_kit_zones` can emit. Kit keys beyond the classic
    shirt/shorts/socks are optional per role in ``render.teams.defaults``:
    ``boots`` (hex), ``gloves`` (hex — presence turns gloves on),
    ``sleeves`` (``short`` default / ``long``). ``appearance`` is the
    per-player ``skin``/``hair`` hex map from ``players.json``
    (``player_names.load_player_appearance``).
    """
    base = resolve_player_colors(teams_cfg, team_class, role_overrides)
    kits = _resolve_kits(teams_cfg, team_class, role_overrides)
    looks = appearance or {}
    out: dict[str, dict] = {}
    for pid, colors in base.items():
        kit = kits[pid]
        app = looks.get(pid, {})
        sleeves = kit.get("sleeves", "short")
        gloves_hex = kit.get("gloves")
        skin_hex = app.get("skin", DEFAULT_SKIN_HEX)
        out[pid] = {
            "colors": {
                **colors,
                "boots": hex_to_linear_rgba(kit.get("boots", DEFAULT_BOOTS_HEX)),
                "gloves": hex_to_linear_rgba(gloves_hex or skin_hex),
                "skin": hex_to_linear_rgba(skin_hex),
                "hair": hex_to_linear_rgba(app.get("hair", DEFAULT_HAIR_HEX)),
            },
            "sleeves": sleeves if sleeves in _SLEEVES else "short",
            "gloves": bool(gloves_hex),
        }
    return out


# SMPL joint indices (src/utils/smpl_skeleton.SMPL_JOINT_NAMES order).
_PELVIS, _L_HIP, _R_HIP, _SPINE1 = 0, 1, 2, 3
_L_KNEE, _R_KNEE, _SPINE2 = 4, 5, 6
_L_ANKLE, _R_ANKLE, _SPINE3 = 7, 8, 9
_L_FOOT, _R_FOOT, _NECK = 10, 11, 12
_L_COLLAR, _R_COLLAR, _HEAD = 13, 14, 15
_L_SHOULDER, _R_SHOULDER = 16, 17
_L_ELBOW, _R_ELBOW = 18, 19
_L_WRIST, _R_WRIST, _L_HAND, _R_HAND = 20, 21, 22, 23

_TORSO = {_PELVIS, _SPINE1, _SPINE2, _SPINE3, _L_COLLAR, _R_COLLAR}
_HANDS = {_L_WRIST, _R_WRIST, _L_HAND, _R_HAND}
_FEET = {_L_FOOT, _R_FOOT}
_LIMB_CHILD = {_L_SHOULDER: _L_ELBOW, _R_SHOULDER: _R_ELBOW,
               _L_HIP: _L_KNEE, _R_HIP: _R_KNEE,
               _L_KNEE: _L_ANKLE, _R_KNEE: _R_ANKLE}

# Rest-pose garment boundaries (metres / fractions along a bone).
SHIRT_HEM_ABOVE_PELVIS_M = 0.04   # shirt/shorts split on torso verts
SHORT_SLEEVE_FRACTION = 0.5       # of shoulder->elbow
SHORTS_LEG_FRACTION = 0.5         # of hip->knee
SOCK_TOP_FRACTION = 0.15          # of knee->ankle (sock starts below the knee)
BOOT_TOP_ABOVE_ANKLE_M = 0.03
# SMPL's head joint sits at the skull base (crown ~ +0.20 m above it,
# face spans z ~ 0..+0.10): hair is a crown cap plus the back of the head
# above the nape; the forehead/face/ears stay skin.
HAIRLINE_ABOVE_HEAD_M = 0.155     # crown cap
HAIR_BEHIND_HEAD_M = 0.01         # back-of-head band (z behind the joint)
HAIR_BACK_MIN_ABOVE_HEAD_M = 0.06 # nape line


def _bone_fraction(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    return float(np.dot(v - a, ab) / max(float(np.dot(ab, ab)), 1e-9))


def anatomical_kit_zones(
    verts: np.ndarray,
    weights: np.ndarray,
    joints: np.ndarray,
    sleeves: str = "short",
    gloves: bool = False,
) -> list[str]:
    """Garment zone per rest-pose SMPL vertex (Y-up, +Z forward).

    Uses each vertex's dominant skinning joint plus its position along
    that bone, so sleeves end mid upper-arm, shorts mid-thigh, socks
    start below the knee and feet get boots — the height-band
    :func:`kit_zone_for_height_fraction` model paints T-posed arms
    (and hands) shirt-coloured and has no boots, sleeves, gloves or
    hair. Zones: shirt, shorts, socks, boots, gloves, skin, hair.
    """
    verts = np.asarray(verts, dtype=float)
    joints = np.asarray(joints, dtype=float)
    dominant = np.asarray(weights).argmax(axis=1)
    long_sleeves = sleeves == "long"
    head = joints[_HEAD]
    zones: list[str] = []
    for v, j in zip(verts, dominant.tolist()):
        if j in _TORSO:
            below_hem = v[1] <= joints[_PELVIS][1] + SHIRT_HEM_ABOVE_PELVIS_M
            waist = j in (_PELVIS, _SPINE1)
            zones.append("shorts" if waist and below_hem else "shirt")
        elif j in (_L_SHOULDER, _R_SHOULDER):
            t = _bone_fraction(v, joints[j], joints[_LIMB_CHILD[j]])
            zones.append("shirt" if long_sleeves or t < SHORT_SLEEVE_FRACTION else "skin")
        elif j in (_L_ELBOW, _R_ELBOW):
            zones.append("shirt" if long_sleeves else "skin")
        elif j in _HANDS:
            zones.append("gloves" if gloves else "skin")
        elif j in (_L_HIP, _R_HIP):
            t = _bone_fraction(v, joints[j], joints[_LIMB_CHILD[j]])
            zones.append("shorts" if t < SHORTS_LEG_FRACTION else "skin")
        elif j in (_L_KNEE, _R_KNEE):
            t = _bone_fraction(v, joints[j], joints[_LIMB_CHILD[j]])
            zones.append("socks" if t > SOCK_TOP_FRACTION else "skin")
        elif j in (_L_ANKLE, _R_ANKLE):
            zones.append("boots" if v[1] < joints[j][1] + BOOT_TOP_ABOVE_ANKLE_M
                         else "socks")
        elif j in _FEET:
            zones.append("boots")
        elif j == _HEAD:
            crown = v[1] > head[1] + HAIRLINE_ABOVE_HEAD_M
            back = (v[2] < head[2] - HAIR_BEHIND_HEAD_M
                    and v[1] > head[1] + HAIR_BACK_MIN_ABOVE_HEAD_M)
            zones.append("hair" if crown or back else "skin")
        else:  # neck
            zones.append("skin")
    return zones


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

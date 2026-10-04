"""Kit-safety lint: will the team kits still read after the post stack?

Pure (numpy + colorsys only, no bpy). Simulates the ``style.post`` chain on
each team's kit colours in CIELAB and reports findings that make a render
unreadable -- two teams merging into one colour, a striped kit collapsing
into a solid, a kit crushed to black, skin tones blotched.

Simulation mirrors ``scripts/blender_render_scene.py``'s compositor order:
saturation -> duotone -> posterize, all on scene-linear values (the
compositor works pre view-transform). Posterize is modelled as
``floor(c * steps) / steps`` (Blender's Posterize node). Distances are
CIE76 dE, which is what the ``min_delta_e`` thresholds are expressed in.

Findings are warnings, never errors: the caller (``resolve_style_payload``)
logs them and writes ``render/<shot>_kit_safety.json``; the operator decides.
"""
from __future__ import annotations

import colorsys
import itertools
from typing import Any, Mapping

import numpy as np

DEFAULT_MIN_DELTA_E = 12.0
# A kit part moving further than this through the post stack no longer
# reads as the club colour (Chelsea blue -> near-black under posterize).
DEFAULT_MAX_SHIFT_DELTA_E = 25.0
# Below this L* a coloured kit part has crushed to black.
CRUSH_L = 14.0
# Representative skin ladder (light -> dark) used for the skin-crush check.
SKIN_TONES = ("#f1c9a5", "#c68863", "#8d5524", "#4a2c17")
TOON_SHADOW_FACTOR = 0.35   # darkest toon-ramp band (blender_render_scene)

_ROLE_PAIRS_EXCLUDED = {frozenset(("home", "home_gk")), frozenset(("away", "away_gk"))}
_KIT_PARTS = ("shirt", "shorts", "socks")
_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def _hex_to_linear(hex_str: str) -> np.ndarray:
    s = str(hex_str).lstrip("#")
    srgb = np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _linear_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = (_RGB2XYZ @ np.clip(rgb, 0.0, 1.0)) / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def apply_post(rgb: np.ndarray, post: Mapping[str, Any] | None) -> np.ndarray:
    """Push one scene-linear colour through saturation -> duotone -> posterize."""
    post = post or {}
    out = np.array(rgb, dtype=float)
    sat_raw = post.get("saturation")
    sat = 1.0 if sat_raw is None else float(sat_raw)
    if sat != 1.0:
        h, s, v = colorsys.rgb_to_hsv(*np.clip(out, 0.0, 1.0))
        out = np.array(colorsys.hsv_to_rgb(h, min(max(s * sat, 0.0), 1.0), v))
    duo = post.get("duotone") or {}
    if duo.get("shadow") and duo.get("highlight"):
        luma = float(np.clip(_RGB2XYZ[1] @ out, 0.0, 1.0))
        shadow, high = _hex_to_linear(duo["shadow"]), _hex_to_linear(duo["highlight"])
        out = shadow + (high - shadow) * luma
    steps = int(post.get("posterize", 0) or 0)
    if steps >= 2:
        out = np.floor(np.clip(out, 0.0, 1.0) * steps) / steps
    return np.clip(out, 0.0, 1.0)


def _lab_after(hex_str: str, post: Mapping[str, Any] | None, shade: float = 1.0) -> np.ndarray:
    return _linear_to_lab(apply_post(_hex_to_linear(hex_str) * shade, post))


def _finding(code: str, roles: list[str], message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "severity": "warn", "roles": roles, "message": message, **extra}


def _kit_hexes(kit: Mapping[str, Any]) -> dict[str, str]:
    """Named colour slots of a KitSpec that the lint tracks."""
    slots = {p: kit[p] for p in _KIT_PARTS if kit.get(p)}
    if kit.get("sleeve_color"):
        slots["sleeve_color"] = kit["sleeve_color"]
    for i, c in enumerate((kit.get("pattern") or {}).get("colors") or []):
        slots[f"pattern_{i}"] = c
    return slots


def _lint_shift(role: str, kit: Mapping[str, Any], post, max_shift: float) -> list[dict]:
    findings = []
    # Saturation/duotone shift colours on purpose (merge/collapse checks cover
    # their failure modes); only quantisation can silently crush a club colour.
    if int((post or {}).get("posterize", 0) or 0) < 2:
        return findings
    for slot, hexv in _kit_hexes(kit).items():
        pre, post_lab = _lab_after(hexv, None), _lab_after(hexv, post)
        shift = delta_e(pre, post_lab)
        crushed = post_lab[0] < CRUSH_L <= pre[0] + 6.0
        if shift > max_shift or crushed:
            findings.append(_finding(
                "kit_crush" if crushed else "kit_shift", [role],
                f"{role}.{slot} {hexv} moves dE {shift:.0f} through the post stack"
                + (" and crushes to black" if crushed else ""),
                part=slot, delta_e=round(shift, 1)))
    return findings


def _lint_intra_kit(role: str, kit: Mapping[str, Any], post, min_de: float) -> list[dict]:
    pairs = []
    pat = (kit.get("pattern") or {}).get("colors") or []
    if len(pat) >= 2:
        pairs.append(("pattern", pat[0], pat[1]))
    if kit.get("sleeve_color") and kit.get("shirt"):
        pairs.append(("shirt/sleeve", kit["shirt"], kit["sleeve_color"]))
    findings = []
    for label, a, b in pairs:
        pre = delta_e(_lab_after(a, None), _lab_after(b, None))
        post_de = delta_e(_lab_after(a, post), _lab_after(b, post))
        if pre >= min_de > post_de:
            findings.append(_finding(
                "intra_kit_collapse", [role],
                f"{role} {label} contrast collapses dE {pre:.0f} -> {post_de:.0f}",
                part=label, delta_e=round(post_de, 1), delta_e_before=round(pre, 1)))
    return findings


def _lint_team_merge(team_kits, post, min_de: float) -> list[dict]:
    roles = [r for r in team_kits if r in ("home", "away", "home_gk", "away_gk", "referee")
             and team_kits[r].get("shirt")]
    findings = []
    for a, b in itertools.combinations(roles, 2):
        if frozenset((a, b)) in _ROLE_PAIRS_EXCLUDED:
            continue
        sa, sb = team_kits[a]["shirt"], team_kits[b]["shirt"]
        pre = delta_e(_lab_after(sa, None), _lab_after(sb, None))
        post_de = delta_e(_lab_after(sa, post), _lab_after(sb, post))
        if pre >= min_de > post_de:
            findings.append(_finding(
                "team_merge", [a, b],
                f"{a} and {b} shirts merge under the post stack (dE {pre:.0f} -> {post_de:.0f})",
                delta_e=round(post_de, 1), delta_e_before=round(pre, 1)))
    return findings


def _lint_skin(post, min_de: float) -> list[dict]:
    if not _post_active(post):
        return []
    # A deliberate duotone remaps every colour by luma; skin cannot "stay true".
    if (post.get("duotone") or {}).get("shadow"):
        return []
    labs_pre = [_lab_after(s, None) for s in SKIN_TONES]
    labs_post = [_lab_after(s, post) for s in SKIN_TONES]
    findings = []
    for tone, pre, after in zip(SKIN_TONES, labs_pre, labs_post):
        if after[0] < CRUSH_L and pre[0] >= CRUSH_L + 6.0:
            findings.append(_finding(
                "skin_crush", ["skin"],
                f"skin tone {tone} crushes to near-black (L* {pre[0]:.0f} -> {after[0]:.0f})",
                tone=tone))
    for (ta, pa, qa), (tb, pb, qb) in itertools.combinations(
            list(zip(SKIN_TONES, labs_pre, labs_post)), 2):
        if delta_e(pa, pb) >= min_de > delta_e(qa, qb):
            findings.append(_finding(
                "skin_crush", ["skin"], f"skin tones {ta} and {tb} merge into one band",
                tones=[ta, tb]))
    return findings


def _post_active(post) -> bool:
    post = post or {}
    return bool(int(post.get("posterize", 0) or 0) >= 2
                or (post.get("saturation") not in (None, 1.0))
                or (post.get("duotone") or {}).get("shadow"))


def lint_kit_safety(
    team_kits: Mapping[str, Mapping[str, Any]],
    post_style: Mapping[str, Any] | None,
    min_delta_e: float = DEFAULT_MIN_DELTA_E,
    *,
    max_shift_delta_e: float = DEFAULT_MAX_SHIFT_DELTA_E,
) -> list[dict[str, Any]]:
    """Return warning findings for ``team_kits`` under ``post_style``.

    ``team_kits`` is ``role -> KitSpec dict`` (hex strings). Findings carry
    ``code`` (``team_merge`` | ``intra_kit_collapse`` | ``kit_crush`` |
    ``kit_shift`` | ``skin_crush``), ``severity`` (always ``warn``),
    ``roles`` and a human ``message``. Empty list = safe.
    """
    if not team_kits or not _post_active(post_style):
        return []
    findings: list[dict[str, Any]] = []
    for role, kit in team_kits.items():
        if not isinstance(kit, Mapping):
            continue
        findings += _lint_shift(role, kit, post_style, max_shift_delta_e)
        findings += _lint_intra_kit(role, kit, post_style, min_delta_e)
    findings += _lint_team_merge(team_kits, post_style, min_delta_e)
    findings += _lint_skin(post_style, min_delta_e)
    return findings

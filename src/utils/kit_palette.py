"""Kit palette solver: colour maths, white balance against pitch lines,
and full-vector library snapping.

Pure numpy (plus the shared camera projection for the pitch-line
sampler). Colours are ``#rrggbb`` hex at the API edge and float
sRGB 0-255 / CIELAB inside.

White balance
    Painted pitch lines are known white (``#f5f5f0``). Sampling the
    bright, low-saturation pixels under the projected line geometry
    gives the scene illuminant; per-channel gains map it back to the
    target. That one multiplication corrects both colour cast and
    exposure, so a dull, dark broadcast sample recovers its true
    saturation.

Snap
    A team's sampled kit (shirt, sleeves, shorts, socks) is compared as
    a whole vector against library candidates with CIEDE2000 (lightness
    de-weighted by ``lightness_weight`` because broadcast shading moves
    L far more than hue). Within ``snap_de`` the library spec wins
    (brand hexes, sleeves, pattern); otherwise the corrected sample is
    kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

WHITE_TARGET_HEX = "#f5f5f0"
SNAP_DELTA_E = 12.0
LIGHTNESS_WEIGHT = 2.0

# Part weights in the full-vector kit distance.
_PART_WEIGHTS = {"shirt": 2.0, "shorts": 1.0, "socks": 1.0, "sleeves": 0.5}

# D65 white, sRGB matrices.
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


# --- colour conversions ------------------------------------------------------

def hex_to_rgb(hex_str: str) -> np.ndarray:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad hex colour {hex_str!r}")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def rgb_to_hex(rgb: Iterable[float]) -> str:
    r, g, b = (int(round(float(np.clip(c, 0, 255)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 ``(..., 3)`` -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    lin = np.clip(xyz @ _M_XYZ2RGB.T, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0, 255)


def hex_to_lab(hex_str: str) -> np.ndarray:
    return srgb_to_lab(hex_to_rgb(hex_str))


def lab_to_hex(lab: np.ndarray) -> str:
    return rgb_to_hex(lab_to_srgb(lab))


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0) -> float:
    """CIEDE2000 between two Lab triples (``kL`` > 1 de-weights lightness)."""
    L1, a1, b1 = (float(v) for v in lab1)
    L2, a2, b2 = (float(v) for v in lab2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p else 0.0
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + (360 if h1p + h2p < 360 else -360)) / 2
    T = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    dTheta = 30 * math.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / math.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2 * dTheta)) * Rc
    tl, tc, th = dLp / (kL * Sl), dCp / Sc, dHp / Sh
    return math.sqrt(tl ** 2 + tc ** 2 + th ** 2 + Rt * tc * th)


def delta_e_hex(a: str, b: str, kL: float = 1.0) -> float:
    return delta_e2000(hex_to_lab(a), hex_to_lab(b), kL)


# --- white balance -----------------------------------------------------------

@dataclass(frozen=True)
class WhiteBalance:
    gains: tuple[float, float, float]
    n_pixels: int
    applied: bool
    reference_hex: str | None = None

    def to_dict(self) -> dict:
        return {"gains": [round(g, 4) for g in self.gains], "n_pixels": self.n_pixels,
                "applied": self.applied, "reference": self.reference_hex}


IDENTITY_WB = WhiteBalance((1.0, 1.0, 1.0), 0, False, None)


def white_balance_gains(
    line_rgb: np.ndarray,
    *,
    target_hex: str = WHITE_TARGET_HEX,
    min_pixels: int = 40,
    gain_clip: tuple[float, float] = (0.8, 1.25),
    percentile: float = 85.0,
    exposure: bool = False,
) -> WhiteBalance:
    """Per-channel gains mapping the pitch-line colour to the white target.

    Thin lines blend with grass, so the reference is a HIGH percentile
    (default 85th, per channel) of the candidate pixels — the unblended
    core of the paint — not the median.

    By default only the colour CAST is removed (gains are divided by their
    geometric mean, so luminance is preserved). The absolute line level
    under-reads white (the paint is a few px wide and blends with grass),
    and applying it as an exposure gain clipped and desaturated bright kits
    (Man City lime read lavender), so ``exposure=True`` is opt-in.

    ``line_rgb`` is ``(N, 3)`` sRGB 0-255 of pixels believed to be painted
    line. Fewer than ``min_pixels`` -> identity (``applied=False``).
    """
    px = np.asarray(line_rgb, dtype=np.float64).reshape(-1, 3)
    if px.shape[0] < min_pixels:
        return WhiteBalance((1.0, 1.0, 1.0), int(px.shape[0]), False, None)
    ref = np.percentile(px, percentile, axis=0)
    gains = hex_to_rgb(target_hex) / np.maximum(ref, 1.0)
    if not exposure:
        gains = gains / float(np.exp(np.mean(np.log(gains))))
    gains = np.clip(gains, *gain_clip)
    return WhiteBalance(tuple(float(g) for g in gains), int(px.shape[0]), True, rgb_to_hex(ref))


def apply_gains(rgb: np.ndarray, wb: WhiteBalance) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64) * np.array(wb.gains), 0, 255)


def line_pixel_candidates(frame_rgb: np.ndarray, pts_uv: np.ndarray,
                          *, patch: int = 3, min_value: float = 165.0,
                          max_chroma: float = 0.12) -> np.ndarray:
    """Brightest low-chroma pixel in a small patch around each projected line point.

    Returns ``(M, 3)`` RGB. Points outside the frame or whose best pixel
    is dark/coloured (occluded by a player, grass shadow) are dropped.
    """
    h, w = frame_rgb.shape[:2]
    out: list[np.ndarray] = []
    for u, v in np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2):
        if not (patch <= u < w - patch and patch <= v < h - patch):
            continue
        x, y = int(round(u)), int(round(v))
        win = frame_rgb[y - patch:y + patch + 1, x - patch:x + patch + 1].reshape(-1, 3).astype(np.float64)
        best = win[np.argmax(win.sum(axis=1))]
        mx, mn = best.max(), best.min()
        if mx < min_value or (mx - mn) / max(mx, 1.0) > max_chroma:
            continue
        out.append(best)
    return np.array(out) if out else np.zeros((0, 3))


# --- kit vectors & snapping --------------------------------------------------

def _effective_shirt_colours(kit: Mapping) -> list[str]:
    pat = kit.get("pattern")
    cols = [kit["shirt"]]
    if pat and pat.get("type") in ("vertical_stripes", "hoops"):
        cols = list(pat["colors"])
    return cols


def _shirt_distance(sample_hex: str, kit: Mapping, kL: float) -> float:
    cols = _effective_shirt_colours(kit)
    labs = [hex_to_lab(c) for c in cols]
    cands = labs + ([np.mean(labs, axis=0)] if len(labs) > 1 else [])
    s = hex_to_lab(sample_hex)
    return min(delta_e2000(s, c, kL) for c in cands)


def kit_distance(sample: Mapping, kit: Mapping, *, lightness_weight: float = LIGHTNESS_WEIGHT) -> float:
    """Weighted-mean ΔE over the full kit vector (shirt, sleeves, shorts, socks)."""
    kL = lightness_weight
    parts: list[tuple[float, float]] = []
    if sample.get("shirt") and kit.get("shirt"):
        parts.append((_PART_WEIGHTS["shirt"], _shirt_distance(sample["shirt"], kit, kL)))
    for key in ("shorts", "socks"):
        if sample.get(key) and kit.get(key):
            parts.append((_PART_WEIGHTS[key], delta_e_hex(sample[key], kit[key], kL)))
    if sample.get("sleeve_color"):
        target = kit.get("sleeve_color") or kit.get("shirt")
        parts.append((_PART_WEIGHTS["sleeves"], delta_e_hex(sample["sleeve_color"], target, kL)))
    if not parts:
        return float("inf")
    wsum = sum(w for w, _ in parts)
    return sum(w * d for w, d in parts) / wsum


@dataclass(frozen=True)
class SnapResult:
    ref: str | None
    delta_e: float
    spec: dict
    snapped: bool

    def to_dict(self) -> dict:
        return {"ref": self.ref, "delta_e": round(self.delta_e, 2), "snapped": self.snapped}


def snap_kit(
    sample: Mapping,
    candidates: Mapping[str, Mapping],
    *,
    snap_de: float = SNAP_DELTA_E,
    lightness_weight: float = LIGHTNESS_WEIGHT,
    ) -> SnapResult:
    """Nearest library candidate by full-vector distance, or the sample itself.

    ``sample`` needs ``shirt`` (and ideally shorts/socks); the returned
    ``spec`` is always a complete KitSpec (sampled fallback fills
    missing parts from the shirt colour / sensible defaults).
    """
    best_ref, best_d = None, float("inf")
    for ref, kit in candidates.items():
        d = kit_distance(sample, kit, lightness_weight=lightness_weight)
        if d < best_d:
            best_ref, best_d = ref, d
    if best_ref is not None and best_d < snap_de:
        spec = dict(candidates[best_ref])
        spec["source"] = "library"
        spec["ref"] = best_ref
        spec["delta_e"] = round(best_d, 2)
        return SnapResult(best_ref, best_d, spec, True)
    shirt = sample.get("shirt", "#808080")
    spec = {
        "shirt": shirt,
        "shorts": sample.get("shorts", shirt),
        "socks": sample.get("socks", sample.get("shorts", shirt)),
        "sleeves": "short",
        "source": "sampled",
    }
    if sample.get("sleeve_color"):
        spec["sleeve_color"] = sample["sleeve_color"]
    return SnapResult(best_ref, best_d, spec, False)


def sample_to_hex_parts(lab_parts: Mapping[str, np.ndarray | None], wb: WhiteBalance) -> dict[str, str]:
    """White-balance Lab part medians and return ``{part: hex}``."""
    out: dict[str, str] = {}
    for key, lab in lab_parts.items():
        if lab is None:
            continue
        rgb = apply_gains(lab_to_srgb(np.asarray(lab)), wb)
        out[key] = rgb_to_hex(rgb)
    return out

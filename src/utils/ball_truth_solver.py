"""Ball Studio solver: authored keys/segments -> dense 3-D track + diagnostics.

Pure (numpy/scipy/cv2 only, no IO). The caller supplies a
:class:`SolveContext` that answers "camera for (shot, shot-local frame)" and
"world position of (player, bone) at reference frame". The router in
``src/web/ball_studio.py`` builds one from an output directory; the tests
build one from synthetic cameras.

Model (see ``docs/superpowers/specs/2026-10-04-ball-studio-design.md``):

* keys are 3-D control points on the reference timeline, derived from pixel
  observations (multi-view triangulation or a single-view ray constraint) so
  they stay consistent with the camera tracks;
* segments between consecutive keys: ``flight`` (endpoint-exact gravity +
  quadratic drag shoot, bounded Magnus fit to soft observations), ``roll``
  (ground, endpoint-exact constant deceleration), ``carried`` (follows a player
  joint), ``linear``, ``static``.

Physics primitives are reused from ``ball_hybrid_physics`` and
``ball_hybrid_spin``; camera math from ``camera_projection``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.utils.ball_hybrid_physics import (
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    fit_roll_segment,
    shoot_arc,
    simulate,
)
from src.utils.ball_hybrid_spin import fit_span_spin
from src.utils.camera_projection import (
    pixel_ray,
    project_point_onto_pixel_ray,
    project_world_to_image,
)

GROUND_Z = 0.11
MAX_KEY_RESIDUAL_PX = 15.0
WEAK_BASELINE_DEG = 3.0
BELOW_GROUND_Z = 0.05
MAX_SPEED_M_S = 45.0
MAX_CURL_ACCEL_M_S2 = 15.0
SOFT_OUTLIER_PX = 12.0
EVENT_JUMP_M_S = 25.0
ROLL_MAX_Z = 0.5
STATIC_MAX_DRIFT_M = 0.3
_MIN_DEPTH_M = 0.1

Vec3 = np.ndarray


# ---------------------------------------------------------------------------
# camera + context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cam:
    """One shot-frame camera: pinhole + 2-coefficient radial distortion."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: tuple[float, float] = (0.0, 0.0)
    image_size: tuple[int, int] | None = None  # (w, h)

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def depth(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return (pts @ self.R.T + self.t)[:, 2]

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(uv (N,2), depth (N,))``; uv is NaN where the point is behind
        the camera."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        depth = self.depth(pts)
        uv = project_world_to_image(self.K, self.R, self.t, self.dist, pts)
        uv = np.where((depth > _MIN_DEPTH_M)[:, None], uv, np.nan)
        return uv, depth

    def ray(self, uv: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        return pixel_ray(tuple(uv), self.K, self.R, self.t, self.dist)


CameraFn = Callable[[str, int], "Cam | None"]
JointFn = Callable[[str, str, int], "np.ndarray | None"]


@dataclass(frozen=True)
class SolveContext:
    fps: float
    offsets: dict[str, int]          # shot_id -> frame_offset (shot = ref + off)
    camera: CameraFn                 # (shot_id, shot_frame) -> Cam | None
    joint: JointFn = lambda pid, bone, ref_frame: None  # type: ignore[assignment]

    def shot_frame(self, shot_id: str, ref_frame: int) -> int:
        return int(ref_frame) + int(self.offsets.get(shot_id, 0))

    def ref_frame(self, shot_id: str, shot_frame: int) -> int:
        return int(shot_frame) - int(self.offsets.get(shot_id, 0))


# ---------------------------------------------------------------------------
# triangulation / single-view constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriResult:
    ok: bool
    reason: str | None
    xyz: np.ndarray | None
    residual_px: list[float]
    ray_angle_deg: float
    rays: list[tuple[np.ndarray, np.ndarray]]
    skew_gap_cm: float = 0.0
    reproj_uv: list[list[float] | None] = None  # type: ignore[assignment]


def _max_pair_angle_deg(dirs: Sequence[np.ndarray]) -> float:
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            c = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            best = max(best, math.degrees(math.acos(c)))
    return best


def _max_skew_gap_cm(rays: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Largest closest-approach distance between any two of the rays (cm)."""
    best = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            (c1, d1), (c2, d2) = rays[i], rays[j]
            n = np.cross(d1, d2)
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                continue
            best = max(best, abs(float(np.dot(c2 - c1, n))) / nn)
    return best * 100.0


def triangulate(views: Sequence[tuple[Cam, Sequence[float]]]) -> TriResult:
    """Least-squares 3-D point from >= 2 (camera, pixel) views.

    Linear closest-point-to-rays start, then Gauss-Newton (LM) on pixel
    reprojection including lens distortion. ``residual_px`` is the per-view
    reprojection error of the returned point.
    """
    if len(views) < 2:
        raise ValueError("triangulate needs >= 2 views")
    rays = [cam.ray(uv) for cam, uv in views]
    dirs = [d for _, d in rays]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    if np.linalg.eigvalsh(A)[0] < 1e-6 * len(views):
        return TriResult(False, "parallel_rays", None, [], 0.0, rays)
    x0 = np.linalg.solve(A, b)

    def resid(x: np.ndarray) -> np.ndarray:
        out = []
        for cam, uv in views:
            p, depth = cam.project(x)
            if not np.isfinite(p).all():
                # behind this camera: push the optimiser back toward the rays
                out.extend([1e3, 1e3])
            else:
                out.extend([p[0, 0] - uv[0], p[0, 1] - uv[1]])
        return np.asarray(out)

    sol = least_squares(resid, x0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=100)
    x = sol.x
    r = resid(x).reshape(-1, 2)
    px = [float(np.hypot(a, b_)) for a, b_ in r]
    angle = _max_pair_angle_deg(dirs)
    gap = _max_skew_gap_cm(rays)
    reproj: list[list[float] | None] = []
    for cam, _ in views:
        p, _d = cam.project(x)
        reproj.append(
            [round(float(p[0, 0]), 2), round(float(p[0, 1]), 2)]
            if np.isfinite(p).all() else None
        )
    if any(cam.depth(x)[0] <= _MIN_DEPTH_M for cam, _ in views):
        return TriResult(False, "behind_camera", x, px, angle, rays, gap, reproj)
    return TriResult(True, None, x, px, angle, rays, gap, reproj)


def ray_plane_point(
    origin: np.ndarray, direction: np.ndarray, axis: str, value: float,
) -> np.ndarray | None:
    """Intersection of a ray with the axis-aligned plane ``axis = value``
    (forward of the origin only)."""
    ai = "xyz".index(axis)
    if abs(direction[ai]) < 1e-9:
        return None
    s = (value - origin[ai]) / direction[ai]
    if s <= 0:
        return None
    return origin + s * direction


def constrain_ray(
    cam: Cam,
    uv: Sequence[float],
    constraint: dict,
    mode: str,
    *,
    ref_frame: int,
    joint: JointFn,
) -> tuple[np.ndarray | None, str | None]:
    """Single-view key: ray intersected with a constraint.

    ``mode`` is one of ``ground | height | plane | depth | player``.
    Returns ``(xyz, error_reason)``.
    """
    C, d = cam.ray(uv)
    if mode == "ground":
        p = ray_plane_point(C, d, "z", GROUND_Z)
        return (p, None) if p is not None else (None, "parallel_rays")
    if mode == "height":
        h = constraint.get("height_m")
        if h is None:
            return None, "missing_constraint"
        p = ray_plane_point(C, d, "z", float(h))
        return (p, None) if p is not None else (None, "parallel_rays")
    if mode == "plane":
        pl = constraint.get("plane")
        if not pl:
            return None, "missing_constraint"
        p = ray_plane_point(C, d, pl["axis"], float(pl["value"]))
        return (p, None) if p is not None else (None, "parallel_rays")
    if mode == "depth":
        dep = constraint.get("depth_m")
        if dep is None:
            return None, "missing_constraint"
        return C + float(dep) * d, None
    if mode == "player":
        pid, bone = constraint.get("player_id"), constraint.get("bone")
        j = joint(pid, bone, ref_frame) if pid and bone else None
        if j is None:
            return None, "unknown_player_joint"
        j = np.asarray(j, dtype=float)
        if constraint.get("offset") is not None:
            j = j + np.asarray(constraint["offset"], dtype=float)
        # pixel authoritative laterally, depth from the joint
        return project_point_onto_pixel_ray(j, uv, cam.K, cam.R, cam.t, cam.dist), None
    return None, "unknown_mode"


def epipolar_polyline(
    origin: np.ndarray,
    direction: np.ndarray,
    other: Cam,
    *,
    near_m: float = 2.0,
    far_m: float = 250.0,
    n_samples: int = 400,
    margin_px: float = 0.0,
    max_points: int = 24,
) -> list[list[float]] | None:
    """The ray ``origin + s*direction`` (s in [near, far]) projected into
    ``other`` and clipped to its image; ``None`` if it never enters it."""
    s = np.geomspace(near_m, far_m, n_samples)
    pts = origin[None, :] + s[:, None] * direction[None, :]
    uv, _ = other.project(pts)
    if other.image_size is not None:
        w, h = other.image_size
    else:
        w, h = other.K[0, 2] * 2.0, other.K[1, 2] * 2.0
    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= -margin_px) & (uv[:, 0] <= w + margin_px)
        & (uv[:, 1] >= -margin_px) & (uv[:, 1] <= h + margin_px)
    )
    if not inside.any():
        return None
    idx = np.nonzero(inside)[0]
    keep = idx[np.unique(np.linspace(0, len(idx) - 1, max_points).astype(int))]
    return [[round(float(uv[i, 0]), 2), round(float(uv[i, 1]), 2)] for i in keep]


# ---------------------------------------------------------------------------
# key resolution
# ---------------------------------------------------------------------------

_MODE_FOR_SOURCE = {
    "ray_ground": "ground", "ray_height": "height", "ray_plane": "plane",
    "ray_depth": "depth", "player": "player",
}


def _flag(level: str, code: str, message: str, *, frame: int | None = None,
          ref: dict | None = None) -> dict:
    out = {"level": level, "code": code, "message": message}
    if frame is not None:
        out["frame"] = int(frame)
    if ref:
        out["ref"] = ref
    return out


def resolve_key(key: dict, ctx: SolveContext) -> tuple[dict, list[dict]]:
    """Re-derive a key's 3-D position from its observations / constraint.

    Returns ``(resolved_key_dict, flags)``. ``manual`` keys (and keys whose
    derivation fails) keep the stored ``xyz``.
    """
    flags: list[dict] = []
    kid, fr = key["id"], key["frame"]
    ref = {"key": kid}
    xyz = np.asarray(key["xyz"], dtype=float)
    residual: dict[str, float] = {}
    angle: float | None = None
    skew_gap: float | None = None
    src = key["source"]
    messages: list[str] = []
    status = "ok"

    def fail(code: str, msg: str, level: str = "error") -> None:
        nonlocal status
        flags.append(_flag(level, code, msg, frame=fr, ref=ref))
        messages.append(msg)
        status = "error" if level == "error" else (status if status == "error" else "warn")

    cams: list[tuple[str, Cam, list[float]]] = []
    for ob in key["observations"]:
        sf = ob["shot_frame"]
        cam = ctx.camera(ob["shot_id"], sf)
        if cam is None:
            fail("no_camera_frame", f"no camera for {ob['shot_id']} frame {sf}")
            continue
        cams.append((ob["shot_id"], cam, ob["uv"]))

    if src == "triangulated":
        if len(cams) >= 2:
            res = triangulate([(c, uv) for _, c, uv in cams])
            if res.xyz is not None:
                xyz = res.xyz
            residual = {sid: round(r, 3) for (sid, _, _), r in zip(cams, res.residual_px)}
            angle = res.ray_angle_deg
            skew_gap = res.skew_gap_cm
            if not res.ok:
                fail(res.reason or "triangulation_failed", f"triangulation failed: {res.reason}")
            else:
                if max(res.residual_px) > MAX_KEY_RESIDUAL_PX:
                    fail("residual_exceeds_limit",
                         f"max view residual {max(res.residual_px):.1f}px > {MAX_KEY_RESIDUAL_PX:.0f}px")
                if angle < WEAK_BASELINE_DEG:
                    fail("weak_baseline", f"rays differ by only {angle:.1f} deg", "warn")
        else:
            fail("no_camera_frame", "triangulated key has < 2 usable views")
    elif src in _MODE_FOR_SOURCE:
        mode = _MODE_FOR_SOURCE[src]
        constraint = dict(key["constraint"])
        if src == "ray_ground":
            constraint["height_m"] = GROUND_Z
        if cams:
            sid, cam, uv = cams[0]
            p, err = constrain_ray(cam, uv, constraint, mode, ref_frame=fr, joint=ctx.joint)
            if p is None:
                fail(err or "constraint_failed", f"{mode} constraint failed: {err}")
            else:
                xyz = p
                proj, depth = cam.project(p)
                if depth[0] <= _MIN_DEPTH_M:
                    fail("behind_camera", "constraint point is behind the camera")
                else:
                    residual = {sid: round(float(np.hypot(*(proj[0] - np.asarray(uv)))), 3)}
        elif src == "player":
            j = ctx.joint(constraint.get("player_id"), constraint.get("bone"), fr)
            if j is None:
                fail("unknown_player_joint", "player joint unavailable at this frame")
            else:
                xyz = np.asarray(j, dtype=float)
                if constraint.get("offset") is not None:
                    xyz = xyz + np.asarray(constraint["offset"], dtype=float)
    # manual: keep xyz as stored

    out = {
        "id": kid, "frame": fr,
        "xyz": [round(float(v), 4) for v in xyz],
        "source": src, "residual_px": residual,
        "ray_angle_deg": None if angle is None else round(angle, 2),
        "skew_gap_cm": None if skew_gap is None else round(skew_gap, 2),
        "status": status, "messages": messages,
    }
    return out, flags


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------

def auto_kind(a_xyz: np.ndarray, b_xyz: np.ndarray) -> str:
    """Default segment kind for two keys: both on the ground -> roll."""
    if a_xyz[2] <= 0.3 and b_xyz[2] <= 0.3:
        return "roll"
    return "flight"


def _magnus_accel(omega: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(DEFAULT_MAGNUS_COEFF * np.cross(omega, v)))


def _eval_flight(
    pa: np.ndarray, pb: np.ndarray, times: np.ndarray, duration: float,
    cd: float, omega: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    v0 = shoot_arc(pa, 0.0, pb, duration, cd=cd, omega=omega)
    return simulate(pa, v0, times, cd=cd, omega=omega), v0


@dataclass
class _SegResult:
    positions: np.ndarray   # (n_frames_inclusive, 3)
    params: dict
    flags: list[dict]


def _soft_obs_in_span(
    soft: Sequence[dict], ctx: SolveContext, fa: int, fb: int,
) -> list[tuple[int, str, list[float], Cam]]:
    out = []
    for ob in soft:
        r = ctx.ref_frame(ob["shot_id"], ob["shot_frame"])
        if fa < r < fb:
            cam = ctx.camera(ob["shot_id"], ob["shot_frame"])
            if cam is not None:
                out.append((r, ob["shot_id"], ob["uv"], cam))
    return out


def _solve_flight(a: dict, b: dict, seg: dict, ctx: SolveContext,
                  soft: Sequence[dict], idx: int) -> _SegResult:
    fa, fb = a["frame"], b["frame"]
    pa, pb = np.asarray(a["xyz"], float), np.asarray(b["xyz"], float)
    frames = np.arange(fa, fb + 1)
    times = (frames - fa) / ctx.fps
    duration = (fb - fa) / ctx.fps
    prm = seg["params"]
    cd = 0.0 if not prm["drag"] else (prm["cd"] if prm["cd"] is not None else CD_DEFAULT)
    flags: list[dict] = []
    omega = None
    spin_info = None
    obs = _soft_obs_in_span(soft, ctx, fa, fb)
    if prm["magnus"] == "auto" and len(obs) >= 3:
        obs_t = np.array([(r - fa) / ctx.fps + i * 1e-9 for i, (r, *_rest) in enumerate(obs)])
        lookup = {float(t): o for t, o in zip(obs_t, obs)}

        def project_fn(t_s: float, xyz: np.ndarray) -> np.ndarray:
            _, _, _, cam = lookup[float(t_s)]
            uv, _ = cam.project(xyz)
            return uv[0] if np.isfinite(uv).all() else np.array([1e4, 1e4])

        try:
            fit = fit_span_spin(
                pa, 0.0, pb, duration, obs_t, [o[2] for o in obs], project_fn,
                cd=cd, min_obs=3,
            )
        except Exception:  # fit is best effort; the no-spin arc stays valid
            fit = None
        if fit is not None:
            omega = np.asarray(fit.omega_world, float)
            spin_info = {"omega": [round(float(x), 3) for x in omega],
                         "delta_bic": round(float(fit.delta_bic), 2)}
    pos, v0 = _eval_flight(pa, pb, times, duration, cd, omega)
    if omega is not None:
        vel = np.gradient(pos, times, axis=0) if len(times) > 2 else np.array([v0])
        curl = max(_magnus_accel(omega, v) for v in vel)
        spin_info["max_curl_accel_m_s2"] = round(curl, 2)
        if curl > MAX_CURL_ACCEL_M_S2:
            flags.append(_flag(
                "warn", "curl_exceeds_limit",
                f"flight needs {curl:.1f} m/s^2 of curl (> {MAX_CURL_ACCEL_M_S2:.0f})",
                frame=fa, ref={"segment": idx}))
    params = {"cd": cd, "v0": [round(float(x), 4) for x in v0], "omega": None,
              "accel_xy": None}
    if spin_info is not None:
        params.update(spin_info)
    return _SegResult(pos, params, flags)


def _solve_roll(a: dict, b: dict, seg: dict, ctx: SolveContext,
                soft: Sequence[dict], idx: int) -> _SegResult:
    fa, fb = a["frame"], b["frame"]
    pa, pb = np.asarray(a["xyz"], float), np.asarray(b["xyz"], float)
    frames = np.arange(fa, fb + 1)
    times = (frames - fa) / ctx.fps
    duration = (fb - fa) / ctx.fps
    flags: list[dict] = []
    if pa[2] > ROLL_MAX_Z or pb[2] > ROLL_MAX_Z:
        flags.append(_flag("error", "segment_infeasible",
                           f"roll needs both keys near the ground (z={pa[2]:.2f}, {pb[2]:.2f})",
                           frame=fa, ref={"segment": idx}))
    z_mean = 0.5 * (pa[2] + pb[2])
    obs_xy = []
    for r, _sid, uv, cam in _soft_obs_in_span(soft, ctx, fa, fb):
        C, d = cam.ray(uv)
        p = ray_plane_point(C, d, "z", z_mean)
        if p is not None:
            obs_xy.append(((r - fa) / ctx.fps, p[:2], 1.0))
    fit = fit_roll_segment(pa[:2], pb[:2], duration, obs=obs_xy)
    pos = fit.eval(times, z_mean)
    # z follows the endpoints (a roll that starts at 0.11 and ends at 0.11
    # stays flat; slightly different endpoints blend linearly)
    pos[:, 2] = pa[2] + (pb[2] - pa[2]) * (times / duration)
    params = {"cd": None, "accel_xy": [round(x, 4) for x in fit.accel_xy],
              "v0": None, "omega": None}
    return _SegResult(pos, params, flags)


def _solve_carried(a: dict, b: dict, seg: dict, ctx: SolveContext, idx: int) -> _SegResult:
    fa, fb = a["frame"], b["frame"]
    pa, pb = np.asarray(a["xyz"], float), np.asarray(b["xyz"], float)
    frames = np.arange(fa, fb + 1)
    prm = seg["params"]
    pid = prm["player_id"] or a["constraint"].get("player_id") or b["constraint"].get("player_id")
    bone = prm["bone"] or a["constraint"].get("bone") or b["constraint"].get("bone")
    flags: list[dict] = []
    lin = pa[None, :] + (pb - pa)[None, :] * ((frames - fa) / (fb - fa))[:, None]
    if not pid or not bone:
        flags.append(_flag("error", "unknown_player_joint",
                           "carried segment needs player_id and bone",
                           frame=fa, ref={"segment": idx}))
        return _SegResult(lin, {}, flags)
    ja, jb = ctx.joint(pid, bone, fa), ctx.joint(pid, bone, fb)
    if ja is None or jb is None:
        flags.append(_flag("error", "unknown_player_joint",
                           f"{pid}/{bone} has no data at the segment ends",
                           frame=fa, ref={"segment": idx}))
        return _SegResult(lin, {}, flags)
    off_a, off_b = pa - np.asarray(ja, float), pb - np.asarray(jb, float)
    pos = np.empty_like(lin)
    missing = 0
    for i, f in enumerate(frames):
        j = ctx.joint(pid, bone, int(f))
        u = (f - fa) / (fb - fa)
        if j is None:
            pos[i] = lin[i]
            missing += 1
        else:
            pos[i] = np.asarray(j, float) + (1 - u) * off_a + u * off_b
    if missing:
        flags.append(_flag("warn", "unknown_player_joint",
                           f"{missing} frame(s) without joint data, linear fill",
                           frame=fa, ref={"segment": idx}))
    return _SegResult(pos, {"player_id": pid, "bone": bone}, flags)


def _solve_linear(a: dict, b: dict) -> _SegResult:
    fa, fb = a["frame"], b["frame"]
    pa, pb = np.asarray(a["xyz"], float), np.asarray(b["xyz"], float)
    frames = np.arange(fa, fb + 1)
    pos = pa[None, :] + (pb - pa)[None, :] * ((frames - fa) / (fb - fa))[:, None]
    return _SegResult(pos, {}, [])


def _solve_static(a: dict, b: dict, idx: int) -> _SegResult:
    fa, fb = a["frame"], b["frame"]
    pa, pb = np.asarray(a["xyz"], float), np.asarray(b["xyz"], float)
    frames = np.arange(fa, fb + 1)
    pos = np.tile(pa, (len(frames), 1))
    pos[-1] = pb
    flags = []
    if np.linalg.norm(pb - pa) > STATIC_MAX_DRIFT_M:
        flags.append(_flag("warn", "segment_infeasible",
                           f"static segment keys are {np.linalg.norm(pb - pa):.2f} m apart",
                           frame=fa, ref={"segment": idx}))
    return _SegResult(pos, {}, flags)


# ---------------------------------------------------------------------------
# full solve
# ---------------------------------------------------------------------------

def _r4(x: float) -> float:
    return round(float(x), 4)


def solve_truth(doc: dict, ctx: SolveContext) -> dict:
    """Solve a (validated, normalised) truth document. See the API doc for
    the response shape."""
    flags: list[dict] = []
    keys_in = sorted(doc["keys"], key=lambda k: k["frame"])
    resolved: list[dict] = []
    for k in keys_in:
        rk, fl = resolve_key(k, ctx)
        resolved.append(rk)
        flags.extend(fl)

    # carry over constraint info needed by segments
    by_id_in = {k["id"]: k for k in keys_in}
    seg_in = {s["from"]: s for s in doc["segments"]}
    seg_results: list[dict] = []
    dense_frames: list[int] = []
    dense_xyz: list[list[float]] = []
    dense_seg: list[int] = []
    dense_kind: list[str] = []
    soft = doc["observations"]

    for i in range(len(resolved) - 1):
        a, b = resolved[i], resolved[i + 1]
        a_full = {**a, "constraint": by_id_in[a["id"]]["constraint"]}
        b_full = {**b, "constraint": by_id_in[b["id"]]["constraint"]}
        declared = seg_in.get(a["id"])
        auto = declared is None or declared["to"] != b["id"]
        pa, pb = np.asarray(a["xyz"]), np.asarray(b["xyz"])
        if declared is not None and declared["to"] != b["id"]:
            flags.append(_flag(
                "error", "segment_infeasible",
                f"segment {declared['from']}->{declared['to']} must join consecutive keys",
                frame=a["frame"], ref={"segment": i}))
        seg = (
            {"from": a["id"], "to": b["id"], "kind": auto_kind(pa, pb),
             "params": {"drag": True, "cd": None, "magnus": "auto",
                        "player_id": None, "bone": None}}
            if auto else declared
        )
        kind = seg["kind"]
        if kind == "flight":
            res = _solve_flight(a_full, b_full, seg, ctx, soft, i)
        elif kind == "roll":
            res = _solve_roll(a_full, b_full, seg, ctx, soft, i)
        elif kind == "carried":
            res = _solve_carried(a_full, b_full, seg, ctx, i)
        elif kind == "static":
            res = _solve_static(a_full, b_full, i)
        else:
            res = _solve_linear(a_full, b_full)
        flags.extend(res.flags)
        frames = list(range(a["frame"], b["frame"] + 1))
        # drop the shared end frame of the previous segment
        start = 1 if dense_frames and dense_frames[-1] == frames[0] else 0
        for f, p in zip(frames[start:], res.positions[start:]):
            dense_frames.append(f)
            dense_xyz.append([_r4(v) for v in p])
            dense_seg.append(i)
            dense_kind.append(kind)
        seg_results.append({
            "index": i, "from": a["id"], "to": b["id"], "kind": kind,
            "auto": bool(auto), "frame_range": [a["frame"], b["frame"]],
            "params": res.params,
            "n_soft_obs": len(_soft_obs_in_span(soft, ctx, a["frame"], b["frame"])),
            "rms_obs_px": None, "max_speed_m_s": None, "status": "ok",
        })
    if len(resolved) == 1:
        dense_frames.append(resolved[0]["frame"])
        dense_xyz.append(list(resolved[0]["xyz"]))
        dense_seg.append(-1)
        dense_kind.append("key")

    frames_arr = np.asarray(dense_frames, dtype=int)
    xyz_arr = np.asarray(dense_xyz, dtype=float).reshape(-1, 3)

    # speeds (finite difference) + sanity flags
    speed = np.zeros(len(frames_arr))
    if len(frames_arr) > 1:
        step = np.diff(xyz_arr, axis=0) * ctx.fps / np.diff(frames_arr)[:, None]
        sp = np.linalg.norm(step, axis=1)
        speed[0], speed[-1] = sp[0], sp[-1]
        if len(sp) > 1:
            speed[1:-1] = 0.5 * (sp[:-1] + sp[1:])
        for si in range(len(seg_results)):
            m = np.asarray(dense_seg) == si
            if m.any():
                seg_results[si]["max_speed_m_s"] = round(float(speed[m].max()), 2)
                if speed[m].max() > MAX_SPEED_M_S:
                    flags.append(_flag(
                        "warn", "speed_exceeds_limit",
                        f"speed {speed[m].max():.1f} m/s > {MAX_SPEED_M_S:.0f}",
                        frame=int(frames_arr[m][np.argmax(speed[m])]), ref={"segment": si}))
        event_frames = {e["frame"] for e in doc["events"]}
        for ki in range(1, len(resolved) - 1):
            f = resolved[ki]["frame"]
            j = int(np.nonzero(frames_arr == f)[0][0]) if (frames_arr == f).any() else None
            if j is None or j == 0 or j >= len(sp):
                continue
            jump = float(np.linalg.norm(step[j] - step[j - 1]))
            if jump > EVENT_JUMP_M_S and not any(abs(f - ef) <= 1 for ef in event_frames):
                flags.append(_flag(
                    "warn", "discontinuity",
                    f"velocity changes by {jump:.1f} m/s at key {resolved[ki]['id']} with no event",
                    frame=f, ref={"key": resolved[ki]["id"]}))
        below = np.nonzero(xyz_arr[:, 2] < BELOW_GROUND_Z)[0]
        if len(below):
            j = int(below[np.argmin(xyz_arr[below, 2])])
            flags.append(_flag("warn", "below_ground",
                               f"z={xyz_arr[j, 2]:.2f} m", frame=int(frames_arr[j]),
                               ref={"segment": int(dense_seg[j])}))

    # projections + observation residuals
    projections: dict[str, dict] = {}
    dense_lookup = {int(f): xyz_arr[i] for i, f in enumerate(frames_arr)}
    for sid in ctx.offsets:
        fr_l, sf_l, uv_l, dp_l = [], [], [], []
        for i, f in enumerate(frames_arr):
            sf = ctx.shot_frame(sid, int(f))
            cam = ctx.camera(sid, sf)
            if cam is None:
                continue
            uv, dep = cam.project(xyz_arr[i])
            fr_l.append(int(f))
            sf_l.append(sf)
            ok = np.isfinite(uv).all()
            uv_l.append([round(float(uv[0, 0]), 2), round(float(uv[0, 1]), 2)] if ok else None)
            dp_l.append(round(float(dep[0]), 3) if ok else None)
        if fr_l:
            projections[sid] = {"frames": fr_l, "shot_frames": sf_l, "uv": uv_l, "depth_m": dp_l}

    obs_out: list[dict] = []

    def _score(kind: str, ob: dict, ref_xyz: np.ndarray | None, extra: dict) -> float | None:
        cam = ctx.camera(ob["shot_id"], ob["shot_frame"])
        proj_uv, resid = None, None
        if cam is not None and ref_xyz is not None:
            uv, _ = cam.project(ref_xyz)
            if np.isfinite(uv).all():
                proj_uv = [round(float(uv[0, 0]), 2), round(float(uv[0, 1]), 2)]
                resid = round(float(np.hypot(*(uv[0] - np.asarray(ob["uv"])))), 3)
        obs_out.append({"kind": kind, **extra, "shot_id": ob["shot_id"],
                        "shot_frame": ob["shot_frame"], "uv": list(ob["uv"]),
                        "projected_uv": proj_uv, "residual_px": resid})
        return resid

    for k_in, rk in zip(keys_in, resolved):
        for ob in k_in["observations"]:
            _score("key", ob, np.asarray(rk["xyz"]), {"key_id": rk["id"]})
    seg_resid: dict[int, list[float]] = {}
    for i, ob in enumerate(soft):
        r = ctx.ref_frame(ob["shot_id"], ob["shot_frame"])
        p = dense_lookup.get(r)
        resid = _score("soft", ob, p, {"index": i})
        if p is None:
            continue
        if resid is not None:
            si = next((s["index"] for s in seg_results
                       if s["frame_range"][0] <= r <= s["frame_range"][1]), None)
            if si is not None:
                seg_resid.setdefault(si, []).append(resid)
            if resid > SOFT_OUTLIER_PX:
                flags.append(_flag("warn", "observation_outlier",
                                   f"soft observation {resid:.1f}px from the solved track",
                                   frame=r, ref={"observation": i}))
    for si, rs in seg_resid.items():
        seg_results[si]["rms_obs_px"] = round(float(np.sqrt(np.mean(np.square(rs)))), 3)

    # statuses
    for f in flags:
        ref = f.get("ref") or {}
        if "segment" in ref and 0 <= ref["segment"] < len(seg_results):
            s = seg_results[ref["segment"]]
            if f["level"] == "error":
                s["status"] = "error"
            elif s["status"] == "ok":
                s["status"] = "warn"
        if "key" in ref and f["level"] != "error":
            for rk in resolved:
                if rk["id"] == ref["key"] and rk["status"] == "ok":
                    rk["status"] = "warn"
    # de-dup flags raised both by resolve_key and the status pass
    ok = not any(f["level"] == "error" for f in flags)
    key_res = [v for rk in resolved for v in rk["residual_px"].values()]
    stats = {
        "n_keys": len(resolved), "n_segments": len(seg_results),
        "n_dense": int(len(frames_arr)),
        "max_key_residual_px": round(max(key_res), 3) if key_res else None,
        "max_speed_m_s": round(float(speed.max()), 2) if len(speed) else None,
    }
    return {
        "ok": ok,
        "keys": resolved,
        "segments": seg_results,
        "dense": {
            "frames": [int(f) for f in frames_arr],
            "xyz": dense_xyz,
            "segment": dense_seg,
            "kind": dense_kind,
            "speed_m_s": [round(float(s), 3) for s in speed],
        },
        "projections": projections,
        "observations": obs_out,
        "flags": flags,
        "stats": stats,
    }

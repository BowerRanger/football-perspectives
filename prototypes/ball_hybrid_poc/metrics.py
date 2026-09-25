"""Pure metric functions over ``(Track, TruthTrack, ClipContext)`` for the
ball hybrid-extraction PoC. See ``CONTRACT.md``'s "Metrics" section for the
list this module implements.

Every function here is a pure function of its JSON-safe inputs (no file
I/O, no global state) so it can be unit-tested with hand-built tiny tracks.
``run_all.py`` is the only place that reads/writes files and assembles the
per-clip ``Results`` dict.

Design note on the flat vs. nested split: the PoC viewer
(``viewer/template.html``) renders ``scenario["metrics"][method]`` as a
generic table and calls ``value.toFixed(...)`` on every entry, so that dict
must stay a flat map of scalars (float/int/None) — never a nested object.
Richer breakdowns (per-state splits, per-event contact gaps, naturalness
violation kinds, ...) go in a sibling ``metrics_detail`` dict instead,
which the current viewer simply ignores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from src.utils.ball_eval import NaturalnessCfg
from src.utils.ball_eval import naturalness_violations as _naturalness_violations

from .types import Track, TruthTrack

BALL_RADIUS_M = 0.11
SINK_Z_M = 0.09
FLOAT_Z_M = 0.25
LE_THRESHOLD_M = 0.20
# z above this is treated as "airborne" for the naturalness adapter, which
# only needs a coarse ground/flight split (see module docstring on Track
# having no native `.state`).
FLIGHT_Z_THRESH_M = 0.15
_CONTACT_KINDS = ("touch", "bounce", "net", "post")

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _track_xyz_by_frame(track: Track) -> dict[int, Optional[Vec3]]:
    return {tf.frame: tf.xyz for tf in track.frames}


def _percentile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


# ---------------------------------------------------------------------------
# 1. per-frame 3-D error + overall/split stats
# ---------------------------------------------------------------------------

def per_frame_error(track: Track, truth: TruthTrack) -> list[Optional[float]]:
    """3-D Euclidean error (m) per truth frame, aligned to ``truth.frames``
    order. A truth frame the method has no estimate for (frame absent from
    the track, or present with ``xyz=None``) is ``None`` — a FAIL, not a
    skip; callers must count it in denominators (``error_stats`` does)."""
    by_frame = _track_xyz_by_frame(track)
    out: list[Optional[float]] = []
    for tf in truth.frames:
        xyz = by_frame.get(tf.frame)
        out.append(_dist3(tf.xyz, xyz) if xyz is not None else None)
    return out


def error_stats(errs: Sequence[Optional[float]],
                 *, le_threshold: float = LE_THRESHOLD_M) -> dict[str, Any]:
    """p50/p95/max over the *successful* (non-``None``) frames, plus
    ``coverage`` and ``pct_le_20cm`` computed against the FULL denominator
    (``len(errs)``) so a null frame counts as a miss for both, per
    CONTRACT.md ("null method frames ... count as FAILS, and report
    coverage")."""
    n = len(errs)
    valid = [e for e in errs if e is not None]
    n_valid = len(valid)
    n_le = sum(1 for e in valid if e <= le_threshold)
    return {
        "n": n,
        "n_valid": n_valid,
        "coverage": (n_valid / n) if n else None,
        "p50": _percentile(valid, 50),
        "p95": _percentile(valid, 95),
        "max": max(valid) if valid else None,
        "pct_le_20cm": (n_le / n) if n else None,
    }


_EMPTY_STATS = {"n": 0, "n_valid": 0, "coverage": None, "p50": None,
                "p95": None, "max": None, "pct_le_20cm": None}


def split_by_state(track: Track, truth: TruthTrack) -> dict[str, dict]:
    """``error_stats()`` computed separately for truth frames whose
    ``.state`` is ``ground`` / ``air`` / ``contact``."""
    errs = per_frame_error(track, truth)
    out: dict[str, dict] = {}
    for state in ("ground", "air", "contact"):
        sub = [e for tf, e in zip(truth.frames, errs) if tf.state == state]
        out[state] = error_stats(sub) if sub else dict(_EMPTY_STATS)
    return out


# ---------------------------------------------------------------------------
# 2. contact gap at truth touch/bounce/net/post events
# ---------------------------------------------------------------------------

def contact_gap(track: Track, truth: TruthTrack) -> dict[str, Any]:
    """Distance from the method's xyz to the truth event's xyz at each
    contact-like event frame (``exact``), plus the min over the event
    frame +/-1 (``min_pm1``) to separate a pure timing error (right
    position, one frame late/early) from a genuine positional miss."""
    by_frame = _track_xyz_by_frame(track)
    exact: list[Optional[float]] = []
    best_pm1: list[Optional[float]] = []
    per_event: list[dict] = []
    for ev in truth.events:
        if ev.kind not in _CONTACT_KINDS:
            continue
        xyz_here = by_frame.get(ev.frame)
        e_exact = _dist3(ev.xyz, xyz_here) if xyz_here is not None else None
        candidates = []
        for df in (-1, 0, 1):
            xyz = by_frame.get(ev.frame + df)
            if xyz is not None:
                candidates.append(_dist3(ev.xyz, xyz))
        e_pm1 = min(candidates) if candidates else None
        exact.append(e_exact)
        best_pm1.append(e_pm1)
        per_event.append({
            "frame": ev.frame, "kind": ev.kind,
            "err_m": e_exact, "err_min_pm1_m": e_pm1,
        })
    return {
        "events": per_event,
        "exact": error_stats(exact) if exact else None,
        "min_pm1": error_stats(best_pm1) if best_pm1 else None,
    }


# ---------------------------------------------------------------------------
# 3. ground float/sink
# ---------------------------------------------------------------------------

def ground_float_sink(track: Track, truth: TruthTrack, *,
                       radius: float = BALL_RADIUS_M,
                       sink_z: float = SINK_Z_M,
                       float_z: float = FLOAT_Z_M) -> dict[str, Any]:
    """Mean/p95 ``|z - radius|`` on truth-ground frames, plus counts of
    frames sinking below ``sink_z`` or floating above ``float_z``."""
    by_frame = _track_xyz_by_frame(track)
    devs: list[float] = []
    n_sink = 0
    n_float = 0
    n_ground = 0
    for tf in truth.frames:
        if tf.state != "ground":
            continue
        n_ground += 1
        xyz = by_frame.get(tf.frame)
        if xyz is None:
            continue
        z = float(xyz[2])
        devs.append(abs(z - radius))
        if z < sink_z:
            n_sink += 1
        if z > float_z:
            n_float += 1
    return {
        "n_ground_frames": n_ground,
        "n_scored": len(devs),
        "mean_abs_dev_m": (sum(devs) / len(devs)) if devs else None,
        "p95_abs_dev_m": _percentile(devs, 95),
        "n_sink": n_sink,
        "n_float": n_float,
    }


# ---------------------------------------------------------------------------
# 4. broadcast reprojection error
# ---------------------------------------------------------------------------

def broadcast_px_error(ctx, track: Track, truth: TruthTrack) -> dict[str, Any]:
    """Pixel distance between the truth and method xyz projected through
    the clip's real per-frame broadcast camera (``ctx.project``). Frames
    without a solved camera, or without a method estimate, are skipped
    (not counted as fails — this is a faithfulness metric, not a coverage
    one)."""
    by_frame = _track_xyz_by_frame(track)
    errs: list[float] = []
    for tf in truth.frames:
        if tf.frame not in ctx.per_frame_K:
            continue
        xyz = by_frame.get(tf.frame)
        if xyz is None:
            continue
        u_truth = ctx.project(tf.frame, tf.xyz)
        u_method = ctx.project(tf.frame, xyz)
        errs.append(float(np.hypot(u_truth[0] - u_method[0],
                                    u_truth[1] - u_method[1])))
    return {"n": len(errs), "p50": _percentile(errs, 50),
            "p95": _percentile(errs, 95)}


# ---------------------------------------------------------------------------
# 5. side-camera (depth-sensitive) screen error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SideCamera:
    """A fixed pinhole camera used only to score depth-sensitive errors
    that a touchline-side broadcast camera can't see. Not the pipeline's
    real camera model."""

    eye: Vec3
    target: Vec3
    hfov_deg: float
    image_size: tuple[int, int]

    def project(self, xyz: Sequence[float]) -> Optional[tuple[float, float]]:
        eye = np.asarray(self.eye, dtype=float)
        target = np.asarray(self.target, dtype=float)
        forward = target - eye
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            return None
        forward = forward / norm
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-9:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        p = np.asarray(xyz, dtype=float) - eye
        cx = float(np.dot(p, right))
        cy = float(np.dot(p, true_up))
        cz = float(np.dot(p, forward))
        if cz <= 1e-6:
            return None  # behind the camera

        w, h = self.image_size
        f = w / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))
        u = f * cx / cz + w / 2.0
        v = -f * cy / cz + h / 2.0  # image y grows downward
        return (u, v)

    def to_json(self) -> dict:
        return {
            "eye": list(self.eye),
            "target": list(self.target),
            "hfov_deg": self.hfov_deg,
            "image_size": list(self.image_size),
        }


def build_side_camera(points: Sequence[Sequence[float]], *,
                       image_size: tuple[int, int] = (1920, 1080),
                       hfov_deg: float = 50.0,
                       distance_m: float = 30.0,
                       height_m: float = 6.0) -> SideCamera:
    """A fixed virtual camera placed ``distance_m`` from ``points``'
    centroid, perpendicular to their principal horizontal direction (PCA
    of the xy scatter), ``height_m`` high, looking at the centroid."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("build_side_camera needs at least one xyz point")
    centroid = pts.mean(axis=0)
    xy = pts[:, :2] - centroid[:2]
    if pts.shape[0] >= 2 and np.linalg.norm(xy) > 1e-9:
        cov = np.cov(xy.T)
        cov = np.atleast_2d(cov)
        vals, vecs = np.linalg.eigh(cov)
        principal = vecs[:, int(np.argmax(vals))]
    else:
        principal = np.array([1.0, 0.0])
    pnorm = np.linalg.norm(principal)
    principal = principal / pnorm if pnorm > 1e-12 else np.array([1.0, 0.0])
    perp = np.array([-principal[1], principal[0]])
    eye_xy = centroid[:2] + distance_m * perp
    eye = (float(eye_xy[0]), float(eye_xy[1]), float(height_m))
    target = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    return SideCamera(eye=eye, target=target, hfov_deg=hfov_deg,
                       image_size=image_size)


def side_px_error(camera: SideCamera, track: Track,
                   truth: TruthTrack) -> dict[str, Any]:
    by_frame = _track_xyz_by_frame(track)
    errs: list[float] = []
    for tf in truth.frames:
        xyz = by_frame.get(tf.frame)
        if xyz is None:
            continue
        u_truth = camera.project(tf.xyz)
        u_method = camera.project(xyz)
        if u_truth is None or u_method is None:
            continue
        errs.append(math.hypot(u_truth[0] - u_method[0],
                                u_truth[1] - u_method[1]))
    return {"n": len(errs), "p50": _percentile(errs, 50),
            "p95": _percentile(errs, 95)}


# ---------------------------------------------------------------------------
# 6. naturalness (reuses src.utils.ball_eval.naturalness_violations)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _NatFrame:
    """Adapts a ``Track``/``TruthTrack`` frame to what
    ``naturalness_violations`` needs: ``.frame``, ``.world_xyz``,
    ``.state`` (only ``"flight"`` vs. anything-else is meaningful there)."""

    frame: int
    world_xyz: Optional[Vec3]
    state: str


def _nat_frames_from_track(track: Track, *,
                            flight_z: float = FLIGHT_Z_THRESH_M) -> list[_NatFrame]:
    out = []
    for tf in track.frames:
        if tf.xyz is None:
            out.append(_NatFrame(tf.frame, None, "ground"))
        else:
            state = "flight" if tf.xyz[2] > flight_z else "ground"
            out.append(_NatFrame(tf.frame, tf.xyz, state))
    return out


def _nat_frames_from_truth(truth: TruthTrack) -> list[_NatFrame]:
    return [
        _NatFrame(tf.frame, tf.xyz, "flight" if tf.state == "air" else "ground")
        for tf in truth.frames
    ]


def _count_by_kind(violations) -> dict[str, int]:
    by_kind: dict[str, int] = {}
    for v in violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    return by_kind


def naturalness_summary(track: Track, truth: TruthTrack, *, fps: float,
                         cfg: NaturalnessCfg = NaturalnessCfg()) -> dict[str, Any]:
    """Naturalness violations on the method track, PLUS the truth track
    itself scored as a control (``naturalness_violations`` assumes no
    drag, so correctly-simulated drag motion may itself be flagged —
    ``violations_minus_truth`` is the number attributable to the method
    beyond what the validator flags on ground truth)."""
    event_frames = [e.frame for e in truth.events]
    method_v = _naturalness_violations(
        _nat_frames_from_track(track), event_frames, fps, cfg=cfg)
    truth_v = _naturalness_violations(
        _nat_frames_from_truth(truth), event_frames, fps, cfg=cfg)
    return {
        "n_violations": len(method_v),
        "n_violations_truth": len(truth_v),
        "violations_minus_truth": len(method_v) - len(truth_v),
        "by_kind": _count_by_kind(method_v),
        "by_kind_truth": _count_by_kind(truth_v),
    }


def naturalness_summary_real(track: Track, *, fps: float,
                              event_frames: Sequence[int] = (),
                              cfg: NaturalnessCfg = NaturalnessCfg()) -> dict[str, Any]:
    """Real-footage variant: no dense truth track to score as a control,
    so only the method's own violation count/kinds are reported.
    ``event_frames`` should be real touch/bounce anchor frames when
    available, so legitimate contacts aren't flagged as breaks."""
    v = _naturalness_violations(
        _nat_frames_from_track(track), event_frames, fps, cfg=cfg)
    return {"n_violations": len(v), "by_kind": _count_by_kind(v)}


# ---------------------------------------------------------------------------
# 7. jitter (p95 of |third difference of position|, m/frame^3)
# ---------------------------------------------------------------------------

def _third_diff_p95(frame_xyz: Sequence[tuple[int, Vec3]]) -> Optional[float]:
    """p95 of the third-difference magnitude over CONTIGUOUS (unit-gap)
    4-frame windows only — a detection gap isn't jitter, so it's skipped
    rather than treated as a huge fake jerk."""
    frames_sorted = sorted(frame_xyz, key=lambda fx: fx[0])
    mags: list[float] = []
    for i in range(len(frames_sorted) - 3):
        f0, f1, f2, f3 = frames_sorted[i:i + 4]
        if f3[0] - f0[0] != 3:
            continue
        p0, p1, p2, p3 = f0[1], f1[1], f2[1], f3[1]
        d3 = [p3[k] - 3 * p2[k] + 3 * p1[k] - p0[k] for k in range(3)]
        mags.append(math.sqrt(sum(v * v for v in d3)))
    return _percentile(mags, 95)


def jitter_p95(track: Track) -> Optional[float]:
    pts = [(tf.frame, tf.xyz) for tf in track.frames if tf.xyz is not None]
    return _third_diff_p95(pts)


def jitter_p95_truth(truth: TruthTrack) -> Optional[float]:
    pts = [(tf.frame, tf.xyz) for tf in truth.frames]
    return _third_diff_p95(pts)


# ---------------------------------------------------------------------------
# assembled per-method metrics blocks (flat for the viewer + nested detail)
# ---------------------------------------------------------------------------

def compute_scenario_metrics(ctx, track: Track, truth: TruthTrack, *,
                              side_camera: SideCamera) -> tuple[dict, dict]:
    """Returns ``(flat, detail)`` for one method against one scenario's
    truth. ``flat`` is safe to drop straight into
    ``scenarios[name]["metrics"][method]`` (scalars only, matching the
    viewer's ``ROW_ORDER`` keys plus a few extra scalar columns it will
    render generically); ``detail`` goes in the sibling
    ``metrics_detail[method]`` key."""
    overall = error_stats(per_frame_error(track, truth))
    by_state = split_by_state(track, truth)
    cgap = contact_gap(track, truth)
    gfs = ground_float_sink(track, truth)
    bpx = broadcast_px_error(ctx, track, truth)
    spx = side_px_error(side_camera, track, truth)
    nat = naturalness_summary(track, truth, fps=ctx.fps)
    jit = jitter_p95(track)
    jit_truth = jitter_p95_truth(truth)

    exact_stats = cgap["exact"]
    pm1_stats = cgap["min_pm1"]

    flat = {
        "p50": overall["p50"],
        "p95": overall["p95"],
        "max": overall["max"],
        "pct_le_20cm": overall["pct_le_20cm"],
        "contact_gap": exact_stats["p50"] if exact_stats else None,
        "ground_float_sink": gfs["mean_abs_dev_m"],
        "broadcast_px_error": bpx["p50"],
        "side_px_error": spx["p50"],
        "naturalness_violations": nat["n_violations"],
        "coverage": overall["coverage"],
        "n_frames": overall["n"],
        "n_valid": overall["n_valid"],
        "p50_ground": by_state["ground"]["p50"],
        "p50_air": by_state["air"]["p50"],
        "p50_contact": by_state["contact"]["p50"],
        "pct_le_20cm_ground": by_state["ground"]["pct_le_20cm"],
        "pct_le_20cm_air": by_state["air"]["pct_le_20cm"],
        "pct_le_20cm_contact": by_state["contact"]["pct_le_20cm"],
        "contact_gap_min_pm1": pm1_stats["p50"] if pm1_stats else None,
        "ground_float_sink_p95": gfs["p95_abs_dev_m"],
        "ground_n_sink": gfs["n_sink"],
        "ground_n_float": gfs["n_float"],
        "broadcast_px_error_p95": bpx["p95"],
        "side_px_error_p95": spx["p95"],
        "naturalness_violations_truth": nat["n_violations_truth"],
        "naturalness_violations_minus_truth": nat["violations_minus_truth"],
        "jitter_p95": jit,
        "jitter_p95_truth": jit_truth,
    }
    detail = {
        "overall": overall,
        "by_state": by_state,
        "contact_gap": cgap,
        "ground_float_sink": gfs,
        "broadcast_px_error": bpx,
        "side_px_error": spx,
        "naturalness": nat,
        "jitter": {"method": jit, "truth": jit_truth},
    }
    return flat, detail


__all__ = [
    "BALL_RADIUS_M", "SINK_Z_M", "FLOAT_Z_M", "LE_THRESHOLD_M",
    "FLIGHT_Z_THRESH_M",
    "per_frame_error", "error_stats", "split_by_state", "contact_gap",
    "ground_float_sink", "broadcast_px_error", "SideCamera",
    "build_side_camera", "side_px_error", "naturalness_summary",
    "naturalness_summary_real", "jitter_p95", "jitter_p95_truth",
    "compute_scenario_metrics",
]

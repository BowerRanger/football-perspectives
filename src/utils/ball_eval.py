"""Metric primitives for grading the ball track against ground truth.

Part of the sub-20cm accuracy campaign (see
docs/superpowers/specs/2026-08-17-ball-sub20cm-accuracy-design.md §4).
Pure numpy — no torch, no video — so everything here runs in the light venv
and inside unit tests with synthetic cameras.

Ground-truth model:
- A clicked/known pixel defines a camera ray the true ball centre must lie on
  (lateral error is measurable for every anchored or detected frame).
- Ground-level states pin full 3-D via ray ∩ the z = ball_radius plane.
- ``player_touch`` pins depth via the contacting joint projected onto the
  clicked ray (lateral from the click, depth from the body — the same
  identifiability the resolver itself relies on).
- Airborne states are ray-only (their depth is what the physics must supply).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.camera_projection import project_world_to_image, undistort_pixel


def pixel_ray(uv, K, R, t, distortion=(0.0, 0.0)):
    """Camera centre and unit world-space ray direction through pixel ``uv``."""
    uv = np.asarray(uv, dtype=float)
    if tuple(distortion) != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    R = np.asarray(R, dtype=float)
    C = -R.T @ np.asarray(t, dtype=float)
    d = R.T @ (np.linalg.inv(np.asarray(K, dtype=float))
               @ np.array([uv[0], uv[1], 1.0]))
    return C, d / np.linalg.norm(d)


def point_ray_distance(P, C, d_hat):
    """(perpendicular distance, along-ray depth) of point ``P`` from a ray."""
    v = np.asarray(P, dtype=float) - C
    along = float(np.dot(v, d_hat))
    return float(np.linalg.norm(v - along * d_hat)), along


def ray_plane_z(C, d_hat, z):
    """Intersect the ray with the horizontal plane ``Z=z`` (forward only)."""
    dz = float(d_hat[2])
    if abs(dz) < 1e-9:
        return None
    s = (float(z) - float(C[2])) / dz
    if s <= 0:
        return None
    return np.asarray(C, dtype=float) + s * np.asarray(d_hat, dtype=float)


# States whose clicked pixel pins full 3-D via the ground plane. ``bounce``
# is at ground level at the bounce instant even though it brackets flight.
_GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}


def anchor_gt_world(anchor: BallAnchor, K, R, t, distortion, *,
                    ball_radius: float, joint_world=None):
    """Best-available ground-truth world position for an anchor.

    Returns ``(xyz | None, kind)`` with kind one of ``"ground_exact"``,
    ``"joint_depth"``, ``"ray_only"``, ``"none"``.
    """
    if anchor.image_xy is None:
        return None, "none"
    C, d = pixel_ray(anchor.image_xy, K, R, t, distortion)
    if anchor.state in _GROUND_EXACT_STATES:
        X = ray_plane_z(C, d, ball_radius)
        return (X, "ground_exact") if X is not None else (None, "ray_only")
    if anchor.state == "player_touch" and joint_world is not None:
        _, along = point_ray_distance(np.asarray(joint_world, float), C, d)
        if along > 0:
            return C + along * d, "joint_depth"
    return None, "ray_only"


@dataclass(frozen=True)
class AnchorEvalRow:
    """Track error graded at one anchor frame."""

    frame: int
    state: str
    kind: str
    held_out: bool
    lateral_m: float | None
    err_3d_m: float | None
    reproj_px: float | None
    depth_m: float | None


@dataclass(frozen=True)
class FixEvalRow:
    """Track error at one cross-replay triangulated 3-D fix."""

    frame: int
    err_3d_m: float | None
    ray_miss_m: float


@dataclass(frozen=True)
class DenseEvalRow:
    """Perpendicular distance from the track to a detection ray."""

    frame: int
    lateral_m: float
    confidence: float
    source: str


def eval_rows_at_anchors(world_by_frame, anchors, cams, *, ball_radius,
                         distortion, joint_world_fn=None,
                         held_out_frames=frozenset()):
    """Grade the track at every anchor frame with a pixel and a camera.

    ``cams`` maps frame → ``(K, R, t)``; ``world_by_frame`` maps frame →
    emitted ``world_xyz``. Anchors whose frame has no camera are skipped;
    anchors whose frame has no emitted world produce a row of ``None``
    errors (visible as ``n_missing`` in the summary).
    """
    rows = []
    for anc in anchors:
        if anc.image_xy is None or anc.frame not in cams:
            continue
        K, R, t = cams[anc.frame]
        joint = None
        if (joint_world_fn is not None and anc.state == "player_touch"
                and anc.player_id and anc.bone):
            joint = joint_world_fn(anc.frame, anc.player_id, anc.bone)
        gt, kind = anchor_gt_world(anc, K, R, t, distortion,
                                   ball_radius=ball_radius, joint_world=joint)
        held = anc.frame in held_out_frames
        w = world_by_frame.get(anc.frame)
        if w is None:
            rows.append(AnchorEvalRow(anc.frame, anc.state, kind, held,
                                      None, None, None, None))
            continue
        P = np.asarray(w, dtype=float)
        C, d = pixel_ray(anc.image_xy, K, R, t, distortion)
        lateral, depth = point_ray_distance(P, C, d)
        uvp = project_world_to_image(K, R, t, distortion, P.reshape(1, 3))[0]
        reproj = float(np.linalg.norm(uvp - np.asarray(anc.image_xy, float)))
        err3d = float(np.linalg.norm(P - gt)) if gt is not None else None
        rows.append(AnchorEvalRow(anc.frame, anc.state, kind, held,
                                  lateral, err3d, reproj, depth))
    return tuple(rows)


def eval_rows_at_fixes(world_by_frame, fixes):
    """Grade the track at triangulated fixes: ``(frame, xyz, ray_miss_m)``."""
    rows = []
    for frame, xyz, ray_miss in fixes:
        w = world_by_frame.get(int(frame))
        err = (float(np.linalg.norm(np.asarray(w, float)
                                    - np.asarray(xyz, float)))
               if w is not None else None)
        rows.append(FixEvalRow(int(frame), err, float(ray_miss)))
    return tuple(rows)


def dense_lateral_rows(world_by_frame, observations, cams, *, distortion,
                       min_confidence):
    """Track→detection-ray distance for confident observations.

    ``observations``: ``(frame, (u, v), confidence, source)`` tuples.
    """
    rows = []
    for frame, uv, conf, source in observations:
        if conf < min_confidence or frame not in cams:
            continue
        w = world_by_frame.get(int(frame))
        if w is None:
            continue
        K, R, t = cams[frame]
        C, d = pixel_ray(uv, K, R, t, distortion)
        lateral, _ = point_ray_distance(np.asarray(w, float), C, d)
        rows.append(DenseEvalRow(int(frame), lateral, float(conf),
                                 str(source)))
    return tuple(rows)


@dataclass(frozen=True)
class NaturalnessCfg:
    """Thresholds for the natural-motion validator (spec A4)."""

    max_heading_change_deg: float = 12.0
    min_speed_m_s: float = 2.0
    event_window_frames: int = 2
    flight_g_tol: float = 0.25
    roll_speedup_tol: float = 0.15
    min_roll_speed_m_s: float = 1.0


@dataclass(frozen=True)
class Violation:
    """One natural-motion violation on the emitted dense track."""

    frame: int
    kind: str
    value: float
    limit: float


def naturalness_violations(frames, event_frames, fps, *,
                           cfg: NaturalnessCfg = NaturalnessCfg()):
    """Flag direction/physics breaks that no event explains.

    ``frames`` is a ``BallTrack.frames``-like sequence (``.frame``,
    ``.world_xyz``, ``.state``); ``event_frames`` are frames where a
    touch/bounce/impact/waypoint legitimately bends the path.
    """
    ev = {int(e) for e in event_frames}

    def near_event(f: int) -> bool:
        return any(abs(f - e) <= cfg.event_window_frames for e in ev)

    by_frame = {f.frame: f for f in frames}
    idx = sorted(by_frame)
    out: list[Violation] = []
    for f in idx:
        a, b, c = by_frame.get(f - 1), by_frame.get(f), by_frame.get(f + 1)
        if not (a and b and c):
            continue
        if a.world_xyz is None or b.world_xyz is None or c.world_xyz is None:
            continue
        pa, pb, pc = (np.asarray(x.world_xyz, float) for x in (a, b, c))
        v_in, v_out = (pb - pa) * fps, (pc - pb) * fps
        sp_in = float(np.linalg.norm(v_in[:2]))
        sp_out = float(np.linalg.norm(v_out[:2]))
        if min(sp_in, sp_out) > cfg.min_speed_m_s and not near_event(f):
            dh = float(np.degrees(abs(np.arctan2(v_out[1], v_out[0])
                                      - np.arctan2(v_in[1], v_in[0]))))
            dh = min(dh, 360.0 - dh)
            if dh > cfg.max_heading_change_deg:
                out.append(Violation(f, "heading_break", dh,
                                     cfg.max_heading_change_deg))
        if (b.state != "flight" and not near_event(f)
                and min(sp_in, sp_out) > cfg.min_roll_speed_m_s
                and sp_out > sp_in * (1.0 + cfg.roll_speedup_tol)):
            out.append(Violation(f, "roll_speedup", sp_out / sp_in,
                                 1.0 + cfg.roll_speedup_tol))
    # Flight runs: the median vertical acceleration must look like gravity.
    run: list[int] = []
    for f in [*idx, None]:
        fr = by_frame.get(f) if f is not None else None
        if fr is not None and fr.state == "flight" and fr.world_xyz is not None:
            run.append(f)
            continue
        if len(run) >= 4:
            zs = np.array([by_frame[r].world_xyz[2] for r in run])
            az_med = float(np.median(np.diff(zs, 2) * fps * fps))
            lo = -9.81 * (1 + cfg.flight_g_tol)
            hi = -9.81 * (1 - cfg.flight_g_tol)
            interior = [r for r in run[1:-1] if not near_event(r)]
            if interior and not (lo <= az_med <= hi):
                out.append(Violation(run[len(run) // 2], "flight_gravity",
                                     az_med, -9.81))
        run = []
    return tuple(out)


def split_anchors(anchors, *, fold, n_folds=2):
    """Deterministic stratified hold-out split → ``(kept, held_out)``.

    Within each state class (sorted by frame), member ``i`` is held out iff
    ``i % n_folds == fold``, so running every fold holds each anchor out
    exactly once while both halves keep every state class populated
    (single-member classes are held out only in their own fold).
    """
    by_state: dict[str, list[BallAnchor]] = {}
    for a in sorted(anchors, key=lambda a: a.frame):
        by_state.setdefault(a.state, []).append(a)
    kept: list[BallAnchor] = []
    held: list[BallAnchor] = []
    for state in sorted(by_state):
        for i, a in enumerate(by_state[state]):
            (held if i % n_folds == fold else kept).append(a)
    kept.sort(key=lambda a: a.frame)
    held.sort(key=lambda a: a.frame)
    return tuple(kept), tuple(held)


def _stats(errs, threshold, n_missing=0):
    vals = np.array([e for e in errs if e is not None], dtype=float)
    if len(vals) == 0:
        return {"n": 0, "p50": None, "p95": None, "max": None,
                "n_over": 0, "n_missing": int(n_missing)}
    return {
        "n": int(len(vals)),
        "p50": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
        "n_over": int((vals > threshold).sum()),
        "n_missing": int(n_missing),
    }


def summarize(anchor_rows, fix_rows, dense_rows, violations, *,
              threshold_m=0.20):
    """JSON-safe per-clip summary of all metric sections.

    Anchor rows grade on ``err_3d_m`` where GT exists, else ``lateral_m``
    (a lower bound on 3-D error, so ``n_over`` never overstates quality).
    """

    def anchor_err(r):
        return r.err_3d_m if r.err_3d_m is not None else r.lateral_m

    def anchor_stats(rows):
        errs = [anchor_err(r) for r in rows]
        st = _stats([e for e in errs if e is not None], threshold_m,
                    n_missing=sum(1 for e in errs if e is None))
        st["n_3d"] = sum(1 for r in rows if r.err_3d_m is not None)
        return st

    held = [r for r in anchor_rows if r.held_out]
    kept = [r for r in anchor_rows if not r.held_out]
    by_kind: dict[str, int] = {}
    for v in violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    return {
        "anchors_held_out": anchor_stats(held),
        "anchors_kept": anchor_stats(kept),
        "fixes": _stats([r.err_3d_m for r in fix_rows], threshold_m,
                        n_missing=sum(1 for r in fix_rows
                                      if r.err_3d_m is None)),
        "dense": _stats([r.lateral_m for r in dense_rows], threshold_m),
        "naturalness": {"n_violations": len(violations), "by_kind": by_kind},
        "threshold_m": threshold_m,
    }


__all__ = [
    "pixel_ray",
    "point_ray_distance",
    "ray_plane_z",
    "anchor_gt_world",
    "AnchorEvalRow",
    "FixEvalRow",
    "DenseEvalRow",
    "eval_rows_at_anchors",
    "eval_rows_at_fixes",
    "dense_lateral_rows",
    "NaturalnessCfg",
    "Violation",
    "naturalness_violations",
    "split_anchors",
    "summarize",
]

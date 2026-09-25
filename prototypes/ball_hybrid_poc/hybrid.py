"""The hybrid ball extractor (Task B / SPIKE core method).

``run_hybrid(ctx, observations, anchors, fixes=(), cfg=None) -> Track``

Pipeline (see CONTRACT.md and the task brief):

  a. Resolve hard 3-D knots (manual anchors via the same semantics as
     ``src.utils.ball_eval.anchor_gt_world``, plus cross-replay fixes)
     and hard RAY constraints (states ``anchor_gt_world`` cannot resolve
     to a 3-D point — airborne_*, catch, goal_impact — lateral-exact,
     depth free); sort by frame.
  b. Robust evidence: real detector observations are graded per-span
     against that span's own physics fit (2-pass: fit, gate outliers by
     reprojection residual, refit); low-confidence/high-residual
     detections never move the path.
  c. Per span between consecutive hard knots: pick roll (both ends
     ground-level, no launch state) or flight (gravity + drag, optional
     Cd fit bounded to ``cd_bounds``, both endpoints always hit exactly
     via boundary-value shooting); split-and-retry once at the
     worst-residual evidence frame when a single arc can't explain the
     span (treated as an internal bounce knot, recursed).
  d. Physics track P_phys(frame) for every frame in [first knot/evidence,
     last knot/evidence]; outside any knot bracket the track holds the
     nearest knot (mode "simulated" — a documented simplification, see
     the module-level NOTE below).
  e. Hybrid blend: delta = faithful_point - P_phys at confident inlier
     evidence (and exactly at hard ray anchors), smoothed by
     ``blend.blend_deltas`` and added back; final = P_phys + delta_s,
     with knot/ray-anchor frames re-snapped exactly afterwards.

NOTE (known simplification, timeboxed): clip head/tail spans (before the
first knot or after the last) hold the nearest knot rather than fitting
a free-ended model against interior evidence. A follow-up could run an
unconstrained LM fit (free v0, fixed one endpoint) there instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.ball_eval import anchor_gt_world, point_ray_distance

from .blend import blend_deltas
from .hybrid_physics import (
    BALL_RADIUS_M,
    CD_BOUNDS,
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    fit_roll_segment,
    shoot_arc,
    simulate,
)
from .types import Observation, Track, TrackFrame

GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}
_LAUNCH_STATES = frozenset({
    "kick", "header", "volley", "chest", "goal_impact", "player_touch",
})

DEFAULT_CFG: dict[str, Any] = {
    "cd": CD_DEFAULT,               # 0.0 = gravity only (drag ablation)
    "fit_cd": True,
    "cd_bounds": CD_BOUNDS,
    "magnus": False,                # bounded Magnus refinement (best-effort)
    "magnus_coeff": DEFAULT_MAGNUS_COEFF,
    "blend_halflife_frames": 5.0,
    "faithful_conf_min": 0.5,
    "inlier_px": 15.0,
    "roll_mu_max": 0.9,
    "max_splits_per_span": 1,
    # Deliberately modest: high enough to saturate confidence exactly at
    # a ray anchor's own frame (which gets force-snapped to "anchor" mode
    # regardless) without inflating "faithful" confidence for several
    # frames around it purely from the anchor's pull rather than actual
    # detector evidence (a wide window there previously mislabelled
    # anchor-dominated frames as "faithful" and made the CLI's faithful-
    # frame-vs-observation reprojection stat misleading — see report).
    "ray_anchor_weight": 1.2,
    "min_obs_for_cd_fit": 5,
    "split_residual_factor": 2.5,
}


# ---------------------------------------------------------------------------
# Knot resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Knot:
    frame: int
    xyz: np.ndarray
    state: str
    kind: str
    is_manual: bool = True


@dataclass(frozen=True)
class _RayAnchor:
    frame: int
    C: np.ndarray
    d_hat: np.ndarray
    uv: tuple[float, float]
    state: str


def _anchor_attrs(a: Any) -> tuple[int, tuple[float, float] | None, str, str | None, str | None]:
    if isinstance(a, Mapping):
        frame = int(a["frame"])
        raw_xy = a.get("image_xy")
        image_xy = (float(raw_xy[0]), float(raw_xy[1])) if raw_xy is not None else None
        state = str(a["state"])
        player_id = a.get("player_id")
        bone = a.get("bone")
    else:
        frame = int(a.frame)
        image_xy = a.image_xy
        state = a.state
        player_id = a.player_id
        bone = a.bone
    return frame, image_xy, state, player_id, bone


def resolve_knots(
    ctx: Any,
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
) -> tuple[list[_Knot], list[_RayAnchor]]:
    """Split ``anchors`` (+ ``fixes``) into hard 3-D knots and hard ray
    constraints, exactly mirroring ``ball_eval.anchor_gt_world``'s own
    classification (``ground_exact``/``joint_depth`` -> hard knot,
    ``ray_only``/``none`` -> ray constraint or dropped)."""
    by_frame: dict[int, _Knot] = {}
    ray_anchors: list[_RayAnchor] = []

    for a in anchors:
        frame, image_xy, state, player_id, bone = _anchor_attrs(a)
        if image_xy is None or frame not in ctx.per_frame_K:
            continue
        K = ctx.per_frame_K[frame]
        R = ctx.per_frame_R[frame]
        t = ctx.per_frame_t[frame]
        joint_world = None
        if state == "player_touch" and player_id and bone:
            pc = ctx.player_context()
            joint_world = pc.joint_world(frame, player_id, bone)
        view = SimpleNamespace(image_xy=image_xy, state=state)
        xyz, kind = anchor_gt_world(
            view, K, R, t, ctx.distortion,
            ball_radius=BALL_RADIUS_M, joint_world=joint_world,
        )
        if kind in ("ground_exact", "joint_depth") and xyz is not None:
            knot = _Knot(frame=frame, xyz=np.asarray(xyz, dtype=float),
                          state=state, kind=kind, is_manual=True)
            if frame not in by_frame:
                by_frame[frame] = knot
        else:
            C, d_hat = ctx.ray(frame, image_xy)
            ray_anchors.append(_RayAnchor(frame=frame, C=C, d_hat=d_hat,
                                           uv=image_xy, state=state))

    for fx in fixes:
        frame = int(fx.frame)
        by_frame[frame] = _Knot(frame=frame, xyz=np.asarray(fx.xyz, dtype=float),
                                 state="fix", kind="fix", is_manual=True)

    hard_knots = sorted(by_frame.values(), key=lambda k: k.frame)
    ray_anchors.sort(key=lambda r: r.frame)
    return hard_knots, ray_anchors


# ---------------------------------------------------------------------------
# Per-span evidence + fitting helpers
# ---------------------------------------------------------------------------

def _reproj_px(ctx: Any, frame: int, xyz: np.ndarray, uv: tuple[float, float]) -> float:
    proj = ctx.project(frame, xyz)
    return float(np.hypot(float(proj[0]) - uv[0], float(proj[1]) - uv[1]))


def _span_evidence(
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    a_frame: int,
    b_frame: int,
    ray_weight: float,
) -> list[tuple[int, tuple[float, float], float]]:
    evid = [(o.frame, o.uv, float(o.conf))
            for o in observations if a_frame < o.frame < b_frame]
    evid += [(r.frame, r.uv, ray_weight)
              for r in ray_anchors if a_frame < r.frame < b_frame]
    return evid


def _choose_model(a_knot: _Knot, b_knot: _Knot,
                   span_rays: Sequence[_RayAnchor]) -> str:
    ground_a = (a_knot.state in GROUND_EXACT_STATES
                or a_knot.xyz[2] <= BALL_RADIUS_M + 0.05)
    ground_b = (b_knot.state in GROUND_EXACT_STATES
                or b_knot.xyz[2] <= BALL_RADIUS_M + 0.05)
    launchy = a_knot.state in _LAUNCH_STATES or b_knot.state in _LAUNCH_STATES
    if any(r.state.startswith("airborne") for r in span_rays):
        return "flight"
    if ground_a and ground_b and not launchy:
        return "roll"
    return "flight"


def _solve_span(
    ctx: Any,
    a_knot: _Knot,
    b_knot: _Knot,
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    cfg: Mapping[str, Any],
    splits_used: int,
    internal_knot_frames: list[int],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    duration_frames = b_knot.frame - a_knot.frame
    if duration_frames <= 0:
        return {}, []
    duration_s = duration_frames / ctx.fps

    span_rays = [r for r in ray_anchors if a_knot.frame < r.frame < b_knot.frame]
    model = _choose_model(a_knot, b_knot, span_rays)

    if model == "roll":
        z_level = 0.5 * (float(a_knot.xyz[2]) + float(b_knot.xyz[2]))
        ground_obs: list[tuple[float, np.ndarray]] = []
        for o in observations:
            if not (a_knot.frame < o.frame < b_knot.frame):
                continue
            C, d = ctx.ray(o.frame, o.uv)
            dz = float(d[2])
            if abs(dz) < 1e-9:
                continue
            s = (z_level - float(C[2])) / dz
            if s <= 0:
                continue
            P = C + s * d
            ground_obs.append(((o.frame - a_knot.frame) / ctx.fps, P[:2]))
        roll = fit_roll_segment(a_knot.xyz[:2], b_knot.xyz[:2], duration_s,
                                 ground_obs, mu_max=cfg["roll_mu_max"])
        frames = list(range(a_knot.frame, b_knot.frame + 1))
        times = [(f - a_knot.frame) / ctx.fps for f in frames]
        positions = roll.eval(times, z_level)
        pts = {f: positions[i] for i, f in enumerate(frames)}
        info = {"span": (a_knot.frame, b_knot.frame), "model": "roll",
                "n_obs": len(ground_obs)}
        return pts, [info]

    # --- flight ---------------------------------------------------------
    evid = _span_evidence(observations, ray_anchors, a_knot.frame, b_knot.frame,
                           cfg["ray_anchor_weight"])
    cd = cfg["cd"]
    if cfg["fit_cd"] and len(evid) >= cfg["min_obs_for_cd_fit"]:
        # bind the frame origin for the cost closure without mutating shared state
        def _cost(cd_val: float, _evid=evid, _a=a_knot.xyz, _b=b_knot.xyz,
                   _T=duration_s, _a_frame=a_knot.frame) -> float:
            v0 = shoot_arc(_a, 0.0, _b, _T, cd=cd_val)
            total = 0.0
            for frame, uv, w in _evid:
                t_s = (frame - _a_frame) / ctx.fps
                p = simulate(_a, v0, [t_s], cd=cd_val)[0]
                total += w * _reproj_px(ctx, frame, p, uv) ** 2
            return total

        lo, hi = cfg["cd_bounds"]
        if lo < hi:
            res = minimize_scalar(_cost, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-3, "maxiter": 15})
            cd = float(res.x)

    v0 = shoot_arc(a_knot.xyz, 0.0, b_knot.xyz, duration_s, cd=cd,
                    magnus_coeff=cfg["magnus_coeff"])
    frames = list(range(a_knot.frame, b_knot.frame + 1))
    times = [(f - a_knot.frame) / ctx.fps for f in frames]
    positions = simulate(a_knot.xyz, v0, times, cd=cd, magnus_coeff=cfg["magnus_coeff"])
    pts = {f: positions[i] for i, f in enumerate(frames)}

    worst: tuple[int, tuple[float, float], float] | None = None
    for frame, uv, _w in evid:
        err = _reproj_px(ctx, frame, pts[frame], uv)
        if worst is None or err > worst[2]:
            worst = (frame, uv, err)

    info = {"span": (a_knot.frame, b_knot.frame), "model": "flight", "cd": cd,
            "n_obs": len(evid), "max_residual_px": worst[2] if worst else None}

    if (worst is not None
            and worst[2] > cfg["inlier_px"] * cfg["split_residual_factor"]
            and splits_used < cfg["max_splits_per_span"]):
        split_frame, split_uv, _err = worst
        C, d = ctx.ray(split_frame, split_uv)
        arc_pt = pts[split_frame]
        _, along = point_ray_distance(arc_pt, C, d)
        faithful = C + max(along, 0.0) * d
        z = max(float(faithful[2]), BALL_RADIUS_M)
        split_xyz = np.array([faithful[0], faithful[1], z])
        split_knot = _Knot(frame=split_frame, xyz=split_xyz, state="bounce",
                            kind="internal", is_manual=False)
        internal_knot_frames.append(split_frame)
        pts1, info1 = _solve_span(ctx, a_knot, split_knot, observations, ray_anchors,
                                   cfg, splits_used + 1, internal_knot_frames)
        pts2, info2 = _solve_span(ctx, split_knot, b_knot, observations, ray_anchors,
                                   cfg, splits_used + 1, internal_knot_frames)
        merged = {**pts1, **pts2}
        return merged, [*info1, *info2]

    return pts, [info]


def _build_physics_track(
    ctx: Any,
    hard_knots: Sequence[_Knot],
    ray_anchors: Sequence[_RayAnchor],
    observations: Sequence[Observation],
    cfg: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], list[dict], list[int]]:
    pts: dict[int, np.ndarray] = {}
    diagnostics: list[dict] = []
    internal_knot_frames: list[int] = []

    for a_knot, b_knot in zip(hard_knots, hard_knots[1:]):
        span_pts, span_info = _solve_span(ctx, a_knot, b_knot, observations,
                                           ray_anchors, cfg, 0, internal_knot_frames)
        pts.update(span_pts)
        diagnostics.extend(span_info)

    if len(hard_knots) == 1:
        pts[hard_knots[0].frame] = hard_knots[0].xyz

    if hard_knots:
        all_frames = sorted(set(list(pts)
                                 + [o.frame for o in observations]
                                 + [r.frame for r in ray_anchors]))
        first_frame = hard_knots[0].frame
        last_frame = hard_knots[-1].frame
        head_candidates = [f for f in all_frames if f < first_frame]
        tail_candidates = [f for f in all_frames if f > last_frame]
        if head_candidates:
            head_start = min(head_candidates)
            for f in range(head_start, first_frame):
                pts[f] = hard_knots[0].xyz
            diagnostics.append({"span": (head_start, first_frame), "model": "hold_head"})
        if tail_candidates:
            tail_end = max(tail_candidates)
            for f in range(last_frame + 1, tail_end + 1):
                pts[f] = hard_knots[-1].xyz
            diagnostics.append({"span": (last_frame, tail_end), "model": "hold_tail"})

    return pts, diagnostics, internal_knot_frames


def _delta_evidence(
    ctx: Any,
    pts: Mapping[int, np.ndarray],
    observations: Sequence[Observation],
    ray_anchors: Sequence[_RayAnchor],
    cfg: Mapping[str, Any],
) -> dict[int, tuple[tuple[float, float, float], float]]:
    evidence: dict[int, tuple[tuple[float, float, float], float]] = {}
    for o in observations:
        if o.conf < cfg["faithful_conf_min"]:
            continue
        p_phys = pts.get(o.frame)
        if p_phys is None:
            continue
        err_px = _reproj_px(ctx, o.frame, p_phys, o.uv)
        if err_px > cfg["inlier_px"]:
            continue
        C, d = ctx.ray(o.frame, o.uv)
        _, along = point_ray_distance(p_phys, C, d)
        faithful = C + max(along, 0.0) * d
        delta = faithful - p_phys
        evidence[o.frame] = (tuple(float(x) for x in delta), float(o.conf))
    for r in ray_anchors:
        p_phys = pts.get(r.frame)
        if p_phys is None:
            continue
        _, along = point_ray_distance(p_phys, r.C, r.d_hat)
        faithful = r.C + max(along, 0.0) * r.d_hat
        delta = faithful - p_phys
        evidence[r.frame] = (tuple(float(x) for x in delta), float(cfg["ray_anchor_weight"]))
    return evidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_hybrid_full(
    ctx: Any,
    observations: Sequence[Observation],
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[Track, dict]:
    full_cfg = dict(DEFAULT_CFG)
    if cfg:
        full_cfg.update(cfg)

    hard_knots, ray_anchors = resolve_knots(ctx, anchors, fixes)
    obs_sorted = sorted(observations, key=lambda o: o.frame)

    pts, span_diag, internal_knot_frames = _build_physics_track(
        ctx, hard_knots, ray_anchors, obs_sorted, full_cfg)

    if not pts:
        empty = Track(clip_id=ctx.clip_id, method="hybrid", frames=())
        return empty, {"spans": [], "n_knots": 0, "n_ray_anchors": 0}

    evidence = _delta_evidence(ctx, pts, obs_sorted, ray_anchors, full_cfg)
    event_frames = ([k.frame for k in hard_knots] + internal_knot_frames)
    frames_sorted = sorted(pts)
    blended = blend_deltas(frames_sorted, evidence,
                            halflife_frames=full_cfg["blend_halflife_frames"],
                            event_frames=event_frames)

    knot_by_frame = {k.frame: k for k in hard_knots}
    ray_by_frame = {r.frame: r for r in ray_anchors}

    out_frames = []
    mode_counts = {"anchor": 0, "faithful": 0, "simulated": 0}
    for f in frames_sorted:
        base = np.asarray(pts[f], dtype=float)
        delta, conf = blended.get(f, ((0.0, 0.0, 0.0), 0.0))
        final = base + np.asarray(delta, dtype=float)
        mode = "faithful" if conf >= 0.5 else "simulated"
        out_conf: float | None = conf

        if f in knot_by_frame:
            final = np.array(knot_by_frame[f].xyz, dtype=float)
            mode, out_conf = "anchor", 1.0
        elif f in ray_by_frame:
            r = ray_by_frame[f]
            _, along = point_ray_distance(final, r.C, r.d_hat)
            final = r.C + max(along, 0.0) * r.d_hat
            mode, out_conf = "anchor", 1.0

        final[2] = max(float(final[2]), BALL_RADIUS_M)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out_frames.append(TrackFrame(
            frame=f, xyz=(float(final[0]), float(final[1]), float(final[2])),
            mode=mode, conf=out_conf))

    track = Track(clip_id=ctx.clip_id, method="hybrid", frames=tuple(out_frames))
    diagnostics = {
        "spans": span_diag,
        "n_knots": len(hard_knots),
        "n_ray_anchors": len(ray_anchors),
        "n_internal_bounces": len(internal_knot_frames),
        "mode_counts": mode_counts,
    }
    return track, diagnostics


def run_hybrid(
    ctx: Any,
    observations: Sequence[Observation],
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    *,
    cfg: Mapping[str, Any] | None = None,
) -> Track:
    """The hybrid ball extractor. See module docstring for the pipeline."""
    track, _diag = _run_hybrid_full(ctx, observations, anchors, fixes, cfg=cfg)
    return track


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_clip_report(clip_id: str, ctx: Any, track: Track, diagnostics: dict,
                        observations: Sequence[Observation], runtime_s: float) -> None:
    # Score "faithful" frames only against observations that were actually
    # eligible to become evidence (conf >= faithful_conf_min) — a frame
    # can be labelled "faithful" from a nearby confident detection while
    # its own frame's observation is a low-confidence outlier that was
    # correctly rejected as evidence; scoring against that rejected point
    # would conflate "matches trusted evidence" with "matches raw noise".
    conf_min = DEFAULT_CFG["faithful_conf_min"]
    by_frame_obs = {}
    for o in observations:
        if o.conf < conf_min:
            continue
        by_frame_obs.setdefault(o.frame, o)

    faithful_errs = []
    for tf in track.frames:
        if tf.mode != "faithful" or tf.xyz is None:
            continue
        o = by_frame_obs.get(tf.frame)
        if o is None:
            continue
        faithful_errs.append(_reproj_px(ctx, tf.frame, np.asarray(tf.xyz), o.uv))

    n = len(track.frames)
    mode_counts = diagnostics.get("mode_counts", {})
    model_counts: dict[str, int] = {}
    for s in diagnostics.get("spans", []):
        model_counts[s["model"]] = model_counts.get(s["model"], 0) + 1

    print(f"\n=== {clip_id} ===")
    print(f"frames: {n}  knots: {diagnostics['n_knots']}  "
          f"ray_anchors: {diagnostics['n_ray_anchors']}  "
          f"internal_bounces: {diagnostics['n_internal_bounces']}")
    if faithful_errs:
        arr = np.array(faithful_errs)
        print(f"faithful-frame reprojection px: "
              f"median={np.median(arr):.2f}  p95={np.percentile(arr, 95):.2f}  "
              f"(n={len(arr)})")
    else:
        print("faithful-frame reprojection px: n/a (no faithful frames with evidence)")
    print("mode split: " + ", ".join(
        f"{k}={100.0 * v / n:.1f}%" for k, v in sorted(mode_counts.items())))
    print("segment models: " + ", ".join(
        f"{k}={v}" for k, v in sorted(model_counts.items())))
    print(f"runtime: {runtime_s:.1f}s")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import time
    from pathlib import Path

    from .ctx import CLIPS, load_clip
    from .types import save_json

    parser = argparse.ArgumentParser(description="Run the hybrid ball extractor on real clips.")
    parser.add_argument("--clips", default=",".join(sorted(CLIPS)),
                         help="comma-separated clip ids")
    parser.add_argument("--no-drag", action="store_true",
                         help="ablation: cd=0, fit_cd=False (gravity-only)")
    parser.add_argument("--output-root", default=None,
                         help="override M/output-ball-poc (defaults to ctx.M)")
    args = parser.parse_args(argv)

    from . import ctx as ctx_mod
    output_root = Path(args.output_root) if args.output_root else Path(ctx_mod.M) / "output-ball-poc"

    cfg = {"cd": 0.0, "fit_cd": False} if args.no_drag else None

    for clip_id in args.clips.split(","):
        clip_id = clip_id.strip()
        if not clip_id:
            continue
        t0 = time.perf_counter()
        clip_ctx = load_clip(clip_id)
        anchors = list(clip_ctx.anchors.anchors)
        fixes = list(clip_ctx.fixes)
        observations = list(clip_ctx.observations)
        track, diagnostics = _run_hybrid_full(clip_ctx, observations, anchors, fixes, cfg=cfg)
        runtime_s = time.perf_counter() - t0

        out_path = output_root / clip_id / "track_hybrid_real_full.json"
        save_json(out_path, track)
        _print_clip_report(clip_id, clip_ctx, track, diagnostics, observations, runtime_s)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

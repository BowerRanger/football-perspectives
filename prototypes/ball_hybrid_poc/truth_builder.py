"""Builds synthetic 3-D ground truth for the ball hybrid-extraction PoC.

Seeded ONLY from operator data (manual anchors, via ``ClipContext.anchors``)
and reconstructed players (``ClipContext.player_context()``) — NEVER from
ball-stage output. ``ClipContext`` (``ctx.py``) already enforces this by
never reading the ball stage's own dense-track, auto-anchor or
sparse-keyframe outputs; this module additionally must not name those
outputs itself (a test greps for them).

Algorithm (see CONTRACT.md and the module docstrings in ``truth_sim.py``):

1. Resolve every manual anchor to either a HARD KNOT (exact 3-D position —
   grounded/kick/bounce via click-ray ∩ ground plane, player_touch via the
   contacting joint projected onto the click ray, goal_impact via goal
   geometry) or a WAYPOINT (only a camera ray — airborne_* anchors, plus
   any anchor whose depth can't be pinned, e.g. ``catch``/``header``/
   ``volley``/``chest`` states, which the real anchor files never carry
   player/bone associations for even though the schema allows it).
2. Between each consecutive pair of hard knots, choose a segment kind
   (roll / carry / flight) and solve it so the simulated arc reaches the
   next knot's exact position at its exact frame, while passing as close
   as possible to any ray waypoints in between.
3. Concatenate the dense per-frame positions into one ``TruthTrack``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.utils.ball_eval import anchor_gt_world, pixel_ray, point_ray_distance, ray_plane_z
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

from . import truth_sim as ts
from .ctx import ClipContext, load_clip
from .types import TruthEvent, TruthFrame, TruthTrack, save_json

# States that get no player/bone association in the real anchor files even
# though the schema permits one (only ``player_touch`` requires it). These
# always demote to ray-only waypoints in this PoC.
_NO_DEPTH_STATES = frozenset({"catch", "header", "volley", "chest"})
_CONTACT_STATES = frozenset({"player_touch", "kick", "bounce", "goal_impact"})

_ROLL_MAX_SPEED_MPS = 14.0
_CARRY_MAX_SPEED_MPS = 6.0
_CARRY_MAX_FRAMES = 20
_FLIGHT_INFEASIBLE_ERR_M = 0.5
_FLIGHT_INFEASIBLE_SPEED_MPS = 40.0


@dataclass(frozen=True)
class _Resolved:
    frame: int
    xyz: np.ndarray | None   # None => waypoint only
    kind: str
    C: np.ndarray | None
    d: np.ndarray | None
    anchor: object


def _resolve_anchor(ctx: ClipContext, anchor) -> _Resolved:
    if anchor.image_xy is None:
        return _Resolved(anchor.frame, None, "none", None, None, anchor)
    frame = anchor.frame
    if frame not in ctx.per_frame_K:
        return _Resolved(frame, None, "no_camera", None, None, anchor)
    K = ctx.per_frame_K[frame]
    R = ctx.per_frame_R[frame]
    t = ctx.per_frame_t[frame]

    if anchor.state == "goal_impact":
        geom = GoalGeometry.from_pitch_config({})
        try:
            xyz = resolve_goal_impact_world(
                anchor.image_xy, anchor.goal_element, K=K, R=R, t=t,
                distortion=ctx.distortion, geometry=geom)
            return _Resolved(frame, np.asarray(xyz, float), "goal_geometry",
                             None, None, anchor)
        except ValueError:
            C, d = pixel_ray(anchor.image_xy, K, R, t, ctx.distortion)
            xyz = ray_plane_z(C, d, geom.crossbar_z)
            kind = "goal_fallback_plane" if xyz is not None else "ray_only"
            return _Resolved(frame, xyz, kind, C, d, anchor)

    joint = None
    if anchor.state == "player_touch":
        joint = ctx.player_context().joint_world(frame, anchor.player_id,
                                                 anchor.bone)
    xyz, kind = anchor_gt_world(anchor, K, R, t, ctx.distortion,
                                ball_radius=ts.BALL_RADIUS_M,
                                joint_world=joint)
    C, d = pixel_ray(anchor.image_xy, K, R, t, ctx.distortion)
    return _Resolved(frame, (np.asarray(xyz, float) if xyz is not None
                             else None), kind, C, d, anchor)


def _roll_like_segment(xa, xb, fa, fb, dt_s, fps, scenario, rng, seg_kind):
    if scenario == "base":
        p = ts.RollParams(friction_decel=0.6, skid_extra_decel=1.4)
    else:
        p = ts.RollParams(
            friction_decel=float(rng.uniform(0.4, 0.8)),
            skid_extra_decel=float(rng.uniform(1.0, 2.0)),
        )
    dist = float(np.linalg.norm(xb - xa))
    direction = (xb - xa) / dist if dist > 1e-9 else np.zeros(3)
    v0 = ts.solve_roll_initial_speed(dist, dt_s, p)
    positions: dict[int, np.ndarray] = {}
    for f in range(fa, fb + 1):
        trel = (f - fa) / fps
        d_along = ts.roll_distance_at(v0, trel, p)
        positions[f] = xa + d_along * direction
    positions[fa] = xa
    positions[fb] = xb
    info = {
        "type": seg_kind, "frame_a": fa, "frame_b": fb, "dt_s": dt_s,
        "dist_m": dist, "v0_m_s": v0,
        "friction_decel": p.friction_decel,
        "skid_extra_decel": p.skid_extra_decel,
    }
    return positions, info


def _flight_segment(ctx: ClipContext, xa, xb, fa, fb, dt_s, wp_between,
                    scenario, rng):
    fps = ctx.fps
    fit_spin = scenario != "base"
    if scenario == "base":
        cd = ts.DragParams(cd_const=0.25, crisis=False)
        e_n = 0.70
        spin_mag = float(rng.uniform(0.0, 5.0)) * 2 * np.pi  # <=5 rev/s
    else:
        cd = ts.DragParams(
            crisis=True,
            cd_low=float(rng.uniform(0.35, 0.50)),
            cd_high=float(rng.uniform(0.15, 0.25)),
            v_low=float(rng.uniform(8.0, 12.0)),
            v_high=float(rng.uniform(18.0, 22.0)),
        )
        e_n = float(rng.uniform(0.60, 0.75))
        spin_mag = float(rng.uniform(0.0, 10.0)) * 2 * np.pi  # <=10 rev/s
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm if axis_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    omega0 = axis * spin_mag
    decay_s = float(rng.uniform(3.0, 6.0)) if scenario != "base" else 4.0

    disp = xb - xa
    vz0_guess = (disp[2] + 0.5 * ts.G * dt_s ** 2) / dt_s
    vxy0_guess = disp[:2] / dt_s
    x0 = np.array([vxy0_guess[0], vxy0_guess[1], vz0_guess])

    def integrate(v0, omega):
        spin = ts.SpinParams(omega0=omega, decay_s=decay_s)
        return ts.simulate_flight(xa, v0, dt_s, cd, spin)

    def residuals(x):
        v0 = x[:3]
        omega = x[3:6] if fit_spin else omega0
        res = integrate(v0, omega)
        end_pos, _ = res.state_at(dt_s)
        r = list((end_pos - xb) * 25.0)
        for f, C, d, _wa in wp_between:
            trel = (f - fa) / fps
            p_i, _ = res.state_at(trel)
            lateral, _ = point_ray_distance(p_i, C, d)
            r.append(lateral * 4.0)
        return np.asarray(r, dtype=float)

    if fit_spin:
        x0f = np.concatenate([x0, omega0])
        bounds = ([-45, -45, -45, -80, -80, -80],
                 [45, 45, 45, 80, 80, 80])
    else:
        x0f = x0
        bounds = ([-45, -45, -45], [45, 45, 45])

    sol = least_squares(residuals, x0f, bounds=bounds, max_nfev=200)
    v0_fit = sol.x[:3]
    omega_fit = sol.x[3:6] if fit_spin else omega0
    res = integrate(v0_fit, omega_fit)
    end_pos, _end_vel = res.state_at(dt_s)
    end_err = float(np.linalg.norm(end_pos - xb))
    infeasible = (end_err > _FLIGHT_INFEASIBLE_ERR_M
                 or float(np.linalg.norm(v0_fit)) > _FLIGHT_INFEASIBLE_SPEED_MPS)

    wp_err_m = None
    if wp_between:
        errs = []
        for f, C, d, _wa in wp_between:
            trel = (f - fa) / fps
            p_i, _ = res.state_at(trel)
            lateral, _ = point_ray_distance(p_i, C, d)
            errs.append(lateral)
        wp_err_m = float(np.max(errs))

    if infeasible:
        positions, info = _roll_like_segment(xa, xb, fa, fb, dt_s, fps,
                                             scenario, rng, "flight_fallback_roll")
        info["notes"] = (
            f"flight shoot infeasible (end_err={end_err:.3f}m, "
            f"|v0|={float(np.linalg.norm(v0_fit)):.1f}m/s); fell back to a "
            "straight-line roll profile between the two hard knots")
        return positions, info

    positions = {}
    for f in range(fa, fb + 1):
        trel = (f - fa) / fps
        p_i, _ = res.state_at(trel)
        positions[f] = p_i
    positions[fa] = xa
    positions[fb] = xb
    info = {
        "type": "flight", "frame_a": fa, "frame_b": fb, "dt_s": dt_s,
        "cd": (cd.cd_const if not cd.crisis else
              {"crisis": True, "cd_low": cd.cd_low, "cd_high": cd.cd_high,
               "v_low": cd.v_low, "v_high": cd.v_high}),
        "bounce_e_n_annotation": e_n,
        "omega0_rad_s": omega_fit.tolist(),
        "spin_decay_s": decay_s,
        "v0_m_s": v0_fit.tolist(),
        "end_err_m": end_err,
        "waypoint_max_err_m": wp_err_m,
    }
    return positions, info


def _segment_kind(anchor_a, anchor_b, xa, xb, dt_s, has_waypoints_between):
    ground_states = {"grounded", "kick", "bounce"}
    dist = float(np.linalg.norm(xb - xa))
    speed = dist / dt_s if dt_s > 0 else float("inf")
    if (not has_waypoints_between and anchor_a.state in ground_states
            and anchor_b.state in ground_states
            and speed <= _ROLL_MAX_SPEED_MPS):
        return "roll"
    if (not has_waypoints_between and anchor_a.state == "player_touch"
            and anchor_b.state == "player_touch"
            and anchor_a.player_id == anchor_b.player_id
            and speed <= _CARRY_MAX_SPEED_MPS
            and (anchor_b.frame - anchor_a.frame) <= _CARRY_MAX_FRAMES):
        return "carry"
    return "flight"


def _event_for_hard_knot(resolved: _Resolved) -> TruthEvent | None:
    a = resolved.anchor
    xyz = tuple(float(v) for v in resolved.xyz)
    if a.state in ("player_touch", "kick"):
        return TruthEvent(frame=resolved.frame, kind="touch", xyz=xyz,
                          player_id=a.player_id, bone=a.bone)
    if a.state == "bounce":
        return TruthEvent(frame=resolved.frame, kind="bounce", xyz=xyz)
    if a.state == "goal_impact":
        kind = "net" if a.goal_element in ("back_net", "side_net") else "post"
        return TruthEvent(frame=resolved.frame, kind=kind, xyz=xyz)
    return None


def build_truth(ctx: ClipContext, scenario: str, seed: int = 0) -> TruthTrack:
    """Build synthetic 3-D truth for ``ctx``'s clip under ``scenario``
    (``"base" | "mismatch" | "sparse"``). ``sparse`` uses IDENTICAL physics
    to ``mismatch`` (only the synthetic detector differs — see
    ``synth_detector.py``); this is implemented by building with the
    ``mismatch`` parameter distributions under any ``seed``, and is exact
    by construction rather than by RNG coincidence.
    """
    if scenario == "sparse":
        # Identical physics to "mismatch" (CONTRACT.md) — reuse the
        # mismatch build outright (same seed => same RNG draw sequence
        # anyway) rather than re-running the shooting solver a second
        # time for every segment.
        import dataclasses as _dc
        mismatch = build_truth(ctx, "mismatch", seed=seed)
        physics = dict(mismatch.physics)
        physics["scenario_requested"] = "sparse"
        return _dc.replace(mismatch, scenario="sparse", physics=physics)
    phys_scenario = scenario
    if phys_scenario not in ("base", "mismatch"):
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng(seed)

    anchors = sorted(ctx.anchors.anchors, key=lambda a: a.frame)
    resolved = [_resolve_anchor(ctx, a) for a in anchors]
    hard = [r for r in resolved if r.xyz is not None]
    waypoints = [r for r in resolved if r.xyz is None and r.C is not None]
    if len(hard) < 2:
        raise ValueError(
            f"{ctx.clip_id}: only {len(hard)} hard knot(s) resolved; "
            "need >= 2 to build a truth track")

    notes: list[str] = []
    for r in waypoints:
        if r.anchor.state in _NO_DEPTH_STATES:
            notes.append(
                f"frame {r.frame}: state={r.anchor.state!r} has no "
                "player/bone association in the real anchor file; treated "
                "as a ray-only waypoint, not a hard knot")
        elif r.anchor.state == "player_touch":
            notes.append(
                f"frame {r.frame}: player_touch joint lookup returned "
                "None (missing pose data); demoted to a ray-only waypoint")
    for r in resolved:
        if r.kind == "no_camera":
            notes.append(
                f"frame {r.frame}: no solved camera at this frame "
                "(outside camera_track's frame range); anchor dropped "
                "entirely (neither hard knot nor waypoint)")

    dense: dict[int, np.ndarray] = {}
    segments: list[dict] = []
    n_fallback = 0
    for ra, rb in zip(hard, hard[1:]):
        fa, fb = ra.frame, rb.frame
        if fb <= fa:
            continue
        dt_s = (fb - fa) / ctx.fps
        wp_between = [(w.frame, w.C, w.d, w.anchor) for w in waypoints
                     if fa < w.frame < fb]
        kind = _segment_kind(ra.anchor, rb.anchor, ra.xyz, rb.xyz, dt_s,
                             bool(wp_between))
        if kind in ("roll", "carry"):
            positions, info = _roll_like_segment(
                ra.xyz, rb.xyz, fa, fb, dt_s, ctx.fps, phys_scenario, rng,
                kind)
        else:
            positions, info = _flight_segment(
                ctx, ra.xyz, rb.xyz, fa, fb, dt_s, wp_between, phys_scenario,
                rng)
            if info["type"] == "flight_fallback_roll":
                n_fallback += 1
        segments.append(info)
        for f, xyz in positions.items():
            dense[f] = xyz

    # Safety clamp: no dense position may sit meaningfully below the pitch.
    n_clamped = 0
    for f in list(dense):
        z = dense[f][2]
        if z < ts.BALL_RADIUS_M - 0.01:
            p = dense[f].copy()
            p[2] = ts.BALL_RADIUS_M - 0.01
            dense[f] = p
            n_clamped += 1
    if n_clamped:
        notes.append(f"{n_clamped} frame(s) clamped to z >= r-1cm "
                     "(simulated arc dipped slightly below ground)")

    contact_frames = {r.frame for r in hard if r.anchor.state in _CONTACT_STATES}
    contact_frames |= {w.frame for w in waypoints
                       if w.anchor.state in _NO_DEPTH_STATES}

    frames_out = []
    for f in sorted(dense):
        xyz = dense[f]
        if f in contact_frames:
            state = "contact"
        elif xyz[2] <= ts.BALL_RADIUS_M + 0.02:
            state = "ground"
        else:
            state = "air"
        frames_out.append(TruthFrame(
            frame=f, xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            state=state))

    events = [e for r in hard if (e := _event_for_hard_knot(r)) is not None]
    if hard[-1].anchor.state == "grounded":
        events.append(TruthEvent(frame=hard[-1].frame, kind="rest",
                                 xyz=tuple(float(v) for v in hard[-1].xyz)))

    physics = {
        "ball_radius_m": ts.BALL_RADIUS_M,
        "ball_mass_kg": ts.BALL_MASS_KG,
        "g": ts.G,
        "rho_air": ts.RHO_AIR,
        "scenario_requested": scenario,
        "scenario_physics": phys_scenario,
        "seed": seed,
        "n_hard_knots": len(hard),
        "n_waypoints": len(waypoints),
        "n_segments": len(segments),
        "n_flight_fallback": n_fallback,
        "segments": segments,
        "notes": notes + [
            "bounce restitution is annotated per flight segment "
            "('bounce_e_n_annotation') for realism, but adjacent segments "
            "are shot independently to their own hard knots rather than "
            "propagated velocity-continuously through the bounce — a "
            "documented PoC simplification (see truth_builder.py)."],
    }

    return TruthTrack(
        clip_id=ctx.clip_id, scenario=scenario, fps=ctx.fps,
        frames=tuple(frames_out), events=tuple(events),
        seed_anchor_frames=tuple(r.frame for r in hard),
        physics=physics,
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", default="gberch,origi01,kroupi01,s013")
    parser.add_argument("--scenarios", default="base,mismatch,sparse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=None,
                        help="Override output root (default: "
                             "$POC_MAIN_REPO/output-ball-poc or "
                             "M/output-ball-poc)")
    args = parser.parse_args()

    import os
    from pathlib import Path

    from .synth_detector import make_synth_run

    m = os.environ.get("POC_MAIN_REPO",
                       "/Users/joebower/workplace/football-perspectives")
    out_root = Path(args.out_root) if args.out_root else Path(m) / "output-ball-poc"

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    for clip_id in clips:
        try:
            ctx = load_clip(clip_id)
        except Exception as exc:  # noqa: BLE001
            print(f"== {clip_id}: SKIP (load_clip failed: {exc})")
            continue
        print(f"== {clip_id} (fps={ctx.fps}, n_frames={ctx.n_frames}, "
             f"n_anchors={len(ctx.anchors.anchors)}, "
             f"n_observations={len(ctx.observations)}) ==")
        for scenario in scenarios:
            try:
                truth = build_truth(ctx, scenario, seed=args.seed)
            except Exception as exc:  # noqa: BLE001
                print(f"  {scenario}: FAILED ({exc})")
                continue
            synth = make_synth_run(ctx, truth, scenario, seed=args.seed)
            truth_path = out_root / clip_id / f"truth_{scenario}.json"
            synth_path = out_root / clip_id / f"synth_obs_{scenario}.json"
            save_json(truth_path, truth)
            save_json(synth_path, synth)

            zs = np.array([f.xyz[2] for f in truth.frames])
            xs = np.array([f.xyz for f in truth.frames])
            speeds = (np.linalg.norm(np.diff(xs, axis=0), axis=1)
                     * truth.fps) if len(xs) > 1 else np.array([0.0])
            by_type: dict[str, int] = {}
            for seg in truth.physics["segments"]:
                by_type[seg["type"]] = by_type.get(seg["type"], 0) + 1
            coverage = (len(synth.observations) / max(1, len(truth.frames)))
            print(f"  {scenario}: n_frames={len(truth.frames)} "
                 f"segments={by_type} max_speed={speeds.max():.1f}m/s "
                 f"max_height={zs.max():.2f}m n_events={len(truth.events)} "
                 f"synth_coverage={coverage:.2f} "
                 f"sigma_px={synth.noise_model.get('sigma_px'):.2f} "
                 f"-> {truth_path}")


if __name__ == "__main__":
    _main()

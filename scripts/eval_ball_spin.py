"""IC-E eval: does the bounded spin fit (``src.utils.ball_hybrid_spin.
fit_span_spin``) help on the ball-hybrid PoC's synthetic ``mismatch``
scenario, across all 4 real-camera clips (gberch, origi01, kroupi01, s013)?

``mismatch`` is the PoC scenario whose truth (``prototypes/ball_hybrid_poc/
truth_sim.py``) deliberately does NOT share code or parameters with the
pipeline's own physics: every flight segment carries a genuine, decaying
(``omega(t) = omega0 * exp(-t/decay_s)``), drag-crisis-coupled Magnus
spin (see ``output-ball-poc/<clip>/truth_mismatch.json``'s ``physics``
block) — the model IC-E's ``fit_span_spin`` (constant omega, constant
``cd``) can only approximate, which is exactly the point: this asks
whether even an imperfect bounded fit earns its keep under real
model-mismatch, camera geometry, and evidence noise, not whether it can
recover the true simulator parameters (that's ``tests/test_ball_hybrid_
spin.py``'s job, on matched-model synthetic data).

Why this script exists (temporary, not production code): IC-A's
production trajectory module (``src/utils/ball_hybrid_trajectory.py``)
hasn't landed yet, so there is no production span-fit call site to wire
``fit_span_spin`` into yet. This script re-orchestrates the ball-hybrid
PoC's existing pipeline (``prototypes/ball_hybrid_poc/hybrid.py``'s
``run_hybrid``/``_run_hybrid_full``) with ONE addition — a bounded spin
fit + reproject for every resolved INTERIOR flight span (a genuine
two-knot-bracketed span; head/tail free-end spans and roll spans are
left untouched) before the existing delta-blend layer. It calls
``hybrid.py``'s own helper functions (``resolve_knots``,
``_build_physics_track``, ``_span_evidence``, ``_smooth_non_event_knot_
windows``, ``_delta_evidence``, ``_is_sharp_knot``) UNMODIFIED — no
``prototypes/`` file is edited — and duplicates only the small
post-blend frame-assembly loop that ``_run_hybrid_full`` doesn't expose
as a standalone function.

Usage (from the repo root, ball-poc-hybrid worktree):
    .venv311/bin/python scripts/eval_ball_spin.py \
        --clips gberch,origi01,kroupi01,s013 --tag ic_e_spin_v1

Writes per-clip artifacts under
``M/output-ball-poc/<clip>/runs/<tag>/`` (M = the main repo, read-only
elsewhere in this script) and prints a spin on/off table to stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prototypes.ball_hybrid_poc import ctx as poc_ctx  # noqa: E402
from prototypes.ball_hybrid_poc import hybrid  # noqa: E402
from prototypes.ball_hybrid_poc import metrics  # noqa: E402
from prototypes.ball_hybrid_poc import types  # noqa: E402
from prototypes.ball_hybrid_poc.blend import blend_deltas, clamp_delta_rate  # noqa: E402
from prototypes.ball_hybrid_poc.hybrid_physics import (  # noqa: E402
    BALL_RADIUS_M,
    shoot_arc,
    simulate,
)

from src.utils.ball_hybrid_spin import fit_span_spin  # noqa: E402

DEFAULT_CLIPS = ("gberch", "origi01", "kroupi01", "s013")
SCENARIO = "mismatch"


# ---------------------------------------------------------------------------
# Spin-augmented re-orchestration of hybrid.py's pipeline.
# ---------------------------------------------------------------------------

def _apply_spin(
    ctx: Any,
    pts: dict[int, np.ndarray],
    span_diag: Sequence[dict],
    observations: Sequence,
    ray_anchors: Sequence,
    cfg: Mapping[str, Any],
    spin_kwargs: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """For every resolved interior FLIGHT span in ``span_diag`` (skips
    roll spans and head/tail open-end spans, whose ``model`` is never
    ``"flight"``), attempt ``fit_span_spin`` on that span's own interior
    evidence (``hybrid._span_evidence`` — the SAME evidence definition
    the span's drag-only fit already used) and, if accepted, splice a
    re-simulated (drag + fitted Magnus) trajectory for the span's full
    frame range into a COPY of ``pts``. Returns ``(pts_with_spin,
    decisions)``; ``pts`` itself is never mutated.
    """
    pts_out = dict(pts)
    decisions: list[dict] = []

    for span in span_diag:
        if span.get("model") != "flight":
            continue
        a_frame, b_frame = span["span"]
        if a_frame not in pts or b_frame not in pts:
            continue
        duration_s = (b_frame - a_frame) / ctx.fps
        if duration_s <= 0:
            continue
        cd = span.get("cd", cfg["cd"])
        p_a = np.asarray(pts[a_frame], dtype=float)
        p_b = np.asarray(pts[b_frame], dtype=float)

        evid = hybrid._span_evidence(observations, ray_anchors, a_frame, b_frame,
                                      cfg["anchor_fit_weight"])
        n_obs = len(evid)
        if not evid:
            decisions.append({"span": [a_frame, b_frame], "accepted": False,
                               "n_obs": 0, "reason": "no_evidence"})
            continue

        obs_frames = np.array([f for f, _, _ in evid], dtype=float)
        obs_times = obs_frames / ctx.fps
        obs_uv = np.array([uv for _, uv, _ in evid], dtype=float)
        obs_conf = np.array([w for _, _, w in evid], dtype=float)

        # obs_times/t_a/t_b are all on the SAME absolute clip-time base
        # (frame / fps) here, so project_fn maps t_s straight to a frame
        # via round(t_s * fps) -- no per-span offset. (fit_span_spin's
        # contract is "project_fn receives whatever base obs_times/t_a/t_b
        # use", not necessarily span-relative; see ball_hybrid_spin.py's
        # module docstring.)
        def project_fn(t_s: float, xyz: np.ndarray) -> np.ndarray:
            frame = int(round(t_s * ctx.fps))
            return ctx.project(frame, xyz)

        t_a = a_frame / ctx.fps
        t_b = b_frame / ctx.fps

        try:
            fit = fit_span_spin(
                tuple(p_a), t_a, tuple(p_b), t_b,
                obs_times, obs_uv, project_fn,
                cd=cd, magnus_coeff=cfg["magnus_coeff"],
                obs_conf=obs_conf, **dict(spin_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 -- eval robustness only
            decisions.append({"span": [a_frame, b_frame], "accepted": False,
                               "n_obs": n_obs, "reason": f"error: {exc!r}"})
            continue

        if fit is None:
            decisions.append({"span": [a_frame, b_frame], "accepted": False,
                               "n_obs": n_obs})
            continue

        omega = np.array(fit.omega_world)
        v0 = shoot_arc(p_a, 0.0, p_b, duration_s, cd=cd, omega=omega,
                        magnus_coeff=cfg["magnus_coeff"])
        frames = list(range(a_frame, b_frame + 1))
        times = [(f - a_frame) / ctx.fps for f in frames]
        positions = simulate(p_a, v0, times, cd=cd, omega=omega,
                              magnus_coeff=cfg["magnus_coeff"])
        for f, pos in zip(frames, positions):
            pts_out[f] = pos

        decisions.append({
            "span": [a_frame, b_frame], "accepted": True, "n_obs": n_obs,
            "rad_s": fit.rad_s, "delta_bic": fit.delta_bic,
            "omega_world": list(fit.omega_world),
        })

    return pts_out, decisions


def run_hybrid_with_spin(
    ctx: Any,
    observations: Sequence,
    anchors: Sequence[Any],
    fixes: Sequence[Any] = (),
    auto_anchors: Sequence[Any] = (),
    *,
    cfg: Mapping[str, Any] | None = None,
    spin_kwargs: Mapping[str, Any] | None = None,
):
    """Mirrors ``hybrid._run_hybrid_full`` (unedited helper calls only),
    inserting ``_apply_spin`` right after ``_build_physics_track`` and
    before the local-knot smoothing / delta-blend layer. Returns
    ``(Track, diagnostics)``; ``diagnostics["spin"]`` is the per-span
    accept/reject list from ``_apply_spin``.
    """
    full_cfg = dict(hybrid.DEFAULT_CFG)
    if cfg:
        full_cfg.update(cfg)
    spin_kwargs = dict(spin_kwargs or {})

    hard_knots, ray_anchors = hybrid.resolve_knots(ctx, anchors, fixes)
    obs_sorted = sorted(observations, key=lambda o: o.frame)

    if auto_anchors:
        auto_events = [a for a in auto_anchors
                        if hybrid._anchor_attrs(a)[2] in hybrid.EVENT_STATES]
        auto_hard, auto_rays = hybrid.resolve_knots(ctx, auto_events, fixes=())
        hard_knots, _accepted, _rejected, _n_dropped, extra_rays = hybrid._integrate_auto_knots(
            ctx, hard_knots, ray_anchors, auto_hard, auto_rays, obs_sorted, full_cfg)
        ray_anchors = sorted(list(ray_anchors) + extra_rays, key=lambda r: r.frame)

    pts, span_diag, internal_knot_frames = hybrid._build_physics_track(
        ctx, hard_knots, ray_anchors, obs_sorted, full_cfg)

    if not pts:
        empty = types.Track(clip_id=ctx.clip_id, method="hybrid_spin", frames=())
        return empty, {"spans": [], "spin": [], "n_knots": 0}

    pts, spin_diag = _apply_spin(ctx, pts, span_diag, obs_sorted, ray_anchors,
                                  full_cfg, spin_kwargs)

    # --- everything below here is an unedited copy of
    # hybrid._run_hybrid_full's post-physics-track pipeline (smoothing,
    # delta-blend, final frame assembly) so the spin-on/off comparison
    # isolates the spin step itself. ------------------------------------
    pts = hybrid._smooth_non_event_knot_windows(ctx, pts, hard_knots, full_cfg)

    evidence = hybrid._delta_evidence(ctx, pts, obs_sorted, ray_anchors, full_cfg)
    sharp_frames = [k.frame for k in hard_knots if hybrid._is_sharp_knot(k)]
    event_frames = sharp_frames + internal_knot_frames
    frames_sorted = sorted(pts)
    blended = blend_deltas(frames_sorted, evidence,
                            halflife_frames=full_cfg["blend_halflife_frames"],
                            event_frames=event_frames)

    max_step_by_frame: dict[int, float] = {}
    frac = full_cfg["blend_max_delta_step_frac"]
    speed_floor = full_cfg["blend_max_delta_step_floor_m_s"]
    for i, f in enumerate(frames_sorted):
        f_prev = frames_sorted[max(i - 1, 0)]
        f_next = frames_sorted[min(i + 1, len(frames_sorted) - 1)]
        dt = (f_next - f_prev) / ctx.fps
        speed = (float(np.linalg.norm(pts[f_next] - pts[f_prev])) / dt
                 if dt > 1e-9 else 0.0)
        max_step_by_frame[f] = frac * max(speed, speed_floor) / ctx.fps
    blended = clamp_delta_rate(frames_sorted, blended,
                                max_step_m=max_step_by_frame,
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

        if f in knot_by_frame and hybrid._is_sharp_knot(knot_by_frame[f]):
            final = np.array(knot_by_frame[f].xyz, dtype=float)
            mode, out_conf = "anchor", 1.0
        elif f in knot_by_frame:
            mode = "anchor"
        elif f in ray_by_frame:
            mode = "anchor"

        final[2] = max(float(final[2]), BALL_RADIUS_M)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out_frames.append(types.TrackFrame(
            frame=f, xyz=(float(final[0]), float(final[1]), float(final[2])),
            mode=mode, conf=out_conf))

    track = types.Track(clip_id=ctx.clip_id, method="hybrid_spin",
                         frames=tuple(out_frames))
    diagnostics = {
        "spans": span_diag,
        "spin": spin_diag,
        "n_knots": len(hard_knots),
        "n_ray_anchors": len(ray_anchors),
        "n_internal_bounces": len(internal_knot_frames),
        "mode_counts": mode_counts,
    }
    return track, diagnostics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _clip_dir(clip_id: str) -> Path:
    return Path(poc_ctx.M) / "output-ball-poc" / clip_id


def evaluate_clip(clip_id: str, spin_kwargs: Mapping[str, Any],
                   run_dir: Path) -> dict[str, Any]:
    clip_ctx = poc_ctx.load_clip(clip_id)
    clip_dir = _clip_dir(clip_id)
    truth = types.TruthTrack.from_json(
        types.load_json(clip_dir / f"truth_{SCENARIO}.json"))
    synth = types.SynthRun.from_json(
        types.load_json(clip_dir / f"synth_obs_{SCENARIO}.json"))

    observations = list(synth.observations)
    anchors = list(synth.anchors)

    t0 = time.perf_counter()
    track_off = hybrid.run_hybrid(clip_ctx, observations, anchors, ())
    t_off = time.perf_counter() - t0

    t0 = time.perf_counter()
    track_on, diag_on = run_hybrid_with_spin(
        clip_ctx, observations, anchors, (), spin_kwargs=spin_kwargs)
    t_on = time.perf_counter() - t0

    side_cam = metrics.build_side_camera([f.xyz for f in truth.frames])
    flat_off, _ = metrics.compute_scenario_metrics(
        clip_ctx, track_off, truth, side_camera=side_cam)
    flat_on, _ = metrics.compute_scenario_metrics(
        clip_ctx, track_on, truth, side_camera=side_cam)

    n_flight_spans = sum(1 for s in diag_on["spans"] if s.get("model") == "flight")
    n_accepted = sum(1 for d in diag_on["spin"] if d.get("accepted"))

    def _sub(flat: dict) -> dict:
        return {k: flat[k] for k in
                ("pct_le_20cm_air", "p50_air", "pct_le_20cm", "p50")}

    result = {
        "clip_id": clip_id,
        "scenario": SCENARIO,
        "runtime_s": {"spin_off": t_off, "spin_on": t_on},
        "n_flight_spans": n_flight_spans,
        "n_spin_accepted": n_accepted,
        "spin_decisions": diag_on["spin"],
        "metrics": {"spin_off": _sub(flat_off), "spin_on": _sub(flat_on)},
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    types.save_json(run_dir / f"{clip_id}_spin_eval.json", result)
    types.save_json(run_dir / f"{clip_id}_track_hybrid_off_{SCENARIO}.json", track_off)
    types.save_json(run_dir / f"{clip_id}_track_hybrid_spin_{SCENARIO}.json", track_on)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fit_span_spin on the ball-hybrid PoC's "
                     "synthetic mismatch scenario across real-camera clips.")
    parser.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    parser.add_argument("--tag", default="ic_e_spin_v1",
                         help="run-record tag; artifacts land under "
                              "M/output-ball-poc/<clip>/runs/<tag>/")
    parser.add_argument("--min-delta-bic", type=float, default=None)
    parser.add_argument("--min-resid-gain", type=float, default=None)
    args = parser.parse_args(argv)

    spin_kwargs: dict[str, Any] = {}
    if args.min_delta_bic is not None:
        spin_kwargs["min_delta_bic"] = args.min_delta_bic
    if args.min_resid_gain is not None:
        spin_kwargs["min_resid_gain"] = args.min_resid_gain

    results = []
    for clip_id in args.clips.split(","):
        clip_id = clip_id.strip()
        if not clip_id:
            continue
        run_dir = _clip_dir(clip_id) / "runs" / args.tag
        print(f"=== {clip_id} ===", flush=True)
        r = evaluate_clip(clip_id, spin_kwargs, run_dir)
        results.append(r)
        off, on = r["metrics"]["spin_off"], r["metrics"]["spin_on"]
        print(f"  flight spans: {r['n_flight_spans']}  "
              f"spin accepted: {r['n_spin_accepted']}  "
              f"runtime off/on: {r['runtime_s']['spin_off']:.2f}s / "
              f"{r['runtime_s']['spin_on']:.2f}s")
        print(f"  pct_le_20cm_air  off={off['pct_le_20cm_air']}  "
              f"on={on['pct_le_20cm_air']}")
        print(f"  p50_air (m)      off={off['p50_air']}  on={on['p50_air']}")
        print(f"  pct_le_20cm(all) off={off['pct_le_20cm']}  "
              f"on={on['pct_le_20cm']}")
        print(f"  p50(all) (m)     off={off['p50']}  on={on['p50']}")
        print(f"  wrote {run_dir}/{clip_id}_spin_eval.json", flush=True)

    print("\n=== summary (mismatch scenario) ===")
    header = (f"{'clip':10s} {'spans':>6s} {'acc':>4s} "
              f"{'20cm_air off':>13s} {'20cm_air on':>12s} "
              f"{'p50_air off':>12s} {'p50_air on':>11s}")
    print(header)
    for r in results:
        off, on = r["metrics"]["spin_off"], r["metrics"]["spin_on"]

        def _fmt(v):
            return f"{v:.3f}" if v is not None else "n/a"

        print(f"{r['clip_id']:10s} {r['n_flight_spans']:>6d} "
              f"{r['n_spin_accepted']:>4d} "
              f"{_fmt(off['pct_le_20cm_air']):>13s} "
              f"{_fmt(on['pct_le_20cm_air']):>12s} "
              f"{_fmt(off['p50_air']):>12s} {_fmt(on['p50_air']):>11s}")


if __name__ == "__main__":
    main()

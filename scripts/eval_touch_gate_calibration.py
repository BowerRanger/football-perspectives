"""Multi-clip touch-gate calibration study (ball-stage campaign, workstream 1).

Recomputes the exact signals the touch-detection gates consume — bone<->ball-
ray 3-D gap, peak foot/knee pixel speed, ball-pixel break/event presence, and
real-detector coverage — at every MANUAL ``player_touch`` anchor across the
four eval dirs, using the CURRENT refined-poses FK, camera track and the
already-persisted ``*_ball_observations.json`` sidecar. No detector or ball-
stage re-run: this is pure offline analysis over cached artifacts (pure
numpy, torch-free).

Contrasts the manual-touch distribution against a BACKGROUND pool built from
the same candidate-generation topology ``propose_touches`` uses: every
local-minimum of the bone<->ball-ray gap series
(``ball_kinematic_touch.local_minima_below`` on a deliberately lax threshold
so nothing is pre-filtered) that is NOT within ``--frame-tol`` of any manual
touch. Each candidate row carries gap3d_m (kinematic_touch.contact_gap_m /
touch_attribution.max_gap_m), pixgap_px (auto_anchors.touch_max_px /
kinematic_touch.touch_relaxed_px — the same 2-D projection of the bone-ray
gap those pixel gates check) and peak_foot_px (kinematic_touch.
kin_min_foot_speed) for foot/knee bones, so it doubles as the background pool
for all of them even though only kinematic_touch's own gates are driven by
its exact local-minima topology.

For each gate reports the smallest/largest threshold (as appropriate) that
captures >= ``--recall-target`` (default 0.90) of the manual touches, and the
background pool's pass-rate at that threshold as a false-positive-rate proxy
(candidate-pool level — an upper bound on the real FP delta, since accepted
candidates still face min_emit_score/NMS/event-evidence downstream).

Usage:
    .venv311/bin/python scripts/eval_touch_gate_calibration.py \
        --out docs/superpowers/notes/ball-accuracy/touch_gate_calibration.json

Re-runnable any time refined_poses/camera/ball observations change; add
``--clips output:gberch`` (repeatable) to restrict to a subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_config  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.stages.ball import (  # noqa: E402
    _auto_event_cfg,
    _kinematic_touch_cfg,
    _touch_attribution_cfg,
)
from src.utils.ball_auto_events import AutoEventCfg, detect_events  # noqa: E402
from src.utils.ball_kinematic_touch import (  # noqa: E402
    FOOT_BONES,
    HEAD_BONES,
    KNEE_BONES,
    KinematicTouchCfg,
    _head_speed_m,
    _peak_foot_speed,
    ball_confirm,
    interpolate_ball_uvs,
    local_minima_below,
    ray_gap_series,
)
from src.utils.ball_player_context import PlayerContext  # noqa: E402
from src.utils.ball_touch_recall import touches_from_anchor_set  # noqa: E402
from src.utils.ball_tracker import TrackerStep  # noqa: E402
from src.utils.goal_geometry import GoalGeometry  # noqa: E402

SPEED_GATED_BONES = frozenset(FOOT_BONES) | frozenset(KNEE_BONES)
REAL_DETECTOR_SOURCES = frozenset({
    "detector", "second_pass", "foot_guided", "strike_window",
})
BREAK_KINDS = frozenset({"touch", "velocity_break", "bounce", "goal_impact"})

# (output_dir, shot_id) pairs with manual ball_anchors.json (pseudo-GT).
DEFAULT_CLIPS: tuple[tuple[str, str], ...] = (
    ("output", "gberch"),
    ("output-kroupi", "kroupi01"),
    ("output-japan", "s013"),
    ("output-origi", "origi01"),
)
# origi02 has no manual ball_anchors.json — background-only, off by default
# (the study is about matching against ground truth; enable with
# --include-background-only for a bigger background sample).
BACKGROUND_ONLY_CLIPS: tuple[tuple[str, str], ...] = (
    ("output-origi", "origi02"),
)


@dataclass
class ShotArtifacts:
    output_dir: Path
    shot_id: str
    per_frame_K: dict
    per_frame_R: dict
    per_frame_t: dict
    distortion: tuple[float, float]
    image_size: tuple[int, int]
    ball_uvs: dict
    confidences: dict
    sources: dict
    player_ctx: PlayerContext
    goal_geometry: GoalGeometry


def load_shot(output_dir: Path, shot_id: str, pitch_cfg: dict) -> "ShotArtifacts | str":
    """Loads one shot's cached artifacts, or returns a skip-reason string."""
    cam_path = output_dir / "camera" / f"{shot_id}_camera_track.json"
    if not cam_path.exists():
        cam_path = output_dir / "camera" / "camera_track.json"
    if not cam_path.exists():
        return f"no camera_track at {cam_path}"
    obs_path = output_dir / "ball" / f"{shot_id}_ball_observations.json"
    if not obs_path.exists():
        return f"no ball observations at {obs_path}"

    camera = CameraTrack.load(cam_path)
    per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
    per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
    t_fallback = np.array(camera.t_world)
    per_frame_t = {
        f.frame: (np.array(f.t) if f.t is not None else t_fallback)
        for f in camera.frames
    }
    distortion = camera.distortion

    obs = json.loads(obs_path.read_text())
    ball_uvs: dict[int, np.ndarray] = {}
    confidences: dict[int, float] = {}
    sources: dict[int, str] = {}
    for row in obs["frames"]:
        fr = int(row["frame"])
        confidences[fr] = float(row.get("confidence", 0.0))
        sources[fr] = str(row.get("source", "none"))
        if row.get("uv") is not None:
            ball_uvs[fr] = np.asarray(row["uv"], dtype=float)

    player_ctx = PlayerContext.load(
        output_dir, shot_id,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion,
    )
    if not player_ctx.player_ids:
        return "no player tracks (refined_poses/hmr_world) found"

    goal_geometry = GoalGeometry.from_pitch_config(pitch_cfg)

    return ShotArtifacts(
        output_dir=output_dir, shot_id=shot_id,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R, per_frame_t=per_frame_t,
        distortion=distortion, image_size=tuple(camera.image_size),
        ball_uvs=ball_uvs, confidences=confidences, sources=sources,
        player_ctx=player_ctx, goal_geometry=goal_geometry,
    )


def _coverage_nearby(shot: ShotArtifacts, frame: int, window: int) -> float:
    frames = range(frame - window, frame + window + 1)
    hits = sum(1 for f in frames if shot.sources.get(f) in REAL_DETECTOR_SOURCES)
    return hits / (2 * window + 1)


def _nearest_break(events, frame: int, window: int) -> int | None:
    best = None
    for e in events:
        if e.kind not in BREAK_KINDS:
            continue
        d = abs(e.frame - frame)
        if d <= window and (best is None or d < best):
            best = d
    return best


def touch_signal_rows(
    shot: ShotArtifacts,
    touches: list[tuple[int, str, str]],
    kin_cfg: KinematicTouchCfg,
    event_cfg: AutoEventCfg,
    confirm_window: int,
    attr_max_gap_m: float,
    attr_margin_m: float,
) -> list[dict]:
    """One signal row per manual touch (deliverable 1's core table)."""
    filled, interp_frames = interpolate_ball_uvs(shot.ball_uvs, kin_cfg.max_ball_gap_frames)
    series = ray_gap_series(
        shot.player_ctx, filled, shot.per_frame_K, shot.per_frame_R,
        shot.per_frame_t, shot.distortion, kin_cfg.min_fk_conf,
    )
    events = detect_events(
        steps=[TrackerStep(frame=f, uv=tuple(uv), p_flight=0.0,
                            is_outlier=False, is_gap_fill=False)
               for f, uv in shot.ball_uvs.items()],
        confidences=shot.confidences, player_ctx=shot.player_ctx,
        per_frame_K=shot.per_frame_K, per_frame_R=shot.per_frame_R,
        per_frame_t=shot.per_frame_t, distortion=shot.distortion,
        goal_geometry=shot.goal_geometry, cfg=event_cfg,
        image_size=shot.image_size,
    )
    detected_frames = frozenset(
        f for f, c in shot.confidences.items() if c > 0.0
    )

    rows: list[dict] = []
    for frame, pid, bone in touches:
        window = range(frame - kin_cfg.kin_window, frame + kin_cfg.kin_window + 1)
        key = (pid, bone)
        per_frame = series.get(key, {})
        in_window = {f: v for f, v in per_frame.items() if f in window}
        row: dict = {
            "output_dir": str(shot.output_dir), "shot_id": shot.shot_id,
            "frame": frame, "player_id": pid, "bone": bone,
        }
        if in_window:
            best_f = min(in_window, key=lambda f: in_window[f][0])
            gap3d, pixgap, fk_conf = in_window[best_f]
            row.update(gap3d_m=gap3d, pixgap_px=pixgap, fk_conf=fk_conf,
                       gap_frame=best_f)
        else:
            row.update(gap3d_m=None, pixgap_px=None, fk_conf=None, gap_frame=None)

        if bone in SPEED_GATED_BONES:
            row["peak_foot_px"] = _peak_foot_speed(
                shot.player_ctx, frame, pid, bone, kin_cfg.kin_window)
            row["peak_head_m"] = None
        elif bone in HEAD_BONES:
            row["peak_foot_px"] = None
            row["peak_head_m"] = _head_speed_m(shot.player_ctx, frame, pid, bone)
        else:
            row["peak_foot_px"] = None
            row["peak_head_m"] = None

        row["coverage_nearby"] = _coverage_nearby(shot, frame, confirm_window)
        row["nearest_break_dist"] = _nearest_break(
            events, frame, event_cfg.event_window_frames)
        row["confirm"] = ball_confirm(
            frame, kin_cfg, frozenset(e.frame for e in events if e.kind in BREAK_KINDS),
            interp_frames, detected_frames)

        # All-bone ranking in the same window: is the claimed bone the
        # argmin, or would another (player, bone) win the ray-gap contest
        # (the touch_attribution failure mode)?
        candidates: dict[tuple[str, str], float] = {}
        for (cpid, cbone), cseries in series.items():
            vals = [g for f, (g, _px, _c) in cseries.items() if f in window]
            if vals:
                candidates[(cpid, cbone)] = min(vals)
        ranked = sorted(candidates.items(), key=lambda kv: kv[1])[:5]
        row["top5_bone_candidates"] = [
            {"player_id": p, "bone": b, "gap_m": round(g, 4)} for (p, b), g in ranked
        ]
        row["claimed_is_argmin"] = bool(ranked) and ranked[0][0] == (pid, bone)
        # Mirrors refine_touch_attribution's actual relabel condition (sans
        # the depth-weighting term, which needs a resolved track): flags a
        # touch as attribution-at-risk only when a DIFFERENT candidate both
        # clears touch_attribution.max_gap_m and beats the claimed bone's
        # own window-min gap by more than margin_m — the same two guards
        # production checks before relabelling.
        current_gap = row["gap3d_m"]
        risk = False
        if ranked and ranked[0][0] != (pid, bone):
            best_key, best_gap = ranked[0]
            risk = (
                best_gap <= attr_max_gap_m
                and (current_gap is None or best_gap + attr_margin_m < current_gap)
            )
        row["attribution_risk"] = risk
        rows.append(row)
    return rows


def kinematic_background_pool(
    shot: ShotArtifacts,
    touches: list[tuple[int, str, str]],
    kin_cfg: KinematicTouchCfg,
    frame_tol: int,
    lax_threshold_m: float,
) -> list[dict]:
    """Every local-minimum candidate NOT near a manual touch (any bone)."""
    filled, _interp = interpolate_ball_uvs(shot.ball_uvs, kin_cfg.max_ball_gap_frames)
    series = ray_gap_series(
        shot.player_ctx, filled, shot.per_frame_K, shot.per_frame_R,
        shot.per_frame_t, shot.distortion, kin_cfg.min_fk_conf,
    )
    touch_frames = [f for f, _p, _b in touches]

    def _near_touch(f: int) -> bool:
        return any(abs(f - tf) <= frame_tol for tf in touch_frames)

    rows: list[dict] = []
    for (pid, bone), per_frame in series.items():
        gaps = {f: g for f, (g, _px, _c) in per_frame.items()}
        for f in local_minima_below(gaps, lax_threshold_m):
            if _near_touch(f):
                continue
            gap3d, pixgap, fk_conf = per_frame[f]
            row = {
                "output_dir": str(shot.output_dir), "shot_id": shot.shot_id,
                "frame": f, "player_id": pid, "bone": bone,
                "gap3d_m": gap3d, "pixgap_px": pixgap, "fk_conf": fk_conf,
            }
            if bone in SPEED_GATED_BONES:
                row["peak_foot_px"] = _peak_foot_speed(
                    shot.player_ctx, f, pid, bone, kin_cfg.kin_window)
            else:
                row["peak_foot_px"] = None
            rows.append(row)
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def calibrate_gap_gate(
    touch_rows: list[dict], background_rows: list[dict],
    *, current: float, recall_target: float, bone_filter=None,
) -> dict:
    """Smallest ``<=`` threshold capturing >= recall_target of touches."""
    vals = [r["gap3d_m"] for r in touch_rows
            if r["gap3d_m"] is not None
            and (bone_filter is None or r["bone"] in bone_filter)]
    n_missing = sum(1 for r in touch_rows if r["gap3d_m"] is None
                    and (bone_filter is None or r["bone"] in bone_filter))
    if not vals:
        return {"n_touches": 0, "n_no_fk": n_missing}
    threshold = _percentile(vals, recall_target * 100.0)
    bg = [r["gap3d_m"] for r in background_rows
          if r["gap3d_m"] is not None
          and (bone_filter is None or r["bone"] in bone_filter)]
    return {
        "n_touches": len(vals),
        "n_no_fk": n_missing,
        "recall_target": recall_target,
        "current_threshold": current,
        "recommended_threshold": round(threshold, 4),
        "touch_values_sorted": sorted(round(v, 4) for v in vals),
        "current_pass_rate_touches": sum(1 for v in vals if v <= current) / len(vals),
        "recommended_pass_rate_touches": sum(1 for v in vals if v <= threshold) / len(vals),
        "n_background": len(bg),
        "current_pass_rate_background": (
            sum(1 for v in bg if v <= current) / len(bg) if bg else None),
        "recommended_pass_rate_background": (
            sum(1 for v in bg if v <= threshold) / len(bg) if bg else None),
    }


def calibrate_speed_gate(
    touch_rows: list[dict], background_rows: list[dict],
    *, current: float, recall_target: float,
) -> dict:
    """Largest ``>=`` threshold still capturing >= recall_target of touches."""
    vals = [r["peak_foot_px"] for r in touch_rows if r["peak_foot_px"] is not None]
    if not vals:
        return {"n_touches": 0}
    threshold = _percentile(vals, (1.0 - recall_target) * 100.0)
    bg = [r["peak_foot_px"] for r in background_rows if r["peak_foot_px"] is not None]
    return {
        "n_touches": len(vals),
        "recall_target": recall_target,
        "current_threshold": current,
        "recommended_threshold": round(threshold, 4),
        "touch_values_sorted": sorted(round(v, 4) for v in vals),
        "current_pass_rate_touches": sum(1 for v in vals if v >= current) / len(vals),
        "recommended_pass_rate_touches": sum(1 for v in vals if v >= threshold) / len(vals),
        "n_background": len(bg),
        "current_pass_rate_background": (
            sum(1 for v in bg if v >= current) / len(bg) if bg else None),
        "recommended_pass_rate_background": (
            sum(1 for v in bg if v >= threshold) / len(bg) if bg else None),
    }


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'clip':<12}{'frame':>6}{'player':>8}{'bone':>10}"
           f"{'gap3d_m':>10}{'peak_px':>9}{'cov':>6}{'brk':>5}{'argmin':>8}")
    print(hdr)
    for r in rows:
        clip = f"{Path(r['output_dir']).name}/{r['shot_id']}"
        gap = f"{r['gap3d_m']:.3f}" if r["gap3d_m"] is not None else "n/a"
        peak = (f"{r['peak_foot_px']:.2f}" if r["peak_foot_px"] is not None else "n/a")
        cov = f"{r['coverage_nearby']:.2f}"
        brk = str(r["nearest_break_dist"]) if r["nearest_break_dist"] is not None else "-"
        argmin = "yes" if r["claimed_is_argmin"] else "NO"
        print(f"{clip:<12}{r['frame']:>6}{r['player_id']:>8}{r['bone']:>10}"
              f"{gap:>10}{peak:>9}{cov:>6}{brk:>5}{argmin:>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", action="append", default=None,
                     help="output_dir:shot_id pair, repeatable (default: the 4 eval dirs)")
    ap.add_argument("--include-background-only", action="store_true",
                     help="also mine origi02 (no manual anchors) for the background pool")
    ap.add_argument("--frame-tol", type=int, default=3,
                     help="frame tolerance for excluding background candidates near a manual touch")
    ap.add_argument("--recall-target", type=float, default=0.90)
    ap.add_argument("--lax-threshold-m", type=float, default=1.2,
                     help="gap cap used only to bound the background local-minima search")
    ap.add_argument("--out", type=Path, default=None,
                     help="write full JSON (rows + calibration) here")
    args = ap.parse_args()

    if args.clips:
        clip_specs = [tuple(c.split(":", 1)) for c in args.clips]
    else:
        clip_specs = list(DEFAULT_CLIPS)
    background_specs = list(BACKGROUND_ONLY_CLIPS) if args.include_background_only else []

    cfg = load_config(None)
    pitch_cfg = cfg.get("pitch", {})
    ball_cfg = cfg.get("ball", {})
    kin_cfg = _kinematic_touch_cfg(ball_cfg.get("kinematic_touch", {}))
    attr_cfg = _touch_attribution_cfg(ball_cfg.get("touch_attribution", {}))
    event_cfg = _auto_event_cfg(
        ball_cfg.get("auto_anchors", {}), ball_cfg.get("segment", {}),
        ball_cfg.get("pose_touch", {}))
    # Lax cfg for background candidate generation: only gates on min_fk_conf +
    # the bounding threshold, so it doesn't pre-filter the very thresholds
    # we're trying to calibrate.
    lax_kin_cfg = replace(kin_cfg, contact_gap_m=args.lax_threshold_m,
                           touch_relaxed_px=10_000.0)

    all_touch_rows: list[dict] = []
    all_background_rows: list[dict] = []
    skipped: list[str] = []

    for output_dir_s, shot_id in clip_specs:
        output_dir = Path(output_dir_s)
        shot = load_shot(output_dir, shot_id, pitch_cfg)
        if isinstance(shot, str):
            skipped.append(f"{output_dir_s}:{shot_id} — {shot}")
            continue
        manual_path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
        if not manual_path.exists():
            skipped.append(f"{output_dir_s}:{shot_id} — no manual anchors")
            continue
        touches = touches_from_anchor_set(manual_path)
        touches = [(f, p, b) for f, p, b in touches if p and b]
        rows = touch_signal_rows(shot, touches, kin_cfg, event_cfg,
                                  confirm_window=kin_cfg.confirm_window,
                                  attr_max_gap_m=attr_cfg.max_gap_m,
                                  attr_margin_m=attr_cfg.margin_m)
        all_touch_rows.extend(rows)
        bg = kinematic_background_pool(
            shot, touches, lax_kin_cfg, args.frame_tol, args.lax_threshold_m)
        all_background_rows.extend(bg)

    for output_dir_s, shot_id in background_specs:
        output_dir = Path(output_dir_s)
        shot = load_shot(output_dir, shot_id, pitch_cfg)
        if isinstance(shot, str):
            skipped.append(f"{output_dir_s}:{shot_id} — {shot}")
            continue
        bg = kinematic_background_pool(
            shot, [], lax_kin_cfg, args.frame_tol, args.lax_threshold_m)
        all_background_rows.extend(bg)

    print(f"loaded {len(all_touch_rows)} manual touches, "
          f"{len(all_background_rows)} background candidates")
    if skipped:
        print("skipped (stale/missing artifacts):")
        for s in skipped:
            print(f"  - {s}")
    print()
    _print_table(all_touch_rows)

    print("\n=== gap3d_m (kinematic_touch.contact_gap_m) — ALL bones ===")
    gap_all = calibrate_gap_gate(
        all_touch_rows, all_background_rows,
        current=kin_cfg.contact_gap_m, recall_target=args.recall_target)
    print(json.dumps(gap_all, indent=2))

    print("\n=== peak_foot_px (kinematic_touch.kin_min_foot_speed) — foot/knee bones ===")
    speed_gate = calibrate_speed_gate(
        all_touch_rows, all_background_rows,
        current=kin_cfg.kin_min_foot_speed, recall_target=args.recall_target)
    print(json.dumps(speed_gate, indent=2))

    print("\n=== touch_attribution: claimed bone is argmin of the ray-gap ranking ===")
    n_argmin = sum(1 for r in all_touch_rows if r["claimed_is_argmin"])
    n_risk = sum(1 for r in all_touch_rows if r["attribution_risk"])
    print(f"{n_argmin}/{len(all_touch_rows)} manual touches have the claimed bone "
          f"as the closest-gap candidate in the window; "
          f"{n_risk}/{len(all_touch_rows)} clear production's actual relabel "
          f"condition (max_gap_m={attr_cfg.max_gap_m}, margin_m={attr_cfg.margin_m})")
    for r in all_touch_rows:
        if r["attribution_risk"]:
            print(f"  ATTRIBUTION-RISK {Path(r['output_dir']).name}/{r['shot_id']} "
                  f"f{r['frame']} claimed=({r['player_id']},{r['bone']}) "
                  f"top5={r['top5_bone_candidates']}")

    payload = {
        "config_snapshot": {
            "kinematic_touch": kin_cfg.__dict__,
            "touch_attribution": attr_cfg.__dict__,
        },
        "touch_rows": all_touch_rows,
        "background_rows": all_background_rows,
        "skipped": skipped,
        "calibration": {
            "contact_gap_m_all_bones": gap_all,
            "kin_min_foot_speed": speed_gate,
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

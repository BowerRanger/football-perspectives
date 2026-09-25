"""2-fold cross-replay FIX hold-out for the ball hybrid trajectory layer.

Fixes (``src/schemas/ball_fixes.py::BallFix``, from cross-replay
triangulation, ``src/utils/ball_cross_replay.py``) are the pipeline's only
ABSOLUTE 3-D ground truth (sub-20cm campaign W5u-z2: 0.04-0.61m vs
operator rays). This harness holds out HALF the fixes as graded ground
truth and feeds the other half in as depth-hard trajectory knots — gated
through ``src.utils.ball_replay_knots.fixes_to_knots`` (operator-click-
conflict + physical-volume drops) — then grades the resulting track at
the held-out half; two folds swap which half is knotted, so every fix is
knotted exactly once and graded exactly once.

It also reports the origi01-style acceptance companion metric: held-out
MANUAL ANCHOR error (same 2-fold ``BE.split_anchors`` procedure the
sub-20cm campaign harness uses) WITH the (gated) fixes knotted alongside
the kept anchors vs WITHOUT any fixes — fixes must not make anchor
accuracy worse.

Two trajectory backends:
  --trajectory prototype (default): ``prototypes/ball_hybrid_poc/hybrid.py``'s
      ``run_hybrid`` — the hybrid-extraction PoC spike (see the
      ``ball-hybrid-poc`` memory note). Its ``fixes=`` parameter only
      reads ``.frame``/``.xyz`` off each item, so the gated ``Knot``
      objects ``fixes_to_knots`` returns are passed straight through with
      no conversion.
  --trajectory hybrid: the PRODUCTION ``src.utils.ball_hybrid_trajectory``
      layer (IC-A's module + the ``ball.trajectory`` stage wiring). Not
      landed at the time this script was written — errors with a clear
      message until it exists; re-run with this flag once
      ``git log`` shows a commit adding ``ball.trajectory`` /
      ``ball_hybrid_trajectory.py``.

Usage:
    .venv311/bin/python scripts/eval_ball_fix_holdout.py \
        --output /path/to/output-origi-global --shot origi01 \
        --json docs/superpowers/notes/ball-accuracy/origi01_fix_holdout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_fixes import BallFix, BallFixSet  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils import ball_eval as BE  # noqa: E402
from src.utils.ball_replay_knots import fixes_to_knots  # noqa: E402

from scripts.eval_ball_accuracy import (  # noqa: E402
    _camera_lookup,
    _load_observations,
)

N_FOLDS = 2
BALL_RADIUS_M = 0.11


# ---------------------------------------------------------------------------
# fix split (mirrors prototypes/ball_hybrid_poc/run_all.py's
# _split_fixes_even_odd / _fix_halves_for_fold exactly, duplicated here so
# this script has no dependency on the PoC package when --trajectory hybrid
# is used against the production path)
# ---------------------------------------------------------------------------

def _split_fixes_even_odd(fixes: Sequence[BallFix]) -> tuple[tuple, tuple]:
    ordered = tuple(sorted(fixes, key=lambda fx: fx.frame))
    half_a = tuple(fx for i, fx in enumerate(ordered) if i % 2 == 0)
    half_b = tuple(fx for i, fx in enumerate(ordered) if i % 2 == 1)
    return half_a, half_b


def _halves_for_fold(fold: int, half_a: tuple, half_b: tuple) -> tuple[tuple, tuple]:
    """fold 0 knots half_a / grades half_b; fold 1 the reverse."""
    return (half_a, half_b) if fold == 0 else (half_b, half_a)


# ---------------------------------------------------------------------------
# clip loading (generic --output/--shot, no CLIPS-dict dependency)
# ---------------------------------------------------------------------------

def _load_clip(output_dir: Path, shot_id: str) -> dict[str, Any]:
    cam = CameraTrack.load(output_dir / "camera" / f"{shot_id}_camera_track.json")
    cams, per_K, per_R, per_t, distortion = _camera_lookup(cam)
    anchors = BallAnchorSet.load(output_dir / "ball" / f"{shot_id}_ball_anchors.json")
    raw_obs = _load_observations(
        output_dir / "ball" / f"{shot_id}_ball_observations.json",
        anchors=anchors.anchors)
    fixes_path = output_dir / "ball" / f"{shot_id}_ball_fixes.json"
    fixes = BallFixSet.load(fixes_path).fixes if fixes_path.exists() else ()
    return {
        "cam": cam, "cams": cams, "per_K": per_K, "per_R": per_R,
        "per_t": per_t, "distortion": distortion, "anchors": anchors,
        "raw_obs": raw_obs, "fixes": fixes,
    }


def _build_poc_clip_context(output_dir: Path, shot_id: str, clip: dict[str, Any]):
    """A ``prototypes.ball_hybrid_poc.ctx.ClipContext`` built directly
    from already-loaded data (no dependency on that module's hardcoded
    ``CLIPS`` mapping, so any ``--output``/``--shot`` pair works)."""
    from prototypes.ball_hybrid_poc import ctx as poc_ctx
    from prototypes.ball_hybrid_poc.types import Observation

    observations = tuple(
        Observation(frame=f, uv=uv, conf=conf, source=source)
        for f, uv, conf, source in clip["raw_obs"]
    )
    cam = clip["cam"]
    return poc_ctx.ClipContext(
        clip_id=shot_id,
        output_dir=output_dir,
        shot_id=shot_id,
        fps=float(cam.fps),
        image_size=(int(cam.image_size[0]), int(cam.image_size[1])),
        frames=tuple(sorted(clip["per_K"])),
        n_frames=len(clip["per_K"]),
        per_frame_K=clip["per_K"],
        per_frame_R=clip["per_R"],
        per_frame_t=clip["per_t"],
        distortion=clip["distortion"],
        camera_track=cam,
        anchors=clip["anchors"],
        observations=observations,
        fixes=clip["fixes"],
        video_path=output_dir / "shots" / f"{shot_id}.mp4",
    ), observations


def _run_prototype_track(clip_ctx, observations, anchors, knot_fixes):
    from prototypes.ball_hybrid_poc import hybrid as poc_hybrid
    return poc_hybrid.run_hybrid(clip_ctx, list(observations), list(anchors),
                                  fixes=list(knot_fixes))


def _run_production_track(clip_ctx, observations, anchors, knot_fixes):
    """Best-effort call into the production trajectory layer IC-A is
    building. Its exact call surface isn't fixed by this script (that
    module hadn't landed when this was written) — this tries the same
    ``run_hybrid(ctx, observations, anchors, fixes=..., cfg=...)`` shape
    the PoC uses (``ball_hybrid_types.py``'s docstring describes it as a
    port of the PoC), and raises a clear, actionable error otherwise."""
    try:
        from src.utils import ball_hybrid_trajectory as prod_traj
    except ImportError as exc:
        raise SystemExit(
            "--trajectory hybrid requires src/utils/ball_hybrid_trajectory.py "
            "(IC-A's production module), which is not present yet in this "
            f"checkout ({exc!r}). Re-run with --trajectory prototype for now, "
            "or once `git log` shows a commit adding ball_hybrid_trajectory.py "
            "/ the `ball.trajectory` config key, re-run with --trajectory hybrid."
        ) from exc
    if hasattr(prod_traj, "run_hybrid"):
        return prod_traj.run_hybrid(clip_ctx, list(observations), list(anchors),
                                     fixes=list(knot_fixes))
    if hasattr(prod_traj, "run_trajectory"):
        return prod_traj.run_trajectory(clip_ctx, list(observations), list(anchors),
                                         fixes=list(knot_fixes))
    raise SystemExit(
        "src.utils.ball_hybrid_trajectory exists but exposes neither "
        "run_hybrid() nor run_trajectory() — update this script's "
        "_run_production_track() to match its actual entry point."
    )


def _pctl(vals: Sequence[Optional[float]], q: float) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def _joint_world_fn_for(clip_ctx):
    def _fn(frame, player_id, bone):
        try:
            return clip_ctx.player_context().joint_world(frame, player_id, bone)
        except Exception:  # noqa: BLE001 — degrades to ray-only GT
            return None
    return _fn


def run_eval(output_dir: Path, shot_id: str, *, trajectory: str,
             tol_px: float, adjacent_frames: int) -> dict[str, Any]:
    clip = _load_clip(output_dir, shot_id)
    all_fixes = clip["fixes"]
    if not all_fixes:
        return {
            "shot": shot_id, "output": str(output_dir),
            "status": f"no fixes at {output_dir / 'ball' / (shot_id + '_ball_fixes.json')} "
                      "— nothing to hold out (run the ball stage's triangulation "
                      "pass on a sync-grouped clip first)",
        }

    clip_ctx, observations = _build_poc_clip_context(output_dir, shot_id, clip)
    all_anchors = clip["anchors"].anchors
    cams = clip["cams"]
    distortion = clip["distortion"]
    joint_world_fn = _joint_world_fn_for(clip_ctx)
    run_track = (_run_prototype_track if trajectory == "prototype"
                 else _run_production_track)

    half_a, half_b = _split_fixes_even_odd(all_fixes)

    fix_err_vals: list[float] = []
    fold_detail: dict[str, Any] = {}
    anchor_rows_with: list = []
    anchor_rows_without: list = []
    n_dropped_total = 0
    drop_reasons: dict[str, int] = {}

    for fold in range(N_FOLDS):
        kept, held = BE.split_anchors(all_anchors, fold=fold, n_folds=N_FOLDS)
        held_frames = frozenset(a.frame for a in held)
        knot_fixes_raw, graded_fixes = _halves_for_fold(fold, half_a, half_b)

        knots, dropped = fixes_to_knots(
            knot_fixes_raw, all_anchors, cams,
            tol_px=tol_px, adjacent_frames=adjacent_frames,
            distortion=distortion,
        )
        n_dropped_total += len(dropped)
        for d in dropped:
            drop_reasons[d["reason"]] = drop_reasons.get(d["reason"], 0) + 1

        track_with = run_track(clip_ctx, observations, kept, knots)
        world_with = {tf.frame: tf.xyz for tf in track_with.frames
                      if tf.xyz is not None}

        fx_rows = BE.eval_rows_at_fixes(
            world_with,
            [(fx.frame, fx.xyz, fx.ray_miss_m) for fx in graded_fixes],
        )
        fold_fix_errs = [r.err_3d_m for r in fx_rows if r.err_3d_m is not None]
        fix_err_vals.extend(fold_fix_errs)

        anchor_rows_with_fold = BE.eval_rows_at_anchors(
            world_with, all_anchors, cams, ball_radius=BALL_RADIUS_M,
            distortion=distortion, joint_world_fn=joint_world_fn,
            held_out_frames=held_frames,
            evidence_frames=frozenset(o.frame for o in observations))
        anchor_rows_with.extend(r for r in anchor_rows_with_fold if r.held_out)

        track_without = run_track(clip_ctx, observations, kept, [])
        world_without = {tf.frame: tf.xyz for tf in track_without.frames
                         if tf.xyz is not None}
        anchor_rows_without_fold = BE.eval_rows_at_anchors(
            world_without, all_anchors, cams, ball_radius=BALL_RADIUS_M,
            distortion=distortion, joint_world_fn=joint_world_fn,
            held_out_frames=held_frames,
            evidence_frames=frozenset(o.frame for o in observations))
        anchor_rows_without.extend(r for r in anchor_rows_without_fold if r.held_out)

        fold_detail[f"fold{fold}"] = {
            "n_knot_fixes_input": len(knot_fixes_raw),
            "n_knot_fixes_gated": len(knots),
            "n_dropped": len(dropped),
            "n_graded_fixes": len(graded_fixes),
            "fix_err_p50_m": _pctl(fold_fix_errs, 50),
        }

    def _anchor_err(r):
        return r.err_3d_m if r.err_3d_m is not None else r.lateral_m

    errs_with = [_anchor_err(r) for r in anchor_rows_with if _anchor_err(r) is not None]
    errs_without = [_anchor_err(r) for r in anchor_rows_without if _anchor_err(r) is not None]

    return {
        "shot": shot_id,
        "output": str(output_dir),
        "trajectory": trajectory,
        "tol_px": tol_px,
        "adjacent_frames": adjacent_frames,
        "n_fixes_total": len(all_fixes),
        "n_dropped_total": n_dropped_total,
        "drop_reasons": drop_reasons,
        "fix_holdout": {
            "n": len(fix_err_vals),
            "p50_m": _pctl(fix_err_vals, 50),
            "p95_m": _pctl(fix_err_vals, 95),
            "max_m": (max(fix_err_vals) if fix_err_vals else None),
            "meets_0_8m_bar": (
                _pctl(fix_err_vals, 50) is not None
                and _pctl(fix_err_vals, 50) <= 0.8
            ),
        },
        "anchor_holdout_with_fixes": {
            "n": len(errs_with), "p50_m": _pctl(errs_with, 50),
            "p95_m": _pctl(errs_with, 95),
        },
        "anchor_holdout_without_fixes": {
            "n": len(errs_without), "p50_m": _pctl(errs_without, 50),
            "p95_m": _pctl(errs_without, 95),
        },
        "fixes_no_worse_for_anchors": (
            _pctl(errs_with, 50) is not None and _pctl(errs_without, 50) is not None
            and _pctl(errs_with, 50) <= _pctl(errs_without, 50) + 1e-9
        ),
        "fold_detail": fold_detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="pipeline output dir")
    ap.add_argument("--shot", required=True, help="shot id (e.g. origi01)")
    ap.add_argument("--trajectory", choices=("prototype", "hybrid"),
                    default="prototype")
    ap.add_argument("--tol-px", type=float, default=40.0,
                    help="fixes_to_knots operator-conflict pixel tolerance")
    ap.add_argument("--adjacent-frames", type=int, default=1)
    ap.add_argument("--json", type=Path, default=None,
                    help="optional path to dump the full result JSON")
    args = ap.parse_args()

    result = run_eval(
        Path(args.output), args.shot, trajectory=args.trajectory,
        tol_px=args.tol_px, adjacent_frames=args.adjacent_frames,
    )

    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

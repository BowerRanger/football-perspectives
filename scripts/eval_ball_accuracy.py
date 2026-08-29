"""Grade the ball stage's dense track against ground truth.

Sub-20cm accuracy campaign harness (spec §4,
docs/superpowers/specs/2026-08-17-ball-sub20cm-accuracy-design.md).

Re-runs the ball stage inside a temp OVERLAY of an output dir — every stage
input symlinked, ``ball/`` replaced with a real dir holding the (optionally
hold-out-filtered) manual anchors — so the real outputs are never touched.
Grades the produced dense track at manual-anchor frames (A1), cross-replay
3-D fixes (A2), detection rays (A3), and with the naturalness validator (A4).

Usage:
    .venv311/bin/python scripts/eval_ball_accuracy.py \
        --output output --shot gberch --holdout --detector noop \
        --json docs/superpowers/notes/ball-accuracy/gberch_holdout.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_fixes import BallFixSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils import ball_eval as BE  # noqa: E402
from src.utils.ball_detection_cache import (  # noqa: E402
    CachingBallDetector as CachingDetector,
)
from src.utils.ball_detector import BallDetector  # noqa: E402

# Top-level entries never linked into an overlay: ball outputs (replaced by
# the overlay's own ball/), heavyweight render/export artifacts, and logs
# (recreated empty so stage logging cannot write into the source dir).
_SKIP_PREFIXES = ("ball",)
_SKIP_NAMES = ("renders", "logs", "export")


class NoopDetector(BallDetector):
    """No detections: anchored frames still reconstruct (anchor-driven)."""

    SUPPORTS_REDETECT = False  # no heatmap to zoom on

    def detect(self, frame):  # noqa: ANN001
        return None


def build_overlay(src_output: Path, tmp_root: Path, shot_id: str,
                  kept: BallAnchorSet | None) -> Path:
    """Create the overlay dir: symlinked inputs + a real ``ball/``."""
    src_output = Path(src_output)
    ov = Path(tmp_root) / "overlay"
    ov.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_output.iterdir()):
        name = entry.name
        if name in _SKIP_NAMES or any(
                name == p or name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        link = ov / name
        if not link.exists():
            link.symlink_to(entry.resolve())
    (ov / "logs").mkdir(exist_ok=True)
    (ov / "ball").mkdir(exist_ok=True)
    anchors_src = src_output / "ball" / f"{shot_id}_ball_anchors.json"
    dst = ov / "ball" / f"{shot_id}_ball_anchors.json"
    if kept is not None:
        kept.save(dst)
    elif anchors_src.exists():
        dst.write_text(anchors_src.read_text())
    return ov


def _make_detector(kind: str, ball_cfg: dict,
                   cache_path: Path | None = None) -> BallDetector:
    if kind == "noop":
        return NoopDetector()
    if kind == "wasb":
        from src.stages.ball import _build_detector
        det = _build_detector(ball_cfg)
        if cache_path is not None:
            det = CachingDetector(det, cache_path)
        return det
    raise ValueError(f"unknown detector kind: {kind!r}")


def _camera_lookup(cam: CameraTrack):
    """Per-frame ``(K, R, t)`` with the clip-shared ``t_world`` fallback."""
    t_fallback = np.asarray(cam.t_world, dtype=float)
    cams = {}
    per_K, per_R, per_t = {}, {}, {}
    for f in cam.frames:
        K = np.asarray(f.K, dtype=float)
        R = np.asarray(f.R, dtype=float)
        t = np.asarray(f.t, dtype=float) if f.t is not None else t_fallback
        cams[f.frame] = (K, R, t)
        per_K[f.frame], per_R[f.frame], per_t[f.frame] = K, R, t
    return cams, per_K, per_R, per_t, tuple(cam.distortion)


def _joint_world_fn(overlay: Path, shot_id: str, per_K, per_R, per_t,
                    distortion):
    """PlayerContext-backed joint lookup; None when context unavailable."""
    try:
        from src.utils.ball_player_context import PlayerContext
        ctx = PlayerContext.load(
            overlay, shot_id, per_frame_K=per_K, per_frame_R=per_R,
            per_frame_t=per_t, distortion=distortion,
        )
        return ctx.joint_world
    except Exception as exc:  # noqa: BLE001 — GT degrades to ray-only
        print(f"  (no player context: {exc}; touch GT degrades to ray-only)")
        return None


def _load_fixes(path: Path):
    if not path.exists():
        return []
    fs = BallFixSet.load(path)
    return [(fx.frame, fx.xyz, fx.ray_miss_m) for fx in fs.fixes]


_DENSE_EVAL_SOURCES = {"detector", "second_pass", "foot_guided"}
# A detection within this many frames of a manual anchor but farther than
# this many pixels from the operator's click is overruled by the operator
# — it is a known-false detection (e.g. gberch's post-kick static cluster
# at f52/53 contradicting the clicks at f51/55) and must not serve as GT.
_ANCHOR_VETO_FRAMES = 2
_ANCHOR_VETO_PX = 60.0


def _load_observations(path: Path, anchors=None):
    """Real-detector observations from the ``frames`` sidecar payload.

    Anchor-sourced and gap-fill entries are excluded: grading the track
    against evidence it was pinned to (or that was synthesized) would
    flatter A3. Detections the operator's adjacent clicks contradict are
    excluded as known-false GT (clicks are ground truth, detections are
    not — this uses anchors only, never the track, so it cannot flatter).
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for e in data.get("frames", []):
        uv = e.get("uv")
        if (uv is None or e.get("frame") is None or e.get("gap_fill")
                or str(e.get("source")) not in _DENSE_EVAL_SOURCES):
            continue
        out.append((int(e["frame"]), (float(uv[0]), float(uv[1])),
                    float(e.get("confidence", 0.0)),
                    str(e.get("source"))))
    if anchors:
        clicks = sorted((a.frame, a.image_xy) for a in anchors
                        if a.image_xy is not None)
        # Expected click-path pixel per frame: exact at anchor frames,
        # linearly interpolated between anchors ≤6 frames apart (fast
        # motion between close clicks is thereby handled correctly).
        expected: dict[int, tuple[float, float]] = {
            f: (float(uv[0]), float(uv[1])) for f, uv in clicks}
        for (fa, ua), (fb, ub) in zip(clicks, clicks[1:]):
            if 0 < fb - fa <= 6:
                for f in range(fa + 1, fb):
                    s = (f - fa) / (fb - fa)
                    expected[f] = (
                        float(ua[0]) + (float(ub[0]) - float(ua[0])) * s,
                        float(ua[1]) + (float(ub[1]) - float(ua[1])) * s)

        def _vetoed(f, uv):
            exp = expected.get(f)
            if exp is None:
                return False
            return ((uv[0] - exp[0]) ** 2 + (uv[1] - exp[1]) ** 2
                    > _ANCHOR_VETO_PX ** 2)

        out = [o for o in out if not _vetoed(o[0], o[1])]
    return out


def _event_frames(keyframes_path: Path) -> set[int]:
    """Frames where the path may legitimately bend: every keyframe and
    segment boundary of the produced sparse set."""
    if not keyframes_path.exists():
        return set()
    data = json.loads(keyframes_path.read_text())
    ev = {int(k["frame"]) for k in data.get("keyframes", [])}
    for s in data.get("segments", []):
        ev.add(int(s["start_frame"]))
        ev.add(int(s["end_frame"]))
    return ev


def _run_fold(src_output: Path, shot_id: str, config: dict, detector: str,
              kept: BallAnchorSet | None, tmp_root: Path,
              det_cache: Path | None = None):
    """Run the ball stage once in an overlay; return produced artifact paths."""
    from src.stages.ball import BallStage

    overlay = build_overlay(src_output, tmp_root, shot_id, kept)
    det = _make_detector(detector, config["ball"], cache_path=det_cache)
    clip = overlay / "shots" / f"{shot_id}.mp4"
    cam_path = overlay / "camera" / f"{shot_id}_camera_track.json"
    track_out = overlay / "ball" / f"{shot_id}_ball_track.json"
    stage = BallStage(config=config, output_dir=overlay, ball_detector=det)
    stage._run_shot(shot_id, clip, cam_path, track_out, config["ball"], det)
    if isinstance(det, CachingDetector):
        det.save()
    return overlay, track_out


def run_and_evaluate(src_output: Path, shot_id: str, *, detector: str,
                     holdout: bool, n_folds: int, config: dict,
                     fixes_path: Path | None = None,
                     threshold_m: float = 0.20,
                     det_cache: Path | None = None) -> dict:
    src_output = Path(src_output)
    anchors_file = src_output / "ball" / f"{shot_id}_ball_anchors.json"
    full_set = BallAnchorSet.load(anchors_file)
    cam = CameraTrack.load(
        src_output / "camera" / f"{shot_id}_camera_track.json")
    cams, per_K, per_R, per_t, distortion = _camera_lookup(cam)
    ball_radius = float(config["ball"].get("ball_radius_m", 0.11))

    fixes_file = fixes_path or (
        src_output / "ball" / f"{shot_id}_ball_fixes.json")
    fixes = _load_fixes(Path(fixes_file))

    folds = list(range(n_folds)) if holdout else [None]
    anchor_rows: list = []
    fix_rows: list = []
    dense_rows: list = []
    violations: list = []
    per_fold = []
    recall: dict = {}

    for fold in folds:
        if fold is None:
            kept_set, held = None, ()
        else:
            kept, held = BE.split_anchors(full_set.anchors, fold=fold,
                                          n_folds=n_folds)
            kept_set = dataclasses.replace(full_set, anchors=tuple(kept))
        held_frames = frozenset(a.frame for a in held)
        with tempfile.TemporaryDirectory(prefix="ball_eval_") as tmp:
            overlay, track_out = _run_fold(
                src_output, shot_id, config, detector, kept_set, Path(tmp),
                det_cache=det_cache)
            track = BallTrack.load(track_out)
            world = {f.frame: f.world_xyz for f in track.frames
                     if f.world_xyz is not None}
            jw = _joint_world_fn(overlay, shot_id, per_K, per_R, per_t,
                                 distortion)
            fold_obs = _load_observations(
                overlay / "ball" / f"{shot_id}_ball_observations.json",
                anchors=full_set.anchors)
            rows = BE.eval_rows_at_anchors(
                world, full_set.anchors, cams, ball_radius=ball_radius,
                distortion=distortion, joint_world_fn=jw,
                held_out_frames=held_frames,
                evidence_frames=frozenset(o[0] for o in fold_obs))
            # Aggregate: held-out rows come from their own fold; kept rows
            # (and fixes/dense/naturalness) only from the first fold so
            # nothing is double-counted.
            first = fold == folds[0]
            anchor_rows.extend(r for r in rows if r.held_out or first)
            if first:
                fix_rows.extend(BE.eval_rows_at_fixes(world, fixes))
                obs = _load_observations(
                    overlay / "ball" / f"{shot_id}_ball_observations.json",
                    anchors=full_set.anchors)
                dense_rows.extend(BE.dense_lateral_rows(
                    world, obs, cams, distortion=distortion,
                    min_confidence=0.5))
                ev = _event_frames(
                    overlay / "ball" / f"{shot_id}_ball_keyframes.json")
                violations.extend(BE.naturalness_violations(
                    track.frames, ev, float(track.fps)))
            if first:
                # A5 support: auto-touch recall against the FULL manual set
                # (relaxed = frame-window only; strict = bone must agree).
                try:
                    from src.utils.ball_touch_recall import (
                        match_touches, touches_from_anchor_set)
                    man = touches_from_anchor_set(anchors_file)
                    aut = touches_from_anchor_set(
                        overlay / "ball" / f"{shot_id}_ball_anchors_auto.json")
                    recall = {
                        "strict": match_touches(man, aut, require_bone=True),
                        "relaxed": match_touches(man, aut,
                                                 require_bone=False),
                    }
                except Exception as exc:  # noqa: BLE001 — report-only
                    recall = {"error": str(exc)}
            per_fold.append({
                "fold": fold,
                "n_kept": len(kept_set.anchors) if kept_set else
                          len(full_set.anchors),
                "n_held_out": len(held_frames),
            })

    summary = BE.summarize(anchor_rows, fix_rows, dense_rows, violations,
                           threshold_m=threshold_m)
    held_by_state: dict[str, dict] = {}
    for state in sorted({r.state for r in anchor_rows if r.held_out}):
        rows = [r for r in anchor_rows if r.held_out and r.state == state]
        errs = [r.err_3d_m if r.err_3d_m is not None else r.lateral_m
                for r in rows]
        held_by_state[state] = BE._stats(
            [e for e in errs if e is not None], threshold_m,
            n_missing=sum(1 for e in errs if e is None))
    summary["held_out_by_state"] = held_by_state
    worst = sorted(
        (r for r in anchor_rows if r.held_out
         and (r.err_3d_m or r.lateral_m) is not None),
        key=lambda r: -(r.err_3d_m if r.err_3d_m is not None
                        else r.lateral_m))[:10]
    return {
        "clip": shot_id,
        "output_dir": str(src_output),
        "detector": detector,
        "holdout": holdout,
        "n_folds": n_folds if holdout else 0,
        "summary": summary,
        "touch_recall": recall,
        "per_fold": per_fold,
        "worst_held_out": [dataclasses.asdict(r) for r in worst],
        "violations": [dataclasses.asdict(v) for v in violations],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--detector", choices=("noop", "wasb"), default="noop")
    ap.add_argument("--fixes", type=Path, default=None,
                    help="BallFixSet JSON to grade against (A2)")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full report JSON here")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config" / "default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="dotted config override, e.g. "
                         "ball.kinematic_touch.enabled=false")
    ap.add_argument("--det-cache", type=Path, default=None,
                    help="content-hash detection cache file (pay detector "
                         "inference once per clip; reruns replay)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    for kv in args.set:
        key, _, raw = kv.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    rep = run_and_evaluate(
        args.output, args.shot, detector=args.detector,
        holdout=args.holdout, n_folds=args.n_folds, config=config,
        fixes_path=args.fixes, threshold_m=args.threshold,
        det_cache=args.det_cache)

    s = rep["summary"]

    def fmt(sec):
        d = s[sec]
        if d["n"] == 0:
            return f"{sec}: n=0 (missing={d.get('n_missing', 0)})"
        return (f"{sec}: n={d['n']} p50={d['p50']:.3f} p95={d['p95']:.3f} "
                f"max={d['max']:.3f} >thr={d['n_over']}"
                f" missing={d.get('n_missing', 0)}")

    print(f"\n=== {rep['clip']} ({rep['detector']}"
          f"{', holdout' if rep['holdout'] else ''}) "
          f"threshold={s['threshold_m']} ===")
    for sec in ("anchors_held_out", "held_out_evidenced",
                "held_out_unevidenced", "anchors_kept", "fixes", "dense"):
        print("  " + fmt(sec))
    nat = s["naturalness"]
    print(f"  naturalness: {nat['n_violations']} violations {nat['by_kind']}")
    if rep["worst_held_out"]:
        print("  worst held-out:")
        for r in rep["worst_held_out"][:5]:
            err = r["err_3d_m"] if r["err_3d_m"] is not None else r["lateral_m"]
            tag = "3d" if r["err_3d_m"] is not None else "lat"
            print(f"    f{r['frame']:>4} {r['state']:<14} {tag}="
                  f"{err:.3f}m ({r['kind']})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=1))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()

"""Per-cue and fused precision/recall/F1 for the single-camera event-cue
modules (audio onset, net energy, blur direction-change) against manual
EVENT anchors, at +-``--tol`` frames (default 3). Also reports recall of
manual events the ball stage's EXISTING auto-anchor layer misses -- the
valuable part, since the cues are only worth wiring in if they catch
events the current pipeline doesn't.

Reads read-only from the main repo's per-clip output dirs (see ``CLIPS``
below); writes cached per-cue outputs and a summary under
``--cache-dir`` (default ``output-ball-poc``, a scratch dir -- never
under a real pipeline output dir).

Usage:
    .venv311/bin/python scripts/eval_ball_event_cues.py \\
        --clips gberch origi01 kroupi01 s013

    # single clip, tighter/looser match tolerance
    .venv311/bin/python scripts/eval_ball_event_cues.py --clips gberch --tol 2

    # marginal-contribution ablation (every audio/net/blur subset's fused
    # score) -- requires cues_{audio,net,blur}.json to already be cached
    # for the clip (i.e. run once without --ablate-only first)
    .venv311/bin/python scripts/eval_ball_event_cues.py --clips gberch --ablate-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.ball_cue_audio import compute_audio_cues  # noqa: E402
from src.utils.ball_cue_blur import compute_blur_cues  # noqa: E402
from src.utils.ball_cue_common import (  # noqa: E402
    iter_video_frames,
    load_auto_event_frames,
    load_camera_frames,
    load_manual_events,
    load_player_boxes,
    load_real_observations,
    probe_speed_factor,
)
from src.utils.ball_cue_fusion import fuse_cues  # noqa: E402
from src.utils.ball_cue_net import net_energy_onsets, net_energy_series  # noqa: E402
from src.utils.goal_geometry import GoalGeometry  # noqa: E402

M = "/Users/joebower/workplace/football-perspectives"
CLIPS: dict[str, tuple[str, str]] = {
    "gberch": (f"{M}/output", "gberch"),
    "origi01": (f"{M}/output-origi-global", "origi01"),
    "kroupi01": (f"{M}/output-kroupi", "kroupi01"),
    "s013": (f"{M}/output-japan", "s013"),
}

_GEOMETRY = GoalGeometry.from_pitch_config({
    "length_m": 105.0, "width_m": 68.0, "goal_height_m": 2.44,
    "goal_width_m": 7.32, "goal_depth_m": 1.5,
})


def _match(candidate_frames: list[int], truth_frames: list[int], *, tol: int) -> tuple[int, int, int]:
    """Greedy 1:1 frame matching within +-``tol``. Returns (tp, fp, fn)."""
    truth_sorted = sorted(truth_frames)
    claimed = [False] * len(truth_sorted)
    tp = 0
    for cf in sorted(candidate_frames):
        best_j, best_d = -1, tol + 1
        for j, tf in enumerate(truth_sorted):
            if claimed[j]:
                continue
            d = abs(cf - tf)
            if d <= tol and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            claimed[best_j] = True
            tp += 1
    fp = len(candidate_frames) - tp
    fn = len(truth_sorted) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def _score(
    frames: list[int], manual_frames: list[int], auto_missed: list[int], *, tol: int,
) -> dict:
    tp, fp, fn = _match(frames, manual_frames, tol=tol)
    p, r, f1 = _prf(tp, fp, fn)
    tp_m, fp_m, fn_m = _match(frames, auto_missed, tol=tol)
    _p_m, r_m, _f1_m = _prf(tp_m, fp_m, fn_m)
    return {
        "n": len(frames), "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "auto_missed_tp": tp_m, "auto_missed_n": len(auto_missed),
        "auto_missed_recall": r_m,
    }


def _sequential_pairs(video_path: Path):
    """Yield ``(idx, prev_bgr, cur_bgr)`` for consecutive decoded frames."""
    prev = None
    prev_idx = None
    for idx, frame_bgr in iter_video_frames(video_path):
        if prev is not None and idx == prev_idx + 1:
            yield idx, prev, frame_bgr
        prev, prev_idx = frame_bgr, idx


def run_clip(clip_id: str, cache_dir: Path, *, tol: int = 3) -> dict:
    output_dir_s, shot_id = CLIPS[clip_id]
    output_dir = Path(output_dir_s)
    out_cache = cache_dir / clip_id
    out_cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cam = load_camera_frames(output_dir, shot_id)
    manual_events = load_manual_events(output_dir, shot_id)
    manual_frames = [f for f, _s in manual_events]
    auto_frames = load_auto_event_frames(output_dir, shot_id)
    auto_missed = [f for f in manual_frames
                   if not any(abs(f - af) <= tol for af in auto_frames)]
    speed_factor = probe_speed_factor(output_dir, shot_id)
    video_path = output_dir / "shots" / f"{shot_id}.mp4"

    result: dict = {
        "clip_id": clip_id, "n_manual_events": len(manual_events),
        "manual_event_states": sorted({s for _f, s in manual_events}),
        "n_auto_events": len(auto_frames), "n_auto_missed": len(auto_missed),
        "speed_factor": speed_factor,
    }

    # ---- audio ----
    t_audio0 = time.time()
    calib, audio_events = compute_audio_cues(
        video_path, cam.fps, manual_frames, speed_factor=speed_factor)
    audio_runtime = time.time() - t_audio0
    (out_cache / "cues_audio.json").write_text(json.dumps({
        "calibration": asdict(calib),
        "events": [asdict(e) for e in audio_events],
    }, indent=2))

    # ---- net ----
    t_net0 = time.time()
    player_boxes = load_player_boxes(output_dir, shot_id)
    cameras_by_frame = {
        f: (cam.per_frame_K[f], cam.per_frame_R[f], cam.per_frame_t[f], cam.distortion)
        for f in cam.frames
    }
    net_series = net_energy_series(
        _sequential_pairs(video_path), cameras_by_frame, _GEOMETRY, cam.image_size,
        player_boxes=player_boxes)
    net_events = net_energy_onsets(net_series, cameras_by_frame, _GEOMETRY)
    net_runtime = time.time() - t_net0
    (out_cache / "cues_net.json").write_text(json.dumps({
        "n_frames_in_view": len(net_series),
        "events": [asdict(e) for e in net_events],
    }, indent=2))

    # ---- blur ----
    t_blur0 = time.time()
    observations = load_real_observations(output_dir, shot_id)
    detections = [(f, uv) for f, uv, _conf, _src in observations]
    needed = {f for f, _uv in detections}
    frame_cache: dict[int, np.ndarray] = {}
    for idx, frame_bgr in iter_video_frames(video_path):
        if idx in needed:
            frame_cache[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_events = compute_blur_cues(
        detections, frame_cache.get, fps=cam.fps)
    blur_runtime = time.time() - t_blur0
    (out_cache / "cues_blur.json").write_text(json.dumps({
        "events": [asdict(e) for e in blur_events],
    }, indent=2))

    # ---- fusion ----
    cue_evidence = {"audio": audio_events, "net": net_events, "blur": blur_events}
    fused = fuse_cues(cue_evidence, auto_frames)
    (out_cache / "cues_fused.json").write_text(json.dumps(
        [asdict(f) for f in fused], indent=2))

    result["audio"] = _score([e.frame for e in audio_events], manual_frames,
                              auto_missed, tol=tol)
    result["audio"]["calibration"] = asdict(calib)
    result["audio"]["runtime_s"] = audio_runtime
    result["net"] = _score([e.frame for e in net_events], manual_frames,
                            auto_missed, tol=tol)
    result["net"]["runtime_s"] = net_runtime
    result["net"]["n_frames_in_view"] = len(net_series)
    result["blur"] = _score([e.frame for e in blur_events], manual_frames,
                             auto_missed, tol=tol)
    result["blur"]["runtime_s"] = blur_runtime
    result["fused"] = _score([f.frame for f in fused], manual_frames,
                              auto_missed, tol=tol)
    result["total_runtime_s"] = time.time() - t0
    return result


def _print_result(res: dict) -> None:
    print(f"=== {res['clip_id']} "
          f"(n_manual_events={res['n_manual_events']} "
          f"n_auto_events={res['n_auto_events']} "
          f"n_auto_missed={res['n_auto_missed']} "
          f"speed_factor={res['speed_factor']}) ===")
    for cue in ("audio", "net", "blur", "fused"):
        s = res[cue]
        print(f"  {cue:6s}  n={s['n']:3d} tp={s['tp']:3d} fp={s['fp']:3d} "
              f"fn={s['fn']:3d}  P={s['precision']:.2f} R={s['recall']:.2f} "
              f"F1={s['f1']:.2f}  auto-missed R={s['auto_missed_recall']:.2f} "
              f"({s['auto_missed_tp']}/{s['auto_missed_n']})  "
              f"{s.get('runtime_s', 0):.1f}s")
    calib = res["audio"]["calibration"]
    if calib["enabled"]:
        print(f"  audio latency: {calib['latency_ms']:.1f}ms "
              f"(n_matched={calib['n_matched']}, std={calib['match_std_ms']:.1f}ms)")
    else:
        print(f"  audio DISABLED: {calib['reason']}")
    print(f"  total runtime: {res['total_runtime_s']:.1f}s")


def _load_cached_cue(cache_dir: Path, clip_id: str, name: str) -> list:
    from src.utils.ball_hybrid_types import CueEvidence
    path = cache_dir / clip_id / f"cues_{name}.json"
    data = json.loads(path.read_text())
    events = data["events"] if isinstance(data, dict) else data
    return [CueEvidence(**e) for e in events]


def run_ablation(clip_id: str, cache_dir: Path, *, tol: int = 3) -> None:
    """Every non-empty subset of {audio, net, blur} fused and scored, to
    isolate each cue's MARGINAL contribution (a single-cue subset can
    only ever produce ``support="cue+auto"`` fused events, since there's
    no second cue to pair with -- this is exactly the "cue alone, only
    when an existing auto anchor corroborates it" arm of the policy).
    Reads the cached ``cues_{audio,net,blur}.json`` this clip's ``run_clip``
    already wrote; run without ``--ablate-only`` first if they're missing.
    """
    output_dir_s, shot_id = CLIPS[clip_id]
    output_dir = Path(output_dir_s)
    manual_events = load_manual_events(output_dir, shot_id)
    manual_frames = [f for f, _s in manual_events]
    auto_frames = load_auto_event_frames(output_dir, shot_id)
    auto_missed = [f for f in manual_frames
                   if not any(abs(f - af) <= tol for af in auto_frames)]
    cues = {n: _load_cached_cue(cache_dir, clip_id, n) for n in ("audio", "net", "blur")}

    print(f"=== {clip_id} ablation (manual={len(manual_frames)} "
          f"auto_missed={len(auto_missed)}) ===")
    for r in range(1, 4):
        for combo in itertools.combinations(["audio", "net", "blur"], r):
            evidence = {k: cues[k] for k in combo}
            fused = fuse_cues(evidence, auto_frames)
            frames = [f.frame for f in fused]
            s = _score(frames, manual_frames, auto_missed, tol=tol)
            print(f"  {'+'.join(combo):18s} n={s['n']:3d} P={s['precision']:.2f} "
                  f"R={s['recall']:.2f} F1={s['f1']:.2f}  auto-missed "
                  f"R={s['auto_missed_recall']:.2f} "
                  f"({s['auto_missed_tp']}/{s['auto_missed_n']})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=list(CLIPS), choices=list(CLIPS))
    ap.add_argument("--cache-dir", type=Path, default=Path(f"{M}/output-ball-poc"))
    ap.add_argument("--tol", type=int, default=3,
                    help="frame tolerance for candidate-vs-manual matching")
    ap.add_argument("--ablate-only", action="store_true",
                    help="skip re-running cues; score cached cues_*.json "
                         "subsets instead (see run_ablation)")
    args = ap.parse_args()

    if args.ablate_only:
        for clip_id in args.clips:
            run_ablation(clip_id, args.cache_dir, tol=args.tol)
        return 0

    all_results = []
    for clip_id in args.clips:
        res = run_clip(clip_id, args.cache_dir, tol=args.tol)
        all_results.append(res)
        _print_result(res)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "event_cues_summary.json").write_text(
        json.dumps(all_results, indent=2))
    print(f"\nsummary written to {args.cache_dir / 'event_cues_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""WS6 continuation go/no-go probe: gberch f343 strike window.

Bounded real-detector smoke test for the new static-lock filter +
kinematic chain search (src/utils/ball_strike_window.py). Loads the
existing gberch observations sidecar (pass-1 track from a prior
production run), demotes the static-lock signature, selects the strike
window around the f343 strike, gathers FULL-FRAME low-threshold
candidates from the real fine-tuned WASB detector across that window
only (bounded: window_radius_frames on each side of the trigger, ~17-20
detector calls), and runs the kinematic chain search — printing the
recovered chain against the baseline sidecar values so the result can be
judged by inspection, not by eyeballing a full pipeline run.

Read-only against the main repo (clip / checkpoint / observations
sidecar); writes nothing outside this worktree.

Usage:
    .venv311/bin/python scripts/probe_strike_window_f343.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.ball_strike_window import (  # noqa: E402
    StrikeWindow,
    StrikeWindowCfg,
    apply_chain_detections,
    find_static_lock_frames,
    find_strike_triggers,
    flanking_knots,
    select_kinematic_chain,
    select_strike_windows,
)

MAIN_REPO = Path("/Users/joebower/workplace/football-perspectives")


def _load_observations(path: Path) -> tuple[
    dict[int, tuple[float, float] | None], dict[int, str], dict[int, float], int,
]:
    payload = json.loads(path.read_text())
    uvs: dict[int, tuple[float, float] | None] = {}
    sources: dict[int, str] = {}
    confs: dict[int, float] = {}
    for row in payload["frames"]:
        f = row["frame"]
        uvs[f] = tuple(row["uv"]) if row["uv"] is not None else None
        sources[f] = row.get("source", "none")
        confs[f] = float(row.get("confidence", 0.0))
    n_frames = max(uvs) + 1
    return uvs, sources, confs, n_frames


def _candidates_for_window(
    clip_path: Path, window, detector, cfg: StrikeWindowCfg,
) -> dict[int, list[tuple[float, float, float]]]:
    """Mirrors BallStage._strike_window_candidates_loop exactly (full
    frame, no crop): primes the detector's temporal buffer starting
    _frames_in - 1 frames before the window, then gathers low-threshold
    candidates for every frame in [window.start, window.end]."""
    out: dict[int, list[tuple[float, float, float]]] = {}
    prime_offset = getattr(detector, "_frames_in", 3) - 1
    prime = max(0, window.start - prime_offset)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, prime)
        detector.reset()
        for f in range(prime, window.end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            cands = detector.detect_candidates(
                frame, cfg.candidate_min_score, cfg.top_k,
            )
            if f >= window.start:
                out[f] = cands
        detector.reset()
    finally:
        detector.reset()
        cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clip", default=str(MAIN_REPO / "output" / "shots" / "gberch.mp4"))
    ap.add_argument(
        "--obs",
        default=str(MAIN_REPO / "output" / "ball" / "gberch_ball_observations.json"))
    ap.add_argument(
        "--checkpoint",
        default=str(
            MAIN_REPO / "third_party" / "wasb_sbdt" / "pretrained_weights"
            / "wasb_soccer_finetuned_v1.pth.tar"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--target-frame", type=int, default=343)
    ap.add_argument(
        "--force-window", default=None,
        help="start,end[,trigger] -- bypass the max_windows_per_shot "
             "ranking/cap and probe this exact window directly (the "
             "trigger-frame ranking scan over the whole clip may not "
             "select the historically-documented window if other breaks "
             "elsewhere in the clip have larger |dv|).")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    obs_path = Path(args.obs)
    assert clip_path.exists(), f"clip not found: {clip_path}"
    assert obs_path.exists(), f"observations sidecar not found: {obs_path}"

    uvs, sources, confs, n_frames = _load_observations(obs_path)
    cfg = StrikeWindowCfg(enabled=True)  # module/config defaults

    static_lock = find_static_lock_frames(uvs, sources, n_frames, cfg=cfg)
    print(f"[static_lock] flagged {len(static_lock)} frame(s): {sorted(static_lock)}")
    demoted_sources = dict(sources)
    for f in static_lock:
        demoted_sources[f] = "static_lock"

    windows = select_strike_windows(uvs, n_frames, cfg)
    print(f"[windows] {len(windows)} window(s) selected under the "
          f"default max_windows_per_shot={cfg.max_windows_per_shot} cap "
          f"(ranked strongest-|dv|-first over the WHOLE clip):")
    for w in windows:
        print(f"    trigger={w.trigger_frame} start={w.start} end={w.end} "
              f"dspeed_px={w.dspeed_px:.1f}")

    all_triggers = sorted(
        find_strike_triggers(uvs, n_frames, cfg), key=lambda t: -t[1])
    f343_rank = next(
        (i for i, (f, _) in enumerate(all_triggers) if f == args.target_frame),
        None)
    if f343_rank is not None:
        print(f"[triggers] frame {args.target_frame}'s own trigger ranks "
              f"#{f343_rank + 1}/{len(all_triggers)} by |dv| clip-wide "
              f"(dspeed_px={all_triggers[f343_rank][1]:.1f}) -- "
              f"{'inside' if f343_rank < cfg.max_windows_per_shot else 'OUTSIDE'} "
              f"the default top-{cfg.max_windows_per_shot} budget.")

    if args.force_window:
        parts = [int(x) for x in args.force_window.split(",")]
        start, end = parts[0], parts[1]
        trigger = parts[2] if len(parts) > 2 else (start + end) // 2
        dspeed = next((dv for f, dv in all_triggers if f == trigger), 0.0)
        window = StrikeWindow(
            trigger_frame=trigger, start=start, end=end, dspeed_px=dspeed)
        print(f"\n[force-window] overriding selection with start={start} "
              f"end={end} trigger={trigger} dspeed_px={dspeed:.1f}")
    else:
        target = [
            w for w in windows
            if w.start <= args.target_frame <= w.end
        ]
        if not target:
            target = sorted(
                windows,
                key=lambda w: abs(w.trigger_frame - args.target_frame))[:1]
        if not target:
            print("No strike window selected at all -- nothing to probe.")
            return
        window = target[0]
    print(f"\n[target window] start={window.start} end={window.end} "
          f"trigger={window.trigger_frame} dspeed_px={window.dspeed_px:.1f}")

    pre, post = flanking_knots(
        uvs, demoted_sources, window, n_frames,
        lookback=cfg.velocity_lookback_frames,
    )
    print(f"[flanking_knots] pre={pre if pre is None else (pre[0], tuple(pre[1]), tuple(pre[2]))}")
    print(f"[flanking_knots] post={post if post is None else (post[0], tuple(post[1]), tuple(post[2]))}")

    print(f"\n[baseline] observations sidecar for frames {window.start}-{window.end}:")
    for f in range(window.start, window.end + 1):
        u = uvs.get(f)
        src = sources.get(f)
        c = confs.get(f)
        print(f"    f={f:4d} uv={u} source={src!r:16s} conf={c}")

    print(f"\n[detector] loading WASB checkpoint={args.checkpoint} device={args.device} ...")
    from src.utils.ball_detector import WASBBallDetector
    detector = WASBBallDetector(
        checkpoint=args.checkpoint, confidence=0.05,
        input_size=(512, 288), device=args.device,
    )

    print(f"[detector] gathering full-frame low-threshold candidates over "
          f"[{window.start}, {window.end}] ...")
    candidates_by_frame = _candidates_for_window(clip_path, window, detector, cfg)
    for f in range(window.start, window.end + 1):
        cands = candidates_by_frame.get(f, [])
        top = sorted(cands, key=lambda c: -c[2])[:3]
        print(f"    f={f:4d} n_candidates={len(cands):2d} top3={top}")

    chain = select_kinematic_chain(candidates_by_frame, pre, post, window, cfg)
    print(f"\n[chain] {len(chain)} frame(s) accepted:")
    for d in chain:
        print(f"    f={d.frame:4d} uv=({d.uv[0]:.2f}, {d.uv[1]:.2f}) "
              f"combined_score={d.combined_score:.3f}")

    # Merge exactly as the stage does: apply_chain_detections honours the
    # "never downgrade" rule EXCEPT for frames the static-lock filter has
    # already demoted (see the module docstring -- the frozen row's
    # confidence can tie the chain's own replacement).
    merged_uv = dict(uvs)
    merged_conf = dict(confs)
    merged_sources = dict(demoted_sources)
    n_replaced = apply_chain_detections(
        chain, static_lock, merged_uv, merged_conf, merged_sources)

    print(f"\n[merged] {n_replaced} frame(s) actually replace baseline evidence "
          f"(after the never-downgrade-except-static-lock rule):")
    still_frozen = []
    for f in range(window.start, window.end + 1):
        base_uv = uvs.get(f)
        new_uv = merged_uv.get(f)
        changed = base_uv != new_uv
        print(f"    f={f:4d} baseline={base_uv} -> merged={new_uv} "
              f"source={merged_sources.get(f)!r:16s} "
              f"{'CHANGED' if changed else ''}")
        if f in static_lock and not changed:
            still_frozen.append(f)

    if chain:
        frames_covered = sorted(d.frame for d in chain)
        print(f"\n[summary] chain covers frames {frames_covered[0]}-{frames_covered[-1]} "
              f"({len(chain)} frames)")
    else:
        print("\n[summary] NO chain accepted for this window.")
    print(f"[summary] static-lock frames still carrying the ORIGINAL frozen "
          f"value after merge: {still_frozen} (expect empty)")


if __name__ == "__main__":
    main()

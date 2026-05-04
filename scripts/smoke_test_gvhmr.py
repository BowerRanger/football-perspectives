#!/usr/bin/env python3
"""End-to-end smoke test: run GVHMR on a tracked player in a short clip.

Uses YOLO (already in the pipeline) to get one player's bounding boxes, then
feeds that sequence to ``GVHMREstimator.estimate_sequence()``.  Dumps a
small JSON summary + writes an HmrResult-compatible .npz.

Run from project root:
    .venv311/bin/python scripts/smoke_test_gvhmr.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.gvhmr_estimator import GVHMREstimator  # noqa: E402
from src.schemas.hmr_result import HmrPlayerTrack, HmrResult  # noqa: E402


def load_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def track_longest_player(frames: list[np.ndarray], model_name: str = "yolov8x.pt") -> list[list[float]]:
    """Run YOLO tracker and return the longest-tracked person's bboxes per frame.

    Returns a list of bboxes [x1, y1, x2, y2]; entries for frames without a
    detection are filled in by linear interp from neighbours.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    per_frame: dict[int, dict[int, list[float]]] = {}

    # Track with persistence enabled so IDs carry across frames
    for i, frame in enumerate(frames):
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        if not results:
            continue
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            tid = int(tid)
            per_frame.setdefault(i, {})[tid] = box.tolist()

    # Pick the track with the most observations
    track_counts: dict[int, int] = {}
    for dets in per_frame.values():
        for tid in dets:
            track_counts[tid] = track_counts.get(tid, 0) + 1
    if not track_counts:
        raise RuntimeError("No player tracks detected")

    best_tid = max(track_counts, key=track_counts.get)
    print(f"[track] Longest track id={best_tid} with {track_counts[best_tid]}/{len(frames)} frames")

    # Pull the bboxes for the chosen track; interpolate across missing frames
    bboxes: list[list[float] | None] = []
    for i in range(len(frames)):
        bboxes.append(per_frame.get(i, {}).get(best_tid))

    # Linear interp for None entries between known bboxes
    last_known = None
    for i, b in enumerate(bboxes):
        if b is not None:
            last_known = (i, b)
            break
    if last_known is None:
        raise RuntimeError(f"Track {best_tid} has no bboxes")

    # Forward fill for leading gap
    for i in range(last_known[0]):
        bboxes[i] = last_known[1]

    # Fill middle gaps by linear interp
    for i in range(last_known[0] + 1, len(bboxes)):
        if bboxes[i] is not None:
            # interp any previous None stretch
            prev_known = i - 1
            while prev_known >= 0 and bboxes[prev_known] is None:
                prev_known -= 1
            # (shouldn't happen with forward-fill, but safe)
            continue
        # find next known
        next_known = i + 1
        while next_known < len(bboxes) and bboxes[next_known] is None:
            next_known += 1
        if next_known >= len(bboxes):
            # trailing gap: back-fill
            prev_known = i - 1
            while prev_known >= 0 and bboxes[prev_known] is None:
                prev_known -= 1
            bboxes[i] = bboxes[prev_known]
            continue
        # interp
        prev_known = i - 1
        while prev_known >= 0 and bboxes[prev_known] is None:
            prev_known -= 1
        lo_b, hi_b = bboxes[prev_known], bboxes[next_known]
        t = (i - prev_known) / (next_known - prev_known)
        bboxes[i] = [lo_b[j] + t * (hi_b[j] - lo_b[j]) for j in range(4)]

    return [b for b in bboxes if b is not None]


def main() -> None:
    video_path = REPO_ROOT / "test-media" / "cleaned_up" / "origi03.mp4"
    assert video_path.exists(), f"Video not found: {video_path}"

    out_dir = REPO_ROOT / "output" / "smoke_gvhmr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {video_path}")
    frames, fps = load_frames(video_path)
    print(f"[load] {len(frames)} frames @ {fps} fps")

    print("[track] Running YOLO tracker")
    t0 = time.time()
    bboxes = track_longest_player(frames)
    print(f"[track] Got {len(bboxes)} bboxes in {time.time() - t0:.1f}s")

    # Trim frames to match bbox count (should be equal already)
    frames = frames[: len(bboxes)]

    print(f"[hmr] Loading GVHMR on cpu")
    t0 = time.time()
    estimator = GVHMREstimator(device="cpu")
    estimator._load_model()
    print(f"[hmr] Model loaded in {time.time() - t0:.1f}s")

    print(f"[hmr] Running inference on {len(frames)} frames")
    t0 = time.time()
    result = estimator.estimate_sequence(frames, bboxes, fps=fps)
    print(f"[hmr] Inference done in {time.time() - t0:.1f}s")

    # Sanity report
    print()
    print("=== Output shapes ===")
    for k, v in result.items():
        print(f"  {k}: {v.shape} ({v.dtype})")

    joints = result["joints_3d"]
    transl = result["transl"]
    print()
    print("=== Sanity ===")
    print(f"  root (pelvis) Y range: {transl[:, 1].min():.3f} to {transl[:, 1].max():.3f} m "
          "(ay frame: negative Y is up)")
    print(f"  ankle Y range: joint 7 {joints[:, 7, 1].min():.3f} to {joints[:, 7, 1].max():.3f} m")
    print(f"  body height (pelvis-to-head): "
          f"{np.abs(joints[:, 15, 1] - joints[:, 0, 1]).mean():.3f} m avg")

    # Save as HmrResult
    player = HmrPlayerTrack(
        track_id="T_smoke",
        player_id="P_smoke",
        player_name="smoke",
        team="A",
        frame_indices=np.arange(len(frames), dtype=np.int32),
        global_orient=result["global_orient"],
        body_pose=result["body_pose"],
        betas=result["betas"],
        transl=result["transl"],
        joints_3d=result["joints_3d"],
        pred_cam=result["pred_cam"],
        bbx_xys=result["bbx_xys"],
        confidences=np.ones(len(frames), dtype=np.float32),
    )
    hmr_result = HmrResult(shot_id="smoke_origi03", fps=fps, players=[player])
    hmr_result.save(out_dir)
    npz_path = out_dir / "smoke_origi03_T_smoke_hmr.npz"
    print(f"\n[save] Wrote {npz_path} ({npz_path.stat().st_size / 1024:.1f} KB)")

    # Reload to prove round-trip works
    loaded = HmrResult.load(out_dir, "smoke_origi03")
    print(f"[load] Reloaded: {len(loaded.players)} player(s), "
          f"joints_3d shape {loaded.players[0].joints_3d.shape}")


if __name__ == "__main__":
    main()

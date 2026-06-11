"""Builds tiny synthetic "highlights reels" for prepare_shots tests.

Segments are visually distinct so PySceneDetect finds the cuts:

- ``green``      : pitch-like (textured green + moving white blob = motion)
- ``green_slow`` : same as ``green`` but the blob moves at half speed
                   (slow-motion replay stand-in)
- ``crowd``      : low-green noise texture (reaction-shot stand-in)
- ``black``      : near-black frames (fade/transition stand-in)
"""
from pathlib import Path

import cv2
import numpy as np

FPS = 25.0
W, H = 192, 108


def _frame(kind: str, t: int, rng: np.ndarray) -> np.ndarray:
    if kind == "black":
        return np.zeros((H, W, 3), np.uint8)
    if kind == "crowd":
        return (rng * 0.5 + 64).astype(np.uint8)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :] = (40, 140, 60)  # BGR pitch green
    frame = frame + (rng * 0.08).astype(np.uint8)  # mow-stripe-ish texture
    step = 2 if kind == "green" else 1  # green_slow: half-speed motion
    x = (10 + t * step) % (W - 20)
    cv2.circle(frame, (x + 10, H // 2), 6, (255, 255, 255), -1)
    return frame


def build_reel(path: Path, segments: list[tuple[str, float]]) -> dict:
    """Write ``segments`` (kind, duration_s) to ``path``.

    Returns ``{"fps", "total_frames", "spans": [{kind, start_frame,
    end_frame}, ...]}`` so tests can assert against ground truth.
    """
    rng = np.random.RandomState(7).rand(H, W, 3) * 255
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H),
    )
    spans: list[dict] = []
    frame_idx = 0
    for kind, dur_s in segments:
        n = int(round(dur_s * FPS))
        spans.append({"kind": kind, "start_frame": frame_idx,
                      "end_frame": frame_idx + n - 1})
        for t in range(n):
            writer.write(_frame(kind, t, rng))
        frame_idx += n
    writer.release()
    return {"fps": FPS, "total_frames": frame_idx, "spans": spans}

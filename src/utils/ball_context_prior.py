"""Context prior for ball detections (spec §4.2).

WASB confidently misdetects scoreboard digits / crowd blobs; cleaning the
track afterwards removes wrong points but cannot add right ones, so the
lever is at detect time. This module computes a multiplicative factor in
[min_factor, 1] per raw detection from three cheap signals available
before the solve pass:

- pitch: the detection's ground-ray intersection lies far off the pitch
  (crowd/stand). Abstains when the ray misses the ground plane — high
  balls legitimately do that.
- player proximity: no tracked person box anywhere near the pixel (only
  when boxes exist for that frame).
- static-in-image: the pixel position is near-constant over a trailing
  window while the camera visibly pans — glued to the IMAGE, not the
  world (overlays, scoreboards).

The factor is a veto signal: the wiring drops a detection only when the
factor itself falls to ``drop_below`` — which only compounded signals
involving the static-under-pan signature can reach — and stored
confidences are never scaled, so low-confidence clips are not penalized.
Signals are deliberately gentle: no single signal drops a confident
detection (see default penalties). Pure and torch-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextPriorCfg:
    enabled: bool = True
    # drop a detection when the prior FACTOR (not conf × factor) falls to/below this
    drop_below: float = 0.35
    pitch_margin_m: float = 5.0
    pitch_penalty: float = 0.7
    player_max_dist_px: float = 180.0
    player_penalty: float = 0.75
    static_window: int = 45
    static_max_px: float = 3.0
    static_min_cam_deg: float = 2.0
    static_penalty: float = 0.45
    min_factor: float = 0.1


def bbox_distance_px(
    uv: tuple[float, float], bbox: tuple[float, float, float, float],
) -> float:
    """Distance from a pixel to a box: 0 inside, else nearest-edge distance."""
    u, v = float(uv[0]), float(uv[1])
    x1, y1, x2, y2 = (float(b) for b in bbox)
    dx = max(x1 - u, 0.0, u - x2)
    dy = max(y1 - v, 0.0, v - y2)
    return float(np.hypot(dx, dy))


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    cos = (float(np.trace(R1.T @ R2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_player_boxes(
    tracks_path: Path,
) -> dict[int, list[tuple[float, float, float, float]]] | None:
    """Per-frame person boxes from the tracking sidecar; None when absent.

    Every class except "ball" counts as a person (the ball's own track
    would make the proximity prior self-confirming).
    """
    if not tracks_path.exists():
        return None
    try:
        data = json.loads(tracks_path.read_text())
        out: dict[int, list[tuple[float, float, float, float]]] = {}
        for track in data.get("tracks", []):
            if track.get("class_name") == "ball":
                continue
            for fr in track.get("frames", []):
                b = fr["bbox"]
                out.setdefault(int(fr["frame"]), []).append(
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                )
        return out
    except Exception as exc:  # noqa: BLE001 — prior input is enrichment
        logger.warning("context prior: unreadable tracks at %s: %s",
                       tracks_path, exc)
        return None


class ContextPrior:
    """Stateful per-shot prior; call ``factor`` once per raw detection in
    frame order (it records the detection for the static-window check)."""

    def __init__(
        self,
        cfg: ContextPriorCfg,
        *,
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float],
        pitch_length_m: float,
        pitch_width_m: float,
        player_boxes_by_frame: dict[
            int, list[tuple[float, float, float, float]]] | None,
        ball_radius_m: float = 0.11,
    ) -> None:
        self._cfg = cfg
        self._K = per_frame_K
        self._R = per_frame_R
        self._t = per_frame_t
        self._distortion = distortion
        self._length = float(pitch_length_m)
        self._width = float(pitch_width_m)
        self._boxes = player_boxes_by_frame
        self._ball_radius = float(ball_radius_m)
        # frame -> uv of the raw detections seen so far (static check).
        self._history: dict[int, tuple[float, float]] = {}

    def factor(self, frame: int, uv: tuple[float, float]) -> float:
        cfg = self._cfg
        u, v = float(uv[0]), float(uv[1])
        if not cfg.enabled:
            return 1.0
        f = 1.0

        # -- pitch signal ---------------------------------------------------
        # Abstains when the ray misses the ground AHEAD of the camera: a
        # near-parallel ray raises, and an above-horizon pixel intersects
        # the plane BEHIND the camera (negative depth) — both are the
        # signature of a legitimately high ball, not a crowd blob.
        K, R, t = self._K.get(frame), self._R.get(frame), self._t.get(frame)
        if K is not None and R is not None and t is not None:
            try:
                world = ankle_ray_to_pitch(
                    (u, v), K=K, R=R, t=t,
                    plane_z=self._ball_radius, distortion=self._distortion,
                )
                depth = float((R @ np.asarray(world) + t)[2])
                if depth > 0.0:
                    m = cfg.pitch_margin_m
                    if not (-m <= world[0] <= self._length + m
                            and -m <= world[1] <= self._width + m):
                        f *= cfg.pitch_penalty
            except ValueError:
                pass  # near-parallel ray: abstain

        # -- player-proximity signal (abstains without boxes that frame) --
        if self._boxes is not None:
            boxes = self._boxes.get(frame)
            if boxes:
                nearest = min(bbox_distance_px((u, v), b) for b in boxes)
                if nearest > cfg.player_max_dist_px:
                    f *= cfg.player_penalty

        # -- static-in-image signal ---------------------------------------
        past_frame = frame - cfg.static_window
        past_uv = self._history.get(past_frame)
        if past_uv is not None:
            moved = float(np.hypot(u - past_uv[0], v - past_uv[1]))
            R_now = self._R.get(frame)
            R_then = self._R.get(past_frame)
            if (moved <= cfg.static_max_px
                    and R_now is not None and R_then is not None
                    and rotation_angle_deg(R_then, R_now)
                    >= cfg.static_min_cam_deg):
                f *= cfg.static_penalty

        self._history[frame] = (u, v)
        return max(cfg.min_factor, f)

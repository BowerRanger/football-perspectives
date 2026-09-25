"""Read-only clip loading helpers shared by the single-camera event-cue
modules (audio/net/blur) and ``scripts/eval_ball_event_cues.py``.

Mirrors the camera-lookup and observation-filtering conventions in
``scripts/eval_ball_accuracy.py`` (imported, not re-derived, per
``prototypes/ball_hybrid_poc/ctx.py``'s precedent) so this module cannot
silently drift from the rest of the ball-eval stack. Cue modules
themselves stay pure (arrays/dataclasses in, no file I/O) -- this module
is the thin path-handling layer that feeds them.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_ball_accuracy import _camera_lookup, _load_observations  # noqa: E402
from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402

# "Event" anchor states: point-in-time contacts/impacts, as opposed to
# the trajectory-description states (grounded/airborne_*/off_screen_flight)
# that describe the ball's state *between* events rather than an event
# itself. Mirrors the task brief's "player_touch, kick, bounce,
# goal_impact, catch, header..." list.
EVENT_STATES = frozenset({
    "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact",
})

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraFrames:
    fps: float
    image_size: tuple[int, int]
    frames: tuple[int, ...]
    per_frame_K: dict[int, np.ndarray]
    per_frame_R: dict[int, np.ndarray]
    per_frame_t: dict[int, np.ndarray]
    distortion: tuple[float, float]


def load_camera_frames(output_dir: str | Path, shot_id: str) -> CameraFrames:
    cam = CameraTrack.load(
        Path(output_dir) / "camera" / f"{shot_id}_camera_track.json")
    _cams, per_K, per_R, per_t, distortion = _camera_lookup(cam)
    frames = tuple(sorted(per_K))
    return CameraFrames(
        fps=float(cam.fps),
        image_size=(int(cam.image_size[0]), int(cam.image_size[1])),
        frames=frames, per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=distortion,
    )


def load_manual_events(output_dir: str | Path, shot_id: str) -> list[tuple[int, str]]:
    """``(frame, state)`` pairs for manual anchors whose state is a real
    point-event (see ``EVENT_STATES``), sorted by frame."""
    path = Path(output_dir) / "ball" / f"{shot_id}_ball_anchors.json"
    aset = BallAnchorSet.load(path)
    return sorted(
        (a.frame, a.state) for a in aset.anchors if a.state in EVENT_STATES)


def load_auto_event_frames(output_dir: str | Path, shot_id: str) -> list[int]:
    """Frame numbers of every anchor in the ball stage's current auto
    sidecar -- the "existing kinematic/velocity_break" evidence layer the
    fusion policy can lean on for single-cue corroboration. The sidecar
    carries no per-anchor origin tag, so this is the whole current
    auto-anchor set, not filtered to velocity-break-sourced events
    specifically (documented limitation -- see the wiring note)."""
    path = Path(output_dir) / "ball" / f"{shot_id}_ball_anchors_auto.json"
    if not path.exists():
        return []
    aset = BallAnchorSet.load(path)
    return sorted(a.frame for a in aset.anchors)


def load_player_boxes(
    output_dir: str | Path, shot_id: str,
) -> dict[int, list[Box]]:
    """``frame -> [ (x0,y0,x1,y1), ... ]`` from the tracking sidecar."""
    path = Path(output_dir) / "tracks" / f"{shot_id}_tracks.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, list[Box]] = {}
    for tr in data.get("tracks", []):
        for fr in tr.get("frames", []):
            bbox = fr.get("bbox")
            if bbox is None:
                continue
            out.setdefault(int(fr["frame"]), []).append(
                tuple(float(v) for v in bbox))  # type: ignore[arg-type]
    return out


def load_real_observations(
    output_dir: str | Path, shot_id: str,
) -> list[tuple[int, tuple[float, float], float, str]]:
    """``(frame, uv, conf, source)`` from the observations sidecar, using
    the exact filtering ``scripts/eval_ball_accuracy.py`` applies (anchor
    / gap-fill entries excluded)."""
    ball_dir = Path(output_dir) / "ball"
    aset = BallAnchorSet.load(ball_dir / f"{shot_id}_ball_anchors.json")
    return _load_observations(
        ball_dir / f"{shot_id}_ball_observations.json", anchors=aset.anchors)


def probe_speed_factor(output_dir: str | Path, shot_id: str) -> float:
    """The shot's extraction ``speed_factor`` from ``shots_manifest.json``
    (1.0 = real time; e.g. 4.0 means the source was slowed 4x and
    retimed to real time at extraction -- audio is not usable there, see
    ``ball_cue_audio``). Defaults to 1.0 when the manifest or shot entry
    is missing (older/manual-only output dirs)."""
    path = Path(output_dir) / "shots" / "shots_manifest.json"
    if not path.exists():
        return 1.0
    data = json.loads(path.read_text())
    for s in data.get("shots", []):
        if s.get("id") == shot_id:
            return float(s.get("speed_factor", 1.0) or 1.0)
    return 1.0


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Sequential BGR-frame reader via OpenCV: ``(frame_idx, frame_bgr)``,
    one frame held in memory at a time."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


__all__ = [
    "EVENT_STATES", "Box", "CameraFrames",
    "load_camera_frames", "load_manual_events", "load_auto_event_frames",
    "load_player_boxes", "load_real_observations", "probe_speed_factor",
    "iter_video_frames",
]

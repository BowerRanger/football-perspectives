"""IO half of the appearance stage: read keypoints / frames / camera and
return per-player region colours plus pitch-line pixels for white balance.

Everything here is read-only against ``output_dir``; the stage owns writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils import kit_palette as kp
from src.utils.team_clustering import sample_player_regions
from src.utils.video_reader import read_frames

logger = logging.getLogger(__name__)


@dataclass
class ShotEvidence:
    shot_id: str
    samples: dict[str, list[dict[str, np.ndarray]]] = field(default_factory=dict)
    line_pixels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    n_frames: int = 0


def load_kp2d(output_dir: Path, shot_id: str) -> dict[str, dict[int, np.ndarray]]:
    """``{pid: {frame: (17, 3) keypoints}}`` from ``hmr_world/{shot}__{pid}_kp2d.json``."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for path in sorted((output_dir / "hmr_world").glob(f"{shot_id}__*_kp2d.json")):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[appearance] unreadable %s: %s", path.name, exc)
            continue
        pid = raw.get("player_id") or path.stem.split("__", 1)[1].rsplit("_kp2d", 1)[0]
        frames = {int(f["frame"]): np.asarray(f["keypoints"], dtype=np.float64)
                  for f in raw.get("frames", []) if len(f.get("keypoints", [])) >= 17}
        if frames:
            out[pid] = frames
    return out


def choose_frames(kp2d: dict[str, dict[int, np.ndarray]], n: int, min_players: int = 3) -> list[int]:
    """``n`` evenly spaced frames among those with at least ``min_players`` tracked."""
    counts: dict[int, int] = {}
    for frames in kp2d.values():
        for f in frames:
            counts[f] = counts.get(f, 0) + 1
    usable = sorted(f for f, c in counts.items() if c >= min_players)
    if not usable:
        return []
    idx = np.unique(np.linspace(0, len(usable) - 1, num=min(n, len(usable))).round().astype(int))
    return [usable[i] for i in idx]


def pitch_line_points(max_per_line: int = 120) -> np.ndarray:
    """Ground-plane (z=0) pitch marking points, ``(N, 3)``."""
    from src.utils.pitch_lines import pitch_polylines

    pts = []
    for line in pitch_polylines():
        if np.all(np.abs(line[:, 2]) < 1e-6):
            step = max(1, len(line) // max_per_line)
            pts.append(line[::step])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _camera_frames(output_dir: Path, shot_id: str):
    path = output_dir / "camera" / f"{shot_id}_camera_track.json"
    if not path.exists():
        return None
    try:
        from src.schemas.camera_track import CameraTrack

        return CameraTrack.load(path)
    except Exception as exc:  # noqa: BLE001 - evidence is best-effort
        logger.warning("[appearance] camera track unreadable for %s: %s", shot_id, exc)
        return None


def _line_pixels(frame_bgr: np.ndarray, cam, frame_idx: int, line_pts: np.ndarray) -> np.ndarray:
    from src.utils.camera_projection import project_world_to_image

    cf = next((f for f in cam.frames if f.frame == frame_idx), None)
    if cf is None:
        return np.zeros((0, 3))
    R = np.asarray(cf.R, dtype=np.float64)
    t = np.asarray(cf.t if cf.t is not None else cam.t_world, dtype=np.float64)
    depth = (line_pts @ R.T + t)[:, 2]
    pts = line_pts[depth > 0.5]
    if len(pts) == 0:
        return np.zeros((0, 3))
    uv = project_world_to_image(np.asarray(cf.K, dtype=np.float64), R, t, tuple(cam.distortion), pts)
    return kp.line_pixel_candidates(frame_bgr[:, :, ::-1], uv)


def collect_shot_evidence(
    output_dir: Path,
    shot,
    *,
    frames_per_shot: int = 16,
    min_conf: float = 0.4,
    wb_frames: int = 8,
) -> ShotEvidence:
    ev = ShotEvidence(shot_id=shot.id)
    kp2d = load_kp2d(output_dir, shot.id)
    chosen = choose_frames(kp2d, frames_per_shot)
    if not chosen:
        return ev
    frames = read_frames(output_dir / shot.clip_file, chosen)
    ev.n_frames = len(frames)
    cam = _camera_frames(output_dir, shot.id)
    line_pts = pitch_line_points() if cam is not None else None
    wb_set = set(chosen[:: max(1, len(chosen) // max(1, wb_frames))][:wb_frames])
    line_px: list[np.ndarray] = []
    for f, img in frames.items():
        for pid, per_frame in kp2d.items():
            if f in per_frame:
                s = sample_player_regions(img, per_frame[f], min_conf=min_conf)
                if s:
                    ev.samples.setdefault(pid, []).append(s)
        if cam is not None and f in wb_set:
            line_px.append(_line_pixels(img, cam, f, line_pts))
    if line_px:
        ev.line_pixels = np.concatenate(line_px)
    return ev


def player_pitch_x(output_dir: Path, pids) -> dict[str, float]:
    """Median pitch x (m) per player from ``refined_poses/{pid}_refined.npz``."""
    out: dict[str, float] = {}
    for pid in pids:
        path = output_dir / "refined_poses" / f"{pid}_refined.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as d:
                x = d["root_t"][:, 0]
            x = x[np.isfinite(x)]
            if len(x):
                out[pid] = float(np.median(x))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[appearance] could not read %s: %s", path.name, exc)
    return out

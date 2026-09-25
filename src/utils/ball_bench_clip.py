"""Per-clip evaluation context for the ball-stage regression bench.

Loads the read-only inputs each clip's bench scenarios need (camera,
manual anchors, real detector observations, origi01's cross-replay fixes)
from the main repo's output dirs, and exposes projection/ray helpers.
Promoted from the ``ball-hybrid-integration`` spike
(``prototypes/ball_hybrid_poc/ctx.py``); self-contained (no dependency on
``scripts/eval_ball_accuracy.py`` — its small camera-lookup and
observation-loading helpers are reproduced here directly) so this module
can live under ``src/utils`` without importing from the top-level
``scripts/`` package.

Deliberately NEVER reads ``*_ball_track.json``, ``*_ball_anchors_auto.json``
or ``*_ball_keyframes.json`` — those are ball-stage outputs under test by
this bench and must not leak into truth-seeding or evaluation context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.schemas.ball_anchor import BallAnchorSet
from src.schemas.ball_fixes import BallFix, BallFixSet
from src.schemas.camera_track import CameraTrack
from src.utils.ball_eval import pixel_ray
from src.utils.ball_player_context import PlayerContext
from src.utils.camera_projection import project_world_to_image

from .ball_bench_types import Observation

# M resolves from POC_MAIN_REPO (kept from the spike's env var name so an
# operator's existing shell setup / scripts keep working) so a checkout
# elsewhere (or a future CI box) can point at a different main-repo copy
# without editing this file.
M = os.environ.get("POC_MAIN_REPO",
                    "/Users/joebower/workplace/football-perspectives")

# clip_id -> (output_dir, shot_id). Each output dir has ball/<shot_id>_
# ball_anchors.json etc. keyed by this exact id even though gberch's
# shots/ dir also holds a second shot (gberch-2, the spidercam replay)
# and output-japan's shots/ dir holds s001..s005 (s013 is the one with
# ball/camera/manual-anchor sidecars).
CLIPS: dict[str, tuple[str, str]] = {
    "gberch": (f"{M}/output", "gberch"),
    "origi01": (f"{M}/output-origi-global", "origi01"),
    "kroupi01": (f"{M}/output-kroupi", "kroupi01"),
    "s013": (f"{M}/output-japan", "s013"),
}

# Only these observation sources count as real detector evidence; anchor
# and gap-fill entries are not evidence (mirrors scripts/eval_ball_accuracy
# .py's _DENSE_EVAL_SOURCES so the two harnesses cannot silently drift).
_DENSE_EVAL_SOURCES = frozenset(
    {"detector", "second_pass", "foot_guided", "strike_window"})
_ANCHOR_VETO_PX = 60.0
_ANCHOR_VETO_GAP_FRAMES = 6


def _camera_lookup(cam: CameraTrack):
    """Per-frame ``(K, R, t)`` dicts with the clip-shared ``t_world``
    fallback for a frame whose track entry has no per-frame ``t``."""
    t_fallback = np.asarray(cam.t_world, dtype=float)
    per_K: dict[int, np.ndarray] = {}
    per_R: dict[int, np.ndarray] = {}
    per_t: dict[int, np.ndarray] = {}
    for f in cam.frames:
        per_K[f.frame] = np.asarray(f.K, dtype=float)
        per_R[f.frame] = np.asarray(f.R, dtype=float)
        per_t[f.frame] = (np.asarray(f.t, dtype=float)
                           if f.t is not None else t_fallback)
    return per_K, per_R, per_t, tuple(cam.distortion)


def _load_observations(path: Path, anchors=None) -> list[tuple]:
    """Real-detector observations from the ``frames`` sidecar payload.

    Anchor-sourced and gap-fill entries are excluded. Detections the
    operator's adjacent clicks contradict are excluded too (clicks are
    ground truth, detections are not): mirrors
    ``scripts/eval_ball_accuracy.py``'s ``_load_observations`` exactly
    so the sub-20cm campaign's veto logic and this bench cannot drift.
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
                    float(e.get("confidence", 0.0)), str(e.get("source"))))
    if anchors:
        clicks = sorted((a.frame, a.image_xy) for a in anchors
                         if a.image_xy is not None)
        expected: dict[int, tuple[float, float]] = {
            f: (float(uv[0]), float(uv[1])) for f, uv in clicks}
        for (fa, ua), (fb, ub) in zip(clicks, clicks[1:]):
            if 0 < fb - fa <= _ANCHOR_VETO_GAP_FRAMES:
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


@dataclass(frozen=True)
class ClipContext:
    """Everything a bench scenario needs to read about one clip, built
    once by :func:`load_clip`. Frozen; the only mutable state is the lazy
    ``PlayerContext`` cache (a dict, so mutating its contents doesn't
    require reassigning a frozen attribute)."""

    clip_id: str
    output_dir: Path
    shot_id: str
    fps: float
    image_size: tuple[int, int]
    frames: tuple[int, ...]          # sorted camera-track frame indices
    n_frames: int
    per_frame_K: dict[int, np.ndarray]
    per_frame_R: dict[int, np.ndarray]
    per_frame_t: dict[int, np.ndarray]
    distortion: tuple[float, float]
    camera_track: CameraTrack
    anchors: BallAnchorSet
    observations: tuple[Observation, ...]
    fixes: tuple[BallFix, ...]
    video_path: Path
    _pc_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def project(self, frame: int, xyz) -> np.ndarray:
        """Project one point (shape ``(3,)``) or many (``(N, 3)``) to
        image pixels for ``frame``'s camera. Mirrors the single-point
        input back out as a single ``(2,)`` array."""
        frame = int(frame)
        K = self.per_frame_K[frame]
        R = self.per_frame_R[frame]
        t = self.per_frame_t[frame]
        pts = np.asarray(xyz, dtype=np.float64)
        single = pts.ndim == 1
        out = project_world_to_image(K, R, t, self.distortion,
                                      pts.reshape(-1, 3))
        return out[0] if single else out

    def ray(self, frame: int, uv) -> tuple[np.ndarray, np.ndarray]:
        """Camera centre + unit world-space ray direction through pixel
        ``uv`` at ``frame``'s camera. ``(C, d_hat)``."""
        frame = int(frame)
        K = self.per_frame_K[frame]
        R = self.per_frame_R[frame]
        t = self.per_frame_t[frame]
        return pixel_ray(uv, K, R, t, self.distortion)

    def camera_centres(self) -> list[list[float] | None]:
        """Per-frame world-space camera centre (``-R^T t``), in
        ``self.frames`` order; ``None`` for a frame missing R/t (should
        not happen — camera_track is dense per solved frame)."""
        out: list[list[float] | None] = []
        for f in self.frames:
            R = self.per_frame_R.get(f)
            t = self.per_frame_t.get(f)
            if R is None or t is None:
                out.append(None)
                continue
            C = -R.T @ t
            out.append([float(C[0]), float(C[1]), float(C[2])])
        return out

    def player_context(self) -> PlayerContext:
        """Lazy, cached ``PlayerContext`` (SMPL FK joint lookup) for this
        clip's shot."""
        if "pc" not in self._pc_cache:
            self._pc_cache["pc"] = PlayerContext.load(
                self.output_dir, self.shot_id,
                per_frame_K=self.per_frame_K, per_frame_R=self.per_frame_R,
                per_frame_t=self.per_frame_t, distortion=self.distortion,
            )
        return self._pc_cache["pc"]


def load_clip(clip_id: str) -> ClipContext:
    """Load a :class:`ClipContext` for ``clip_id`` (one of ``CLIPS``)."""
    if clip_id not in CLIPS:
        raise KeyError(f"unknown clip_id {clip_id!r}; known: {sorted(CLIPS)}")
    output_dir, shot_id = CLIPS[clip_id]
    output_dir = Path(output_dir)

    cam = CameraTrack.load(
        output_dir / "camera" / f"{shot_id}_camera_track.json")
    per_K, per_R, per_t, distortion = _camera_lookup(cam)
    frames = tuple(sorted(per_K))

    anchors = BallAnchorSet.load(
        output_dir / "ball" / f"{shot_id}_ball_anchors.json")

    raw_obs = _load_observations(
        output_dir / "ball" / f"{shot_id}_ball_observations.json",
        anchors=anchors.anchors)
    observations = tuple(
        Observation(frame=f, uv=uv, conf=conf, source=source)
        for f, uv, conf, source in raw_obs)

    fixes_path = output_dir / "ball" / f"{shot_id}_ball_fixes.json"
    fixes = BallFixSet.load(fixes_path).fixes if fixes_path.exists() else ()

    video_path = output_dir / "shots" / f"{shot_id}.mp4"

    return ClipContext(
        clip_id=clip_id,
        output_dir=output_dir,
        shot_id=shot_id,
        fps=float(cam.fps),
        image_size=(int(cam.image_size[0]), int(cam.image_size[1])),
        frames=frames,
        n_frames=len(frames),
        per_frame_K=per_K,
        per_frame_R=per_R,
        per_frame_t=per_t,
        distortion=distortion,
        camera_track=cam,
        anchors=anchors,
        observations=observations,
        fixes=fixes,
        video_path=video_path,
    )


__all__ = ["M", "CLIPS", "ClipContext", "load_clip"]

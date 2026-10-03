"""Pre-render framing checks for 9:16 Shorts passes.

Pure functions on camera-track arrays: before paying 3-5 minutes for a
Blender pass, project the ball and the focus subject through the candidate
virtual camera and reject draft framings that were previously found by eye
(design doc G19): a drone so high the players are specks, an over-the-shoulder
camera inside the striker, a chase cam the ball outruns, an eye-line cam
whose view of the ball is blocked by a body.

Projection mirrors the render: the portrait pass keeps the landscape lens
(``sensor_fit=VERTICAL``), so the *vertical* field of view of the 1080x1920
frame equals the track's horizontal FOV.

Pitch metres, z up; players are vertical capsules standing on the pitch at
the refined-poses root xy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

PORTRAIT_SIZE = (1080, 1920)
SAFE_TOP_PX = 250       # matches short_compositor caption safe area
SAFE_BOTTOM_PX = 430
SAFE_SIDE_PX = 60
BALL_DIAMETER_M = 0.22
PLAYER_HEIGHT_M = 1.8


@dataclass(frozen=True)
class FramingLimits:
    safe_side_px: int = SAFE_SIDE_PX
    safe_top_px: int = SAFE_TOP_PX
    safe_bottom_px: int = SAFE_BOTTOM_PX
    max_ball_out_frames: int = 6        # consecutive frames outside the safe area
    max_occluded_frames: int = 12       # consecutive frames the ball is blocked
    max_camera_in_player_frames: int = 1
    capsule_radius_m: float = 0.45      # body + swinging limbs (camera-inside check)
    occlusion_radius_m: float = 0.28    # torso-ish: a sight line this close to the axis is blocked
    check_ball: bool = True             # False for subject-driven establishing shots
    camera_clearance_m: float = 0.10
    min_subject_px: float = 120.0       # projected player height (of 1920)
    min_ball_px: float = 12.0           # projected ball diameter (of 1920)

    def merged(self, overrides: Mapping | None) -> "FramingLimits":
        if not overrides:
            return self
        valid = set(self.__dataclass_fields__)
        bad = sorted(set(overrides) - valid)
        if bad:
            raise ValueError(f"unknown framing limit(s) {bad}; valid: {sorted(valid)}")
        return FramingLimits(**{**self.__dict__, **overrides})


@dataclass(frozen=True)
class CameraArrays:
    frames: np.ndarray          # (N,) scene frame numbers, ascending
    R: np.ndarray               # (N,3,3) world->camera
    t: np.ndarray               # (N,3)
    fov_deg: float              # landscape horizontal FOV == portrait vertical FOV

    def centres(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)

    def focal_px(self, size: tuple[int, int] = PORTRAIT_SIZE) -> float:
        return (size[1] / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)


def camera_arrays_from_track(track) -> CameraArrays:
    """Build ``CameraArrays`` from a ``CameraTrack`` (or look-alike)."""
    fr = list(track.frames)
    if not fr:
        raise ValueError("camera track has no frames")
    k = fr[0].K
    w = float(track.image_size[0])
    fov = math.degrees(2.0 * math.atan((w / 2.0) / float(k[0][0])))
    return CameraArrays(
        frames=np.array([f.frame for f in fr], dtype=int),
        R=np.array([f.R for f in fr], dtype=float),
        t=np.array([f.t if f.t is not None else track.t_world for f in fr], dtype=float),
        fov_deg=fov,
    )


@dataclass(frozen=True)
class FramingFailure:
    check: str                  # ball_out_of_safe_area | ball_occluded | camera_in_player | subject_too_small
    frames: tuple[int, int]     # first/last offending frame
    detail: str

    def to_dict(self) -> dict:
        return {"check": self.check, "frames": list(self.frames), "detail": self.detail}


@dataclass(frozen=True)
class FramingResult:
    ok: bool
    failures: tuple[FramingFailure, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "failures": [f.to_dict() for f in self.failures],
                "metrics": dict(self.metrics)}


def project_points(cam: CameraArrays, pts: np.ndarray, idx: np.ndarray,
                   size: tuple[int, int] = PORTRAIT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Project world ``pts`` (M,3) with camera rows ``idx`` (M,) -> (uv (M,2), depth (M,))."""
    pc = np.einsum("mij,mj->mi", cam.R[idx], pts) + cam.t[idx]
    z = pc[:, 2]
    f = cam.focal_px(size)
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = f * pc[:, 0] / safe + size[0] / 2.0
    v = f * pc[:, 1] / safe + size[1] / 2.0
    return np.stack([u, v], axis=1), z


def _longest_run(mask: np.ndarray, frames: np.ndarray) -> tuple[int, int, int]:
    """Longest run of consecutive-frame True values: (length, first, last)."""
    best = (0, 0, 0)
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        contiguous = i + 1 < len(mask) and mask[i + 1] and frames[i + 1] == frames[i] + 1
        if m and not contiguous:
            n = i - start + 1
            if n > best[0]:
                best = (n, int(frames[start]), int(frames[i]))
            start = None
    return best


def _seg_axis_dist(p0: np.ndarray, p1: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Min distance between segments p0->p1 and a0->a1, row-wise (M,3 each).
    Closest-points via the standard clamped solution (Ericson)."""
    d1, d2, r = p1 - p0, a1 - a0, p0 - a0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1)
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1), 0, 1)
    return np.linalg.norm((p0 + d1 * s[:, None]) - (a0 + d2 * t[:, None]), axis=1)


def _player_pos(players: Mapping[str, tuple[np.ndarray, np.ndarray]], pid: str,
                frames: np.ndarray) -> np.ndarray | None:
    """Root xy (M,2) of ``pid`` at ``frames`` (nearest-frame, NaN outside coverage)."""
    if pid not in players:
        return None
    pf, pxy = players[pid]
    pf = np.asarray(pf)
    pxy = np.asarray(pxy, dtype=float)[:, :2]
    idx = np.clip(np.searchsorted(pf, frames), 0, len(pf) - 1)
    ok = (frames >= pf[0]) & (frames <= pf[-1])
    out = pxy[idx].copy()
    out[~ok] = np.nan
    return out


def check_framing(
    cam: CameraArrays,
    ball_xyz: Mapping[int, Sequence[float]],
    players: Mapping[str, tuple[np.ndarray, np.ndarray]],
    start: int,
    end: int,
    *,
    subject_pid: str | None = None,
    exclude_pids: Sequence[str] = (),
    limits: FramingLimits | None = None,
) -> FramingResult:
    """Check a candidate camera over scene frames ``[start, end]``.

    ``players[pid] = (frames, xy[N,2])``. ``exclude_pids`` are bodies the
    camera is mounted on / hides in the render (``eyes:<PID>``) - ignored by
    the occlusion and inside-body checks. ``subject_pid`` is the player the
    pass is about (size check); ``None`` checks the ball only.
    """
    lim = limits or FramingLimits()
    win = (cam.frames >= start) & (cam.frames <= end)
    cam_rows = np.where(win)[0]
    failures: list[FramingFailure] = []
    metrics: dict[str, float] = {}
    if len(cam_rows) == 0:
        return FramingResult(False, (FramingFailure(
            "no_camera_frames", (start, end), "camera track does not cover the window"),), {})
    frames = cam.frames[cam_rows]

    # --- ball in the 9:16 safe area -------------------------------------
    brows = [(i, ball_xyz[int(f)]) for i, f in zip(cam_rows, frames) if int(f) in ball_xyz]
    if brows and lim.check_ball:
        bi = np.array([b[0] for b in brows])
        bpts = np.array([b[1] for b in brows], dtype=float).reshape(-1, 3)
        bfr = cam.frames[bi]
        uv, depth = project_points(cam, bpts, bi)
        w, h = PORTRAIT_SIZE
        outside = ((depth <= 0.1) | (uv[:, 0] < lim.safe_side_px) | (uv[:, 0] > w - lim.safe_side_px)
                   | (uv[:, 1] < lim.safe_top_px) | (uv[:, 1] > h - lim.safe_bottom_px))
        n, a, b = _longest_run(outside, bfr)
        metrics["ball_out_frames"] = float(outside.sum())
        if n > lim.max_ball_out_frames:
            failures.append(FramingFailure(
                "ball_out_of_safe_area", (a, b),
                f"ball outside the 9:16 safe area for {n} consecutive frames (max {lim.max_ball_out_frames})"))
        ball_px = BALL_DIAMETER_M * cam.focal_px() / np.maximum(depth, 1e-6)
        med_ball = float(np.median(ball_px[depth > 0.1])) if (depth > 0.1).any() else 0.0
        metrics["ball_px_median"] = med_ball
        if med_ball < lim.min_ball_px:
            failures.append(FramingFailure(
                "subject_too_small", (int(bfr[0]), int(bfr[-1])),
                f"ball is {med_ball:.1f}px across (min {lim.min_ball_px:.0f}px)"))

        # --- ball occluded by a player body ------------------------------
        centres = cam.centres()[bi]
        to_ball = bpts - centres
        dist = np.linalg.norm(to_ball, axis=1, keepdims=True)
        stop = bpts - to_ball / np.maximum(dist, 1e-9) * np.minimum(0.5, dist * 0.5)
        blocked = np.zeros(len(bi), dtype=bool)
        for pid in players:
            if pid in exclude_pids:
                continue
            pos = _player_pos(players, pid, bfr)
            if pos is None:
                continue
            valid = ~np.isnan(pos[:, 0])
            a0 = np.concatenate([pos, np.zeros((len(pos), 1))], axis=1)
            a1 = np.concatenate([pos, np.full((len(pos), 1), PLAYER_HEIGHT_M)], axis=1)
            d = _seg_axis_dist(centres, stop, a0, a1)
            blocked |= valid & (d < lim.occlusion_radius_m)
        n, a, b = _longest_run(blocked, bfr)
        metrics["ball_occluded_frames"] = float(blocked.sum())
        if n > lim.max_occluded_frames:
            failures.append(FramingFailure(
                "ball_occluded", (a, b),
                f"ball hidden behind a player for {n} consecutive frames (max {lim.max_occluded_frames})"))

    # --- camera inside a player capsule ---------------------------------
    centres_all = cam.centres()[cam_rows]
    inside = np.zeros(len(cam_rows), dtype=bool)
    for pid in players:
        if pid in exclude_pids:
            continue
        pos = _player_pos(players, pid, frames)
        if pos is None:
            continue
        valid = ~np.isnan(pos[:, 0])
        horiz = np.linalg.norm(centres_all[:, :2] - np.nan_to_num(pos), axis=1)
        in_height = (centres_all[:, 2] > -0.1) & (centres_all[:, 2] < PLAYER_HEIGHT_M + 0.1)
        inside |= valid & in_height & (horiz < lim.capsule_radius_m + lim.camera_clearance_m)
    n, a, b = _longest_run(inside, frames)
    metrics["camera_in_player_frames"] = float(inside.sum())
    if n > lim.max_camera_in_player_frames:
        failures.append(FramingFailure(
            "camera_in_player", (a, b), f"camera is inside a player's body for {n} frames"))

    # --- subject large enough to read ------------------------------------
    if subject_pid and subject_pid not in exclude_pids:
        pos = _player_pos(players, subject_pid, frames)
        if pos is not None and (~np.isnan(pos[:, 0])).any():
            ok = ~np.isnan(pos[:, 0])
            idx = cam_rows[ok]
            foot = np.concatenate([pos[ok], np.zeros((ok.sum(), 1))], axis=1)
            head = foot + np.array([0.0, 0.0, PLAYER_HEIGHT_M])
            uvf, zf = project_points(cam, foot, idx)
            uvh, zh = project_points(cam, head, idx)
            front = (zf > 0.1) & (zh > 0.1)
            heights = np.abs(uvf[:, 1] - uvh[:, 1])[front]
            med = float(np.median(heights)) if len(heights) else 0.0
            metrics["subject_px_median"] = med
            if med < lim.min_subject_px:
                failures.append(FramingFailure(
                    "subject_too_small", (int(frames[ok][0]), int(frames[ok][-1])),
                    f"{subject_pid} is {med:.0f}px tall in the 9:16 frame (min {lim.min_subject_px:.0f}px)"))

    return FramingResult(not failures, tuple(failures), metrics)

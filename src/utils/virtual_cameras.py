"""Synthesised player POV / over-the-shoulder cameras.

Pure math + rig builders. No file I/O — the export stage handles reading
selections and writing CameraTrack JSON. Conventions match the broadcast
camera: ``R`` is world->camera (OpenCV: +Z optical ray into scene, +X
right, +Y down); per-frame ``t`` satisfies camera-centre ``C = -R.T @ t``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.pitch import FIFA_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH
from src.utils.smpl_skeleton import compute_joint_world_pose

WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_UP.flags.writeable = False

_FrameTuple = tuple[int, np.ndarray, np.ndarray, float]  # (frame_idx, R_world2cam, t, confidence)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def intrinsics_from_fov(fov_deg: float, image_size: tuple[int, int]) -> list[list[float]]:
    """3x3 K from a horizontal field of view. Principal point centred."""
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    w, h = int(image_size[0]), int(image_size[1])
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def look_at_view(
    center: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target``.

    Rows of ``R`` are the camera axes in world coords: right (+X), down
    (+Y), forward (+Z). ``t = -R @ center``.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(target - center)) < 1e-9:
        raise ValueError(
            f"look_at_view: center and target are coincident (center={center}, target={target})"
        )
    z = _normalize(target - center)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    x = np.cross(z, up)
    if float(np.linalg.norm(x)) < 1e-9:
        # Optical axis parallel to up — pick an arbitrary stable basis.
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(x)) < 1e-9:
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = _normalize(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ center
    return R, t


def _look_at_safe(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``look_at_view`` wrapper that nudges a degenerate (coincident)
    ``center``/``target`` pair apart by an epsilon instead of raising.

    ``look_at_view`` intentionally raises ``ValueError`` on a coincident
    center/target (see its docstring) — correct for rigs whose geometry
    guarantees separation by construction (e.g. ``build_drone_track``'s
    non-zero ``drone_back_m``/``drone_height_m``). ``build_chase_track``
    can't make that guarantee: under a degenerate config
    (``chase_back_m=0`` and ``chase_height_m=0``) with a static/missing
    ball, its camera center and look-at target both collapse to the
    same point. Rather than let that raise mid-render, nudge the center
    up by 1mm — imperceptible on screen, and keeps the rig always
    produce a valid frame.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(target - center)) < 1e-6:
        center = center + np.array([0.0, 0.0, 1e-3])
    return look_at_view(center, target)


HEAD_JOINT_IDX = 15
# SMPL canonical (y-up) facing axis. +Z is "forward" out of the torso for the
# rest pose; sign may need flipping after the first real export — kept as a
# module constant so tuning is a one-line change.
FACE_AXIS_CANONICAL = np.array([0.0, 0.0, 1.0])
FACE_AXIS_CANONICAL.flags.writeable = False


@dataclass(frozen=True)
class RigConfig:
    pov_fov_deg: float = 75.0
    ots_fov_deg: float = 60.0
    ots_back_m: float = 0.4
    ots_up_m: float = 0.3
    ots_right_m: float = 0.0
    ball_target_max_occlusion_frames: int = 10
    drone_fov_deg: float = 55.0
    drone_height_m: float = 40.0
    drone_back_m: float = 25.0
    drone_smooth_frames: int = 25
    # Low behind-goal camera: fixed x/y behind the named goal's line,
    # looking down-pitch and panning with the smoothed action centroid
    # (shares drone_smooth_frames for that smoothing window).
    goal_fov_deg: float = 40.0
    goal_height_m: float = 1.2
    goal_back_m: float = 8.0
    # On the goal line itself, offset in from the near post — low and
    # dramatic. Also pans with the smoothed action centroid.
    goalline_fov_deg: float = 50.0
    goalline_height_m: float = 0.4
    goalline_post_offset_m: float = 1.5
    # Arc sweep at fixed radius/height around the smoothed action
    # centroid, azimuth interpolated linearly across the shot's frames.
    orbit_fov_deg: float = 50.0
    orbit_radius_m: float = 15.0
    orbit_height_m: float = 6.0
    orbit_sweep_deg: float = 180.0
    # Ball-chase cam: trails the smoothed ball's horizontal velocity
    # vector; falls back to the smoothed action centroid (using
    # drone_smooth_frames) when the ball is missing or its smoothed
    # speed is below chase_min_speed_m_s ("static").
    chase_fov_deg: float = 45.0
    chase_back_m: float = 6.0
    chase_height_m: float = 2.0
    chase_smooth_frames: int = 9
    chase_min_speed_m_s: float = 0.5
    # Low sideline dolly: fixed y near the nearside touchline (negative
    # = just off the pitch, y=0 is the touchline itself), x tracks the
    # smoothed action centroid, long lens.
    dolly_fov_deg: float = 30.0
    dolly_y_m: float = -3.0
    dolly_height_m: float = 1.0
    tactical_fov_deg: float = 55.0
    sideline_fov_deg: float = 55.0
    sideline_height_m: float = 14.0
    corner_fov_deg: float = 58.0
    corner_height_m: float = 10.0
    # Eye-line cam (``eyes:<PID>``): a player's smoothed eye position
    # (head joint + up/forward offsets, clear of the player's own head
    # mesh) aimed at the smoothed ball — "what the keeper saw". Unlike
    # ``pov`` (raw head facing) it keeps the ball in frame.
    eyes_fov_deg: float = 70.0
    eyes_up_m: float = 0.10
    eyes_forward_m: float = 0.18
    eyes_smooth_frames: int = 5
    # Shot-planning knobs for the goal/orbit rigs (shorts editing):
    # ``focus`` picks what they frame — "centroid" (all players + ball,
    # the default), "ball", or a player id (e.g. "P006", the scorer) —
    # smoothed over ``focus_smooth_frames``. ``orbit_start_frame``/
    # ``orbit_end_frame`` confine the orbit sweep to a frame window (the
    # angle holds at the ends outside it); -1 = the shot's first/last frame.
    focus: str = "centroid"
    focus_smooth_frames: int = 9
    orbit_start_frame: int = -1
    orbit_end_frame: int = -1


def _head_pose_world(
    track: "SmplWorldTrack",
    i: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(head_pos, head_R, ok)`` for frame index ``i``.

    ``ok`` is ``True`` on the normal FK path and ``False`` when FK raises
    and we fall back to root-only head pose. Callers should halve frame
    confidence when ``ok`` is ``False``.
    """
    try:
        pos, R = compute_joint_world_pose(
            track.thetas[i], track.root_R[i], track.root_t[i], HEAD_JOINT_IDX
        )
        return pos, R, True
    except (ValueError, IndexError, np.linalg.LinAlgError):
        pos = np.asarray(track.root_t[i], dtype=np.float64) + np.array([0.0, 0.0, 1.6])
        return pos, np.asarray(track.root_R[i], dtype=np.float64), False


def _ball_xyz_by_frame(ball_track: object) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if ball_track is None:
        return out
    for f in getattr(ball_track, "frames", ()):
        xyz = getattr(f, "world_xyz", None)
        if xyz is not None:
            out[int(f.frame)] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return out


def _make_track(
    clip_id: str,
    image_size: tuple[int, int],
    fps: float,
    K: list[list[float]],
    per_frame: list[_FrameTuple],
) -> CameraTrack:
    frames = tuple(
        CameraFrame(
            frame=int(fr),
            K=[list(map(float, row)) for row in K],
            R=[list(map(float, row)) for row in R],
            confidence=float(conf),
            is_anchor=False,
            t=[float(x) for x in t],
        )
        for (fr, R, t, conf) in per_frame
    )
    if per_frame:
        centres = np.array(
            [-(np.asarray(R)).T @ np.asarray(t) for (_, R, t, _) in per_frame]
        )
        t_world = centres.mean(axis=0).tolist()
    else:
        t_world = [0.0, 0.0, 0.0]
    return CameraTrack(
        clip_id=clip_id,
        fps=float(fps),
        image_size=(int(image_size[0]), int(image_size[1])),
        t_world=t_world,
        frames=frames,
    )


def build_pov_track(
    track: "SmplWorldTrack",
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Build a first-person (POV) CameraTrack from a player's SmplWorldTrack.

    The camera is placed at the player's head joint and aimed in the
    player's facing direction (SMPL canonical forward rotated into world).
    """
    K = intrinsics_from_fov(cfg.pov_fov_deg, image_size)
    per_frame: list[_FrameTuple] = []
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R, ok = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        R, t = look_at_view(head_pos, head_pos + facing)
        conf = float(track.confidence[i]) * (1.0 if ok else 0.5)
        per_frame.append((int(fr), R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_ots_track(
    track: "SmplWorldTrack",
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Build an over-the-shoulder (OTS) CameraTrack from a player's SmplWorldTrack.

    Camera is positioned slightly behind, above, and optionally to the side
    of the player's head. Target is the ball when available (with short
    occlusion bridging), otherwise falls back to a point ahead of the player.
    """
    K = intrinsics_from_fov(cfg.ots_fov_deg, image_size)
    ball_xyz = _ball_xyz_by_frame(ball_track)
    per_frame: list[_FrameTuple] = []
    last_target: np.ndarray | None = None
    frames_since_ball = 0
    for i, fr in enumerate(np.asarray(track.frames).tolist()):
        head_pos, head_R, ok = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        facing_ground = _normalize(np.array([facing[0], facing[1], 0.0]))
        right_ground = _normalize(np.cross(facing_ground, WORLD_UP))
        center = (
            head_pos
            - cfg.ots_back_m * facing_ground
            + cfg.ots_up_m * WORLD_UP
            + cfg.ots_right_m * right_ground
        )
        target = ball_xyz.get(int(fr))
        if target is not None:
            last_target = target
            frames_since_ball = 0
        elif (
            last_target is not None
            and frames_since_ball < cfg.ball_target_max_occlusion_frames
        ):
            target = last_target
            frames_since_ball += 1
        else:
            target = head_pos + facing * 10.0
        R, t = look_at_view(center, target)
        conf = float(track.confidence[i]) * (1.0 if ok else 0.5)
        per_frame.append((int(fr), R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered, edge-padded moving average over axis 0 (same length)."""
    arr = np.asarray(arr, dtype=np.float64)
    win = max(1, int(window))
    if win == 1 or len(arr) == 0:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    return np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(arr.shape[1])],
        axis=1,
    )[: len(arr)]


def build_eyes_track(
    track: "SmplWorldTrack",
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Eye-line camera: the player's (smoothed) eye position aimed at the
    (smoothed) ball — e.g. ``eyes:<keeper>`` for a "would you save this?"
    shot.

    The eye point is the head joint lifted ``eyes_up_m`` and pushed
    ``eyes_forward_m`` along the ground-projected facing so the near
    clip never lands inside the player's own head/outline hull. The ball
    target bridges short occlusions (``ball_target_max_occlusion_frames``)
    then falls back to a point ahead of the player; both the eye path and
    the target path are moving-averaged over ``eyes_smooth_frames`` so
    GVHMR head jitter doesn't shake the frame.
    """
    K = intrinsics_from_fov(cfg.eyes_fov_deg, image_size)
    ball_xyz = _ball_xyz_by_frame(ball_track)
    frames = [int(f) for f in np.asarray(track.frames).tolist()]
    if not frames:
        return _make_track(clip_id, image_size, fps, K, [])
    eyes, targets, confs = [], [], []
    last_target: np.ndarray | None = None
    since_ball = 0
    for i, fr in enumerate(frames):
        head_pos, head_R, ok = _head_pose_world(track, i)
        facing = _normalize(head_R @ FACE_AXIS_CANONICAL)
        facing_ground = _normalize(np.array([facing[0], facing[1], 0.0]))
        eye = head_pos + cfg.eyes_up_m * WORLD_UP + cfg.eyes_forward_m * facing_ground
        target = ball_xyz.get(fr)
        if target is not None:
            last_target, since_ball = target, 0
        elif last_target is not None and since_ball < cfg.ball_target_max_occlusion_frames:
            target = last_target
            since_ball += 1
        else:
            target = eye + facing_ground * 10.0
        eyes.append(eye)
        targets.append(target)
        confs.append(float(track.confidence[i]) * (1.0 if ok else 0.5))
    eyes_s = _moving_average(np.asarray(eyes), cfg.eyes_smooth_frames)
    targets_s = _moving_average(np.asarray(targets), cfg.eyes_smooth_frames)
    per_frame: list[_FrameTuple] = []
    for fr, eye, target, conf in zip(frames, eyes_s, targets_s, confs):
        R, t = _look_at_safe(eye, target)
        per_frame.append((fr, R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def _smoothed_centroid(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    smooth_frames: int,
) -> tuple[list[int], np.ndarray]:
    """Per-frame centroid of all player root positions (+ ball when
    tracked), smoothed with a centered moving average over
    ``smooth_frames`` frames (edge-padded) so single-frame jitter never
    reaches a camera that follows it.

    Shared "follow the action" primitive for ``build_drone_track`` and
    the goal/goalline/orbit/dolly rigs. Returns ``(frames, smoothed)``:
    ``frames`` is the sorted union of frame indices across ``tracks``
    (empty when ``tracks`` contributes none at all); ``smoothed`` is an
    ``(N, 3)`` array aligned to it.
    """
    ball_xyz = _ball_xyz_by_frame(ball_track) if ball_track is not None else {}

    # Union of frame indices across tracks.
    all_frames = sorted({int(f) for tr in tracks
                         for f in np.asarray(tr.frames).tolist()})
    if not all_frames:
        return [], np.zeros((0, 3))

    # Raw per-frame centroid.
    by_frame_pos: dict[int, list[np.ndarray]] = {f: [] for f in all_frames}
    for tr in tracks:
        idx = {int(f): i for i, f in enumerate(np.asarray(tr.frames).tolist())}
        for f, i in idx.items():
            by_frame_pos[f].append(np.asarray(tr.root_t[i], dtype=np.float64))
    raw = []
    for f in all_frames:
        pts = list(by_frame_pos[f])
        if f in ball_xyz:
            pts.append(np.asarray(ball_xyz[f], dtype=np.float64))
        raw.append(np.mean(pts, axis=0))
    raw_arr = np.asarray(raw)

    # Centered moving average (edge-padded).
    win = max(1, int(smooth_frames))
    pad = win // 2
    padded = np.pad(raw_arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    smooth = np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)],
        axis=1,
    )[: len(all_frames)]
    return all_frames, smooth


def _focus_path(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
) -> tuple[list[int], np.ndarray]:
    """``(frames, smoothed_xyz)`` the goal/orbit rigs aim at, per
    ``cfg.focus``: the action centroid (default), the ball (held across
    gaps, leading gap back-filled with the first sighting; falls back to
    the centroid when the ball is never seen), or one player's root."""
    all_frames, centroid = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    focus = (cfg.focus or "centroid").strip()
    if focus == "centroid" or not all_frames:
        return all_frames, centroid
    if focus == "ball":
        ball_xyz = _ball_xyz_by_frame(ball_track) if ball_track is not None else {}
        if not ball_xyz:
            return all_frames, centroid
        first = ball_xyz[min(ball_xyz)]
        held, last = [], first
        for f in all_frames:
            last = ball_xyz.get(f, last)
            held.append(last)
        return all_frames, _moving_average(np.asarray(held), cfg.focus_smooth_frames)
    player = next((t for t in tracks if t.player_id == focus), None)
    if player is None:
        raise ValueError(f"rig focus {focus!r}: no such player track")
    idx = {int(f): i for i, f in enumerate(np.asarray(player.frames).tolist())}
    raw, last = [], None
    for k, f in enumerate(all_frames):
        if f in idx:
            last = np.asarray(player.root_t[idx[f]], dtype=np.float64)
        raw.append(last if last is not None else centroid[k])
    return all_frames, _moving_average(np.asarray(raw), cfg.focus_smooth_frames)


def _orbit_fraction(frame: int, i: int, n: int, cfg: RigConfig, frames: list[int]) -> float:
    start = cfg.orbit_start_frame if cfg.orbit_start_frame >= 0 else frames[0]
    end = cfg.orbit_end_frame if cfg.orbit_end_frame >= 0 else frames[-1]
    if end <= start:
        return i / (n - 1) if n > 1 else 0.0
    return min(1.0, max(0.0, (frame - start) / (end - start)))


def build_drone_track(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Elevated tactical camera tracking the smoothed action centroid.

    Per frame the action centroid is the mean of all player root
    positions present on that frame plus the ball (when tracked); the
    camera sits ``drone_back_m`` toward the near touchline (-y) and
    ``drone_height_m`` up, looking at the centroid. The centroid is
    smoothed with a centered moving average over ``drone_smooth_frames``
    frames so single-frame jitter never reaches the camera.
    """
    K = intrinsics_from_fov(cfg.drone_fov_deg, image_size)
    all_frames, smooth = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    per_frame: list[_FrameTuple] = []
    for f, target in zip(all_frames, smooth):
        centre = np.array([target[0],
                           target[1] - cfg.drone_back_m,
                           cfg.drone_height_m])
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_goal_track(
    side: str,
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Low behind-goal camera (``goal:left`` / ``goal:right``).

    Fixed at ``goal_back_m`` behind the named goal's line, centred on
    the goal mouth (pitch y = width/2), ``goal_height_m`` up — low and
    looking straight down the length of the pitch. Pans (but does not
    translate) with the smoothed action centroid, same smoothing
    primitive and window (``drone_smooth_frames``) as ``build_drone_track``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"build_goal_track: side must be 'left' or 'right', got {side!r}")
    K = intrinsics_from_fov(cfg.goal_fov_deg, image_size)
    all_frames, smooth = _focus_path(tracks, ball_track, cfg)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    goal_x = 0.0 if side == "left" else PITCH_LENGTH
    behind = -1.0 if side == "left" else 1.0  # step away from the pitch
    centre = np.array([
        goal_x + behind * cfg.goal_back_m,
        PITCH_WIDTH / 2.0,
        cfg.goal_height_m,
    ])
    per_frame: list[_FrameTuple] = []
    for f, target in zip(all_frames, smooth):
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_goalline_track(
    side: str,
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Goal-line camera (``goalline:left`` / ``goalline:right``): on the
    goal line, near the (pitch.py-convention) "near" post, low and
    dramatic.

    Anchored on ``FIFA_LANDMARKS["{side}_goal_near_post_base"]`` (the
    single-source-of-truth pitch geometry — never hand-duplicated),
    offset ``goalline_post_offset_m`` in from the post toward the goal
    centre so the camera isn't literally inside the goal frame, at
    ``goalline_height_m``. Pans with the smoothed action centroid like
    ``build_goal_track``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"build_goalline_track: side must be 'left' or 'right', got {side!r}")
    K = intrinsics_from_fov(cfg.goalline_fov_deg, image_size)
    all_frames, smooth = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    post = FIFA_LANDMARKS[f"{side}_goal_near_post_base"]
    goal_y_centre = PITCH_WIDTH / 2.0
    inward_y = 1.0 if post[1] < goal_y_centre else -1.0
    centre = np.array([
        float(post[0]),
        float(post[1]) + inward_y * cfg.goalline_post_offset_m,
        cfg.goalline_height_m,
    ])
    per_frame: list[_FrameTuple] = []
    for f, target in zip(all_frames, smooth):
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_orbit_track(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Arc sweep at fixed radius/height around a pivot — the smoothed
    action centroid by default (same primitive as ``build_drone_track``).

    Azimuth interpolates linearly from ``-orbit_sweep_deg/2`` to
    ``+orbit_sweep_deg/2`` across the full frame span (angle 0 matches
    the drone's "toward the near touchline (-y)" convention), at
    ``orbit_radius_m``/``orbit_height_m`` from the pivot, always
    looking at it.
    """
    K = intrinsics_from_fov(cfg.orbit_fov_deg, image_size)
    all_frames, smooth = _focus_path(tracks, ball_track, cfg)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    n = len(all_frames)
    per_frame: list[_FrameTuple] = []
    for i, (f, target) in enumerate(zip(all_frames, smooth)):
        frac = _orbit_fraction(int(f), i, n, cfg, all_frames)
        angle = math.radians(-cfg.orbit_sweep_deg / 2.0 + cfg.orbit_sweep_deg * frac)
        centre = np.array([
            target[0] + cfg.orbit_radius_m * math.sin(angle),
            target[1] - cfg.orbit_radius_m * math.cos(angle),
            cfg.orbit_height_m,
        ])
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_chase_track(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Ball-chase camera: trails the smoothed ball's horizontal velocity
    vector, falling back to the smoothed action centroid when the ball
    is missing on a frame or its smoothed speed is below
    ``chase_min_speed_m_s`` ("static").

    The ball position sequence is smoothed with the same centered
    moving-average primitive as the centroid (window
    ``chase_smooth_frames``), holding the last known position across
    gaps so a velocity is still computable either side of a short
    occlusion; frames before the first ever ball sighting use the
    centroid fallback outright. The trailing direction persists across
    fallback frames (rather than snapping to a default) so the camera
    doesn't jerk when the ball reappears.
    """
    K = intrinsics_from_fov(cfg.chase_fov_deg, image_size)
    all_frames, centroid = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    ball_xyz = _ball_xyz_by_frame(ball_track) if ball_track is not None else {}

    # Ball position per frame, held across gaps; leading gap (before the
    # first sighting) stays None so those frames fall back outright.
    raw_ball: list[np.ndarray | None] = []
    last: np.ndarray | None = None
    for f in all_frames:
        pos = ball_xyz.get(f)
        if pos is not None:
            last = np.asarray(pos, dtype=np.float64)
        raw_ball.append(last)

    have_ball = [p is not None for p in raw_ball]
    smooth_ball: list[np.ndarray | None] = list(raw_ball)
    if any(have_ball):
        first_known = next(p for p in raw_ball if p is not None)
        filled = np.asarray([p if p is not None else first_known for p in raw_ball])
        win = max(1, int(cfg.chase_smooth_frames))
        pad = win // 2
        padded = np.pad(filled, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(win) / win
        smoothed = np.stack(
            [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)],
            axis=1,
        )[: len(all_frames)]
        smooth_ball = [smoothed[i] if have_ball[i] else None for i in range(len(all_frames))]

    n = len(all_frames)
    default_dir = np.array([0.0, -1.0, 0.0])  # matches drone/dolly "toward near touchline"
    trailing_dir = default_dir
    per_frame: list[_FrameTuple] = []
    for i, f in enumerate(all_frames):
        ball_here = smooth_ball[i]
        target = None
        if ball_here is not None:
            prev_i = max(0, i - 1)
            next_i = min(n - 1, i + 1)
            if smooth_ball[prev_i] is not None and smooth_ball[next_i] is not None and next_i > prev_i:
                vel = (smooth_ball[next_i] - smooth_ball[prev_i]) / (next_i - prev_i)
                vel_horizontal = np.array([vel[0], vel[1], 0.0])
                speed = float(np.linalg.norm(vel_horizontal)) * fps
                if speed >= cfg.chase_min_speed_m_s:
                    trailing_dir = _normalize(vel_horizontal)
                    target = ball_here
        conf = 1.0
        if target is None:
            target = centroid[i]
            conf = 0.5
        centre = target - cfg.chase_back_m * trailing_dir + np.array([0.0, 0.0, cfg.chase_height_m])
        R, t = _look_at_safe(centre, target)
        per_frame.append((int(f), R, t, conf))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_dolly_track(
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Low sideline dolly: fixed y near the nearside touchline
    (``dolly_y_m``, negative = just off the pitch), x tracking the
    smoothed action centroid, long lens, low to the ground.
    """
    K = intrinsics_from_fov(cfg.dolly_fov_deg, image_size)
    all_frames, smooth = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    if not all_frames:
        return _make_track(clip_id, image_size, fps, K, [])

    per_frame: list[_FrameTuple] = []
    for f, target in zip(all_frames, smooth):
        centre = np.array([target[0], cfg.dolly_y_m, cfg.dolly_height_m])
        R, t = look_at_view(centre, target)
        per_frame.append((int(f), R, t, 1.0))
    return _make_track(clip_id, image_size, fps, K, per_frame)


def build_stadium_track(
    camera_id: str,
    tracks: Sequence["SmplWorldTrack"],
    ball_track: object,
    cfg: RigConfig,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Stable stadium rigs: whole-pitch overhead, two touchlines and corners.

    Tactical framing includes a 5m run-off on every side at any aspect ratio.
    Sideline/corner cameras sit inside the board/stand perimeter and above
    the goals so that dressing does not obstruct their view of play.
    """
    frames, targets = _smoothed_centroid(tracks, ball_track, cfg.drone_smooth_frames)
    if camera_id == "tactical":
        fov = cfg.tactical_fov_deg
        half_tan = math.tan(math.radians(fov)/2)
        aspect = image_size[0]/image_size[1]
        height = max((PITCH_LENGTH/2+5)/half_tan,
                     (PITCH_WIDTH/2+5)*aspect/half_tan)
        center = np.array([PITCH_LENGTH/2, PITCH_WIDTH/2, height])
        target = np.array([PITCH_LENGTH/2, PITCH_WIDTH/2, 0.])
        R, t = look_at_view(center, target, up=np.array([0.,1.,0.]))
        per_frame = [(int(f), R, t, 1.) for f in frames]
    else:
        if camera_id in ("sideline:near", "sideline:far"):
            far = camera_id.endswith(":far")
            center = np.array([PITCH_LENGTH/2, PITCH_WIDTH+6 if far else -6,
                               cfg.sideline_height_m])
            fov = cfg.sideline_fov_deg
        elif camera_id in ("corner:left", "corner:right"):
            center = np.array([-6 if camera_id.endswith(":left") else PITCH_LENGTH+6,
                               -5, cfg.corner_height_m])
            fov = cfg.corner_fov_deg
        else:
            raise ValueError(f"Unknown stadium camera {camera_id!r}")
        per_frame = []
        for f, target in zip(frames, targets):
            R, t = _look_at_safe(center, target)
            per_frame.append((int(f), R, t, 1.))
    return _make_track(clip_id, image_size, fps,
                       intrinsics_from_fov(fov,image_size), per_frame)

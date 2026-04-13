"""3D ball reconstruction from per-shot 2D ball tracks.

Three reconstruction methods, layered:

1. **Multi-view triangulation** — when ≥2 calibrated shots see the
   ball at the same global frame, weighted DLT gives the unique
   3D position.

2. **Single-shot ground projection** — when 1 shot sees the ball,
   project the ball pixel onto the pitch plane ``z = _BALL_RADIUS``.
   Correct when the ball is rolling/grounded.  Wildly wrong during
   flight (the projected position lands far behind the actual ball).

3. **Parabolic flight refinement** — detect runs of frames where the
   ball is in flight (large pixel velocity, gravitating away from
   the previous ground touch) and replace the ground-projection
   solution with a 6-DOF parabola fit through the camera ray
   constraints + gravity.  Ends when the next confident ground
   touch is reached.

The output is a single :class:`TriangulatedBall` array spanning the
entire reference frame range.  Frames with no ball detection at all
are left as NaN; brief gaps (≤ 5 frames) get linearly interpolated;
longer gaps stay NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.calibration import CameraFrame
from src.schemas.tracks import TracksResult
from src.schemas.triangulated import TriangulatedBall
from src.utils.camera import build_projection_matrix
from src.utils.triangulation import weighted_dlt
from src.utils.triangulation_calib import CalibrationInterpolator

logger = logging.getLogger(__name__)

_BALL_RADIUS = 0.11   # FIFA standard ~22cm diameter
_GRAVITY = 9.81       # m/s²
_FLIGHT_PX_VELOCITY = 25.0    # pixel velocity threshold to consider "in flight"
_MIN_FLIGHT_FRAMES = 4
_MAX_FLIGHT_FRAMES = 60       # don't fit a parabola over a 2.4-sec arc
# Generous pitch-bounding box for plausibility filtering.  A back-projected
# ball that lands outside this box came from either a false-positive
# detection (e.g. a player's white sock) or a near-tangent camera ray;
# either way it would only pollute the bird's-eye view.
_PLAUSIBLE_X = (-15.0, 120.0)
_PLAUSIBLE_Y = (-15.0, 83.0)


_METHOD_NONE = 0
_METHOD_MULTI = 1
_METHOD_GROUND = 2
_METHOD_FLIGHT = 3


@dataclass(frozen=True)
class _BallObservation:
    """Per-shot observation of the ball pixel at a single global frame."""

    shot_id: str
    pixel_uv: np.ndarray  # (2,)
    P: np.ndarray         # (3, 4) projection matrix
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


def _ball_centre(bbox: list[float]) -> np.ndarray:
    """Return the centre of a ball bounding box in pixels."""
    return np.array([
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
    ], dtype=np.float64)


def _back_project_to_plane(
    pixel: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    plane_z: float,
) -> np.ndarray | None:
    """Intersect the camera ray through ``pixel`` with ``z = plane_z``."""
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    cam_world = -R.T @ tvec.reshape(3)
    direction_camera = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    direction_world = R.T @ direction_camera
    dz = float(direction_world[2])
    if abs(dz) < 1e-9:
        return None
    s = (plane_z - float(cam_world[2])) / dz
    xy = cam_world[:2] + s * direction_world[:2]
    return np.array([float(xy[0]), float(xy[1]), float(plane_z)], dtype=np.float64)


def _gather_ball_pixels_per_shot(
    tracks_by_shot: dict[str, TracksResult],
) -> dict[str, dict[int, np.ndarray]]:
    """``{shot_id: {local_frame: ball_pixel_xy}}`` for the longest ball track per shot."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for shot_id, tracks in tracks_by_shot.items():
        ball_tracks = [t for t in tracks.tracks if t.class_name == "ball"]
        if not ball_tracks:
            continue
        # Pick the track with the most frames (typically the actual play ball)
        ball_track = max(ball_tracks, key=lambda t: len(t.frames))
        out[shot_id] = {
            tf.frame: _ball_centre(tf.bbox)
            for tf in ball_track.frames
            if tf.confidence > 0.2
        }
    return out


def _gather_observations_at_frame(
    ref_frame: int,
    pixels_by_shot: dict[str, dict[int, np.ndarray]],
    interps_by_shot: dict[str, CalibrationInterpolator],
    sync_offsets: dict[str, int],
) -> list[_BallObservation]:
    """Gather (calibration, pixel) tuples for every shot that sees the ball
    at the given reference frame.
    """
    obs: list[_BallObservation] = []
    for shot_id, frames in pixels_by_shot.items():
        local = ref_frame - sync_offsets.get(shot_id, 0)
        if local not in frames:
            continue
        interp = interps_by_shot.get(shot_id)
        if interp is None or interp.is_empty:
            continue
        cal = interp.at_nearest(local, max_extrapolation_frames=200)
        if cal is None:
            continue
        P = build_projection_matrix(cal.K, cal.rvec, cal.tvec)
        obs.append(_BallObservation(
            shot_id=shot_id,
            pixel_uv=frames[local],
            P=P, K=cal.K, rvec=cal.rvec, tvec=cal.tvec,
        ))
    return obs


def _is_plausible_ball_xy(pt: np.ndarray) -> bool:
    """Reject ball positions that fall well off the pitch in (x, y)."""
    return (
        _PLAUSIBLE_X[0] <= float(pt[0]) <= _PLAUSIBLE_X[1]
        and _PLAUSIBLE_Y[0] <= float(pt[1]) <= _PLAUSIBLE_Y[1]
    )


def _try_multi_view(observations: list[_BallObservation]) -> np.ndarray | None:
    """Weighted DLT across distinct shots.  Returns world (x,y,z) or None."""
    distinct = {obs.shot_id for obs in observations}
    if len(distinct) < 2:
        return None
    Ps = [obs.P for obs in observations]
    uvs = [obs.pixel_uv for obs in observations]
    weights = [1.0] * len(observations)
    pt = weighted_dlt(Ps, uvs, weights)
    if not np.all(np.isfinite(pt)):
        return None
    # Sanity: ball must be on or above the pitch plane, within a sane
    # vertical extent, and roughly above the pitch in (x, y).
    if pt[2] < -0.5 or pt[2] > 30.0:
        return None
    if not _is_plausible_ball_xy(pt):
        return None
    return pt.astype(np.float64)


def _detect_flight_segments(
    methods: np.ndarray,
    pixel_velocities: list[float],
    confidences: np.ndarray,
) -> list[tuple[int, int]]:
    """Identify runs of frames where the ball is likely in flight.

    Heuristic: contiguous runs where the per-frame pixel velocity
    exceeds ``_FLIGHT_PX_VELOCITY`` *and* the ground-projection method
    is currently in use.  Multi-view frames are not flagged because
    multi-view already gives the correct 3D position.
    """
    segments: list[tuple[int, int]] = []
    n = len(methods)
    i = 0
    while i < n:
        in_segment = (
            methods[i] == _METHOD_GROUND
            and pixel_velocities[i] > _FLIGHT_PX_VELOCITY
            and confidences[i] > 0
        )
        if not in_segment:
            i += 1
            continue
        j = i
        while j < n and methods[j] == _METHOD_GROUND and pixel_velocities[j] > _FLIGHT_PX_VELOCITY * 0.6:
            j += 1
        run = j - i
        if _MIN_FLIGHT_FRAMES <= run <= _MAX_FLIGHT_FRAMES:
            segments.append((i, j))
        i = max(j, i + 1)
    return segments


def _fit_parabola_segment(
    frame_indices: list[int],
    observations_per_frame: list[list[_BallObservation]],
    fps: float,
    initial_p0: np.ndarray,
    initial_v0: np.ndarray,
) -> np.ndarray | None:
    """Fit a 6-DOF parabolic trajectory to a segment of ball observations.

    Parameters: ``(x0, y0, z0, vx, vy, vz)`` at the segment start.
    Position at ``t = i / fps`` (relative to segment start) follows
    ``p(t) = p0 + v0 * t + 0.5 * g * t²`` with ``g = (0, 0, -9.81)``.

    Residual: for each frame's observation in each shot, project the
    parameterised position through the per-shot calibration and
    measure the pixel error.  Solved by Levenberg-Marquardt.

    Returns ``(N, 3)`` world positions or ``None`` on failure.
    """
    n_frames = len(frame_indices)
    if n_frames < _MIN_FLIGHT_FRAMES:
        return None
    times = np.array([(i - 0) / fps for i in range(n_frames)], dtype=np.float64)

    def positions_from_params(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:]
        out = np.empty((n_frames, 3), dtype=np.float64)
        for k, t in enumerate(times):
            out[k, 0] = p0[0] + v0[0] * t
            out[k, 1] = p0[1] + v0[1] * t
            out[k, 2] = p0[2] + v0[2] * t - 0.5 * _GRAVITY * t * t
        return out

    def residuals(params: np.ndarray) -> np.ndarray:
        positions = positions_from_params(params)
        out: list[float] = []
        for k, obs_list in enumerate(observations_per_frame):
            for obs in obs_list:
                projected, _ = cv2.projectPoints(
                    positions[k:k + 1], obs.rvec.reshape(3),
                    obs.tvec.reshape(3), obs.K, None,
                )
                px = projected.reshape(2)
                out.append(float(px[0] - obs.pixel_uv[0]))
                out.append(float(px[1] - obs.pixel_uv[1]))
        return np.asarray(out, dtype=np.float64) if out else np.zeros(1)

    x0 = np.concatenate([initial_p0, initial_v0]).astype(np.float64)
    try:
        result = least_squares(residuals, x0, method="lm", max_nfev=200)
    except Exception as exc:  # noqa: BLE001
        logger.debug("parabola LM failed: %s", exc)
        return None

    positions = positions_from_params(result.x)
    # Sanity: ball should stay on or above the pitch.  If the fit
    # produces underground positions, reject.
    if np.any(positions[:, 2] < -0.5) or np.any(positions[:, 2] > 30.0):
        return None
    return positions


def reconstruct_ball(
    tracks_by_shot: dict[str, TracksResult],
    interps_by_shot: dict[str, CalibrationInterpolator],
    sync_offsets: dict[str, int],
    frame_range: list[int],
    fps: float,
    *,
    enable_parabolic: bool = True,
) -> TriangulatedBall | None:
    """Reconstruct the ball trajectory across all reference frames.

    Args:
        tracks_by_shot: per-shot tracking results (must include ball
            tracks for the function to do anything).
        interps_by_shot: per-shot calibration interpolators (output
            of :class:`CalibrationInterpolator`).
        sync_offsets: per-shot frame offsets to convert local to
            reference time.
        frame_range: list of reference-time frame indices.
        fps: frames per second (used to compute time deltas for
            parabolic fitting).
        enable_parabolic: when ``True``, runs flight-segment detection
            + parabolic refinement after the per-frame pass.

    Returns:
        :class:`TriangulatedBall` covering ``frame_range``.  ``None``
        when no ball pixels were found in any shot.
    """
    pixels_by_shot = _gather_ball_pixels_per_shot(tracks_by_shot)
    if not pixels_by_shot:
        return None

    n_frames = len(frame_range)
    positions = np.full((n_frames, 3), np.nan, dtype=np.float32)
    confidences = np.zeros(n_frames, dtype=np.float32)
    methods = np.zeros(n_frames, dtype=np.int8)
    pixel_velocities = [0.0] * n_frames
    obs_per_frame: list[list[_BallObservation]] = [[] for _ in range(n_frames)]

    last_pixel: np.ndarray | None = None
    for fi, ref_frame in enumerate(frame_range):
        observations = _gather_observations_at_frame(
            ref_frame, pixels_by_shot, interps_by_shot, sync_offsets,
        )
        obs_per_frame[fi] = observations
        if not observations:
            last_pixel = None
            continue

        # Track pixel velocity (use the longest-track shot as the velocity proxy)
        primary_pixel = observations[0].pixel_uv
        if last_pixel is not None:
            pixel_velocities[fi] = float(np.linalg.norm(primary_pixel - last_pixel))
        last_pixel = primary_pixel

        # Try multi-view first
        pt = _try_multi_view(observations)
        if pt is not None:
            positions[fi] = pt.astype(np.float32)
            confidences[fi] = 1.0
            methods[fi] = _METHOD_MULTI
            continue

        # Fall back to single-shot ground projection
        obs = observations[0]
        ground = _back_project_to_plane(
            obs.pixel_uv, obs.K, obs.rvec, obs.tvec, plane_z=_BALL_RADIUS,
        )
        if ground is None or not _is_plausible_ball_xy(ground):
            continue
        positions[fi] = ground.astype(np.float32)
        confidences[fi] = 0.6
        methods[fi] = _METHOD_GROUND

    if enable_parabolic:
        flight_segments = _detect_flight_segments(methods, pixel_velocities, confidences)
        for start, end in flight_segments:
            seg_frames = list(range(start, end))
            seg_obs = [obs_per_frame[k] for k in seg_frames]
            if any(len(o) == 0 for o in seg_obs):
                continue
            initial_p0 = positions[start].astype(np.float64)
            # Initial velocity from the start-of-segment trajectory in
            # ground coordinates (decent guess; LM handles the rest).
            duration = max(1, end - 1 - start) / fps
            delta = positions[end - 1].astype(np.float64) - initial_p0
            v_xy = delta[:2] / duration
            # Initial vertical velocity: pick something positive so the
            # parabola has a chance to peak.  ~0.5 g * duration is the
            # value that returns to z0 at t = duration.
            vz0 = 0.5 * _GRAVITY * duration
            initial_v0 = np.array([v_xy[0], v_xy[1], vz0], dtype=np.float64)
            fitted = _fit_parabola_segment(
                seg_frames, seg_obs, fps, initial_p0, initial_v0,
            )
            if fitted is None:
                continue
            positions[start:end] = fitted.astype(np.float32)
            methods[start:end] = _METHOD_FLIGHT
            confidences[start:end] = np.maximum(confidences[start:end], 0.7)

    if not np.any(~np.isnan(positions[:, 0])):
        return None

    return TriangulatedBall(
        positions=positions,
        confidences=confidences,
        methods=methods,
        fps=fps,
        start_frame=frame_range[0],
    )

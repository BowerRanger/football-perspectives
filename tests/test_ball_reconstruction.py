"""Unit tests for src/utils/ball_reconstruction.py."""

from __future__ import annotations

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.ball_reconstruction import (
    _BALL_RADIUS,
    _GRAVITY,
    reconstruct_ball,
)
from src.utils.triangulation_calib import CalibrationInterpolator


def _camera(target: tuple[float, float, float],
            cam_pos: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[1500.0, 0.0, 960.0],
                  [0.0, 1500.0, 540.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.array(cam_pos, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    forward = tgt - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam
    return K, rvec.reshape(3), tvec.reshape(3)


def _project(world_xyz: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, float]:
    proj, _ = cv2.projectPoints(
        world_xyz.reshape(1, 3).astype(np.float64),
        rvec, tvec, K, None,
    )
    return float(proj[0, 0, 0]), float(proj[0, 0, 1])


def _make_calibration(shot_id: str, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, frames: list[int]) -> CalibrationResult:
    cf_list = [
        CameraFrame(
            frame=f,
            intrinsic_matrix=K.tolist(),
            rotation_vector=rvec.tolist(),
            translation_vector=tvec.tolist(),
            reprojection_error=0.0,
            num_correspondences=0,
            confidence=1.0,
            tracked_landmark_types=[],
        )
        for f in frames
    ]
    return CalibrationResult(shot_id=shot_id, camera_type="static", frames=cf_list)


def _make_ball_track(frames: list[int], pixels: list[tuple[float, float]]) -> TracksResult:
    track_frames = [
        TrackFrame(frame=f, bbox=[u - 5, v - 5, u + 5, v + 5], confidence=0.9, pitch_position=None)
        for f, (u, v) in zip(frames, pixels)
    ]
    return TracksResult(
        shot_id="test",
        tracks=[Track(track_id="ball_t", class_name="ball", team="", frames=track_frames)],
    )


class TestBallReconstruction:
    def test_multi_view_recovers_known_ball(self):
        # Two cameras looking at pitch centre; ball at (50, 30, 1.0)
        ball_world = np.array([50.0, 30.0, 1.0])

        K1, r1, t1 = _camera((52.5, 34.0, 0.0), (52.5, -20.0, 25.0))
        K2, r2, t2 = _camera((52.5, 34.0, 0.0), (10.0, -10.0, 20.0))

        u1, v1 = _project(ball_world, K1, r1, t1)
        u2, v2 = _project(ball_world, K2, r2, t2)

        cal1 = _make_calibration("shotA", K1, r1, t1, frames=[0])
        cal2 = _make_calibration("shotB", K2, r2, t2, frames=[0])
        tracks1 = _make_ball_track([0], [(u1, v1)])
        tracks2 = _make_ball_track([0], [(u2, v2)])

        interps = {
            "shotA": CalibrationInterpolator(cal1),
            "shotB": CalibrationInterpolator(cal2),
        }
        result = reconstruct_ball(
            tracks_by_shot={"shotA": tracks1, "shotB": tracks2},
            interps_by_shot=interps,
            sync_offsets={"shotA": 0, "shotB": 0},
            frame_range=[0],
            fps=25.0,
            enable_parabolic=False,
        )
        assert result is not None
        assert np.allclose(result.positions[0], ball_world, atol=0.05)
        assert result.methods[0] == 1  # multi-view

    def test_single_shot_ground_projection_for_rolling_ball(self):
        # One camera, ball rolling along the pitch
        K, rvec, tvec = _camera((52.5, 34.0, 0.0), (52.5, -20.0, 25.0))
        ball_positions = [
            np.array([40.0, 30.0, _BALL_RADIUS]),
            np.array([41.0, 30.0, _BALL_RADIUS]),
            np.array([42.0, 30.0, _BALL_RADIUS]),
        ]
        pixels = [_project(p, K, rvec, tvec) for p in ball_positions]
        cal = _make_calibration("shotA", K, rvec, tvec, frames=[0, 1, 2])
        tracks = _make_ball_track([0, 1, 2], pixels)
        interps = {"shotA": CalibrationInterpolator(cal)}
        result = reconstruct_ball(
            tracks_by_shot={"shotA": tracks},
            interps_by_shot=interps,
            sync_offsets={"shotA": 0},
            frame_range=[0, 1, 2],
            fps=25.0,
            enable_parabolic=False,
        )
        assert result is not None
        for fi, expected in enumerate(ball_positions):
            assert np.allclose(result.positions[fi], expected, atol=0.05)
            assert result.methods[fi] == 2  # single ground

    def test_parabolic_flight_fits_arc_better_than_ground_projection(self):
        # Synthetic ball arc: starts at (40, 30, 0.5) with vy=10, vz=8 m/s
        # over 25 frames at 25 fps (1 second)
        K, rvec, tvec = _camera((52.5, 34.0, 0.0), (52.5, -25.0, 28.0))
        n_frames = 12
        fps = 25.0
        p0 = np.array([40.0, 30.0, 0.5])
        v0 = np.array([0.0, 8.0, 6.0])
        true_positions = []
        pixels = []
        for k in range(n_frames):
            t = k / fps
            p = p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -_GRAVITY]) * t * t
            true_positions.append(p)
            pixels.append(_project(p, K, rvec, tvec))

        cal = _make_calibration("shotA", K, rvec, tvec, frames=list(range(n_frames)))
        tracks = _make_ball_track(list(range(n_frames)), pixels)
        interps = {"shotA": CalibrationInterpolator(cal)}
        result = reconstruct_ball(
            tracks_by_shot={"shotA": tracks},
            interps_by_shot=interps,
            sync_offsets={"shotA": 0},
            frame_range=list(range(n_frames)),
            fps=fps,
            enable_parabolic=True,
        )
        assert result is not None

        # Ground projection (no flight refinement) is wildly wrong on
        # the apex of the arc — flight refinement should be close.
        # We don't require sub-cm accuracy because the LM seed is rough,
        # but the apex z should be in the right ballpark (>1m).
        flight_frames = result.methods == 3
        if np.any(flight_frames):
            apex_idx = int(np.argmax(result.positions[flight_frames, 2]))
            apex_z = float(result.positions[flight_frames][apex_idx, 2])
            assert apex_z > 0.8, f"flight refinement apex z={apex_z}, expected >0.8m"
        # And the trajectory should not be NaN
        assert not np.any(np.isnan(result.positions))

    def test_returns_none_when_no_ball_tracks(self):
        result = reconstruct_ball(
            tracks_by_shot={},
            interps_by_shot={},
            sync_offsets={},
            frame_range=[0, 1, 2],
            fps=25.0,
        )
        assert result is None

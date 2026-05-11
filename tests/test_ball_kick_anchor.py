"""Unit tests for the Layer 3 kick-anchor helper."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_kick_anchor import (
    KickAnchorCfg,
    find_kick_anchor,
)


def _cfg(**over) -> KickAnchorCfg:
    base = dict(
        enabled=True,
        max_pixel_distance_px=30.0,
        lookahead_frames=4,
        min_pixel_acceleration_px_per_frame=6.0,
        foot_anchor_z_m=0.11,
    )
    base.update(over)
    return KickAnchorCfg(**base)


def _camera():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def test_kick_detected_when_foot_close_and_acceleration_present():
    K, R, t = _camera()
    # Ball pixel positions: stationary, then rapid acceleration.
    ball_uvs = {10: (640.0, 360.0), 11: (645.0, 365.0), 12: (660.0, 380.0), 13: (685.0, 405.0), 14: (720.0, 440.0)}
    foot_uvs = {10: (635.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(),
    )
    assert anchor is not None
    # p0.z should match foot_anchor_z_m.
    assert anchor[2] == pytest.approx(0.11, abs=1e-6)


def test_no_kick_when_foot_far():
    K, R, t = _camera()
    ball_uvs = {10: (640.0, 360.0), 11: (645.0, 365.0), 12: (660.0, 380.0), 13: (685.0, 405.0)}
    foot_uvs = {10: (100.0, 100.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(max_pixel_distance_px=30.0),
    )
    assert anchor is None


def test_no_kick_when_no_acceleration():
    K, R, t = _camera()
    # Constant 1 px/frame motion — no acceleration jump.
    ball_uvs = {i: (640.0 + (i - 10), 360.0) for i in range(10, 15)}
    foot_uvs = {10: (640.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(min_pixel_acceleration_px_per_frame=6.0),
    )
    assert anchor is None


def test_disabled_returns_none():
    K, R, t = _camera()
    ball_uvs = {10: (640.0, 360.0), 11: (660.0, 380.0)}
    foot_uvs = {10: (640.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(enabled=False),
    )
    assert anchor is None


def test_missing_segment_start_frame_returns_none():
    K, R, t = _camera()
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs={11: (660.0, 380.0)},
        foot_uvs_by_frame={10: (640.0, 363.0)},
        K=K, R=R, t=t,
        cfg=_cfg(),
    )
    assert anchor is None

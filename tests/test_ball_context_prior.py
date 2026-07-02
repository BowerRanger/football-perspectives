"""Context prior: pitch / player-proximity / static-in-image penalties,
gentle single-signal semantics, factor floor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.ball_context_prior import (
    ContextPrior,
    ContextPriorCfg,
    bbox_distance_px,
    load_player_boxes,
    rotation_angle_deg,
)

CFG = ContextPriorCfg()


def _camera_pose(yaw_deg: float = 0.0):
    """Broadcast-ish pose; optional yaw about world z to simulate panning."""
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    a = np.deg2rad(yaw_deg)
    yaw = np.array([[np.cos(a), -np.sin(a), 0.0],
                    [np.sin(a), np.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    R = R @ yaw
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _prior(n_frames: int = 90, yaw_per_frame: float = 0.0,
           boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
           cfg: ContextPriorCfg = CFG) -> tuple[ContextPrior, dict]:
    Ks, Rs, ts = {}, {}, {}
    for i in range(n_frames):
        K, R, t = _camera_pose(yaw_deg=i * yaw_per_frame)
        Ks[i], Rs[i], ts[i] = K, R, t
    prior = ContextPrior(
        cfg, per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), pitch_length_m=105.0, pitch_width_m=68.0,
        player_boxes_by_frame=boxes,
    )
    return prior, {"K": Ks, "R": Rs, "t": ts}


def test_on_pitch_detection_with_player_nearby_is_unpenalized():
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    boxes = {0: [(uv[0] - 30, uv[1] - 80, uv[0] + 30, uv[1] + 10)]}
    prior, _ = _prior(boxes=boxes)
    assert prior.factor(0, uv) == pytest.approx(1.0)


def test_single_signal_never_drops_a_confident_detection():
    # No player box within reach (boxes exist that frame) — player penalty
    # alone must keep 0.8 * factor >= drop_below.
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    boxes = {0: [(uv[0] + 500, uv[1] + 500, uv[0] + 560, uv[1] + 620)]}
    prior, _ = _prior(boxes=boxes)
    f = prior.factor(0, uv)
    assert f == pytest.approx(CFG.player_penalty)
    assert 0.8 * f >= CFG.drop_below


def test_off_pitch_ground_intersection_penalized():
    K, R, t = _camera_pose()
    # A point 30 m beyond the far touchline at ground level.
    uv = _project(np.array([52.5, 68.0 + 30.0, 0.11]), K, R, t)
    prior, _ = _prior()
    assert prior.factor(0, uv) == pytest.approx(CFG.pitch_penalty)


def test_unresolvable_ground_ray_is_not_penalized():
    # A pixel above the horizon intersects the ground plane BEHIND the
    # camera (negative depth) — the pitch signal must abstain (airborne
    # balls legitimately do this). Verified empirically: ankle_ray_to_pitch
    # returns a behind-camera point rather than raising for such pixels.
    prior, _ = _prior()
    f = prior.factor(0, (640.0, -2000.0))
    # No boxes provided and pitch abstains -> only static could fire, and
    # there's no history yet.
    assert f == pytest.approx(1.0)


def test_static_in_image_under_pan_penalized_combined_with_player():
    # Same pixel for 60 frames while the camera pans 0.2 deg/frame, and no
    # player anywhere near: static * player must drop a 0.8-conf blob.
    uv = (640.0, 40.0)
    boxes = {i: [(100.0, 600.0, 160.0, 700.0)] for i in range(90)}
    prior, _ = _prior(yaw_per_frame=0.2, boxes=boxes)
    fs = [prior.factor(i, uv) for i in range(60)]
    f_late = fs[-1]
    assert f_late <= CFG.static_penalty * CFG.player_penalty + 1e-9
    assert 0.8 * f_late < CFG.drop_below


def test_static_not_triggered_when_camera_still():
    # Camera fixed: a world-static resting ball is image-static too — the
    # static signal must NOT fire (cam rotation below static_min_cam_deg).
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    prior, _ = _prior(yaw_per_frame=0.0)
    fs = [prior.factor(i, uv) for i in range(60)]
    assert fs[-1] == pytest.approx(1.0)


def test_factor_floor():
    cfg = ContextPriorCfg(pitch_penalty=0.1, player_penalty=0.1,
                          static_penalty=0.1, min_factor=0.1)
    K, R, t = _camera_pose()
    uv = _project(np.array([52.5, 120.0, 0.11]), K, R, t)
    boxes = {0: [(0.0, 0.0, 10.0, 10.0)]}
    prior, _ = _prior(boxes=boxes, cfg=cfg)
    assert prior.factor(0, uv) >= cfg.min_factor


def test_disabled_returns_one():
    prior, _ = _prior(cfg=ContextPriorCfg(enabled=False))
    assert prior.factor(0, (9999.0, -9999.0)) == 1.0


def test_bbox_distance():
    assert bbox_distance_px((5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == 0.0
    assert bbox_distance_px((13.0, 14.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(5.0)


def test_rotation_angle():
    K, R0, _ = _camera_pose(0.0)
    _, R1, _ = _camera_pose(3.0)
    assert rotation_angle_deg(R0, R0) == pytest.approx(0.0, abs=1e-6)
    assert rotation_angle_deg(R0, R1) == pytest.approx(3.0, abs=1e-3)


def test_load_player_boxes_excludes_ball_and_missing_file(tmp_path: Path):
    payload = {
        "shot_id": "play",
        "tracks": [
            {"track_id": "1", "class_name": "player", "team": "A",
             "player_id": "P001", "player_name": "",
             "frames": [{"frame": 3, "bbox": [1.0, 2.0, 3.0, 4.0],
                         "confidence": 0.9, "pitch_position": None,
                         "interpolated": False}]},
            {"track_id": "2", "class_name": "ball", "team": "unknown",
             "player_id": "", "player_name": "",
             "frames": [{"frame": 3, "bbox": [9.0, 9.0, 10.0, 10.0],
                         "confidence": 0.9, "pitch_position": None,
                         "interpolated": False}]},
        ],
    }
    p = tmp_path / "play_tracks.json"
    p.write_text(json.dumps(payload))
    boxes = load_player_boxes(p)
    assert boxes == {3: [(1.0, 2.0, 3.0, 4.0)]}
    assert load_player_boxes(tmp_path / "missing.json") is None


def test_config_block_keys():
    import yaml
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["context_prior"]
    assert cfg["enabled"] is True
    assert cfg["drop_below"] == 0.35
    assert cfg["min_factor"] == 0.1

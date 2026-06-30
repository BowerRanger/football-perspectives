"""End-to-end origi01-like scenario through the automatic pipeline.

One physically-consistent play, no manual anchors:

    roll -> (detection gap) -> kick by a tracked player's foot ->
    ballistic arc -> landing bounce -> roll out (short detection gap
    bridged by appearance NCC).

Asserts the whole chain cooperates: detection + IMM + appearance
bridge, PlayerContext FK, automatic touch/bounce events, auto-anchor
generation, and the piecewise solver (flight bracketed by the touch and
the bounce, rolling elsewhere, continuity throughout).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector

FPS = 30.0


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0]); look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _full_cfg() -> dict:
    return {
        "ball": {
            "detector": "fake",
            # Validates the dense piecewise solver (flight bracketed by the
            # touch and the bounce); pin it now that ``events`` is default.
            "solver": "piecewise",
            "ball_radius_m": 0.11,
            "max_gap_frames": 2,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
                "initial_p_flight": 0.5,
            },
            "spin": {"enabled": False},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
            # Bridge fills the 3-frame post-landing gap (60-62) via NCC.
            "appearance_bridge": {"enabled": True, "max_gap_frames": 3, "template_size_px": 32, "search_radius_px": 64, "min_ncc": 0.6, "template_max_age_frames": 30, "template_update_confidence": 0.5},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_origi01_like_scenario(tmp_path: Path):
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip_path = out / "shots" / "play.mp4"
    n_frames = 90
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    # Physically continuous plan:
    # 0..22   roll toward the kicker; 23..29 detection gap (ball keeps
    #         rolling in the video)
    # 30      kick at p0_kick by P001's left foot
    # 30..59  ballistic arc landing exactly at frame 59
    # 59..89  roll out after the bounce (60..62 detections missing;
    #         appearance bridge recovers them from the rendered frames)
    kick_frame, land_frame = 30, 59
    t_flight = (land_frame - kick_frame) / FPS
    p0_kick = np.array([37.0, 25.0, 0.11])
    v0_kick = np.array([0.0, 8.0, 0.5 * 9.81 * t_flight])
    g_vec = np.array([0.0, 0.0, -9.81])
    landing = p0_kick + v0_kick * t_flight + 0.5 * g_vec * t_flight**2
    v_roll_out = np.array([0.0, 2.0, 0.0])

    def world_at(i: int) -> np.ndarray:
        if i < kick_frame:
            return np.array([40.0 - 0.1 * i, 25.0, 0.11])
        if i <= land_frame:
            dt = (i - kick_frame) / FPS
            return p0_kick + v0_kick * dt + 0.5 * g_vec * dt**2
        dt = (i - land_frame) / FPS
        return landing + v_roll_out * dt

    detections: list[tuple[float, float, float] | None] = []
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720)
    )
    for i in range(n_frames):
        img = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
        u, v = _project(world_at(i), K, R, t)
        cv2.circle(img, (int(u), int(v)), 8, (240, 240, 240), -1)
        writer.write(img)
        if 23 <= i <= 29 or 60 <= i <= 62:
            detections.append(None)
        else:
            detections.append((u, v, 0.85))
    writer.release()

    CameraTrack(
        clip_id="origi-like", fps=FPS, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=1.0, is_anchor=(i == 0))
            for i in range(n_frames)
        ),
    ).save(out / "camera" / "play_camera_track.json")

    ShotsManifest(
        source_file="fake.mp4",
        fps=FPS,
        total_frames=n_frames,
        shots=[
            Shot(
                id="play",
                clip_file="shots/play.mp4",
                start_frame=0,
                end_frame=n_frames - 1,
                start_time=0.0,
                end_time=(n_frames - 1) / FPS,
            )
        ],
    ).save(out / "shots" / "shots_manifest.json")

    # The kicker: SMPL track placing the left foot at the kick point so
    # the automatic contact pipeline (PlayerContext -> touch event ->
    # auto player_touch anchor) supplies the launch node.
    from src.schemas.smpl_world import SmplWorldTrack
    from src.utils.ball_anchor_heights import BONE_TO_SMPL_INDEX
    from src.utils.smpl_skeleton import compute_joint_world

    base_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    thetas0 = np.zeros((24, 3), dtype=np.float32)
    foot_at_origin = compute_joint_world(
        thetas0, base_R, np.zeros(3), BONE_TO_SMPL_INDEX["l_foot"]
    )
    root_t = (p0_kick - foot_at_origin).astype(np.float32)
    frames = np.arange(n_frames, dtype=np.int64)
    SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.stack([thetas0] * n_frames),
        root_R=np.stack([base_R.astype(np.float32)] * n_frames),
        root_t=np.stack([root_t] * n_frames),
        confidence=np.full(n_frames, 0.8, dtype=np.float32),
        shot_id="play",
    ).save(out / "hmr_world" / "play__P001_smpl_world.npz")

    BallStage(
        config=_full_cfg(),
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")

    # No runaway positions anywhere (pitch frame: 0..105 x 0..68 + margin).
    for f in track.frames:
        if f.world_xyz is None:
            continue
        x, y, z = f.world_xyz
        assert -5.0 <= x <= 110.0, f"f={f.frame} x={x}"
        assert -5.0 <= y <= 73.0, f"f={f.frame} y={y}"
        assert -1.0 <= z <= 50.0, f"f={f.frame} z={z}"

    # The kick was detected automatically and launches a flight.
    from src.schemas.ball_anchor import BallAnchorSet
    auto = BallAnchorSet.load(out / "ball" / "play_ball_anchors_auto.json")
    touch_frames = [a.frame for a in auto.anchors if a.state == "player_touch"]
    assert any(abs(tf - kick_frame) <= 1 for tf in touch_frames), (
        f"expected an auto touch near frame {kick_frame}; got {touch_frames}"
    )

    # The arc is reconstructed with believable height (true apex 1.15 m).
    apex_zs = [
        f.world_xyz[2] for f in track.frames
        if kick_frame <= f.frame <= land_frame and f.world_xyz is not None
    ]
    assert apex_zs, "no world positions in the flight range"
    assert max(apex_zs) >= 0.9, f"expected apex >= 0.9 m, got {max(apex_zs):.2f}"
    assert any(
        s.frame_range[0] >= kick_frame - 1 and s.frame_range[1] <= land_frame + 3
        for s in track.flight_segments
    ), f"expected a flight segment inside the arc; got {[s.frame_range for s in track.flight_segments]}"

    # Appearance bridge recovered the post-landing detection gap.
    gap = [f.state for f in track.frames if 60 <= f.frame <= 62]
    assert all(s != "missing" for s in gap), f"appearance bridge missed gap: {gap}"

    # Accuracy: solved RMS within 35 cm of the analytic truth over the
    # play (the arc may absorb the few roll frames after landing when
    # the landing anchor falls a couple of frames late).
    errs = []
    for f in track.frames:
        if f.world_xyz is None or not (kick_frame <= f.frame <= 85):
            continue
        errs.append(float(np.linalg.norm(np.array(f.world_xyz) - world_at(f.frame))))
    assert errs and float(np.sqrt(np.mean(np.square(errs)))) <= 0.35, (
        f"RMS error {float(np.sqrt(np.mean(np.square(errs)))):.2f} m over the play"
    )

    # Continuity: no frame-to-frame teleport anywhere positions exist.
    prev = None
    for f in track.frames:
        if f.world_xyz is None:
            prev = None
            continue
        cur = np.array(f.world_xyz)
        if prev is not None:
            assert np.linalg.norm(cur - prev) < 1.5, (
                f"teleport at frame {f.frame}"
            )
        prev = cur

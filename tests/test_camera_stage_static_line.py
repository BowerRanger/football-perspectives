"""Integration test: the camera stage's static-camera line-solve path
produces a track with exactly one camera centre."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraTrack
from src.stages.camera import CameraStage
from src.utils.camera_projection import project_world_to_image
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
C_TRUE = np.array([52.5, -30.0, 30.0])
IMAGE_SIZE = (1280, 720)
FPS = 30.0
_LINE_NAMES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "near_touchline",
]
# Six landmarks (two non-coplanar) so the anchor solve is identifiable.
_LANDMARKS = [
    ("near_left_corner", np.array([0, 0, 0.0])),
    ("near_right_corner", np.array([105, 0, 0.0])),
    ("far_left_corner", np.array([0, 68, 0.0])),
    ("far_right_corner", np.array([105, 68, 0.0])),
    ("near_left_corner_flag_top", np.array([0, 0, 1.5])),
    ("left_goal_crossbar_left", np.array([0, 30.34, 2.44])),
]


def _yaw(angle_deg):
    a = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    return R_BASE @ Ry.T


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _frame(K, R, t):
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)
    for name in _LINE_NAMES:
        seg = np.array(PITCH_LINE_CATALOGUE[name], dtype=float)
        cam = seg @ R.T + t
        if (cam[:, 2] <= 0.1).any():
            continue
        proj = project_world_to_image(K, R, t, (0.0, 0.0), seg)
        a = tuple(int(round(v)) for v in proj[0])
        b = tuple(int(round(v)) for v in proj[1])
        cv2.line(img, a, b, (255, 255, 255), thickness=5)
    return img


def _write_manifest(output_dir, shot_id, n_frames):
    from src.schemas.shots import Shot, ShotsManifest
    end = max(0, n_frames - 1)
    ShotsManifest(
        source_file="test", fps=FPS, total_frames=n_frames,
        shots=[Shot(id=shot_id, start_frame=0, end_frame=end,
                    start_time=0.0, end_time=(end + 1) / FPS,
                    clip_file=f"shots/{shot_id}.mp4")],
    ).save(output_dir / "shots" / "shots_manifest.json")


@pytest.mark.integration
def test_static_line_solve_track_has_single_camera_centre(tmp_path: Path) -> None:
    n_frames = 12
    fx = 900.0
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    yaws = np.linspace(-6.0, 6.0, n_frames)

    shots = tmp_path / "shots"
    shots.mkdir(parents=True)
    vw = cv2.VideoWriter(
        str(shots / "play.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE_SIZE
    )
    Rs, ts = [], []
    for yaw in yaws:
        R = _yaw(float(yaw))
        t = -R @ C_TRUE
        Rs.append(R)
        ts.append(t)
        vw.write(_frame(K, R, t))
    vw.release()
    _write_manifest(tmp_path, "play", n_frames)

    # Anchors on the first, middle, last frame — point landmarks only.
    anchor_frames = [0, n_frames // 2, n_frames - 1]
    anchors = []
    for af in anchor_frames:
        lms = tuple(
            LandmarkObservation(
                name=name,
                image_xy=_project(K, Rs[af], ts[af], world),
                world_xyz=tuple(world),
            )
            for name, world in _LANDMARKS
        )
        anchors.append(Anchor(frame=af, landmarks=lms))
    AnchorSet(clip_id="play", image_size=IMAGE_SIZE,
              anchors=tuple(anchors)).save(tmp_path / "camera" / "play_anchors.json")

    stage = CameraStage(
        config={"camera": {
            "static_camera": True,
            "line_extraction": True,
            "lens_from_anchor": False,
        }},
        output_dir=tmp_path,
    )
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / "play_camera_track.json")
    # The static-C line solve must report exactly one camera centre.
    assert track.camera_centre is not None
    C = np.array(track.camera_centre)
    # Every per-frame (R, t) must satisfy -R.T @ t == that single C.
    for f in track.frames:
        if f.t is None:
            continue
        R = np.array(f.R)
        t = np.array(f.t)
        c_frame = -R.T @ t
        assert np.linalg.norm(c_frame - C) < 1e-3, (
            f"frame {f.frame}: camera centre {c_frame} != track C {C}"
        )

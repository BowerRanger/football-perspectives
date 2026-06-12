"""Circle-aided propagation into a line-sparse clip start.

origi01's start in miniature: the clip is anchored only at its (box-end)
tail; the start is a midfield view whose only features are the halfway line
and the CENTRE CIRCLE (plus, early on, a parallel 18yd pair) — too few /
degenerate straight lines for a line-only solve. The camera pans at constant
rate from midfield to the box. Backward propagation must walk into the start
using the centre circle as the disambiguating constraint and recover the true
orientation there; without the circle the start frames either stay uncovered
or drift along the parallel-line direction.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraTrack
from src.stages.camera import CameraStage
from src.utils.camera_projection import project_world_to_image
from src.utils.circle_detector import (
    CENTRE_CIRCLE_CENTRE,
    CENTRE_CIRCLE_RADIUS,
)
from src.utils.line_camera_refine import PITCH_LINE_CATALOGUE

IMAGE_SIZE = (1280, 720)
FPS = 30.0
C_TRUE = np.array([52.5, -28.0, 16.0])
N_FRAMES = 14

# World content painted on every frame — what is visible falls out of the
# projection. No touchlines: the midfield view must offer ONLY the halfway
# line + centre circle (origi01's start geometry).
_WORLD_LINES = [
    "left_18yd_front", "left_18yd_near_edge", "left_18yd_far_edge",
    "left_6yd_front", "left_goal_line", "halfway_line",
]

# Anchor landmarks for the box-end frames — all IN FRONT of the camera there
# (a landmark behind the camera produces a garbage click and poisons the
# anchor solve).
_LANDMARKS = [
    ("near_left_corner", np.array([0, 0, 0.0])),
    ("far_left_corner", np.array([0, 68, 0.0])),
    ("left_18yd_near_corner", np.array([16.5, 13.84, 0.0])),
    ("left_18yd_far_corner", np.array([16.5, 54.16, 0.0])),
    ("left_goal_crossbar_left", np.array([0, 30.34, 2.44])),
    ("left_goal_crossbar_right", np.array([0, 37.66, 2.44])),
]

# Constant-rate pan: view bearing sweeps from the midfield (centre spot) at
# frame 0 to the left box at the last frame, ~2.6 deg/frame.
_BOX_TARGET = np.array([10.0, 34.0, 0.0])
_MID_TARGET = np.array([52.5, 34.0, 0.0])


def _true_R(i: int) -> np.ndarray:
    w = i / (N_FRAMES - 1)
    target = (1 - w) * _MID_TARGET + w * _BOX_TARGET
    fwd = target - C_TRUE
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.array([right, down, fwd], dtype=float)


def _project(K, R, t, world):
    cam = R @ np.asarray(world, float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _frame(K, R, t):
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)
    for name in _WORLD_LINES:
        seg = np.array(PITCH_LINE_CATALOGUE[name], dtype=float)
        cam = seg @ R.T + t
        if (cam[:, 2] <= 0.1).any():
            continue
        proj = project_world_to_image(K, R, t, (0.0, 0.0), seg)
        a = tuple(int(round(v)) for v in proj[0])
        b = tuple(int(round(v)) for v in proj[1])
        cv2.line(img, a, b, (255, 255, 255), thickness=5)
    cx, cy, _cz = CENTRE_CIRCLE_CENTRE
    ang = np.linspace(0.0, 2 * np.pi, 180)
    world = np.stack([
        cx + CENTRE_CIRCLE_RADIUS * np.cos(ang),
        cy + CENTRE_CIRCLE_RADIUS * np.sin(ang),
        np.zeros_like(ang),
    ], axis=1)
    cam = world @ R.T + t
    vis = cam[:, 2] > 0.1
    proj = project_world_to_image(K, R, t, (0.0, 0.0), world)
    pts = proj[vis]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, tuple(int(round(v)) for v in a),
                 tuple(int(round(v)) for v in b), (255, 255, 255),
                 thickness=5)
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
def test_circle_propagation_covers_linepoor_start(tmp_path: Path) -> None:
    # fx=1800 -> ~39 deg FOV: the midfield start then sees ONLY the halfway
    # line + centre circle (at 900 the FOV is so wide the box lines never
    # leave the view and the start is not line-sparse at all).
    fx = 1800.0
    w, h = IMAGE_SIZE
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])

    Rs, ts = [], []
    shots = tmp_path / "shots"
    shots.mkdir(parents=True)
    vw = cv2.VideoWriter(
        str(shots / "play.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
        IMAGE_SIZE)
    for i in range(N_FRAMES):
        R = _true_R(i)
        t = -R @ C_TRUE
        Rs.append(R)
        ts.append(t)
        vw.write(_frame(K, R, t))
    vw.release()
    _write_manifest(tmp_path, "play", N_FRAMES)

    # Anchors ONLY on the box-end tail (mimics PnLCalib coverage on origi01).
    anchor_frames = [10, 11, 12, 13]
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
              anchors=tuple(anchors)).save(
        tmp_path / "camera" / "play_anchors.json")

    stage = CameraStage(
        config={"camera": {
            "static_camera": True,
            "line_extraction": True,
            "lens_from_anchor": False,
            "line_extraction_circle_lens": False,
            "line_extraction_pnlcalib_bootstrap": False,
            "line_extraction_extend_coverage": False,
            "line_extraction_smooth_window": 0,
            # synthetic clip: rendered lines are short/sparse — relax the
            # detector gates (legitimate per-clip knobs) so the box frames
            # solve; the start frames stay line-sparse by construction.
            "line_extraction_min_lines_per_frame": 3,
            "line_extraction_det_min_n_samples": 20,
        }},
        output_dir=tmp_path,
    )
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / "play_camera_track.json")
    by_frame = {f.frame: f for f in track.frames}

    # The line-sparse start must be covered...
    for i in range(0, 5):
        assert i in by_frame, f"start frame {i} not covered"
    # ...and accurately: project the centre spot under the solved camera and
    # compare to its location under the TRUE camera. Without the circle the
    # 18yd parallel pair lets the solve drift along the line direction.
    spot = np.array([[52.5, 34.0, 0.0]])
    for i in range(0, 5):
        fr = by_frame[i]
        proj = project_world_to_image(
            np.array(fr.K), np.array(fr.R), np.array(fr.t), (0.0, 0.0),
            spot)[0]
        truth = project_world_to_image(K, Rs[i], ts[i], (0.0, 0.0), spot)[0]
        err = float(np.linalg.norm(proj - truth))
        assert err < 40.0, (
            f"frame {i}: centre spot {err:.0f}px from truth — start was not "
            "circle-solved"
        )

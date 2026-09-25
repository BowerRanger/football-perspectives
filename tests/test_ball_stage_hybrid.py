"""Integration tests for BallStage's ``ball.trajectory: hybrid`` wiring.

Reuses test_ball_stage.py's synthetic-scenario pattern (fake camera +
blank clip + FakeBallDetector) so both trajectory values run through the
REAL stage end-to-end, not a mocked trajectory layer. ``reference``
(default) must be completely unaffected by this feature (T1c's brief:
keep tests/test_ball_stage*.py green for BOTH trajectory values); the
``hybrid`` runs additionally check the stage never crashes, z stays
>= ball radius, and the diag sidecar carries the hybrid block with no
``error`` key (i.e. the hybrid path actually ran, rather than silently
falling back after an exception).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _save_camera_track(path: Path, K, R, t, n: int, clip_id: str = "play",
                        fps: float = 30.0) -> None:
    track = CameraTrack(
        clip_id=clip_id, fps=fps, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=1.0,
                        is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(path)


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                              (1280, 720))
    for _ in range(n):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K, R, t) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _grounded_roll_scenario(tmp_path: Path, n: int = 60, fps: float = 30.0):
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    positions = [np.array([30.0 + 0.2 * i, 34.0, 0.11]) for i in range(n)]
    detections = []
    for p in positions:
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    # Manual anchors at both ends -- exercises resolve_knots' "grounded"
    # (non-event, hard-but-not-sharp) branch in the hybrid path.
    anchors = BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=0, image_xy=_project(positions[0], K, R, t), state="grounded"),
            BallAnchor(frame=n - 1, image_xy=_project(positions[-1], K, R, t), state="grounded"),
        ),
    )
    (tmp_path / "ball").mkdir(parents=True, exist_ok=True)
    anchors.save(tmp_path / "ball" / "ball_anchors.json")
    return K, R, t, detections


@pytest.mark.integration
def test_reference_trajectory_unaffected_by_hybrid_feature(tmp_path: Path):
    """Default config (no ball.trajectory key at all) must behave exactly
    as before -- the hybrid wiring is opt-in and must not change the
    reference path's output shape or crash."""
    _, _, _, detections = _grounded_roll_scenario(tmp_path)
    stage = BallStage(config={"ball": {"detector": "fake"}}, output_dir=tmp_path,
                       ball_detector=FakeBallDetector(detections))
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    assert len(out.frames) == 60
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["trajectory"] == "reference"
    assert "hybrid_trajectory" not in diag


@pytest.mark.integration
def test_hybrid_trajectory_runs_end_to_end_without_error(tmp_path: Path):
    _, _, _, detections = _grounded_roll_scenario(tmp_path)
    stage = BallStage(
        config={"ball": {"detector": "fake", "trajectory": "hybrid"}},
        output_dir=tmp_path, ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    assert len(out.frames) == 60
    for f in out.frames:
        if f.world_xyz is not None:
            assert f.world_xyz[2] >= 0.11 - 1e-6

    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["trajectory"] == "hybrid"
    assert "hybrid_trajectory" in diag
    assert "error" not in diag["hybrid_trajectory"]
    assert diag["hybrid_trajectory"]["n_frames_covered"] > 0


@pytest.mark.integration
def test_hybrid_trajectory_honours_manual_anchors(tmp_path: Path):
    """Operator input always wins. A 'grounded' anchor is a NON-EVENT
    knot (ball_hybrid_trajectory.py's module docstring: exact position
    for FITTING, but the local Hermite velocity-smoothing window can
    nudge it by the documented, narrowly-scoped tolerance tracked as
    anchor_residuals/anchor_not_honoured — never silently, always
    surfaced in the diag). Checked two ways: (1) reprojection stays
    within anchor_not_honoured_px (the design's own flag threshold,
    looser than the C4 airborne-snap mechanism's ray_faithful_tolerance_
    px, which is a different mechanism for a different anchor class);
    (2) the trajectory layer's own honesty check agrees nothing was
    flagged not-honoured."""
    K, R, t, detections = _grounded_roll_scenario(tmp_path)
    stage = BallStage(
        config={"ball": {"detector": "fake", "trajectory": "hybrid"}},
        output_dir=tmp_path, ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    by_frame = {f.frame: f for f in out.frames}
    anchors = BallAnchorSet.load(tmp_path / "ball" / "ball_anchors.json")
    for a in anchors.anchors:
        world = by_frame[a.frame].world_xyz
        assert world is not None
        uv = _project(np.array(world), K, R, t)
        err_px = float(np.hypot(uv[0] - a.image_xy[0], uv[1] - a.image_xy[1]))
        assert err_px <= 4.0  # config/default.yaml: ball.hybrid.anchor_not_honoured_px

    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["hybrid_trajectory"]["anchor_not_honoured"] == []


@pytest.mark.integration
def test_hybrid_trajectory_keyframes_and_export_schema_unchanged(tmp_path: Path):
    """The keyframe sidecar (consumed by export/render/web/UE) must still
    validate against its schema when the hybrid trajectory replaced the
    dense track -- ball_keyframe_builder is unchanged either way."""
    from src.schemas.ball_keyframes import BallKeyframeSet

    _, _, _, detections = _grounded_roll_scenario(tmp_path)
    stage = BallStage(
        config={"ball": {"detector": "fake", "trajectory": "hybrid"}},
        output_dir=tmp_path, ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    kf_path = tmp_path / "ball" / "ball_keyframes.json"
    assert kf_path.exists()
    kf_set = BallKeyframeSet.load(kf_path)
    assert len(kf_set.keyframes) > 0

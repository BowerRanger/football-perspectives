"""BallStage shot-chain integration: auto proposals reach the auto sidecar;
manual + auto chains are validated into the diag sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 60
FPS = 30.0


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _build_scene(tmp_path: Path):
    out = tmp_path / "out"
    K, R, t = _camera_pose()
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=FPS, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(N_FRAMES)),
    ).save(out / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")
    detections = []
    for i in range(N_FRAMES):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    return out, detections


def _cfg() -> dict:
    return {
        "ball": {"detector": "fake",
                 "appearance_bridge": {"enabled": False}},
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_auto_proposal_reaches_auto_sidecar(tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)

    synthetic = (
        BallEvent(frame=15, kind="touch", score=0.8,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=40, kind="goal_impact", score=0.9,
                  goal_element="back_net"),
    )
    monkeypatch.setattr(
        "src.stages.ball.detect_events", lambda **kwargs: synthetic)
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    auto = json.loads(
        (out / "ball" / "play_ball_anchors_auto.json").read_text())
    assert [15, 40] in [list(c) for c in auto.get("shot_chains", [])]

    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    auto_chains = [c for c in diag["shot_chains"] if c["source"] == "auto"]
    assert any(c["frames"] == [15, 40] for c in auto_chains)


@pytest.mark.integration
def test_no_ghost_auto_chains_when_auto_anchors_disabled(
        tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)

    synthetic = (
        BallEvent(frame=15, kind="touch", score=0.8,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=40, kind="goal_impact", score=0.9,
                  goal_element="back_net"),
    )
    monkeypatch.setattr(
        "src.stages.ball.detect_events", lambda **kwargs: synthetic)
    cfg = _cfg()
    cfg["ball"]["auto_anchors"] = {"enabled": False}
    BallStage(config=cfg, output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    # No auto sidecar should be written when auto anchors are disabled...
    assert not (out / "ball" / "play_ball_anchors_auto.json").exists()

    # ...and the diag must not report source="auto" chains that exist in no
    # sidecar (ghost entries — see fix for post-review issue #2).
    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    auto_chains = [c for c in diag["shot_chains"] if c["source"] == "auto"]
    assert auto_chains == []


@pytest.mark.integration
def test_manual_chain_validated_into_diag(tmp_path: Path):
    out, detections = _build_scene(tmp_path)
    # Manual anchors 20 frames apart, ~4 m apart on the roll -> ~6 m/s,
    # below the 8 m/s shot floor -> launch_speed warning expected.
    K, R, t = _camera_pose()
    uv_a = _project(np.array([34.0, 34.0, 0.11]), K, R, t)
    uv_b = _project(np.array([38.0, 34.0, 0.11]), K, R, t)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=20, image_xy=uv_a, state="grounded"),
            BallAnchor(frame=40, image_xy=uv_b, state="grounded"),
        ),
        shot_chains=((20, 40),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    manual = [c for c in diag["shot_chains"] if c["source"] == "manual"]
    assert len(manual) == 1
    assert manual[0]["frames"] == [20, 40]
    kinds = {w["kind"] for w in manual[0]["warnings"]}
    assert "launch_speed" in kinds

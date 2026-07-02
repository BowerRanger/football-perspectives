"""BallStage wiring guard for touch-attribution refinement: it runs on the
merged events, its output feeds the diag, the flag disables it, and a
crash degrades with a warning."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.smpl_world import SmplWorldTrack
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
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(N_FRAMES)),
    ).save(out / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")

    base_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    thetas0 = np.zeros((24, 3), dtype=np.float32)
    frames = np.arange(N_FRAMES, dtype=np.int64)
    SmplWorldTrack(
        player_id="P001", frames=frames,
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.stack([thetas0] * N_FRAMES),
        root_R=np.stack([base_R.astype(np.float32)] * N_FRAMES),
        root_t=np.stack([np.array([40.0, 34.0, 1.0], dtype=np.float32)] * N_FRAMES),
        confidence=np.full(N_FRAMES, 0.8, dtype=np.float32),
        shot_id="play",
    ).save(out / "hmr_world" / "play__P001_smpl_world.npz")

    detections = []
    for i in range(N_FRAMES):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    return out, detections


def _cfg(**ball_extra) -> dict:
    cfg = {"ball": {"detector": "fake",
                    "appearance_bridge": {"enabled": False}},
           "pitch": {"length_m": 105.0, "width_m": 68.0}}
    cfg["ball"].update(ball_extra)
    return cfg


@pytest.mark.integration
def test_refinement_runs_and_relabel_reaches_diag(tmp_path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}

    def fake_refine(events, **kwargs):
        calls["n"] = len(events)
        import dataclasses as dc
        return tuple(
            dc.replace(e, bone="head")
            if e.kind == "touch" and e.bone else e
            for e in events
        )

    synthetic = (BallEvent(frame=20, kind="touch", score=0.8,
                           player_id="P001", bone="r_foot"),)
    monkeypatch.setattr("src.stages.ball.detect_events",
                        lambda **kwargs: synthetic)
    monkeypatch.setattr("src.stages.ball.refine_touch_attribution",
                        fake_refine)
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    assert "n" in calls, "refine_touch_attribution was never invoked"
    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    assert any(e["kind"] == "touch" and e["bone"] == "head"
               for e in diag["events"])


@pytest.mark.integration
def test_refinement_disabled_by_flag(tmp_path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}
    monkeypatch.setattr(
        "src.stages.ball.refine_touch_attribution",
        lambda events, **kwargs: calls.setdefault("hit", True) and tuple(events),
    )
    BallStage(config=_cfg(touch_attribution={"enabled": False}),
              output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    assert "hit" not in calls


@pytest.mark.integration
def test_refinement_crash_degrades_with_warning(tmp_path, monkeypatch, caplog):
    out, detections = _build_scene(tmp_path)

    def broken(events, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.stages.ball.refine_touch_attribution", broken)
    with caplog.at_level("WARNING"):
        BallStage(config=_cfg(), output_dir=out,
                  ball_detector=FakeBallDetector(detections)).run()
    assert (out / "ball" / "play_ball_track.json").exists()
    assert any("touch attribution refinement failed" in r.message
               for r in caplog.records)

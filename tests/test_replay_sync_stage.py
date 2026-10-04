"""replay_sync stage on a synthetic two-shot output dir."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import HighlightGroup, Shot, ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.stages.replay_sync import ReplaySyncStage
from tests.test_replay_speed import _players, _views

IMG = (1920, 1080)
FPS = 25.0


def look_at(centre, target, fx=2000.0):
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    K = np.array([[fx, 0, IMG[0] / 2], [0, fx, IMG[1] / 2], [0, 0, 1.0]])
    return K, R, -R @ centre


CAMS = {"a": look_at([52.5, -30, 40], [52.5, 30, 0]), "b": look_at([30, 80, 30], [52.5, 30, 0])}


def _write_shot(out: Path, sid: str, pts: dict[int, np.ndarray], n: int, camera: bool = True):
    K, R, t = CAMS[sid]
    tracks = []
    for pi in range(40):
        frames = []
        for f in range(n):
            p = pts.get(f)
            if p is None or pi >= len(p):
                continue
            u, v, w = K @ (R @ np.array([p[pi][0], p[pi][1], 0.0]) + t)
            u, v = u / w, v / w
            frames.append({"frame": f, "bbox": [u - 10, v - 60, u + 10, v], "interpolated": False})
        if frames:
            tracks.append({"track_id": f"T{pi}", "class_name": "player", "frames": frames})
    (out / "tracks").mkdir(exist_ok=True)
    (out / "tracks" / f"{sid}_tracks.json").write_text(json.dumps({"shot_id": sid, "tracks": tracks}))
    if camera:
        (out / "camera").mkdir(exist_ok=True)
        frames = tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=0.9,
                                   is_anchor=False, t=t.tolist()) for i in range(n))
        CameraTrack(clip_id=sid, fps=FPS, image_size=IMG, t_world=t.tolist(), frames=frames,
                    distortion=(0.0, 0.0)).save(out / "camera" / f"{sid}_camera_track.json")
    vw = cv2.VideoWriter(str(out / "shots" / f"{sid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (32, 24))
    for _ in range(n):
        vw.write(np.zeros((24, 32, 3), np.uint8))
    vw.release()


def build(out: Path, rate: float, offset: float, *, camera_b: bool = True,
          method: str | None = None):
    (out / "shots").mkdir(parents=True)
    traj = _players(22, 320)
    n_rep = int(150 / max(rate, 0.4))
    live, replay = _views(traj, rate, offset, n_rep=n_rep)
    _write_shot(out, "a", live, 320)
    _write_shot(out, "b", replay, n_rep, camera=camera_b)
    ShotsManifest(source_file="x", fps=FPS, total_frames=320 + n_rep,
                  shots=[Shot(s, 0, 9, 0, 1, f"shots/{s}.mp4", group_id="g") for s in ("a", "b")],
                  groups=[HighlightGroup("g", "g", ["a", "b"])]).save(
        out / "shots" / "shots_manifest.json")
    if method:
        SyncMap(groups=[GroupSync("g", "a", [Alignment("a", 0), Alignment("b", -3, method)])]).save(
            out / "shots" / "sync_map.json")
    return n_rep


def run(out: Path, **cfg):
    ReplaySyncStage({"replay_sync": cfg}, out).run()
    return (json.loads((out / "shots" / "replay_sync.json").read_text()),
            SyncMap.load(out / "shots" / "sync_map.json"))


def test_real_time_replay_stores_rate_and_offset(tmp_path):
    build(tmp_path, 1.0, 60.0)
    rep, sm = run(tmp_path)
    m = rep["groups"][0]["members"][0]
    assert m["decision"] == "applied" and m["estimate"]["rate"] == pytest.approx(1.0, rel=0.03)
    a = next(a for a in sm.group("g").alignments if a.shot_id == "b")
    assert a.method == "player_formation" and a.frame_offset == pytest.approx(-60, abs=2)
    assert a.playback_rate == pytest.approx(1.0, rel=0.03)
    assert not ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json").shots[1].retimed


def test_manual_alignment_is_kept(tmp_path):
    build(tmp_path, 1.0, 60.0, method="manual")
    rep, sm = run(tmp_path)
    assert rep["groups"][0]["members"][0]["decision"] == "kept_manual"
    assert rep["groups"][0]["members"][0]["estimate"] is not None
    a = next(a for a in sm.group("g").alignments if a.shot_id == "b")
    assert a.method == "manual" and a.frame_offset == -3


def test_slow_replay_is_retimed_and_tracks_remapped(tmp_path):
    n_rep = build(tmp_path, 0.4, 85.0)
    rep, sm = run(tmp_path)
    m = rep["groups"][0]["members"][0]
    assert m["decision"] == "applied_retimed"
    assert m["estimate"]["rate"] == pytest.approx(0.4, rel=0.05)
    a = next(a for a in sm.group("g").alignments if a.shot_id == "b")
    assert a.playback_rate == 1.0 and a.frame_offset == pytest.approx(-85, abs=3)
    shot = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json").shots[1]
    assert shot.retimed and shot.native_frames == n_rep
    assert shot.speed_factor == pytest.approx(1 / 0.4, rel=0.05)
    cam = json.loads((tmp_path / "camera" / "b_camera_track.json").read_text())
    assert len(cam["frames"]) < n_rep * 0.5
    assert (tmp_path / "shots" / "native" / "b.mp4").exists()


def test_auto_retime_off_keeps_rate_only(tmp_path):
    build(tmp_path, 0.4, 85.0)
    rep, sm = run(tmp_path, auto_retime=False)
    assert rep["groups"][0]["members"][0]["decision"] == "applied"
    a = next(a for a in sm.group("g").alignments if a.shot_id == "b")
    assert a.playback_rate == pytest.approx(0.4, rel=0.05)


def test_no_camera_skips(tmp_path):
    build(tmp_path, 1.0, 60.0, camera_b=False)
    rep, sm = run(tmp_path)
    assert rep["groups"][0]["members"][0]["decision"] == "no_camera"
    assert not any(a.method == "player_formation" for g in sm.groups for a in g.alignments)


def test_no_tracks_skips(tmp_path):
    build(tmp_path, 1.0, 60.0)
    (tmp_path / "tracks" / "b_tracks.json").unlink()
    rep, _ = run(tmp_path)
    assert rep["groups"][0]["members"][0]["decision"] == "no_tracks"

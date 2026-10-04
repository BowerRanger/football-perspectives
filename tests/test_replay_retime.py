"""Retime a slow-motion clip to real time and remap its sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.utils.replay_retime import restore_native, retime_shot

N = 40
FPS = 25.0
W, H = 64, 48


def _level(i: int) -> int:
    return 20 + 5 * i  # native frame i has this grey level


def _build(out: Path) -> None:
    (out / "shots").mkdir(parents=True)
    (out / "tracks").mkdir()
    (out / "camera").mkdir()
    vw = cv2.VideoWriter(str(out / "shots" / "r.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        vw.write(np.full((H, W, 3), _level(i), np.uint8))
    vw.release()
    ShotsManifest(source_file="x", fps=FPS, total_frames=N, shots=[
        Shot("r", 0, N - 1, 0.0, N / FPS, "shots/r.mp4")]).save(out / "shots" / "shots_manifest.json")
    tracks = {"shot_id": "r", "tracks": [
        {"track_id": "T1", "class_name": "player", "frames": [
            {"frame": i, "bbox": [i, 0, i + 5, 9]} for i in range(N)]},
        {"track_id": "T2", "class_name": "player", "frames": [
            {"frame": i, "bbox": [0, 0, 1, 1]} for i in range(3)]},  # only early frames
    ]}
    (out / "tracks" / "r_tracks.json").write_text(json.dumps(tracks))
    cam = {"clip_id": "r", "fps": FPS, "image_size": [W, H], "t_world": [1, 2, 3],
           "distortion": [0.1, 0.2], "camera_centre": None, "principal_point": [1, 2],
           "frames": [{"frame": i, "K": [[1]], "R": [[1]], "confidence": 0.9,
                       "is_anchor": i == 0, "t": [i, 0, 0]} for i in range(N)]}
    (out / "camera" / "r_camera_track.json").write_text(json.dumps(cam))


def _frames(path: Path) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    vals = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vals.append(float(f.mean()))
    return vals


def test_retime_resamples_clip_tracks_and_camera(tmp_path: Path):
    _build(tmp_path)
    res = retime_shot(tmp_path, "r", 0.5)
    # k in 0..floor((N-1)*0.5) -> 20 frames, native frame round(k/0.5) = 2k
    assert res.n_native == N and res.n_new == int((N - 1) * 0.5) + 1
    assert res.frame_map == [2 * k for k in range(res.n_new)]
    vals = _frames(tmp_path / "shots" / "r.mp4")
    assert len(vals) == res.n_new
    for k in (0, 5, 19):
        assert vals[k] == pytest.approx(_level(2 * k), abs=6)
    cap = cv2.VideoCapture(str(tmp_path / "shots" / "r.mp4"))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS, abs=0.01)
    assert (tmp_path / "shots" / "native" / "r.mp4").exists()

    tr = json.loads((tmp_path / "tracks" / "r_tracks.json").read_text())
    t1 = next(t for t in tr["tracks"] if t["track_id"] == "T1")
    assert [f["frame"] for f in t1["frames"]] == list(range(res.n_new))
    assert [f["bbox"][0] for f in t1["frames"]][:4] == [0, 2, 4, 6]  # native 2k
    t2 = next(t for t in tr["tracks"] if t["track_id"] == "T2")
    assert [f["frame"] for f in t2["frames"]] == [0, 1]  # native 0 and 2 only

    cam = json.loads((tmp_path / "camera" / "r_camera_track.json").read_text())
    assert [f["frame"] for f in cam["frames"]] == list(range(res.n_new))
    assert cam["frames"][3]["t"] == [6, 0, 0]
    assert cam["t_world"] == [1, 2, 3] and cam["distortion"] == [0.1, 0.2]
    assert cam["principal_point"] == [1, 2]

    m = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json").shots[0]
    assert m.retimed and m.native_frames == N and m.speed_factor == pytest.approx(2.0)


def test_retime_twice_does_not_compound(tmp_path: Path):
    _build(tmp_path)
    retime_shot(tmp_path, "r", 0.5)
    res = retime_shot(tmp_path, "r", 0.25)
    assert res.n_new == int((N - 1) * 0.25) + 1
    tr = json.loads((tmp_path / "tracks" / "r_tracks.json").read_text())
    t1 = next(t for t in tr["tracks"] if t["track_id"] == "T1")
    assert [f["bbox"][0] for f in t1["frames"]][:3] == [0, 4, 8]


def test_restore_native_puts_everything_back(tmp_path: Path):
    _build(tmp_path)
    before = (tmp_path / "tracks" / "r_tracks.json").read_text()
    retime_shot(tmp_path, "r", 0.5)
    restore_native(tmp_path, "r")
    assert (tmp_path / "tracks" / "r_tracks.json").read_text() == before
    assert len(_frames(tmp_path / "shots" / "r.mp4")) == N
    m = ShotsManifest.load(tmp_path / "shots" / "shots_manifest.json").shots[0]
    assert not m.retimed and m.speed_factor == 1.0
    assert (tmp_path / "shots" / "native" / "r.mp4").exists()  # never deleted


def test_retime_without_sidecars_and_bad_input(tmp_path: Path):
    _build(tmp_path)
    (tmp_path / "tracks" / "r_tracks.json").unlink()
    (tmp_path / "camera" / "r_camera_track.json").unlink()
    assert retime_shot(tmp_path, "r", 0.4).n_new > 0
    with pytest.raises(ValueError):
        retime_shot(tmp_path, "r", 0.0)
    with pytest.raises(KeyError):
        retime_shot(tmp_path, "nope", 0.5)
    with pytest.raises(ValueError):
        retime_shot(tmp_path, "r", float("nan"))

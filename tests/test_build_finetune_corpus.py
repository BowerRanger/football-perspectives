"""Corpus builder: gold+weak merge per clip, manifest with holdout split."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from scripts.build_finetune_corpus import build_clip_entry

N = 30


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _fake_output(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    K, R, t = _camera()
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for _ in range(N):
        writer.write(np.full((720, 1280, 3), 90, dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(N)),
    ).save(out / "camera" / "play_camera_track.json")
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=10, image_xy=(640.0, 400.0),
                            state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")
    BallTrack(
        clip_id="play", fps=30.0,
        frames=tuple(
            BallFrame(frame=i, world_xyz=(30.0 + 0.2 * i, 34.0, 0.11),
                      state="grounded", confidence=0.9)
            for i in range(N)
        ),
        flight_segments=(),
    ).save(out / "ball" / "play_ball_track.json")
    return out


def test_build_clip_entry_merges_gold_and_weak(tmp_path: Path):
    out = _fake_output(tmp_path)
    corpus = tmp_path / "corpus"
    entry = build_clip_entry(out, "play", corpus, window=5, min_conf=0.5)
    assert entry["clip_id"] == "play"
    assert entry["n_gold"] == 1
    assert entry["n_weak"] == 10  # frames 5..15 minus the gold frame
    assert entry["n_frames"] == N
    # Frames extracted with the 5-digit naming the WASB layout expects.
    assert (corpus / "frames" / "play" / "00000.png").exists()
    # XML contains gold + weak, gold pixel authoritative at frame 10.
    root = ET.parse(corpus / "annos" / "play.xml").getroot()
    pts = {int(p.attrib["frame"]): p.attrib["points"]
           for p in root.find("track").findall("points")}
    assert len(pts) == 11
    assert pts[10] == "640.00,400.00"


def test_skip_frames_reuses_existing(tmp_path: Path):
    out = _fake_output(tmp_path)
    corpus = tmp_path / "corpus"
    build_clip_entry(out, "play", corpus, window=5, min_conf=0.5)
    marker = corpus / "frames" / "play" / "00000.png"
    before = marker.stat().st_mtime
    entry = build_clip_entry(out, "play", corpus, window=5, min_conf=0.5,
                             skip_frames=True)
    assert marker.stat().st_mtime == before
    assert entry["n_gold"] == 1

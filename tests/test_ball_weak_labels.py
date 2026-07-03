"""Weak-label densification from the solved ball track."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack
from src.utils.ball_weak_labels import (
    labels_to_cvat_xml,
    merge_labels,
    weak_labels_from_track,
)

IMG = (1280, 720)


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _track(frames: list[BallFrame]) -> BallTrack:
    return BallTrack(clip_id="play", fps=30.0, frames=tuple(frames),
                     flight_segments=())


def _bf(frame: int, world, state="grounded", conf=0.9) -> BallFrame:
    return BallFrame(frame=frame, world_xyz=world, state=state,
                     confidence=conf)


def _mats(n: int):
    K, R, t = _camera()
    return ({i: K for i in range(n)}, {i: R for i in range(n)},
            {i: t for i in range(n)})


def test_window_and_gold_exclusion():
    Ks, Rs, ts = _mats(100)
    frames = [_bf(i, (30.0 + 0.2 * i, 34.0, 0.11)) for i in range(100)]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={50}, window=5,
    )
    assert set(out) == {45, 46, 47, 48, 49, 51, 52, 53, 54, 55}


def test_state_conf_and_missing_world_gates():
    Ks, Rs, ts = _mats(10)
    frames = [
        _bf(0, (40.0, 34.0, 0.11)),                       # ok
        _bf(1, (40.2, 34.0, 0.11), conf=0.2),             # low conf
        _bf(2, None, state="missing"),                    # no world
        _bf(3, (40.6, 34.0, 0.11), state="occluded"),     # bad state
        _bf(4, (40.8, 34.0, 2.0), state="flight"),        # ok (flight)
    ]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={2}, window=10,
    )
    assert set(out) == {0, 4}


def test_off_image_projection_rejected():
    Ks, Rs, ts = _mats(4)
    frames = [
        _bf(0, (52.5, 34.0, 0.11)),      # centre-ish, in image
        _bf(1, (52.5, 300.0, 0.11)),     # projects far outside
    ]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={0, 1}, window=3,
    )
    # both frames are gold -> excluded anyway; use neighbours instead
    frames2 = [
        _bf(0, (52.5, 34.0, 0.11)),
        _bf(1, (52.5, 300.0, 0.11)),
        _bf(2, (52.5, 34.5, 0.11)),
    ]
    out = weak_labels_from_track(
        _track(frames2), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={2}, window=3,
    )
    assert 0 in out and 1 not in out


def test_merge_gold_wins():
    merged = merge_labels({5: (1.0, 2.0)}, {5: (9.0, 9.0), 6: (3.0, 4.0)})
    assert merged == {5: (1.0, 2.0), 6: (3.0, 4.0)}


def test_xml_parses_and_matches_exporter_dialect():
    xml = labels_to_cvat_xml("play", {7: (100.5, 200.25), 3: (1.0, 2.0)})
    root = ET.fromstring(xml)
    track = root.find("track")
    assert track is not None and track.attrib["label"] == "ball"
    pts = track.findall("points")
    assert [int(p.attrib["frame"]) for p in pts] == [3, 7]  # ascending
    for p in pts:
        assert p.attrib["outside"] == "0"
        assert p.attrib["occluded"] == "0"
        attr = p.find("attribute")
        assert attr is not None and attr.attrib["name"] == "used_in_game"
        assert attr.text == "1"
    assert pts[1].attrib["points"] == "100.50,200.25"

"""Unit tests for src.utils.track_backfill — the backward track-extension
pass (ball-stage campaign Workstream 4).

Motivating case: on gberch, P008 (Hato)'s track starts at frame 62
though the player is visible earlier, so a manual ball-touch anchor at
frame 56 has nothing to attach to. This module walks backward from a
track's ORIGINAL first frame and prepends matched detections without
ever touching track ids, frame ordering, or operator-assigned
annotations (player_id/player_name/team) — see the HARD CONSTRAINT
in the module docstring.

The detector is behind ``PlayerDetector`` so these tests use small
scripted fakes instead of real YOLO (GPU-dependent, slow); a
``_MarkerDetector`` decodes a marker pixel embedded in each synthetic
in-memory frame to look up canned per-frame detections, keeping the
tests fast and fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.player_detector import Detection, PlayerDetector
from src.utils.track_backfill import (
    BackfillConfig,
    VideoFrameSource,
    backfill_track,
    backfill_tracks_result,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _DictFrameSource:
    """FrameSource backed by an in-memory {frame_idx: ndarray} dict."""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self._frames = frames

    def get(self, frame_idx: int) -> np.ndarray | None:
        return self._frames.get(frame_idx)


class _MarkerDetector(PlayerDetector):
    """Decodes a marker value stashed at the bottom-right pixel (channel 0)
    of the frame to select which canned detections to return — lets a
    test drive per-frame detector output without any real image content
    or model, while still exercising the real ``detect(frame)`` call
    boundary the production code uses."""

    def __init__(self, detections_by_marker: dict[int, list[Detection]]) -> None:
        self._by_marker = detections_by_marker

    def detect(self, frame: np.ndarray) -> list[Detection]:
        marker = int(frame[-1, -1, 0])
        return list(self._by_marker.get(marker, []))


def _marker_frame(marker: int, size: int = 20) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[-1, -1, 0] = marker
    return img


def _colored_marker_frame(
    marker: int, block_bbox: tuple[float, float, float, float], color: tuple[int, int, int],
    size: int = 20,
) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in block_bbox)
    img[y1:y2, x1:x2] = color
    img[-1, -1, 0] = marker
    return img


def _track(
    frames: list[TrackFrame],
    track_id: str = "T001",
    player_id: str = "P001",
    player_name: str = "Hato",
    class_name: str = "player",
    team: str = "A",
) -> Track:
    return Track(
        track_id=track_id, class_name=class_name, team=team,
        player_id=player_id, player_name=player_name, frames=frames,
    )


def _shifted_bbox(
    base: tuple[float, float, float, float], dx_per_step: float, steps: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = base
    shift = dx_per_step * steps
    return (x1 - shift, y1, x2 - shift, y2)


# ---------------------------------------------------------------------------
# Clean backward extension
# ---------------------------------------------------------------------------


def test_clean_matchable_detections_extend_exactly_to_true_start() -> None:
    """True visible range is frames 3..10; frames 0-2 have nothing for
    this class. The walk must stop exactly at frame 3, not wander into
    the frames with no real evidence."""
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    track = _track([TrackFrame(frame=10, bbox=list(base_bbox), confidence=0.9, pitch_position=None)])

    detections_by_marker: dict[int, list[Detection]] = {}
    for k in range(3, 10):  # frames 3..9 all have a clean, close-by candidate
        steps_back = 10 - k
        bbox = _shifted_bbox(base_bbox, dx_per_step=4.0, steps=steps_back)
        detections_by_marker[k] = [Detection(bbox=bbox, confidence=0.9, class_name="player")]
    # frames 0, 1, 2 intentionally have no matching detection at all.

    frame_source = _DictFrameSource({k: _marker_frame(k) for k in range(0, 10)})
    detector = _MarkerDetector(detections_by_marker)

    new_track, report = backfill_track(track, frame_source, detector, BackfillConfig())

    assert report.frames_added == 7
    assert report.stop_reason == "miss_patience"
    assert report.original_start_frame == 10
    assert report.new_start_frame == 3
    assert [f.frame for f in new_track.frames[:7]] == [3, 4, 5, 6, 7, 8, 9]
    assert all(f.source == "backfill" for f in new_track.frames[:7])
    # The original frame is untouched and still present, in order, after the new ones.
    assert new_track.frames[7].frame == 10
    assert new_track.frames[7].bbox == list(base_bbox)
    assert new_track.frames[7].source == "detector"
    # Operator-relevant fields never change.
    assert new_track.track_id == "T001"
    assert new_track.player_id == "P001"
    assert new_track.player_name == "Hato"


def test_max_backfill_frames_caps_the_walk() -> None:
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    track = _track([TrackFrame(frame=20, bbox=list(base_bbox), confidence=0.9, pitch_position=None)])
    detections_by_marker = {
        k: [Detection(
            bbox=_shifted_bbox(base_bbox, dx_per_step=1.0, steps=20 - k),
            confidence=0.9, class_name="player",
        )]
        for k in range(0, 20)
    }
    frame_source = _DictFrameSource({k: _marker_frame(k) for k in range(0, 20)})
    detector = _MarkerDetector(detections_by_marker)
    cfg = BackfillConfig(max_backfill_frames=3)

    new_track, report = backfill_track(track, frame_source, detector, cfg)

    assert report.frames_added == 3
    assert report.stop_reason == "max_frames"
    assert report.new_start_frame == 17


# ---------------------------------------------------------------------------
# Ambiguous / crowded starts stop conservatively
# ---------------------------------------------------------------------------


def test_ambiguous_immediate_crowd_adds_nothing() -> None:
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    track = _track([TrackFrame(frame=10, bbox=list(base_bbox), confidence=0.9, pitch_position=None)])
    # Two equally-good candidates right behind the track's first frame —
    # can't tell which is the real player without guessing.
    detections_by_marker = {
        9: [
            Detection(bbox=base_bbox, confidence=0.9, class_name="player"),
            Detection(bbox=base_bbox, confidence=0.85, class_name="player"),
        ],
    }
    frame_source = _DictFrameSource({9: _marker_frame(9)})
    detector = _MarkerDetector(detections_by_marker)

    new_track, report = backfill_track(track, frame_source, detector, BackfillConfig())

    assert report.frames_added == 0
    assert report.stop_reason == "ambiguous"
    assert report.new_start_frame == 10
    assert new_track is track  # untouched: same object, not a copy


def test_ambiguous_after_partial_extension_keeps_only_the_clean_prefix() -> None:
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    track = _track([TrackFrame(frame=10, bbox=list(base_bbox), confidence=0.9, pitch_position=None)])
    frame9_bbox = _shifted_bbox(base_bbox, dx_per_step=4.0, steps=1)  # clean, single candidate
    detections_by_marker = {
        9: [Detection(bbox=frame9_bbox, confidence=0.9, class_name="player")],
        8: [
            Detection(bbox=_shifted_bbox(frame9_bbox, dx_per_step=4.0, steps=1), confidence=0.9, class_name="player"),
            Detection(bbox=_shifted_bbox(frame9_bbox, dx_per_step=6.0, steps=1), confidence=0.85, class_name="player"),
        ],
    }
    frame_source = _DictFrameSource({9: _marker_frame(9), 8: _marker_frame(8)})
    detector = _MarkerDetector(detections_by_marker)

    new_track, report = backfill_track(track, frame_source, detector, BackfillConfig())

    assert report.frames_added == 1
    assert report.stop_reason == "ambiguous"
    assert report.new_start_frame == 9
    assert [f.frame for f in new_track.frames] == [9, 10]


# ---------------------------------------------------------------------------
# Class-name gating
# ---------------------------------------------------------------------------


def test_different_class_candidate_is_not_a_match() -> None:
    """A perfectly-overlapping box of a DIFFERENT class (e.g. referee)
    must not be treated as evidence the player track continues."""
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    track = _track([TrackFrame(frame=10, bbox=list(base_bbox), confidence=0.9, pitch_position=None)])
    detections_by_marker = {9: [Detection(bbox=base_bbox, confidence=0.9, class_name="referee")]}
    frame_source = _DictFrameSource({9: _marker_frame(9)})
    detector = _MarkerDetector(detections_by_marker)
    cfg = BackfillConfig(patience=0)

    new_track, report = backfill_track(track, frame_source, detector, cfg)

    assert report.frames_added == 0
    assert report.stop_reason == "miss_patience"
    assert new_track is track


# ---------------------------------------------------------------------------
# Eligibility / byte-stability for untouched tracks
# ---------------------------------------------------------------------------


def test_track_already_starting_near_shot_start_is_not_eligible() -> None:
    track = _track([TrackFrame(frame=1, bbox=[0, 0, 10, 10], confidence=0.9, pitch_position=None)])
    frame_source = _DictFrameSource({})
    detector = _MarkerDetector({})

    new_track, report = backfill_track(track, frame_source, detector, BackfillConfig())

    assert new_track is track
    assert report.frames_added == 0
    assert report.stop_reason == "not_eligible"


def test_backfill_tracks_result_preserves_order_ids_and_untouched_tracks() -> None:
    base_bbox = (100.0, 100.0, 200.0, 300.0)
    t1 = _track(
        [TrackFrame(frame=10, bbox=list(base_bbox), confidence=0.9, pitch_position=None)],
        track_id="T001", player_id="P001", player_name="Hato",
    )
    t2 = _track(
        [TrackFrame(frame=0, bbox=[5, 5, 15, 15], confidence=0.8, pitch_position=None)],
        track_id="T002", player_id="P002", player_name="Salah",
    )
    t3 = _track(
        [TrackFrame(frame=12, bbox=[50, 50, 60, 60], confidence=0.8, pitch_position=None)],
        track_id="T003", player_id="P003", player_name="Origi",
    )
    result = TracksResult(shot_id="gberch", tracks=[t1, t2, t3])

    detections_by_marker = {
        9: [Detection(bbox=_shifted_bbox(base_bbox, 4.0, 1), confidence=0.9, class_name="player")],
    }
    frame_source = _DictFrameSource({9: _marker_frame(9)})
    detector = _MarkerDetector(detections_by_marker)

    # Only select P001's track — P002 is already at frame 0 (not eligible
    # regardless), P003 is late-starting but intentionally excluded here.
    new_result, reports = backfill_tracks_result(
        result, frame_source, detector, BackfillConfig(patience=0),
        select=lambda t: t.player_id == "P001",
    )

    assert [t.track_id for t in new_result.tracks] == ["T001", "T002", "T003"]
    assert new_result.tracks[1] is t2  # not selected -> untouched, same object
    assert new_result.tracks[2] is t3  # not selected -> untouched, same object
    assert new_result.tracks[0] is not t1  # selected and extended -> new object
    assert new_result.tracks[0].player_id == "P001"
    assert new_result.tracks[0].player_name == "Hato"
    assert len(new_result.tracks[0].frames) == 2
    # Only the selected track produced a report.
    assert len(reports) == 1
    assert reports[0].track_id == "T001"


# ---------------------------------------------------------------------------
# Optional appearance-distance gate
# ---------------------------------------------------------------------------


def test_appearance_gate_rejects_a_color_mismatched_candidate() -> None:
    block = (0.0, 0.0, 10.0, 10.0)
    reference_frame = _colored_marker_frame(marker=10, block_bbox=block, color=(0, 0, 220))  # red-ish BGR
    candidate_frame = _colored_marker_frame(marker=9, block_bbox=block, color=(220, 0, 0))  # blue-ish BGR
    track = _track([TrackFrame(frame=10, bbox=list(block), confidence=0.9, pitch_position=None)])
    detections_by_marker = {9: [Detection(bbox=block, confidence=0.9, class_name="player")]}
    frame_source = _DictFrameSource({10: reference_frame, 9: candidate_frame})
    detector = _MarkerDetector(detections_by_marker)
    cfg = BackfillConfig(patience=0, use_appearance_gate=True, max_appearance_distance=0.6)

    new_track, report = backfill_track(track, frame_source, detector, cfg)

    assert report.frames_added == 0
    assert report.stop_reason == "miss_patience"
    assert new_track is track


def test_appearance_gate_accepts_a_color_matched_candidate() -> None:
    block = (0.0, 0.0, 10.0, 10.0)
    reference_frame = _colored_marker_frame(marker=10, block_bbox=block, color=(0, 0, 220))
    # Same hue/saturation (pure, fully-saturated red), only V (brightness)
    # differs — the histogram's H/S channels make this a lighting-
    # invariant match, unlike the hue swap in the reject test above.
    candidate_frame = _colored_marker_frame(marker=9, block_bbox=block, color=(0, 0, 210))
    track = _track([TrackFrame(frame=10, bbox=list(block), confidence=0.9, pitch_position=None)])
    detections_by_marker = {9: [Detection(bbox=block, confidence=0.9, class_name="player")]}
    frame_source = _DictFrameSource({10: reference_frame, 9: candidate_frame})
    detector = _MarkerDetector(detections_by_marker)
    cfg = BackfillConfig(patience=0, use_appearance_gate=True, max_appearance_distance=0.6)

    new_track, report = backfill_track(track, frame_source, detector, cfg)

    assert report.frames_added == 1
    assert report.new_start_frame == 9
    assert new_track.frames[0].source == "backfill"


# ---------------------------------------------------------------------------
# VideoFrameSource smoke test (real cv2 I/O, no detector/model involved)
# ---------------------------------------------------------------------------


def test_video_frame_source_reads_frames_and_returns_none_out_of_range(tmp_path: Path) -> None:
    clip_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for i in range(8):
        writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
    writer.release()

    source = VideoFrameSource(clip_path)
    try:
        frame0 = source.get(0)
        assert frame0 is not None
        assert frame0.shape == (24, 32, 3)
        assert source.get(-1) is None
        assert source.get(1000) is None
    finally:
        source.close()


def test_backfill_config_from_dict_applies_overrides_and_defaults() -> None:
    cfg = BackfillConfig.from_dict({"min_iou": 0.5, "enabled": True})
    assert cfg.min_iou == pytest.approx(0.5)
    assert cfg.enabled is True
    assert cfg.patience == BackfillConfig().patience  # untouched default carried through

    assert BackfillConfig.from_dict(None) == BackfillConfig()

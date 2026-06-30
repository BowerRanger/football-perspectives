"""Quality report surfaces the ball solver mode and derived-track flag."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.quality_report import _ball_shot_entry
from src.schemas.ball_track import BallFrame, BallTrack


def _write_track(track_path: Path) -> None:
    BallTrack(
        clip_id="play", fps=25.0,
        frames=(BallFrame(frame=0, world_xyz=(1.0, 2.0, 0.11),
                          state="grounded", confidence=1.0),),
        flight_segments=(),
    ).save(track_path)


def test_ball_entry_reports_events_mode_and_derived(tmp_path: Path):
    track_path = tmp_path / "play_ball_track.json"
    _write_track(track_path)
    diag_path = tmp_path / "play_ball_diag.json"
    diag_path.write_text(json.dumps({"solver": "events", "derived": True}))

    entry = _ball_shot_entry(track_path, "play")
    assert entry["mode"] == "events"
    assert entry["derived"] is True


def test_ball_entry_defaults_for_legacy_diag(tmp_path: Path):
    track_path = tmp_path / "play_ball_track.json"
    _write_track(track_path)
    diag_path = tmp_path / "play_ball_diag.json"
    diag_path.write_text(json.dumps({"solver": "piecewise"}))

    entry = _ball_shot_entry(track_path, "play")
    assert entry["mode"] == "piecewise"
    assert entry["derived"] is False

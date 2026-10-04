"""D6.4: the quality report fails loudly when a goal shot's trajectory misses
the mouth / is over the bar / never crosses the line."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.pipeline.quality_report import _ball_section, _ball_shot_entry, _goal_check_alerts
from src.schemas.ball_track import BallFrame, BallTrack


def _write(tmp: Path, shot: str, goal_check: dict | None) -> Path:
    track_path = tmp / "ball" / f"{shot}_ball_track.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    BallTrack(
        clip_id=shot, fps=30.0,
        frames=(BallFrame(frame=0, world_xyz=(1.0, 2.0, 0.11),
                          state="grounded", confidence=1.0),),
        flight_segments=(),
    ).save(track_path)
    diag = {"solver": "events", "hybrid_trajectory": {"direction_gate": {"n_dropped": 3}}}
    if goal_check is not None:
        diag["goal_check"] = goal_check
    (tmp / "ball" / f"{shot}_ball_diag.json").write_text(json.dumps(diag))
    return track_path


def test_entry_carries_goal_check_and_gate_count(tmp_path: Path):
    gc = {"status": "ok", "goal_frame": 402,
          "line_cross": {"frame": 394.0, "xyz": [0.0, 37.1, 1.8]}}
    entry = _ball_shot_entry(_write(tmp_path, "gberch", gc), "gberch")
    assert entry["goal_check"] == gc
    assert entry["direction_gate_dropped"] == 3


def test_entry_without_goal_event_has_none(tmp_path: Path):
    entry = _ball_shot_entry(_write(tmp_path, "s1", None), "s1")
    assert entry["goal_check"] is None


def test_ok_goal_check_raises_no_alert(tmp_path: Path):
    p = _write(tmp_path, "gberch", {"status": "ok", "goal_frame": 402, "line_cross": None})
    entries = [_ball_shot_entry(p, "gberch")]
    assert _goal_check_alerts(entries) == []


def test_failures_become_loud_alerts(tmp_path: Path, caplog):
    for shot, status in (("a", "misses_mouth"), ("b", "over_crossbar"), ("c", "no_line_cross")):
        gc = {"status": status, "goal_frame": 100,
              "line_cross": None if status == "no_line_cross" else {"frame": 98.0, "xyz": [0, 45, 3.8]}}
        _write(tmp_path, shot, gc)
    entries = [_ball_shot_entry(tmp_path / "ball" / f"{s}_ball_track.json", s) for s in "abc"]
    alerts = _goal_check_alerts(entries)
    assert [a["status"] for a in alerts] == ["misses_mouth", "over_crossbar", "no_line_cross"]
    assert "MISSES THE MOUTH" in alerts[0]["message"]
    assert "OVER THE CROSSBAR" in alerts[1]["message"]
    assert "NEVER CROSSES" in alerts[2]["message"]


def test_ball_section_aggregates_goal_checks(tmp_path: Path, caplog):
    _write(tmp_path, "good", {"status": "ok", "goal_frame": 10, "line_cross": None})
    _write(tmp_path, "bad", {"status": "misses_mouth", "goal_frame": 10, "line_cross": None})
    (tmp_path / "ball" / "ball_track.json").unlink(missing_ok=True)

    from types import SimpleNamespace
    manifest = SimpleNamespace(active_shots=lambda: [SimpleNamespace(id="good"),
                                                     SimpleNamespace(id="bad")])
    with caplog.at_level(logging.WARNING):
        section = _ball_section(tmp_path, manifest)
    assert section["goal_checks"] == 2
    assert [a["shot_id"] for a in section["goal_check_failures"]] == ["bad"]
    assert any("GOAL CHECK FAILED" in r.message for r in caplog.records)

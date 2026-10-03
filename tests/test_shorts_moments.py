"""shorts_moments: synthetic-data unit tests + an integration check against
the real gberch scratch output (skipped when it is not linked)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils import shorts_moments as sm

FPS = 30.0
REPO = Path(__file__).resolve().parents[1]


def _shot_track():
    """Ball sits still to 100, is struck at 100 (30 m/s) towards the x=0 goal
    (mouth centre y=34), crosses the line at ~f127, stops in the net."""
    fr = {}
    for f in range(0, 101):
        fr[f] = (25.0, 30.0, 0.11)
    for f in range(101, 140):
        x = max(25.0 - 1.0 * (f - 100), -1.2)
        fr[f] = (x, 30.0 + 0.16 * min(f - 100, 26), 1.0)
    return fr


def _roots(frames, keeper_dive_at=120):
    n = 140
    fr = np.arange(n)
    keeper = np.zeros((n, 2)); keeper[:, 0] = 1.0; keeper[:, 1] = 34.0
    for i in range(keeper_dive_at - 3, keeper_dive_at + 3):
        keeper[i:, 1] += 0.5          # a lateral burst around the dive
    scorer = np.zeros((n, 2)); scorer[:, 0] = 25; scorer[:, 1] = 30
    other_gk = np.zeros((n, 2)); other_gk[:, 0] = 100; other_gk[:, 1] = 34
    return {"K": (fr, keeper), "S": (fr, scorer), "Z": (fr, other_gk)}


ROLES = {"K": "away_gk", "Z": "home_gk", "S": "home", "T": "home", "O": "away"}


def _derive(events=None, anchors=None, goal_check=None, roles=ROLES, frames=None):
    frames = _shot_track() if frames is None else frames
    return sm.derive_from_data(
        ball_frames=frames, fps=FPS, events=events or [], operator_anchors=anchors or [],
        roles=roles, root_xy=_roots(frames), goal_check=goal_check)


def test_strike_is_last_touch_with_velocity_change_not_shoulder_graze():
    ev = [
        {"kind": "touch", "frame": 100, "player_id": "S", "bone": "r_foot", "score": 0.4},
        {"kind": "touch", "frame": 110, "player_id": "T", "bone": "r_shoulder", "score": 0.9},
        {"kind": "goal_impact", "frame": 135, "goal_element": "back_net"},
    ]
    m = _derive(events=ev)
    assert m["strike"] == 100 and m["scorer_pid"] == "S"
    assert m["impact"] == 135
    assert m["sources"]["strike"] == "velocity_change"


def test_operator_shot_anchor_beats_diag_touches_and_net_beats_post():
    anchors = [
        {"frame": 100, "state": "player_touch", "player_id": "S", "touch_type": "shot"},
        {"frame": 135, "state": "goal_impact", "goal_element": "back_net"},
    ]
    ev = [{"kind": "goal_impact", "frame": 126, "goal_element": "post"},
          {"kind": "touch", "frame": 105, "player_id": "T", "bone": "head", "score": 1.0}]
    m = _derive(events=ev, anchors=anchors)
    assert (m["strike"], m["impact"]) == (100, 135)
    assert m["sources"] == {**m["sources"], "strike": "operator_shot_anchor", "impact": "operator_anchor"}


def test_defending_keeper_and_team_never_score():
    """origi01: keeper parry (operator anchor) then the tap-in, then a later
    auto touch credited to the keeper — the scorer is the tap-in, not the GK."""
    anchors = [
        {"frame": 95, "state": "player_touch", "player_id": "K", "bone": "r_hand"},
        {"frame": 100, "state": "player_touch", "player_id": "S", "bone": "r_foot"},
        {"frame": 135, "state": "goal_impact", "goal_element": "back_net"},
    ]
    ev = [{"kind": "touch", "frame": 112, "player_id": "K", "bone": "l_hand", "score": 0.9},
          {"kind": "touch", "frame": 114, "player_id": "O", "bone": "r_foot", "score": 0.9}]
    m = _derive(events=ev, anchors=anchors)
    assert (m["strike"], m["scorer_pid"], m["keeper_pid"]) == (100, "S", "K")
    assert m["sources"]["strike"] == "operator_touch"


def test_later_attacking_auto_touch_still_beats_an_older_operator_touch():
    anchors = [{"frame": 90, "state": "player_touch", "player_id": "T", "bone": "r_foot"},
               {"frame": 135, "state": "goal_impact", "goal_element": "back_net"}]
    ev = [{"kind": "touch", "frame": 100, "player_id": "S", "bone": "r_foot", "score": 0.5}]
    m = _derive(events=ev, anchors=anchors)
    assert (m["strike"], m["scorer_pid"]) == (100, "S")


def test_line_cross_from_goal_check_wins_else_track_crossing():
    gc = {"goal_frame": 135, "goal_end_x": 0.0,
          "line_cross": {"frame": 126, "xyz": [0.0, 34.0, 1.0]}, "status": "ok"}
    m = _derive(goal_check=gc)
    assert m["line_cross"] == 126 and m["goal_end"] == "left" and m["impact"] == 135
    m2 = _derive(events=[{"kind": "goal_impact", "frame": 135, "goal_element": "back_net"}])
    assert m2["line_cross"] == 125 and m2["goal_end"] == "left"
    assert m2["sources"]["line_cross"] == "track_crossing"


def test_track_crossing_ignores_balls_outside_the_mouth():
    frames = {f: (10.0 - f, 5.0, 0.2) for f in range(0, 20)}      # exits at y=5: wide
    assert sm.find_line_crossing(frames) is None
    frames = {f: (10.0 - f, 34.0, 3.5) for f in range(0, 20)}     # over the bar
    assert sm.find_line_crossing(frames) is None


def test_track_clamped_exactly_on_the_line_counts_as_the_crossing():
    frames = {f: (max(5.0 - f, 0.0) + (3.7e-11 if f >= 5 else 0.0), 34.0, 1.0) for f in range(10)}
    assert sm.find_line_crossing(frames)[0] == 5


def test_keeper_is_the_gk_defending_the_scored_goal():
    m = _derive(events=[{"kind": "goal_impact", "frame": 135, "goal_element": "back_net"}])
    assert m["goal_end"] == "left" and m["keeper_pid"] == "K"      # not home_gk Z at x=100


def test_keeper_dive_is_peak_lateral_velocity_in_window():
    m = _derive(events=[{"kind": "touch", "frame": 100, "player_id": "S", "bone": "r_foot", "score": .5},
                        {"kind": "goal_impact", "frame": 135, "goal_element": "back_net"}])
    assert m["keeper_dive"] is not None and 115 <= m["keeper_dive"] <= 125


def test_buildup_start_is_first_touch_of_team_chain_else_strike_minus_75():
    ev = [{"kind": "touch", "frame": f, "player_id": p, "bone": "r_foot", "score": .5}
          for f, p in ((10, "T"), (35, "T"), (60, "S"), (100, "S"))]
    ev.append({"kind": "touch", "frame": 5, "player_id": "O", "bone": "r_foot", "score": .5})
    ev.append({"kind": "goal_impact", "frame": 135, "goal_element": "back_net"})
    m = _derive(events=ev)
    assert m["buildup_start"] == 10 and m["sources"]["buildup_start"] == "possession_chain"
    short_chain = [e for e in ev if e.get("frame") not in (10, 35, 60)]
    m2 = _derive(events=short_chain)
    assert m2["buildup_start"] == 100 - 75


def test_opponent_touch_breaks_the_chain():
    ev = [{"kind": "touch", "frame": f, "player_id": p, "bone": "r_foot", "score": .5}
          for f, p in ((10, "T"), (30, "O"), (45, "T"), (70, "T"), (100, "S"))]
    ev.append({"kind": "goal_impact", "frame": 135, "goal_element": "back_net"})
    # chain after the O touch starts at 45 -> lead 55 >= 45
    assert _derive(events=ev)["buildup_start"] == 45


def test_no_goal_yields_none_moments_without_raising():
    m = _derive(frames={f: (50.0, 30.0, 0.1) for f in range(50)})
    assert m["impact"] is None and m["line_cross"] is None and m["keeper_dive"] is None


@pytest.mark.skipif(
    not (REPO / "output-shorts/ball/gberch_ball_track.json").exists(),
    reason="gberch scratch output not linked")
def test_gberch_real_output():
    m = sm.derive_moments(REPO / "output-shorts", "gberch")
    assert m["strike"] == 371
    assert m["line_cross"] == 394
    assert m["impact"] == 402
    assert m["scorer_pid"] == "P006"
    assert m["keeper_pid"] == "P005"          # players.json away_gk
    assert m["goal_end"] == "left"
    assert 371 - 10 <= m["keeper_dive"] <= 402
    assert m["buildup_start"] < 371 - 40

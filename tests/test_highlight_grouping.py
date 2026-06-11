"""Grouping: one group per attack passage (GT-derived rules).

Bournemouth 1-1 Man City ground truth (operator-annotated groups A-H):
every group starts with a long wide live shot (small players, >= 6 s)
and collects the shorter/closer replay shots that follow. Hard-cut
continuations (gap ~0) never start a group even when wide.
"""
from src.utils.highlight_grouping import GroupingInput, group_shots


def _gi(sid, *, kind="gameplay", scale="wide", speed=1.0,
        start=0.0, end=10.0, person=0.15):
    return GroupingInput(shot_id=sid, kind=kind, scale=scale,
                         speed_factor=speed, source_start_s=start,
                         source_end_s=end, max_person_height=person)


def test_live_wide_after_replays_starts_new_group():
    shots = [
        _gi("live1", start=0, end=10),                       # group 1
        _gi("rep1", start=10.4, end=15, person=0.55),        # replay
        _gi("live2", start=15.4, end=24),                    # group 2
        _gi("rep2", start=24.4, end=29, person=0.60),
    ]
    groups = group_shots(shots)
    assert [g.shot_ids for g in groups] == [["live1", "rep1"],
                                            ["live2", "rep2"]]
    assert groups[1].boundary_rule == "live_wide"


def test_hard_cut_continuation_stays_in_group():
    """Consecutive live shots split by a hard cut (gap ~0) are one
    passage — the goal-mouth angle of the same move."""
    shots = [
        _gi("live", start=0, end=10),
        _gi("cont", start=10.0, end=18),     # gap 0: continuation
        _gi("rep", start=18.4, end=23, person=0.6),
    ]
    groups = group_shots(shots)
    assert len(groups) == 1
    assert groups[0].shot_ids == ["live", "cont", "rep"]


def test_short_wide_shot_is_a_replay_not_a_new_group():
    """Replays are short even when wide-framed (GT: non-initial shots
    all <= ~5.9 s; initials >= 6 s)."""
    shots = [
        _gi("live", start=0, end=12),
        _gi("widereplay", start=12.4, end=17.6, person=0.17),  # 5.2 s
    ]
    groups = group_shots(shots)
    assert len(groups) == 1


def test_person_dominant_shot_never_starts_group():
    shots = [
        _gi("live", start=0, end=10),
        _gi("close", start=12, end=20, person=0.6),  # long but close
    ]
    assert len(group_shots(shots)) == 1


def test_long_celebration_gap_does_not_split_event():
    """GT group C: an 8.7 s dropped celebration sits INSIDE the goal
    event — its replays still belong to the goal."""
    shots = [
        _gi("live", start=0, end=10),
        _gi("rep1", start=18.7, end=24, person=0.5),  # after 8.7 s hole
    ]
    assert len(group_shots(shots)) == 1


def test_non_gameplay_shots_never_grouped():
    shots = [
        _gi("live", start=0, end=10),
        _gi("x", kind="reaction", start=10.4, end=12, person=0.9),
    ]
    groups = group_shots(shots)
    assert groups[0].shot_ids == ["live"]


def test_reference_is_first_live_wide():
    shots = [
        _gi("live", start=0, end=10),
        _gi("rep", start=10.4, end=15, person=0.6),
    ]
    assert group_shots(shots)[0].reference_shot == "live"


def test_groups_get_sequential_ids_and_labels():
    shots = [
        _gi("a", start=0, end=10),
        _gi("b", start=10.4, end=22),
    ]
    groups = group_shots(shots)
    assert [g.id for g in groups] == ["g01", "g02"]
    assert groups[0].label == "Highlight 1"

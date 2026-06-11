"""Rule-based grouping of shots into highlight events."""
from src.utils.highlight_grouping import GroupingInput, group_shots


def _gi(sid, *, kind="gameplay", scale="wide", speed=1.0,
        start=0.0, end=4.0):
    return GroupingInput(shot_id=sid, kind=kind, scale=scale,
                         speed_factor=speed, source_start_s=start,
                         source_end_s=end)


def test_single_run_is_one_group():
    groups = group_shots([
        _gi("a"),
        _gi("b", scale="medium", start=4, end=8),
    ])
    assert len(groups) == 1
    assert groups[0].shot_ids == ["a", "b"]
    assert groups[0].id == "g01"
    assert groups[0].label == "Highlight 1"
    assert groups[0].boundary_rule == "start"


def test_transition_between_shots_starts_new_group():
    shots = [_gi("a"),
             _gi("t", kind="transition", start=4, end=5),
             _gi("b", start=5, end=9)]
    groups = group_shots(shots)
    assert [g.shot_ids for g in groups] == [["a"], ["b"]]
    assert groups[1].boundary_rule == "transition"
    assert groups[1].id == "g02"


def test_large_source_gap_starts_new_group():
    # an excluded reaction span leaves a 6 s hole between kept shots
    groups = group_shots([_gi("a", end=4.0), _gi("b", start=10.0, end=14.0)],
                         gap_boundary_s=5.0)
    assert len(groups) == 2
    assert groups[1].boundary_rule == "gap"


def test_small_gap_does_not_split():
    groups = group_shots([_gi("a", end=4.0), _gi("b", start=7.0, end=11.0)],
                         gap_boundary_s=5.0)
    assert len(groups) == 1


def test_live_wide_after_replay_starts_new_group():
    shots = [_gi("a"),
             _gi("r", scale="medium", speed=1.8, start=4, end=8),
             _gi("b", start=8, end=12)]
    groups = group_shots(shots)
    assert [g.shot_ids for g in groups] == [["a", "r"], ["b"]]
    assert groups[1].boundary_rule == "live_after_replay"


def test_live_wide_without_prior_replay_stays_in_group():
    # wide -> medium -> wide with no replay: ordinary build-up cutting
    shots = [_gi("a"),
             _gi("m", scale="medium", start=4, end=8),
             _gi("b", start=8, end=12)]
    assert len(group_shots(shots)) == 1


def test_reaction_shots_never_grouped():
    groups = group_shots([_gi("a"), _gi("x", kind="reaction", start=4, end=6)])
    assert len(groups) == 1
    assert groups[0].shot_ids == ["a"]


def test_reference_prefers_wide_realtime_member():
    # medium live build-up then the wide live angle: the wide one
    # anchors the group timeline even though it isn't first.
    shots = [_gi("m", scale="medium", end=3),
             _gi("a", start=3, end=9)]
    groups = group_shots(shots)
    assert len(groups) == 1
    assert groups[0].reference_shot == "a"


def test_reference_falls_back_to_longest():
    shots = [_gi("r1", scale="tight", speed=1.8, end=3),
             _gi("r2", scale="medium", speed=1.6, start=3, end=12)]
    g = group_shots(shots)[0]
    assert g.reference_shot == "r2"


def test_boundary_confidences():
    shots = [_gi("a"),
             _gi("t", kind="transition", start=4, end=5),
             _gi("b", start=5, end=9)]
    groups = group_shots(shots)
    assert groups[0].boundary_confidence == 1.0
    assert groups[1].boundary_confidence == 0.9

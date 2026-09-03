from scripts.ball_touch_recall_report import (
    proposer_only_touches,
    recall_table,
    strict_recall_table,
)


def test_strict_table_flags_wrong_player_legacy_does_not():
    # Regression case: gberch f343 -- legacy credits a same-frame/same-bone
    # hit from the WRONG player (P016 vs manual P006); strict must not.
    manual = [(343, "P006", "r_foot")]
    break_only: list = []
    proposer_only = [(342, "P016", "r_foot")]
    union = break_only + proposer_only
    legacy = recall_table(manual, break_only, proposer_only, union, frame_tol=2)
    strict = strict_recall_table(manual, break_only, proposer_only, union, frame_tol=2)
    assert legacy["union"]["true_positive"] == 1
    assert legacy["union"]["recall"] == 1.0
    assert strict["union"]["true_positive"] == 0
    assert strict["union"]["recall"] == 0.0
    # recall_table's own numbers are untouched by strict_recall_table
    # existing (same call, same result — historical comparability).
    assert legacy == recall_table(manual, break_only, proposer_only, union, frame_tol=2)


def test_union_recall_at_least_break_only():
    # pseudo-ground-truth: three touches
    manual = [(10, "P1", "r_foot"), (40, "P1", "l_foot"), (70, "P2", "head")]
    # ball-break path found only the first
    break_only = [(10, "P1", "r_foot")]
    # proposer recovered the two the ball missed
    proposer_only = [(41, "P1", "l_foot"), (70, "P2", "head")]
    union = break_only + proposer_only
    table = recall_table(manual, break_only, proposer_only, union, frame_tol=2)
    assert table["break_only"]["recall"] <= table["union"]["recall"]
    assert table["union"]["recall"] > table["break_only"]["recall"]
    assert table["union"]["recall"] == 1.0


def test_proposer_only_is_union_minus_break_only():
    break_only = [(10, "P1", "r_foot")]
    union = [(10, "P1", "r_foot"), (41, "P1", "l_foot"), (70, "P2", "head")]
    assert proposer_only_touches(break_only, union) == [
        (41, "P1", "l_foot"), (70, "P2", "head")]

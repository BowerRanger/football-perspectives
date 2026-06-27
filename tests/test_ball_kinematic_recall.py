from scripts.ball_touch_recall_report import recall_table


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

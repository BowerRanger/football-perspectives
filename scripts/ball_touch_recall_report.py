"""Report touch-detection recall/precision for three configurations
(break-only / proposer-only / union) against a manual anchor set used as
pseudo-ground-truth. See the body-kinematics-touch-proposer spec, section 8.

Usage:
    python scripts/ball_touch_recall_report.py \
        output/ball/<shot>_ball_anchors.json \
        output/ball/<shot>_ball_anchors_auto_break_only.json \
        output/ball/<shot>_ball_anchors_auto_union.json
"""

from __future__ import annotations

import sys

from src.utils.ball_touch_recall import match_touches, touches_from_anchor_set

Touch = tuple[int, str, str]


def recall_table(
    manual: list[Touch],
    break_only: list[Touch],
    proposer_only: list[Touch],
    union: list[Touch],
    *,
    frame_tol: int = 2,
) -> dict[str, dict]:
    """recall/precision for each config against ``manual``."""
    return {
        name: match_touches(manual, auto, frame_tol=frame_tol, require_bone=True)
        for name, auto in (
            ("break_only", break_only),
            ("proposer_only", proposer_only),
            ("union", union),
        )
    }


def _print_table(table: dict[str, dict]) -> None:
    print(f"{'config':<16}{'recall':>8}{'precision':>11}{'tp':>5}{'fp':>5}")
    for name, m in table.items():
        print(f"{name:<16}{m['recall']:>8.3f}{m['precision']:>11.3f}"
              f"{m['true_positive']:>5}{m['false_positive']:>5}")


def proposer_only_touches(break_only: list[Touch], union: list[Touch]) -> list[Touch]:
    """Touches present in the proposer-on (union) set but not in the
    proposer-off (break-only) set, matched exactly on (frame, player, bone)."""
    bo = set(break_only)
    return [t for t in union if t not in bo]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "usage: ball_touch_recall_report.py "
            "<manual.json> <break_only_auto.json> <union_auto.json>\n"
            "  break_only_auto.json: auto anchors produced with "
            "ball.kinematic_touch.enabled=false\n"
            "  union_auto.json:      auto anchors produced with "
            "ball.kinematic_touch.enabled=true"
        )
        raise SystemExit(2)
    manual = touches_from_anchor_set(sys.argv[1])
    break_only = touches_from_anchor_set(sys.argv[2])
    union = touches_from_anchor_set(sys.argv[3])
    proposer_only = proposer_only_touches(break_only, union)
    _print_table(recall_table(manual, break_only, proposer_only, union))

"""Report touch-detection recall/precision for three configurations
(break-only / proposer-only / union) against a manual anchor set used as
pseudo-ground-truth. See the body-kinematics-touch-proposer spec, section 8.

Usage:
    python scripts/ball_touch_recall_report.py \
        output/ball/<shot>_ball_anchors.json \
        output/ball/<shot>_ball_anchors_auto.json
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ball_touch_recall_report.py <manual.json> <auto.json>")
        raise SystemExit(2)
    manual = touches_from_anchor_set(sys.argv[1])
    auto = touches_from_anchor_set(sys.argv[2])
    # With only the merged auto set on disk we report union vs the empty
    # break-only baseline; pass a break-only file as auto to compare paths.
    table = recall_table(manual, [], [], auto)
    _print_table(table)

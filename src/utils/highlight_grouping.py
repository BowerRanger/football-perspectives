"""Group reel-ordered shots into highlight events (attack passages).

Rules derived from operator-annotated ground truth (Bournemouth 1-1
Man City, groups A-H over 20 kept shots): every group begins with the
**live wide build-up shot** of a new passage — long (the broadcast
follows the move develop) with small players (true wide framing) —
and collects everything that follows until the next one: replay shots
(shorter, closer framing), goal-mouth angles, and live continuations.

Measured separations on that ground truth:

- group-initial shots: trimmed duration >= 5.7 s, max-person-height
  0.15-0.19
- non-initial shots: duration 1.6-7.8 s (one 7.8 s live continuation),
  most with person height >= 0.5 (replay framing)
- hard-cut continuations (gap ~0 s) are always the same passage,
  whatever their stats
- dropped-content holes do NOT mark passage boundaries (an 8.7 s
  celebration sits INSIDE the goal event, while several passages are
  separated by a 0.4 s fade alone)

Known limit: two consecutive live passages with no replay between them
(GT groups A|B) need event semantics no pixel statistic carries — the
rule splits them into their live shots and the dashboard's merge
control resolves it (never wrongly attaching replays).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_RULE_CONFIDENCE = {
    "start": 1.0,
    "live_wide": 0.85,
}


@dataclass(frozen=True)
class GroupingInput:
    """The per-shot facts the rules need, in reel order."""

    shot_id: str
    kind: str
    scale: str
    speed_factor: float
    source_start_s: float
    source_end_s: float
    # Median max-person-height from the classifier (0.0 = unmeasured).
    max_person_height: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.source_end_s - self.source_start_s)


@dataclass
class GroupedHighlight:
    id: str
    label: str
    shot_ids: list[str] = field(default_factory=list)
    boundary_rule: str = "start"
    boundary_confidence: float = 1.0
    reference_shot: str = ""


def group_shots(
    shots: list[GroupingInput],
    *,
    live_wide_max_person_height: float = 0.25,
    live_wide_min_duration_s: float = 5.7,
    continuation_max_gap_s: float = 0.1,
) -> list[GroupedHighlight]:
    """Partition gameplay shots into one group per attack passage."""

    def is_live_wide(s: GroupingInput) -> bool:
        person_ok = (
            s.max_person_height <= live_wide_max_person_height
            if s.max_person_height > 0
            else s.scale == "wide"  # person check disabled: scale proxy
        )
        return (s.scale == "wide" and person_ok
                and s.duration_s >= live_wide_min_duration_s)

    groups: list[GroupedHighlight] = []
    members: list[GroupingInput] = []
    open_rule = "start"
    prev_kept: GroupingInput | None = None

    def _close_group() -> None:
        nonlocal members
        if not members:
            return
        idx = len(groups) + 1
        groups.append(GroupedHighlight(
            id=f"g{idx:02d}",
            label=f"Highlight {idx}",
            shot_ids=[m.shot_id for m in members],
            boundary_rule=open_rule,
            boundary_confidence=_RULE_CONFIDENCE.get(open_rule, 1.0),
            reference_shot=_pick_reference(
                members, live_wide_max_person_height,
                live_wide_min_duration_s,
            ),
        ))
        members = []

    for shot in shots:
        if shot.kind != "gameplay":
            continue
        starts_new = False
        if members and is_live_wide(shot):
            gap = (shot.source_start_s - prev_kept.source_end_s
                   if prev_kept is not None else 0.0)
            # A hard-cut continuation (gap ~0) is the same passage —
            # the broadcast cut to another live angle mid-move.
            starts_new = gap > continuation_max_gap_s
        if starts_new:
            _close_group()
            open_rule = "live_wide"
        members.append(shot)
        prev_kept = shot

    _close_group()
    return groups


def _pick_reference(
    members: list[GroupingInput],
    live_wide_max_person_height: float,
    live_wide_min_duration_s: float,
) -> str:
    """First live-wide member (the passage's build-up shot), else the
    longest member."""
    for m in members:
        person_ok = (
            m.max_person_height <= live_wide_max_person_height
            if m.max_person_height > 0
            else m.scale == "wide"
        )
        if (m.scale == "wide" and person_ok
                and m.duration_s >= live_wide_min_duration_s):
            return m.shot_id
    longest = max(members, key=lambda m: m.duration_s)
    return longest.shot_id

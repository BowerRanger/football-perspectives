"""Group reel-ordered shots into highlight events.

A highlights reel interleaves each event's live footage with replays
from other angles, separated by transitions/reactions. With reaction
and transition shots dropped, a highlight is a contiguous run of
gameplay shots; the rules below decide where one run ends and the next
begins. Every boundary records which rule fired and a confidence so the
dashboard can flag uncertain groupings for operator review.

Boundary rules (checked in order, first hit wins) — a new group opens
before gameplay shot *i* when:

- R1 ``transition``        : a transition shot (fade/graphic) sits
  between *i-1* and *i* in the reel. Broadcast packaging wraps replay
  sequences in transitions, so this is the strongest signal (0.9).
- R2 ``gap``               : the source-time hole between kept shots
  exceeds ``gap_boundary_s`` — several dropped shots in a row usually
  means the reel moved on (0.6, the weakest rule: a long crowd
  celebration mid-highlight can also produce a gap).
- R3 ``live_after_replay`` : *i* is a wide real-time shot and the
  current group already contains a replay — "replays finished, back to
  live action" (0.75).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_RULE_CONFIDENCE = {
    "start": 1.0,
    "transition": 0.9,
    "gap": 0.6,
    "live_after_replay": 0.75,
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
    gap_boundary_s: float = 5.0,
    replay_min_speed_factor: float = 1.25,
) -> list[GroupedHighlight]:
    """Partition gameplay shots into highlight groups (reel order).

    Non-gameplay shots are never members: transitions mark a pending
    boundary, reactions are skipped outright (their absence shows up in
    the R2 source-time gap instead).
    """
    groups: list[GroupedHighlight] = []
    members: list[GroupingInput] = []
    open_rule = "start"
    transition_pending = False
    prev_kept: GroupingInput | None = None

    def _is_replay(s: GroupingInput) -> bool:
        return s.speed_factor >= replay_min_speed_factor

    def _close_group() -> None:
        nonlocal members
        if not members:
            return
        gid = f"g{len(groups) + 1:02d}"
        groups.append(GroupedHighlight(
            id=gid,
            label=f"Highlight {len(groups) + 1}",
            shot_ids=[m.shot_id for m in members],
            boundary_rule=open_rule,
            boundary_confidence=_RULE_CONFIDENCE.get(open_rule, 1.0),
            reference_shot=_pick_reference(members, replay_min_speed_factor),
        ))
        members = []

    for shot in shots:
        if shot.kind == "transition":
            transition_pending = True
            continue
        if shot.kind != "gameplay":
            continue

        rule = None
        if members:
            if transition_pending:
                rule = "transition"
            elif (prev_kept is not None
                  and shot.source_start_s - prev_kept.source_end_s
                  > gap_boundary_s):
                rule = "gap"
            elif (shot.scale == "wide" and not _is_replay(shot)
                  and any(_is_replay(m) for m in members)):
                rule = "live_after_replay"

        if rule is not None:
            _close_group()
            open_rule = rule
        members.append(shot)
        transition_pending = False
        prev_kept = shot

    _close_group()
    return groups


def _pick_reference(
    members: list[GroupingInput],
    replay_min_speed_factor: float,
) -> str:
    """First wide real-time member, else the longest member.

    The reference anchors the group's sync timeline at offset 0; a wide
    live shot is the natural choice because every replay re-covers a
    subset of its time range.
    """
    for m in members:
        if m.scale == "wide" and m.speed_factor < replay_min_speed_factor:
            return m.shot_id
    longest = max(members, key=lambda m: m.source_end_s - m.source_start_s)
    return longest.shot_id

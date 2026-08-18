"""Uncovered-bend event proposals (sub-20cm campaign W5k).

The natural-motion validator flags frames where the resolved track bends
or accelerates with no covering event. Those violations are exactly where
an event is MISSING — a real touch or bounce the detectors' break search
did not surface. This module turns each violation cluster into a
second-chance :class:`BallEvent` candidate (touch when a joint is within
contact range of the bend, bounce when the bend happens at ground level),
which then flows through the standard auto-anchor gates (hard-evidence
window, flight gate, contact gap, reachability, operator suppression) —
a proposal is a question, the gates remain the answer.

Pure numpy; the validator math is shared with ``ball_eval`` so the
renderer, the validator, and the proposer can never disagree about what
counts as unnatural.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.ball_auto_events import BallEvent
from src.utils.ball_eval import naturalness_violations

_PROPOSAL_SCORE = 0.5
_CLUSTER_GAP_FRAMES = 3
_BOUNCE_MAX_Z_M = 0.5


@dataclass(frozen=True)
class _Frame:
    frame: int
    world_xyz: tuple[float, float, float] | None
    state: str


def propose_bend_events(
    *,
    world_by_frame: dict,
    state_by_frame: dict,
    event_frames,
    fps: float,
    player_ctx,
    contact_max_gap_m: float,
) -> tuple[BallEvent, ...]:
    """Return candidate events at unexplained bends of the resolved track."""
    frames = [
        _Frame(f, tuple(w) if w is not None else None,
               str(state_by_frame.get(f, "grounded")))
        for f, w in sorted(world_by_frame.items())
    ]
    violations = naturalness_violations(frames, event_frames, fps)
    if not violations:
        return ()
    # Cluster nearby violation frames; the peak-magnitude frame speaks.
    clusters: list[list] = []
    for v in sorted(violations, key=lambda v: v.frame):
        if clusters and v.frame - clusters[-1][-1].frame <= _CLUSTER_GAP_FRAMES:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    out: list[BallEvent] = []
    for cluster in clusters:
        peak = max(cluster, key=lambda v: abs(float(v.value)))
        f = int(peak.frame)
        w = world_by_frame.get(f)
        if w is None:
            continue
        ball = np.asarray(w, dtype=float)
        best = None
        for s in player_ctx.joints_at(f):
            if getattr(s, "world_xyz", None) is None:
                continue
            gap = float(np.linalg.norm(
                np.asarray(s.world_xyz, dtype=float) - ball))
            if gap <= contact_max_gap_m and (best is None or gap < best[0]):
                best = (gap, s.player_id, s.bone)
        if best is not None:
            out.append(BallEvent(
                frame=f, kind="touch", score=_PROPOSAL_SCORE,
                player_id=best[1], bone=best[2],
            ))
        elif float(ball[2]) <= _BOUNCE_MAX_Z_M:
            out.append(BallEvent(
                frame=f, kind="bounce", score=_PROPOSAL_SCORE,
            ))
    return tuple(out)


__all__ = ["propose_bend_events"]

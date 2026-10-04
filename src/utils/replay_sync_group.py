"""Per-group replay sync decisions (pure: no IO, no retiming).

Given each member's feet-on-pitch table (or why it has none) and the
operator alignments already on disk, estimate every non-reference member
against the reference, chaining through already-placed members when the
reference does not show the moment, and decide what the stage should do with
each result. See docs/superpowers/specs/2026-10-04-replay-speed.md.

Convention: a placement is ``(rate, offset)`` with ``ref = offset + rate *
shot_frame``; the sync map stores ``frame_offset = -offset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from src.utils.replay_speed import SpeedEstimate, estimate_speed

# A feet table, or "no_camera" / "no_tracks" when the shot cannot take part.
Feet = Mapping[int, np.ndarray]

DEFAULTS = {
    "enabled": True,
    "min_confidence": 0.5,
    "max_cost_m": 2.2,
    "auto_retime": True,
    "retime_tolerance": 0.08,
    "retime_min_confidence": 0.6,
}


@dataclass
class MemberResult:
    shot_id: str
    against: str = ""
    estimate: SpeedEstimate | None = None
    decision: str = "low_confidence"
    reason: str = ""
    placement: tuple[float, float] | None = None   # (rate, offset) onto the reference
    retime_rate: float | None = None


@dataclass
class GroupResult:
    group_id: str
    reference_shot: str
    members: list[MemberResult] = field(default_factory=list)


def _confident(est: SpeedEstimate | None, cfg: Mapping) -> bool:
    return (est is not None and est.confidence >= float(cfg["min_confidence"])
            and est.cost_m <= float(cfg["max_cost_m"]))


def _compose(est: SpeedEstimate, rate: float, offset: float) -> SpeedEstimate:
    """Pair estimate (member onto ``other``) + other's placement -> reference."""
    return replace(est, rate=rate * est.rate, offset=rate * est.offset + offset,
                   rate_first=rate * est.rate_first, rate_second=rate * est.rate_second)


def solve_group(
    group_id: str,
    reference: str,
    feet: Mapping[str, "Feet | str"],
    prior_manual: Mapping[str, tuple[float, int]],
    cfg: Mapping,
    only: str | None = None,
) -> GroupResult:
    """``prior_manual[sid] = (playback_rate, frame_offset)`` of operator
    alignments; those are never changed, only reported against."""
    cfg = {**DEFAULTS, **dict(cfg)}
    out = GroupResult(group_id, reference)
    ref_feet = feet.get(reference)
    placed: dict[str, tuple[float, float]] = {reference: (1.0, 0.0)}
    for sid, (rate, off) in prior_manual.items():
        placed.setdefault(sid, (float(rate), float(-off)))

    members = [MemberResult(s) for s in sorted(feet)
               if s != reference and (only is None or s == only)]

    def pair(m: MemberResult, other: str) -> SpeedEstimate | None:
        a, b = feet.get(other), feet.get(m.shot_id)
        if a is None or b is None or isinstance(a, str) or isinstance(b, str):
            return None
        return estimate_speed(a, b)

    for m in members:
        own = feet.get(m.shot_id)
        if isinstance(own, str):
            m.decision, m.reason = own, f"{m.shot_id}: {own.replace('_', ' ')}"
        elif isinstance(ref_feet, str):
            m.decision, m.reason = ref_feet, f"reference {reference}: {ref_feet.replace('_', ' ')}"
        else:
            m.against = reference
            m.estimate = pair(m, reference)
            if _confident(m.estimate, cfg):
                m.placement = (m.estimate.rate, m.estimate.offset)
                if m.shot_id not in prior_manual:
                    placed[m.shot_id] = m.placement

    # Chain: retry unplaced members against any placed member.
    open_ms = [m for m in members if m.placement is None
               and m.decision not in ("no_camera", "no_tracks")]
    progress = True
    while open_ms and progress:
        progress = False
        for m in list(open_ms):
            best = None
            for other, (rate, off) in placed.items():
                if other in (reference, m.shot_id):
                    continue
                est = pair(m, other)
                if _confident(est, cfg) and (best is None or est.confidence > best[0].confidence):
                    best = (est, other, rate, off)
            if best is not None:
                est, other, rate, off = best
                m.against, m.estimate = other, _compose(est, rate, off)
                m.placement = (m.estimate.rate, m.estimate.offset)
                if m.shot_id not in prior_manual:
                    placed[m.shot_id] = m.placement
                open_ms.remove(m)
                progress = True

    for m in members:
        _decide(m, prior_manual, cfg)
    out.members = members
    return out


def _decide(m: MemberResult, prior_manual: Mapping, cfg: Mapping) -> None:
    if m.shot_id in prior_manual:
        m.decision, m.reason = "kept_manual", "operator alignment is never overwritten"
        m.placement = m.retime_rate = None
        return
    if m.decision in ("no_camera", "no_tracks"):
        return
    est = m.estimate
    if est is None or m.placement is None:
        m.decision = "low_confidence"
        m.reason = ("too little overlapping player data" if est is None else
                    f"confidence {est.confidence:.2f}, cost {est.cost_m:.2f} m")
        return
    if est.ramp:
        m.decision = "ramp_not_applied"
        m.reason = f"speed ramp {est.rate_first:.2f}x -> {est.rate_second:.2f}x"
        m.placement = None
        return
    slow = est.rate < 1.0 - float(cfg["retime_tolerance"])
    if cfg["auto_retime"] and slow and est.confidence >= float(cfg["retime_min_confidence"]):
        m.decision, m.retime_rate = "applied_retimed", est.rate
    else:
        m.decision = "applied"

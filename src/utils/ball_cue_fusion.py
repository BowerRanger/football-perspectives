"""Combine per-cue ``CueEvidence`` candidate lists into fused event
proposals.

Policy (per the IC-D task brief): no single cue mints an event on its
own. A frame cluster becomes a fused event when it draws evidence from
>= 2 distinct cues within +-``frame_tol`` frames of each other, OR from
exactly 1 cue whose cluster frame lands within +-``frame_tol`` of an
existing auto/kinematic anchor frame (the ball stage's current
auto-anchor layer -- see ``ball_cue_common.load_auto_event_frames``).
This keeps the fused layer additive and conservative: it either
corroborates two independent single-camera signals, or nudges an
existing kinematic proposal with independent supporting evidence: it
never invents an event out of one noisy cue alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.ball_cue_types import CueEvidence

_FRAME_TOL = 2


@dataclass(frozen=True)
class FusedEvent:
    frame: int
    kind: str
    cues: tuple[str, ...]
    conf: float
    xyz: tuple[float, float, float] | None
    uv: tuple[float, float] | None
    support: str  # "cues" (>=2 distinct cues) or "cue+auto" (1 cue + existing auto anchor)


def _cluster(evidence: list[CueEvidence], *, frame_tol: int) -> list[list[CueEvidence]]:
    """Greedy 1-D clustering by frame: sorted, start a new cluster
    whenever the gap to the cluster's *last* member exceeds
    ``frame_tol`` (a chain-linked window, not a fixed-width bucket)."""
    clusters: list[list[CueEvidence]] = []
    for e in sorted(evidence, key=lambda e: e.frame):
        if clusters and e.frame - clusters[-1][-1].frame <= frame_tol:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


def _near_auto(frame: int, auto_frames: list[int], *, frame_tol: int) -> bool:
    return any(abs(frame - af) <= frame_tol for af in auto_frames)


def _pick_kind(members: list[CueEvidence]) -> str:
    # The net-energy cue is unambiguous about "goal_impact"; prefer it.
    # Otherwise take the most common non-generic kind, else fall back to
    # the generic "contact" label single-camera audio/blur cues use.
    kinds = [m.kind for m in members]
    if "goal_impact" in kinds:
        return "goal_impact"
    non_generic = [k for k in kinds if k != "contact"]
    if non_generic:
        return max(set(non_generic), key=non_generic.count)
    return "contact"


def _fuse_conf(members: list[CueEvidence]) -> float:
    # Noisy-OR across member confidences: 1 - prod(1 - c_i).
    p = 1.0
    for m in members:
        p *= (1.0 - max(0.0, min(1.0, m.conf)))
    return float(1.0 - p)


def fuse_cues(
    cue_evidence: dict[str, list[CueEvidence]],
    auto_event_frames: list[int],
    *,
    frame_tol: int = _FRAME_TOL,
) -> list[FusedEvent]:
    """``cue_evidence`` maps a cue-module name (e.g. ``"audio"``,
    ``"net"``, ``"blur"``) to its candidate ``CueEvidence`` list;
    ``auto_event_frames`` is the existing auto/kinematic anchor frame
    set. Returns fused events sorted by frame; a cluster with only one
    distinct ``CueEvidence.cue`` value and no nearby auto anchor is
    dropped entirely (not returned with a low confidence -- the policy
    is a hard gate, not a soft score)."""
    all_evidence: list[CueEvidence] = [
        e for items in cue_evidence.values() for e in items]
    clusters = _cluster(all_evidence, frame_tol=frame_tol)

    out: list[FusedEvent] = []
    for members in clusters:
        distinct_cues = sorted({m.cue for m in members})
        center_frame = int(round(sum(m.frame for m in members) / len(members)))
        if len(distinct_cues) >= 2:
            support = "cues"
        elif _near_auto(center_frame, auto_event_frames, frame_tol=frame_tol):
            support = "cue+auto"
        else:
            continue
        xyz = next((m.xyz for m in members if m.xyz is not None), None)
        uv = next((m.uv for m in members if m.uv is not None), None)
        out.append(FusedEvent(
            frame=center_frame, kind=_pick_kind(members),
            cues=tuple(distinct_cues), conf=_fuse_conf(members),
            xyz=xyz, uv=uv, support=support,
        ))
    return sorted(out, key=lambda f: f.frame)


__all__ = ["FusedEvent", "fuse_cues"]
